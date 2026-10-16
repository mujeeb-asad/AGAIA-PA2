#the prompt format, and the conversion of each raw dataset into single turn rows

import hashlib
import json
from pathlib import Path

#one template for sft, dpo and generation. plain text rather than chatml -> the base checkpoint was never trained on
#the chatml control tokens, while <|endoftext|> is already how it ends a document
PROMPT_TEMPLATE = '### Instruction:\n{instruction}\n\n### Response:\n'


def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def ultrachat_row(row):
    #ultrachat conversations are multi turn, only the first user -> assistant exchange is kept
    messages = row['messages']
    if len(messages) < 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
        return None
    return {'id': row['prompt_id'],
            'instruction': messages[0]['content'].strip(),
            'response': messages[1]['content'].strip()}


def split_hh_transcript(text):
    #hh-rlhf stores a whole transcript as one string -> '\n\nHuman: ...\n\nAssistant: ...'
    #exactly one of each marker means single turn, anything else is multi turn and gets dropped
    if text.count('\n\nHuman:') != 1 or text.count('\n\nAssistant:') != 1:
        return None
    head, response = text.split('\n\nAssistant:')
    return head.split('\n\nHuman:')[1].strip(), response.strip() #(instruction, response)


def hh_row(row):
    chosen = split_hh_transcript(row['chosen'])
    rejected = split_hh_transcript(row['rejected'])
    #both sides must be single turn and must answer the same instruction
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    #hh-rlhf has no id column, so the id is a hash of the content. that also makes duplicate rows collide
    digest = hashlib.sha1('\x1f'.join([chosen[0], chosen[1], rejected[1]]).encode('utf-8')).hexdigest()[:12]
    return {'id': f'hh-{digest}', 'instruction': chosen[0], 'chosen': chosen[1], 'rejected': rejected[1]}


def ultrafeedback_row(row):
    #the binarized set stores each side as a [user, assistant] message list, longer lists would be multi turn
    if len(row['chosen']) != 2 or len(row['rejected']) != 2:
        return None
    return {'id': row['prompt_id'],
            'instruction': row['prompt'].strip(),
            'chosen': row['chosen'][1]['content'].strip(),
            'rejected': row['rejected'][1]['content'].strip()}


def is_usable(example, tokenizer, max_prompt_tokens):
    #an sft row has one response, a dpo row has a chosen and a rejected one
    responses = [example[key] for key in ('response', 'chosen', 'rejected') if key in example]
    #nothing to learn from an empty field, or from a pair whose two answers are the same text
    if not example['instruction'] or not all(responses) or len(set(responses)) != len(responses):
        return False
    #long prompts are dropped rather than cut -> a cut prompt would lose the '### Response:' marker at its end, and
    #the cap guarantees every row keeps at least half of the 512 token window for its response
    prompt_ids = tokenizer.encode(format_prompt(example['instruction']), add_special_tokens=False)
    return len(prompt_ids) <= max_prompt_tokens


def select_examples(dataset, convert, tokenizer, count, seed, max_prompt_tokens):
    #seeded shuffle, then take the first `count` rows that convert cleanly. the same seed always gives the same subset
    selected, seen_ids = [], set()
    for row in dataset.shuffle(seed=seed):
        example = convert(row)
        if example is None or example['id'] in seen_ids or not is_usable(example, tokenizer, max_prompt_tokens):
            continue
        seen_ids.add(example['id'])
        selected.append(example)
        if len(selected) == count:
            break
    return selected
