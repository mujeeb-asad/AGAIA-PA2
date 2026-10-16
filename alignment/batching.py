#response masked batches. sft and dpo both build their batches here, so the masking rules exist in one place only

import torch

IGNORE_INDEX = -100 #the label value cross entropy skips, it marks prompt and padding positions


def encode_example(tokenizer, prompt, response, max_length):
    #prompt and response are tokenized separately so the boundary between them is known exactly, no searching for it
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))

    #eos goes on the response before truncation, that is how the model learns where an answer ends
    if getattr(tokenizer, 'eos_token_id', None) is not None:
        response_ids.append(tokenizer.eos_token_id)

    #truncate from the right. a response that overflows max_length therefore loses its eos, which the manual allows
    input_ids = (prompt_ids + response_ids)[:max_length]

    #labels copy the response tokens and hide the prompt. masking is by position, not by token id -> qwen's pad token
    #is its eos token, so masking every pad id would also hide the real eos and the model would never learn to stop
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=False):
    #strict zip so a prompt list and a response list of different lengths fail loudly instead of dropping rows
    rows = [encode_example(tokenizer, prompt, response, max_length)
            for prompt, response in zip(prompts, responses, strict=True)]

    #training pads to the longest row to save compute, the grading adapter pads all the way to max_length
    width = max_length if pad_to_max_length else max((len(input_ids) for input_ids, _ in rows), default=0)

    #some tokenizers have no pad token. eos is the usual stand in, and the value never matters since it is masked out
    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, 'eos_token_id', None) or 0

    #start from an all padding batch, then write each row into the left side (right padding, prompt at position 0)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long) #(n, t)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long) #(n, t), 0 on padding
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long) #(n, t), -100 on prompt and padding
    for row, (row_input_ids, row_labels) in enumerate(rows): #enumerate keeps the rows in input order
        length = len(row_input_ids)
        input_ids[row, :length] = torch.tensor(row_input_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(row_labels, dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to_max_length=False):
    #the same prompts go into both halves, so chosen and rejected are scored against an identical context
    return {
        'chosen': build_lm_batch(tokenizer, prompts, chosen_responses, max_length, pad_to_max_length),
        'rejected': build_lm_batch(tokenizer, prompts, rejected_responses, max_length, pad_to_max_length),
    }
