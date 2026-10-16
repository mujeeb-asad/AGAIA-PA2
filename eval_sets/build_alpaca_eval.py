# task 2's fixed alpacaeval subset -> 300 of the 805 alpacaeval prompts, sampled with seed 42
#
# the subset is committed next to this file and every generation run reads that copy, so this never has to run again.
# by default it rebuilds the subset in memory and checks it against the committed file, which shows the file is
# exactly what this code produces. it downloads one small json from the hugging face hub and needs no gpu:
#     python eval_sets/build_alpaca_eval.py
#
# --write builds the file from scratch instead, for another size or seed:
#     python eval_sets/build_alpaca_eval.py --write --size 300 --seed 42

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=300) #the manual's default
parser.add_argument('--seed', type=int, default=42) #the same seed as every other run
parser.add_argument('--write', action='store_true')
args = parser.parse_args()

#the official alpacaeval prompt file, the same 805 instructions the public leaderboard uses
path = hf_hub_download('tatsu-lab/alpaca_eval', 'alpaca_eval.json', repo_type='dataset')
rows = json.loads(Path(path).read_text(encoding='utf-8'))

#its own seeded generator, so the sample depends on the seed alone. source_index keeps the link back to the full set,
#and dataset is the part of alpacaeval the prompt came from (helpful_base, koala, oasst, selfinstruct, vicuna)
indices = random.Random(args.seed).sample(range(len(rows)), args.size)
subset = [{'id': f'alpaca-eval-{args.seed}-{position:04d}', 'instruction': rows[index]['instruction'],
           'dataset': rows[index]['dataset'], 'source_index': index}
          for position, index in enumerate(indices)]

output_path = Path(__file__).with_name(f'alpaca_eval_{args.size}_seed{args.seed}.json')
if args.write:
    output_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(subset)} of {len(rows)} prompts -> {output_path}')
else:
    committed = json.loads(output_path.read_text(encoding='utf-8'))
    assert committed == subset, f'{output_path.name} differs from a fresh build with seed {args.seed}'
    print(f'{output_path.name} matches a fresh build: {len(subset)} of {len(rows)} prompts, seed {args.seed}')
