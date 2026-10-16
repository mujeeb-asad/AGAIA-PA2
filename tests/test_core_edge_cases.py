#edge cases for batching and scoring that the public smoke tests do not cover

import math
from types import SimpleNamespace

import torch
from torch import nn

import submission_adapter as adapter
from alignment import batching, losses
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class BigramLM(nn.Module):
    #logits depend on the current token. the public toy model ignores its input, so an off by one in the
    #label shift would pass there and only fail here

    def __init__(self, vocab_size, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, generator=generator))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(logits=self.table[input_ids])


def _expected_logp(model, tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer.encode(prompt)
    ids = (prompt_ids + tokenizer.encode(response) + [tokenizer.eos_token_id])[:max_length]
    log_table = torch.log_softmax(model.table.detach(), dim=-1)
    return sum(log_table[ids[t - 1], ids[t]].item() for t in range(max(len(prompt_ids), 1), len(ids)))


def test_logprobs_match_a_manual_bigram_computation_with_mixed_lengths():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size)
    prompts = ["Q: ", "A much longer prompt: ", "x"]
    responses = ["yes", "no", "a longer response here"]

    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=40)

    expected = [_expected_logp(model, tokenizer, p, r, 40) for p, r in zip(prompts, responses)]
    torch.testing.assert_close(logps, torch.tensor(expected), rtol=1e-5, atol=1e-5)


def test_truncation_drops_eos_and_scores_only_surviving_tokens():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size)
    batch = adapter.build_sft_batch(tokenizer, ["ab: "], ["cdefgh"], max_length=7)

    assert batch["input_ids"].shape == (1, 7)
    assert batch["attention_mask"].all()
    assert tokenizer.eos_token_id not in batch["input_ids"][0].tolist()
    assert batch["labels"][0].tolist() == [-100] * 4 + tokenizer.encode("cde")

    logps = adapter.compute_response_logprobs(model, tokenizer, ["ab: "], ["cdefgh"], max_length=7)
    assert math.isclose(logps.item(), _expected_logp(model, tokenizer, "ab: ", "cdefgh", 7), rel_tol=1e-5)


def test_eos_survives_when_the_sequence_exactly_fits():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["ab: "], ["cd"], max_length=7)

    assert batch["labels"][0].tolist() == [-100] * 4 + tokenizer.encode("cd") + [tokenizer.eos_token_id]


def test_prompt_filling_the_window_gives_zero_logprob_and_finite_loss():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size)
    prompts, responses = ["abcdefgh"], ["xyz"]

    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=8)
    assert batch["labels"].eq(-100).all()
    assert adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=8).item() == 0.0
    assert adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=8).item() == 0.0


def test_eos_is_supervised_when_it_is_also_the_pad_token():
    tokenizer = ToyTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    batch = adapter.build_sft_batch(tokenizer, ["Q: ", "Q: "], ["a", "abc"], max_length=10)

    assert batch["labels"].ne(-100).sum(dim=1).tolist() == [2, 4]
    assert batch["labels"][0, 4] == tokenizer.eos_token_id
    assert batch["attention_mask"].sum(dim=1).tolist() == [5, 7]


def test_tokenizer_without_eos_or_pad_still_builds_a_batch():
    tokenizer = ToyTokenizer()
    tokenizer.eos_token_id = None
    tokenizer.pad_token_id = None
    batch = adapter.build_sft_batch(tokenizer, ["Q: ", "Q: "], ["a", "abc"], max_length=10)

    assert batch["labels"].ne(-100).sum(dim=1).tolist() == [1, 3]
    assert batch["attention_mask"].sum(dim=1).tolist() == [4, 6]


def test_row_order_is_preserved_under_permutation():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size)
    prompts = ["a: ", "bb: ", "ccc: ", "dddd: "]
    responses = ["one", "three", "five five", "7"]
    order = [2, 0, 3, 1]

    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=24)
    permuted = adapter.compute_response_logprobs(
        model, tokenizer, [prompts[i] for i in order], [responses[i] for i in order], max_length=24
    )
    torch.testing.assert_close(permuted, logps[order])


def test_padding_width_does_not_change_scores():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size)
    prompts, responses = ["a: ", "bb: "], ["short", "a longer one"]

    dynamic = batching.build_lm_batch(tokenizer, prompts, responses, max_length=64)
    padded = batching.build_lm_batch(tokenizer, prompts, responses, max_length=64, pad_to_max_length=True)

    assert dynamic["input_ids"].shape[1] < padded["input_ids"].shape[1] == 64
    torch.testing.assert_close(losses.response_logprobs(model, dynamic), losses.response_logprobs(model, padded))
    torch.testing.assert_close(losses.sft_loss(model, dynamic), losses.sft_loss(model, padded))


def test_sft_loss_is_the_token_mean_of_response_nll():
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size)
    prompts, responses = ["a: ", "bb: "], ["x", "a longer one"]

    loss = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=32)
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=32)
    token_count = sum(len(tokenizer.encode(r)) + 1 for r in responses)
    torch.testing.assert_close(loss, -logps.sum() / token_count)


def test_dpo_loss_from_logps_matches_the_closed_form():
    policy_chosen = torch.tensor([-1.0, -4.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -2.0, -2.5])
    reference_chosen = torch.tensor([-2.0, -3.0, -2.0])
    reference_rejected = torch.tensor([-2.5, -3.5, -1.0])
    beta = 0.3

    loss = adapter.compute_dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )
    gaps = [(-1.0 + 3.0) - (-2.0 + 2.5), (-4.0 + 2.0) - (-3.0 + 3.5), (-2.0 + 2.5) - (-2.0 + 1.0)]
    expected = sum(math.log(1 + math.exp(-beta * gap)) for gap in gaps) / 3
    assert math.isclose(loss.item(), expected, rel_tol=1e-6)


def test_dpo_loss_composes_logprobs_and_keeps_the_reference_out_of_the_graph():
    tokenizer = ToyTokenizer()
    policy = BigramLM(tokenizer.vocab_size, seed=1)
    reference = BigramLM(tokenizer.vocab_size, seed=2)
    prompts, chosen, rejected = ["Pick: ", "Q: "], ["good one", "yes"], ["bad", "no thanks"]

    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.1, max_length=32)
    expected = adapter.compute_dpo_loss_from_logps(
        adapter.compute_response_logprobs(policy, tokenizer, prompts, chosen, 32),
        adapter.compute_response_logprobs(policy, tokenizer, prompts, rejected, 32),
        adapter.compute_response_logprobs(reference, tokenizer, prompts, chosen, 32),
        adapter.compute_response_logprobs(reference, tokenizer, prompts, rejected, 32),
        beta=0.1,
    )
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert policy.table.grad.abs().sum() > 0
    assert reference.table.grad is None


def test_identical_policy_and_reference_start_at_log_two_with_zero_reward_accuracy():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    dpo_batch = batching.build_dpo_batch(tokenizer, ["Q: ", "Q: "], ["yes", "ok"], ["no", "nah"], max_length=16)

    logps = losses.dpo_forward(policy, reference, dpo_batch)
    margins = losses.dpo_logits(*logps, beta=0.1)

    #every pair is tied at the start, so the loss is log 2 and the strict r_w > r_l accuracy is 0, not 0.5
    torch.testing.assert_close(losses.dpo_loss_from_logps(*logps, beta=0.1), torch.log(torch.tensor(2.0)))
    assert margins.abs().max() == 0
    assert not margins.gt(0).any()
