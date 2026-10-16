#the stable interface the private grader calls. the six signatures are the staff's and must not change, each
#function is only a thin wrapper around the implementation in alignment/

from alignment import batching, losses


def build_sft_batch(tokenizer, prompts, responses, max_length):
    #padded to max_length here, the manual's contract reads "pad shorter examples ... to max_length"
    return batching.build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    #(n,) summed log probability of each response, in input row order
    return losses.response_logprobs(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    #scalar, the mean over all unmasked response tokens in the batch
    return losses.sft_loss(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    #{'chosen': batch, 'rejected': batch}, each with input_ids, attention_mask and labels
    return batching.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length,
                                    pad_to_max_length=True)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    #scalar batch mean of -log sigmoid(beta * (policy gap - reference gap))
    return losses.dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                                      reference_chosen_logps, reference_rejected_logps, beta)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    #the reference passes run under no_grad inside dpo_forward, so only the policy receives gradients
    dpo_batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return losses.dpo_loss(policy_model, reference_model, dpo_batch, beta)
