#response only log probabilities, and the sft and dpo objectives built on top of them

import torch
import torch.nn.functional as F

from alignment.batching import IGNORE_INDEX


def token_logprobs(model, batch):
    #the batch is built on cpu, so it is moved to wherever this model lives (policy and reference can be on two gpus)
    device = next(model.parameters()).device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits #(n, t, vocab)

    #causal shift -> the logits at position i predict the token at position i + 1
    shift_logits = logits[:, :-1, :].float() #(n, t - 1, vocab), float32 so the log softmax is exact under fp16/bf16
    shift_labels = labels[:, 1:] #(n, t - 1)

    #cross entropy with no reduction is -log p(label) per position, and it returns exactly 0 wherever the label is -100
    logps = -F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, ignore_index=IGNORE_INDEX, reduction='none')
    return logps, shift_labels.ne(IGNORE_INDEX) #both (n, t - 1), the mask is True on the scored response tokens


def response_logprobs(model, batch):
    logps, _ = token_logprobs(model, batch)
    return logps.sum(dim=-1) #(n,) log pi(y | x), prompt and padding positions contribute 0


def sft_loss(model, batch):
    #reduction used everywhere in this repo -> mean over every response token in the batch (not a mean of row sums)
    logps, mask = token_logprobs(model, batch)
    return -logps.sum() / mask.sum().clamp_min(1) #clamp so a batch with no response tokens gives 0, not nan


def dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    #how much more the policy prefers chosen over rejected than the frozen reference already did
    policy_gap = policy_chosen_logps - policy_rejected_logps #(n,)
    #the reference may sit on a second gpu, its numbers are brought over to the policy's device
    reference_gap = (reference_chosen_logps - reference_rejected_logps).to(policy_gap.device) #(n,)
    #this quantity is also the reward margin r_w - r_l from the manual's metrics section
    return beta * (policy_gap - reference_gap) #(n,)


def dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    logits = dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta)
    #-log sigmoid, averaged over the batch. logsigmoid stays finite where log(sigmoid(x)) would underflow
    return -F.logsigmoid(logits).mean()


def dpo_forward(policy_model, reference_model, dpo_batch):
    #four scores per pair. the two policy passes carry gradients
    policy_chosen_logps = response_logprobs(policy_model, dpo_batch['chosen'])
    policy_rejected_logps = response_logprobs(policy_model, dpo_batch['rejected'])
    #the two reference passes do not, the reference is the fixed anchor and must never be updated
    with torch.no_grad():
        reference_chosen_logps = response_logprobs(reference_model, dpo_batch['chosen'])
        reference_rejected_logps = response_logprobs(reference_model, dpo_batch['rejected'])
    return policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps


def dpo_loss(policy_model, reference_model, dpo_batch, beta):
    return dpo_loss_from_logps(*dpo_forward(policy_model, reference_model, dpo_batch), beta)
