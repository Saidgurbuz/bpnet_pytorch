import torch
from torch.distributions.multinomial import Multinomial
import torch.nn.functional as F


def one_hot_encoder(dna_sequence):
    labels = {"a": 0, "A": 0, "c": 1, "C": 1, "g": 2, "G": 2, "t": 3, "T": 3}
    labeled_dna_sequence = [
        labels[i] for i in dna_sequence
    ]  # ask for dna_sequence.lower()
    return torch.eye(4)[labeled_dna_sequence]


def multinomial_nll(logits, true_counts):
    """
    Compute the multinomial negative log-likelihood along the sequence (axis=1)
    and sum the values across all each channels

    Args:
            true_counts: observed count values (batch, seqlen, channels)
            logits: predicted logit values (batch, seqlen, channels)
    """
    logits_perm = logits.permute(0, 2, 1)
    true_counts_perm = true_counts.permute(0, 2, 1)

    counts_per_example = true_counts_perm.sum(dim=-1)

    # since pytorch multinomial does not support batches for multinomial we will use for loop
    batch_size = true_counts_perm.size(0)
    num_channels = true_counts_perm.size(1)
    loss = 0.0
    for i in range(batch_size):
        channel_loss = 0.0
        for j in range(num_channels):
            counts_per_example = true_counts_perm[i, j].sum().item()
            multinomial_dist = Multinomial(counts_per_example, logits=logits_perm[i, j])
            neg_log_likelihood = -multinomial_dist.log_prob(true_counts_perm[i, j])
            channel_loss += neg_log_likelihood.sum()
        loss += channel_loss / float(num_channels)

    loss /= float(batch_size)

    return loss


num_tasks = 4
loss_functions = [multinomial_nll, F.mse_loss] * num_tasks
loss_weights = [1, 10] * num_tasks


def custom_loss(outputs, targets):
    total_loss = 0.0
    for i, loss_fn in enumerate(loss_functions):
        total_loss += loss_weights[i] * loss_fn(outputs[i], targets[i])
    return total_loss
