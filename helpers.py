import torch
from torch.distributions.multinomial import Multinomial
import torch.nn.functional as F



ONE_HOT_EMBED = torch.zeros(1000, 4)
ONE_HOT_EMBED[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
ONE_HOT_EMBED[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
ONE_HOT_EMBED[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("x")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("o")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("p")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])
ONE_HOT_EMBED[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
ONE_HOT_EMBED[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
ONE_HOT_EMBED[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("X")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("O")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("P")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])
ONE_HOT_EMBED[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

def seq_to_one_hot(seq: str) -> torch.Tensor:
    seq_chrs = torch.tensor(list(map(ord, list(seq))), dtype=torch.long)
    return ONE_HOT_EMBED[seq_chrs]



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
