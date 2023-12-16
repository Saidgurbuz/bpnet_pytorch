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
            counts_per_example = int(true_counts_perm[i, j].sum().item())
            multinomial_dist = Multinomial(
                counts_per_example, logits=logits_perm[i, j], validate_args=False)
            neg_log_likelihood = - \
                multinomial_dist.log_prob(true_counts_perm[i, j])
            channel_loss += neg_log_likelihood.sum()
        loss += channel_loss / float(num_channels)

    loss /= float(batch_size)

    return loss


num_tasks = 4
loss_functions = [multinomial_nll, F.mse_loss] * num_tasks
chip_seq_loss_weight = 1.0
bias_loss_weight = 10.0


def custom_loss(outputs, targets):
    total_loss = 0.0
    # change view to get task dimension first, batch dimension second
    chip_seq_targets = targets[0].permute(1, 0, 2, 3)
    bias_targets = targets[1].permute(1, 0, 2)
    chip_seq_outputs = outputs[0].permute(1, 0, 2, 3)
    bias_outputs = outputs[1].permute(1, 0, 2)

    for i in range(bias_targets.size(0)):
        total_loss += multinomial_nll(chip_seq_outputs[i], chip_seq_targets[i]) * chip_seq_loss_weight
        total_loss += F.mse_loss(bias_outputs[i], torch.log(1 + bias_targets[i])) * bias_loss_weight

    return total_loss

    # TODO: alternative once we figure out how to parallelize multinomial_nll over batch dimension
    # chip_seq_outputs = torch.stack(outputs[0::2])
    # bias_outputs = torch.stack(outputs[1::2])
    #
    # chip_seq_loss = 0.
    # bias_loss = F.mse_loss(bias_outputs, bias_targets)
    #
    # loss = chip_seq_loss_weight * chip_seq_loss + bias_loss_weight * bias_loss


def collate_fn(data):
    """
    Collates the data into a batch
    :param data: this is a list of outputs of __getitem__ function in the dataset
    :return: Tuple of batched data
    """
    # assume the dna seqs are the first item in the tuple
    dna_seqs = torch.stack([item[0] for item in data])

    # get chip-seq targets
    # shape: (batch_size, 1000, 2)
    chip_seq_targets = torch.stack([item[1] for item in data])

    # get bias targets
    # shape: (batch_size, 1, 2)
    bias_targets = torch.stack([item[2] for item in data])

    return dna_seqs, chip_seq_targets, bias_targets


def get_prot_embeddings_attention_mask(prot_seq_lens):
    """
    Returns the attention mask for the protein embeddings
    :param prot_seq_lens: list of protein sequence lengths
    :return: attention mask for the protein embeddings
    """
    max_prot_seq_len = max(prot_seq_lens)
    prot_att_mask = torch.zeros(len(prot_seq_lens), max_prot_seq_len)
    for i, prot_seq_len in enumerate(prot_seq_lens):
        prot_att_mask[i, :prot_seq_len] = 1
    return prot_att_mask
