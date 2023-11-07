import torch
from torch.utils.data import Dataset
from helpers import one_hot_encoder
from helpers import seq_to_one_hot


class DNADataset(Dataset):
    def __init__(self, inputs, targets, sequence_length, num_tasks):
        self.inputs = inputs
        self.targets = targets
        self.sequence_length = sequence_length
        self.num_tasks = num_tasks

    def __len__(self):
        total_samples = 0
        for chr in self.inputs:
            total_samples += len(chr.seq) // self.sequence_length
        return total_samples

    def __getitem__(self, idx):

        # get the current chromosome
        chr_idx = 0
        offset = idx * self.sequence_length
        while offset >= len(self.inputs[chr_idx].seq):
            offset -= len(self.inputs[chr_idx].seq)
            chr_idx += 1
        chr = self.inputs[chr_idx]

        start = offset
        end = start + self.sequence_length
        input = seq_to_one_hot(str(chr.seq[start:end]))
        # TODO will configure getting target (actually multiple target since we have multiple tasks and multiple head)
        target = torch.zeros(self.num_tasks * 2, self.sequence_length, 2)

        name_chr = self.targets[0][chr_idx]
        for task in range(self.num_tasks):
            # get profile shape for both positive and negative strands
            ps_pos = torch.Tensor(
                self.targets[1][task].values(name_chr, start, end))
            ps_neg = torch.Tensor(
                self.targets[2][task].values(name_chr, start, end))
            ps_pos[ps_pos != ps_pos] = 0
            ps_neg[ps_neg != ps_neg] = 0
            target[task * 2] = torch.stack([ps_pos, ps_neg], dim=1)

            # get total counts
            tc_pos = ps_pos.sum()
            tc_neg = ps_neg.sum()
            target[task * 2 + 1][0] = torch.Tensor([tc_pos, tc_neg])

        return input, target
