import torch
from torch.utils.data import Dataset
from helpers import one_hot_encoder
from helpers import seq_to_one_hot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # get input
        seq_input = seq_to_one_hot(str(chr.seq[start:end])).to(device)

        # get profile shape and bias targets
        chip_seq_target = torch.zeros(self.num_tasks, self.sequence_length, 2).to(device)
        bias_target = torch.zeros(self.num_tasks, 2).to(device)
        name_chr = self.targets[0][chr_idx]
        for task in range(self.num_tasks):
            # get profile shape for both positive and negative strands
            ps_pos = torch.Tensor(
                self.targets[1][task].values(name_chr, start, end))
            ps_neg = torch.Tensor(
                self.targets[2][task].values(name_chr, start, end))
            ps_pos[ps_pos != ps_pos] = 0
            ps_neg[ps_neg != ps_neg] = 0
            chip_seq_target[task] = torch.stack([ps_pos, ps_neg], dim=1)

            # get total counts
            tc_pos = ps_pos.sum()
            tc_neg = ps_neg.sum()
            bias_target[task] = torch.Tensor([tc_pos, tc_neg])
        return seq_input, chip_seq_target, bias_target
