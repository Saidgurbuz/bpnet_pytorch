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

        start = idx * self.sequence_length
        end = start + self.sequence_length
        input = seq_to_one_hot(str(chr[start:end]))
        # TODO will configure getting target (actually multiple target since we have multiple tasks and multiple head)
        
        
        target = self.targets[idx]
        input_tensor = torch.tensor(input, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return input_tensor, target_tensor
