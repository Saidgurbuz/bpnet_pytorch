import torch
from torch.utils.data import Dataset
from helpers import one_hot_encoder


class DNADataset(Dataset):
    def __init__(self, inputs, targets, sequence_length, num_tasks):
        self.inputs = inputs
        self.targets = targets
        self.sequence_length = sequence_length
        self.num_tasks = num_tasks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = start + self.sequence_length
        input = one_hot_encoder(str(self.inputs[start:end]))
        # TODO will configure getting target (actually multiple target since we have multiple tasks and multiple head)
        target = self.targets[idx]
        input_tensor = torch.tensor(input, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return input_tensor, target_tensor
