from torch.utils.data import Dataset


class DNADataset(Dataset):
    def __init__(self, inputs, targets_chip_seq, targets_bias, sequence_length, num_tasks):
        self.inputs = inputs
        self.targets_chip_seq = targets_chip_seq
        self.targets_bias = targets_bias
        self.sequence_length = sequence_length
        self.num_tasks = num_tasks

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets_chip_seq[idx], self.targets_bias[idx]
