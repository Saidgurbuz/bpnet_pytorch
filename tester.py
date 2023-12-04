from model import BPNet
from helpers import custom_loss
import torch.nn.functional as F
import torch


def test_bpnet_model():
    batch_size = 16
    sequence_length = 1000
    num_channels = 4
    num_tasks = 4

    model = BPNet(num_tasks)

    dna_seq = torch.randint(num_channels, (batch_size * sequence_length,))
    dna_batch = F.one_hot(
        dna_seq, num_classes=num_channels).type(torch.FloatTensor)
    outputs = model(
        dna_batch.unsqueeze(0)
        .view(batch_size, sequence_length, num_channels)
    )
    assert outputs[0].shape == (
        batch_size, sequence_length, 2)  # profile head output
    assert outputs[1].shape == (batch_size, 2)  # total counts head output

    chip_seq_targets = torch.randint(2, (batch_size, num_tasks, sequence_length, 2))
    bias_targets = torch.rand(batch_size, num_tasks, 2)

    loss = custom_loss(outputs=outputs, targets=(chip_seq_targets, bias_targets))
    loss.backward()
    assert loss > 0.0
    return
