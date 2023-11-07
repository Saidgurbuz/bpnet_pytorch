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
        .permute(0, 2, 1)
    )
    assert outputs[0].shape == (
        batch_size, sequence_length, 2)  # profile head output
    assert outputs[1].shape == (batch_size, 2)  # total counts head output

    targets = []
    for i in range(num_tasks * 2):
        if i % 2 == 0:
            targets.append(torch.randint(2, (batch_size, sequence_length, 2)))
        else:
            targets.append(torch.rand(batch_size, 2))
    loss = custom_loss(outputs=outputs, targets=targets)
    loss.backward()
    assert loss > 0.0
    return
