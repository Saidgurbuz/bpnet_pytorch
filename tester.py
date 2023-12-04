from model import BPNet, BPNetWithProteinEmbeddings
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


def test_bpnet_model_with_protein_embeddings():
    batch_size = 16
    sequence_length = 1000
    num_channels = 4
    n_prots = 4  # number of proteins, corresponds to the number of tasks
    protein_embedding_dim = 256

    model = BPNetWithProteinEmbeddings(protein_embedding_dim)

    dna_seq = torch.randint(num_channels, (batch_size * sequence_length,))
    dna_batch = F.one_hot(
        dna_seq, num_classes=num_channels).type(torch.FloatTensor)
    prot_embeddings = torch.randn(n_prots, protein_embedding_dim)
    outputs = model(
        dna_seq=dna_batch.unsqueeze(0).view(batch_size, sequence_length, num_channels),
        prot_embeddings=prot_embeddings,
    )
    assert outputs[0].shape == (
        batch_size, sequence_length, 2)  # profile head output
    assert outputs[1].shape == (batch_size, 2)  # total counts head output

    chip_seq_targets = torch.randint(2, (batch_size, n_prots, sequence_length, 2))
    bias_targets = torch.rand(batch_size, n_prots, 2)

    loss = custom_loss(outputs=outputs, targets=(chip_seq_targets, bias_targets))
    loss.backward()
    assert loss > 0.0
