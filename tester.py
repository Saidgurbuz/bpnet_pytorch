import json

from model import BPNet, BPNetWithProteinEmbeddings
from helpers import custom_loss, get_prot_embeddings_attention_mask
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
    chip_seq_outputs, bias_outputs = model(
        dna_batch.unsqueeze(0)
        .view(batch_size, sequence_length, num_channels)
    )
    assert chip_seq_outputs.shape == (
        batch_size, num_tasks, sequence_length, 2)  # profile head output
    assert bias_outputs.shape == (batch_size, num_tasks, 2)  # total counts head output

    chip_seq_targets = torch.randint(2, (batch_size, num_tasks, sequence_length, 2))
    bias_targets = torch.rand(batch_size, num_tasks, 2)

    loss = custom_loss(outputs=(chip_seq_outputs, bias_outputs), targets=(chip_seq_targets, bias_targets))
    loss.backward()
    assert loss > 0.0


def test_bpnet_model_with_protein_embeddings():
    batch_size = 16
    sequence_length = 1000
    num_channels = 4
    n_prots = 4  # number of proteins, corresponds to the number of tasks
    n_amino_acids = 10  # max nr of amino acids in a protein
    protein_embedding_dim = 1280

    model = BPNetWithProteinEmbeddings(protein_embedding_dim)

    dna_seq = torch.randint(num_channels, (batch_size * sequence_length,))
    dna_batch = F.one_hot(
        dna_seq, num_classes=num_channels).type(torch.FloatTensor)
    prot_embeddings = torch.randn(n_prots, n_amino_acids, protein_embedding_dim)
    chip_seq_outputs, bias_outputs = model(
        dna_seq=dna_batch.unsqueeze(0).view(batch_size, sequence_length, num_channels),
        prot_embeddings=prot_embeddings,
    )
    assert chip_seq_outputs.shape == (
        batch_size, n_prots, sequence_length, 2)  # profile head output
    assert bias_outputs.shape == (batch_size, n_prots, 2)  # total counts head output

    chip_seq_targets = torch.randint(2, (batch_size, n_prots, sequence_length, 2))
    bias_targets = torch.rand(batch_size, n_prots, 2)

    loss = custom_loss(outputs=(chip_seq_outputs, bias_outputs), targets=(chip_seq_targets, bias_targets))
    loss.backward()
    assert loss > 0.0

    with open('prot_data/prot_idx_to_metadata.json', 'r') as f:
        prot_idx_to_metadata = json.load(f)

    prot_seq_lens = [item['dna_binding_domain_aa_seq_len'] for item in prot_idx_to_metadata.values()]
    prot_att_mask = get_prot_embeddings_attention_mask(prot_seq_lens)
    prot_embeddings = torch.load(
        'prot_data/dna_binding_domain_aa_prot_embeds.pt', map_location='cpu')

    chip_seq_outputs, bias_outputs = model(
        dna_seq=dna_batch.unsqueeze(0).view(batch_size, sequence_length, num_channels),
        prot_embeddings=prot_embeddings,
        prot_attention_mask=prot_att_mask,
    )
    assert chip_seq_outputs.shape == (
        batch_size, n_prots, sequence_length, 2)  # profile head output
    assert bias_outputs.shape == (batch_size, n_prots, 2)  # total counts head output

    chip_seq_targets = torch.randint(2, (batch_size, n_prots, sequence_length, 2))
    bias_targets = torch.rand(batch_size, n_prots, 2)

    loss = custom_loss(outputs=(chip_seq_outputs, bias_outputs), targets=(chip_seq_targets, bias_targets))
    loss.backward()
    assert loss > 0.0
