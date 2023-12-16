import os
from model import BPNet, BPNetWithProteinEmbeddings
from dataset import DNADataset
from helpers import collate_fn, custom_loss, get_prot_embeddings_attention_mask
from torch.utils.data import DataLoader
import torch
import json
import torch.optim as optim
from Bio import SeqIO
import pyBigWig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, model, num_epochs, optimizer, loss_fn, val_loader=None, with_prot_emb=False, prot_embs=None, prot_att_mask=None):
    if val_loader:
        patience = 10
        best_val_loss = float("inf")
        counter = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, chip_seq_targets, bias_targets = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()

            if(with_prot_emb):
                chip_seq_outputs, bias_outputs = model(inputs, prot_embs, prot_att_mask)
            chip_seq_outputs, bias_outputs = model(inputs)
            loss = loss_fn((chip_seq_outputs, bias_outputs), (chip_seq_targets, bias_targets))
            print(
                "Epoch [{}/{}], Batch [{}/{}] train loss: {}".format(
                    epoch + 1, num_epochs, i+1, len(train_loader), loss
                )
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        if val_loader:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for index, data in enumerate(val_loader):
                    inputs, chip_seq_targets, bias_targets = data[0].to(device), data[1].to(device), data[2].to(device)
                    if(with_prot_emb):
                        chip_seq_outputs, bias_outputs = model(inputs, prot_embs, prot_att_mask)
                    chip_seq_outputs, bias_outputs = model(inputs)
                    loss = loss_fn((chip_seq_outputs, bias_outputs), (chip_seq_targets, bias_targets))
                    total_val_loss += loss
                avg_val_loss = total_val_loss / (index + 1)
            print(
                "Epoch [{}/{}], train loss: {}, validation loss: {}".format(
                    epoch + 1, num_epochs, avg_loss, avg_val_loss
                )
            )

            # for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping after {epoch+1} epochs without improvement.")
                    return model

        else:
            print(
                "Epoch [{}/{}], train loss: {}".format(
                    epoch + 1, num_epochs, avg_loss)
            )
    return model


if __name__ == "__main__":

    with_prot_emb = False
    prot_embeddings = None
    prot_att_mask = None
    sequence_length = 1000
    num_tasks = 4
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.004

    tasks = ['oct4', 'sox2', 'nanog', 'klf4']

    chromosomes = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
                   'chr16', 'chr17', 'chr18', 'chr19', 'chr2', 'chr3', 'chr4',
                   'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX', 'chrY']

    train_chrs = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
                  'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9']

    val_chrs = ['chr16', 'chr17', 'chr18']
    test_chrs = ['chr19', 'chrX', 'chrY']

    # Set the path to the dna_data directory
    dna_data_dir = 'dna_data/'

    dna_file = os.path.join(dna_data_dir, 'mm10.fa')

    # get labels
    pos_counts = []
    neg_counts = []
    for task in tasks:
        pos_counts.append(pyBigWig.open(os.path.join(dna_data_dir, f'{task}-counts.pos.bw')))
        neg_counts.append(pyBigWig.open(os.path.join(dna_data_dir, f'{task}-counts.neg.bw')))

    # dna_sequences
    with open(dna_file, 'r') as file:
        dna = list(SeqIO.parse(file, 'fasta'))

    train_sequences = list(filter(lambda c: c.id in train_chrs, dna))
    val_sequences = list(filter(lambda c: c.id in val_chrs, dna))
    test_sequences = list(filter(lambda c: c.id in test_chrs, dna))

    # targets
    train_targets = (train_chrs, pos_counts, neg_counts)
    val_targets = (val_chrs, pos_counts, neg_counts)
    test_targets = (test_chrs, pos_counts, neg_counts)

    train_dataset = DNADataset(
        train_sequences, train_targets, sequence_length, num_tasks)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = DNADataset(
        val_sequences, val_targets, sequence_length, num_tasks)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    test_dataset = DNADataset(
        test_sequences, test_targets, sequence_length, num_tasks)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if(with_prot_emb):
        protein_embedding_dim = 1280

        with open('prot_data/prot_idx_to_metadata.json', 'r') as f:
            prot_idx_to_metadata = json.load(f)
        prot_seq_lens = [item['dna_binding_domain_aa_seq_len'] for item in prot_idx_to_metadata.values()]
        prot_att_mask = get_prot_embeddings_attention_mask(prot_seq_lens)
        prot_embeddings = torch.load(
            'prot_data/dna_binding_domain_aa_prot_embeds.pt', map_location='cpu')
        model = BPNetWithProteinEmbeddings(protein_embedding_dim).to(device)
    else:
        model = BPNet(num_tasks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = custom_loss

    # train
    train(train_loader, model, num_epochs, optimizer, loss_fn, val_loader=val_loader, 
          with_prot_emb=with_prot_emb, prot_embs=prot_embeddings, prot_att_mask=prot_att_mask)
