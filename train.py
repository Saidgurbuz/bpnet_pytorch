from model_try import custom_loss
from model_try import BPNet
from dataset import DNADataset
from helpers import custom_loss
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from Bio import SeqIO
from Bio import Seq
import pyBigWig


def train(train_loader, model, num_epochs, optimizer, loss_fn, val_loader=None):
    if val_loader:
        patience = 5
        best_val_loss = float("inf")
        counter = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, targets = data

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        if val_loader:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for index, data in enumerate(val_loader):
                    inputs, targets = data
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
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
                    print(f"Early stopping after {epoch+1} epochs without improvement.")
                    return model

        else:
            print(
                "Epoch [{}/{}], train loss: {}".format(epoch + 1, num_epochs, avg_loss)
            )
    return model


if __name__ == "__main__":
    # defining hyperparameters
    sequence_length = 1000
    num_tasks = 4
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.004

    tasks = ['Oct4', 'Sox2', 'Nanog', 'Klf4']
    # TODO
    choromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 
                    'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14',
                    'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chrX', 'chrY']
    
    train_chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                  'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15']
    val_chrs = ['chr16', 'chr17', 'chr18']
    test_chrs = ['chr19', 'chrX', 'chrY']

    dna_file = 'mm10.fa'

    # get labels
    pos_counts = []
    neg_counts = []
    for task in tasks:
        pos_counts.append(pyBigWig.open(f'{task}_counts.pos.bw'))
        neg_counts.append(pyBigWig.open(f'{task}_counts.neg.bw'))

    # dna_sequences = ...  will be implemented
    with open(dna_file, 'r') as file:
        dna = list(SeqIO.parse(file, 'fasta'))
    # dna_sequences = Seq('')
    # with open(dna_file, 'r') as dna:
    #     for sequence in SeqIO.parse(dna, 'fasta'):
    #         dna_sequences += sequence.seq
    train_sequences = list(filter(lambda c: c.id in train_chrs, dna))
    val_sequences = list(filter(lambda c: c.id in val_chrs, dna))
    test_sequences = list(filter(lambda c: c.id in test_chrs, dna))

    # targets = ... will be implemented
    train_targets = (train_chrs, pos_counts, neg_counts)
    val_targets = (val_chrs, pos_counts, neg_counts)
    test_targets = (test_chrs, pos_counts, neg_counts)

    train_dataset = DNADataset(train_sequences, train_targets, sequence_length, num_tasks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = DNADataset(train_sequences, val_targets, sequence_length, num_tasks)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = DNADataset(train_sequences, test_targets, sequence_length, num_tasks)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model = BPNet(num_tasks)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = custom_loss

    # train
    train(train_loader, model, num_epochs, optimizer, loss_fn, val_loader)
