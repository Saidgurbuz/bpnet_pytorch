from model import BPNet
from dataset_v2 import DNADataset
from helpers import collate_fn
from helpers import custom_loss
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, num_epochs, optimizer, loss_fn, val_loader=None):
    if val_loader:
        patience = 20
        best_val_loss = float("inf")
        counter = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, chip_seq_targets, bias_targets = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, (chip_seq_targets, bias_targets))
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
                    outputs = model(inputs)
                    loss = loss_fn(outputs, (chip_seq_targets, bias_targets))
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

    sequence_length = 1000
    num_tasks = 4
    batch_size = 2048
    num_epochs = 100
    learning_rate = 0.004
    tasks = ['oct4', 'sox2', 'nanog', 'klf4']

    train_inputs = torch.load('train_inputs.pt')
    train_targets_chip_seq = torch.load('train_targets_chip_seq.pt')
    train_targets_bias = torch.load('train_targets_bias.pt')

    val_inputs = torch.load('val_inputs.pt')
    val_targets_chip_seq = torch.load('val_targets_chip_seq.pt')
    val_targets_bias = torch.load('val_targets_bias.pt')

    test_inputs = torch.load('test_inputs.pt')
    test_targets_chip_seq = torch.load('test_targets_chip_seq.pt')
    test_targets_bias = torch.load('test_targets_bias.pt')


    train_dataset = DNADataset(
        train_inputs, train_targets_chip_seq, train_targets_bias, sequence_length, num_tasks)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    val_dataset = DNADataset(
        val_inputs, val_targets_chip_seq, val_targets_bias, sequence_length, num_tasks)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    test_dataset = DNADataset(
        test_inputs, test_targets_chip_seq, test_targets_bias, sequence_length, num_tasks)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    model = BPNet(num_tasks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = custom_loss

    # train
    train(train_loader, model, num_epochs, optimizer,
          loss_fn, val_loader=val_loader)
