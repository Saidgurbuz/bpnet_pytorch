from model_try import custom_loss
from model_try import BPNet
from dataset import DNADataset
from helpers import custom_loss
from torch.utils.data import DataLoader
import torch
import torch.optim as optim


def train(data_loader, model, num_epochs, optimizer, loss_fn, validation_loader=None):
    if validation_loader:
        patience = 5
        best_val_loss = float("inf")
        counter = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs, targets = data

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        if validation_loader:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for index, data in enumerate(validation_loader):
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

    # TODO
    # dna_sequences = ...  will be implemented
    # targets = ... will be implemented

    dna_dataset = DNADataset(dna_sequences, targets, sequence_length, num_tasks)
    data_loader = DataLoader(dna_dataset, batch_size=batch_size, shuffle=True)

    model = BPNet(num_tasks)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = custom_loss

    # train
    train(data_loader, model, num_epochs, optimizer, loss_fn)
