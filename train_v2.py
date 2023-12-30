from model import BPNet, BPNetWithProteinEmbeddings
from dataset_v2 import DNADataset
from helpers import collate_fn, custom_loss, get_prot_embeddings_attention_mask
from torch.utils.data import DataLoader
import torch
import json
import torch.optim as optim
from torchmetrics import PearsonCorrCoef
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, num_epochs, optimizer, loss_fn, val_loader=None, with_prot_emb=False, prot_embs=None, prot_att_mask=None, model_path='best_model.pt'):
    """Train the model and save the best model based on validation loss."""
    pearson_corr = PearsonCorrCoef(2000).to(device)
    train_losses = []
    train_corrs = []

    if val_loader:
        patience = 100
        best_val_loss = float("inf")
        counter = 0
        val_losses = []
        val_corrs = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_corr = 0.0
        for i, data in enumerate(train_loader):
            inputs, chip_seq_targets, bias_targets = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()

            if(with_prot_emb):
                chip_seq_outputs, bias_outputs = model(inputs, prot_embs.to(device), prot_att_mask.to(device))
            else:
                chip_seq_outputs, bias_outputs = model(inputs)
            loss = loss_fn((chip_seq_outputs, bias_outputs), (chip_seq_targets, bias_targets))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Compute Pearson correlation coefficient
            d0, d1, d2, d3 = chip_seq_outputs.shape
            corr = pearson_corr(chip_seq_outputs.view(d0 * d1, d2 * d3), chip_seq_targets.view(d0 * d1, d2 * d3))
            total_corr += corr.mean().item()
            print(
                "Epoch [{}/{}], Batch [{}/{}] train loss: {:.2f}, train corr: {:.5f}".format(
                    epoch + 1, num_epochs, i+1, len(train_loader), loss, corr.mean()
                )
            )

        avg_loss = total_loss / (i + 1)
        avg_corr = total_corr / (i + 1)

        train_losses.append(avg_loss)
        train_corrs.append(avg_corr)

        if val_loader:
            val_loss, val_corr = test(val_loader, model, loss_fn, with_prot_emb, prot_embs, prot_att_mask, pearson_corr)
            val_losses.append(val_loss)
            val_corrs.append(val_corr)
            print(
                "Epoch [{}/{}], validation loss: {:.2f}, validation correlation: {:.5f}".format(
                    epoch + 1, num_epochs, val_loss, val_corr
                )
            )
            # for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save the model with the best validation loss
                torch.save(model.state_dict(), model_path)
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping after {epoch+1} epochs without improvement.")
                    return model, train_losses, train_corrs, val_losses, val_corrs, pearson_corr

        else:
            print(
                "Epoch [{}/{}], train loss: {}".format(
                    epoch + 1, num_epochs, avg_loss)
            )
    return model, train_losses, train_corrs, val_losses, val_corrs, pearson_corr

def test(data_loader, model, loss_fn, with_prot_emb=False, prot_embs=None, prot_att_mask=None, pearson_corr=None):
    """Test the model on the test or validation set."""
    model.eval()

    total_loss = 0.0
    total_corr = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, chip_seq_targets, bias_targets = data[0].to(device), data[1].to(device), data[2].to(device)
            if with_prot_emb:
                chip_seq_outputs, bias_outputs = model(inputs, prot_embs.to(device), prot_att_mask.to(device))
            else:
                chip_seq_outputs, bias_outputs = model(inputs)

            loss = loss_fn((chip_seq_outputs, bias_outputs), (chip_seq_targets, bias_targets))
            total_loss += loss

            # Compute Pearson correlation coefficient
            d0, d1, d2, d3 = chip_seq_outputs.shape
            corr = pearson_corr(chip_seq_outputs.view(d0 * d1, d2 * d3), chip_seq_targets.view(d0 * d1, d2 * d3))
            total_corr += corr.mean().item()

        avg_loss = total_loss / (i + 1)
        avg_corr = total_corr / (i + 1)
    return avg_loss, avg_corr


if __name__ == "__main__":

    with_prot_emb = False
    prot_embs = None
    prot_att_mask = None
    model_path = 'best_model.pt'
    sequence_length = 1000
    num_tasks = 4
    batch_size = 1024
    num_epochs = 100
    learning_rate = 0.004
    tasks = ['oct4', 'sox2', 'nanog', 'klf4']

    dna_data_dir = 'dna_data/'

    train_inputs = torch.load(f'{dna_data_dir}train_inputs.pt')
    train_targets_chip_seq = torch.load(f'{dna_data_dir}train_targets_chip_seq.pt')
    train_targets_bias = torch.load(f'{dna_data_dir}train_targets_bias.pt')

    val_inputs = torch.load(f'{dna_data_dir}val_inputs.pt')
    val_targets_chip_seq = torch.load(f'{dna_data_dir}val_targets_chip_seq.pt')
    val_targets_bias = torch.load(f'{dna_data_dir}val_targets_bias.pt')

    test_inputs = torch.load(f'{dna_data_dir}test_inputs.pt')
    test_targets_chip_seq = torch.load(f'{dna_data_dir}test_targets_chip_seq.pt')
    test_targets_bias = torch.load(f'{dna_data_dir}test_targets_bias.pt')


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

    if(with_prot_emb):
        protein_embedding_dim = 1280

        with open('prot_data/prot_idx_to_metadata.json', 'r') as f:
            prot_idx_to_metadata = json.load(f)
        prot_seq_lens = [item['dna_binding_domain_aa_seq_len'] for item in prot_idx_to_metadata.values()]
        prot_att_mask = get_prot_embeddings_attention_mask(prot_seq_lens)
        prot_embs = torch.load(
            'prot_data/dna_binding_domain_aa_prot_embeds.pt', map_location='cpu')
        model = BPNetWithProteinEmbeddings(protein_embedding_dim).to(device)
    else:
        model = BPNet(num_tasks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = custom_loss

    # train
    model, train_losses, train_corrs, val_losses, val_corrs, pearson_corr = train(train_loader, model, 
                num_epochs, optimizer, loss_fn, val_loader, with_prot_emb, prot_embs, prot_att_mask)
    
    # test
    test_loss, test_corr = test(test_loader, model, loss_fn, with_prot_emb, prot_embs, prot_att_mask, pearson_corr)
    print(f"Test loss: {test_loss}, Test correlation: {test_corr}")

    # Plotting
    epochs = list(range(1, len(train_losses)+1))

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses.cpu().numpy(), label='Training Loss')
    plt.plot(epochs, val_losses.cpu().numpy(), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')

    # Plot Training and Validation Pearson Correlation
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_corrs.cpu().numpy(), label='Training Pearson Correlation')
    plt.plot(epochs, val_corrs.cpu().numpy(), label='Validation Pearson Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.title('Training and Validation Pearson Correlation')
    plt.legend()
    plt.savefig('correlation_plot.png')