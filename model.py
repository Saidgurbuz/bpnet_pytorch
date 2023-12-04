import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=25, padding="same")
        self.relu = nn.ReLU()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(64, 64, kernel_size=3, padding="same", dilation=2**i)
                for i in range(1, 10)
            ]
        )

    def forward(self, x):
        x = self.relu(self.conv1(x.permute(0, 2, 1)))
        for conv_layer in self.conv_layers:
            x = self.relu(conv_layer(x))
        return x


class BPNet(nn.Module):
    def __init__(self, num_tasks):
        super(BPNet, self).__init__()
        self.num_tasks = num_tasks

        self.conv_layers = ConvEncoder()

        # Heads (Output Layers)
        self.profile_shape_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        64, 2, kernel_size=(25, 1), padding=(12, 0)),
                    # nn.Flatten(), # check doing without this layer
                )
                for _ in range(num_tasks)
            ]
        )

        self.total_counts_heads = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool1d(
                    1), nn.Flatten(), nn.Linear(64, 2))
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x):
        # Body
        x = self.conv_layers(x)

        # Profile Shape and Total Counts Heads
        outputs = []
        for index in range(self.num_tasks):
            px = (
                self.profile_shape_heads[index](x.unsqueeze(3))
                .squeeze(-1)
                .permute(0, 2, 1)
            )
            outputs.append(px)  # profile shape output appended
            cx = self.total_counts_heads[index](x)
            outputs.append(cx)  # total counts output appended

        return outputs


class BPNetWithProteinEmbeddings(nn.Module):
    def __init__(self, num_tasks, protein_embedding_dim=256):
        super(BPNetWithProteinEmbeddings, self).__init__()
        self.num_tasks = num_tasks
        self.protein_embedding_dim = protein_embedding_dim

        self.conv_layers = ConvEncoder()
        # TODO: potentially replace with a flattening layer instead of ConvTranspose2d
        self.conv_transpose = nn.ConvTranspose2d(
            64, 1, kernel_size=(25, 1), padding=(12, 0))

        self.protein_embedder = nn.Sequential(
            nn.Linear(protein_embedding_dim, 1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.profile_head = nn.Linear(1000, 2)
        self.total_counts_head = nn.Linear(1000, 2)

    def forward(self, dna_seq: torch.Tensor, prot_embeddings: torch.Tensor):
        # DNA sequence
        dna_emb = self.conv_layers(dna_seq)
        dna_emb = self.conv_transpose(dna_emb.unsqueeze(3)).squeeze(-1).permute(0, 2, 1)

        # Protein embeddings
        prot_emb = self.protein_embedder(prot_embeddings)

        # TODO: implement cross-attention here
        # TODO: repeat the dna_emb to match the number of unique prot_emb
        # TODO: calculate cross-attention
        batch_size, dim = dna_emb.shape
        n_prot, _ = prot_emb.shape
        prot_dna_cross_att_output = torch.randn(batch_size, n_prot, dim)

        # final layers
        profile_shape_output = self.profile_head(prot_dna_cross_att_output)
        total_counts_output = self.total_counts_head(prot_dna_cross_att_output)

        # TODO: adapt the outputs to be similar as the BPNet above
        return
