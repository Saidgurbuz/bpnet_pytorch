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
        chip_seq_outputs = []
        bias_outputs = []
        for index in range(self.num_tasks):
            px = (
                self.profile_shape_heads[index](x.unsqueeze(3))
                .squeeze(-1)
                .permute(0, 2, 1)
            )
            chip_seq_outputs.append(px)  # profile shape output appended
            cx = self.total_counts_heads[index](x)
            bias_outputs.append(cx)  # total counts output appended

        # Outputs are stacked along the second dimension
        chip_seq_outputs = torch.stack(chip_seq_outputs, dim=1)
        bias_outputs = torch.stack(bias_outputs, dim=1)

        return chip_seq_outputs, bias_outputs


class BPNetWithProteinEmbeddings(nn.Module):
    def __init__(self, protein_embedding_dim: int = 1280):
        super(BPNetWithProteinEmbeddings, self).__init__()
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

        self.multihead_attention = nn.MultiheadAttention(1000, num_heads=1, batch_first=True)

        self.profile_head_pos = nn.Linear(1000, 1000)
        self.profile_head_neg = nn.Linear(1000, 1000)
        self.total_counts_head = nn.Linear(1000, 2)

    def forward(
            self,
            dna_seq: torch.Tensor,
            prot_embeddings: torch.Tensor,
            prot_attention_mask: torch.Tensor = None,
    ):
        """
        :param dna_seq: DNA seq torch.Tensor of shape [batch_size, sequence_length, 4]
        :param prot_embeddings: Protein embeddings torch.Tensor of shape [n_prots, n_amino_acids, amino_acid_emb_dim]
        :param prot_embeddings_attention_mask: Attention mask for the protein embeddings torch.Tensor of shape
            [n_prots, n_amino_acids]. This is used to mask the padding tokens in the protein embeddings.
        """
        # DNA sequence
        dna_emb = self.conv_layers(dna_seq)
        dna_emb = self.conv_transpose(dna_emb.unsqueeze(3)).squeeze(-1).squeeze(1)

        # Protein embeddings
        prot_emb = self.protein_embedder(prot_embeddings)

        # Cross-attention
        batch_size, dim = dna_emb.shape
        n_prot, n_amino_acids, _ = prot_emb.shape
        # Repeat the protein embeddings along the batch dimension to match the batch size of dna_emb
        prot_emb_repeated = prot_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # Repeat dna_emb along the second dimension (n_prot)
        dna_emb_repeated = dna_emb.unsqueeze(1).expand(-1, n_prot, -1)
        # Reshape prot_emb to (batch_size * n_prot, n_amino_acids, embed_dim)
        prot_emb_reshaped = prot_emb_repeated.reshape(-1, n_amino_acids, dim)
        # Reshape dna_emb_repeated to (batch_size * n_prot, 1, embed_dim)
        dna_emb_reshaped = dna_emb_repeated.reshape(-1, 1, dim)
        # Perform cross-attention for all proteins
        prot_dna_cross_att_output, _ = self.multihead_attention(dna_emb_reshaped, prot_emb_reshaped, prot_emb_reshaped)
        # Reshape output to (batch_size, n_prot, embed_dim)
        prot_dna_cross_att_output = prot_dna_cross_att_output.view(batch_size, n_prot, dim)

        # final layers
        profile_pred_pos = self.profile_head_pos(prot_dna_cross_att_output)
        profile_pred_neg = self.profile_head_neg(prot_dna_cross_att_output)
        profile_pred = torch.stack([profile_pred_pos, profile_pred_neg], dim=3)
        total_counts_pred = self.total_counts_head(prot_dna_cross_att_output)

        return profile_pred, total_counts_pred
