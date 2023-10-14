import torch.nn as nn


class BPNet(nn.Module):
    def __init__(self, num_tasks):
        super(BPNet, self).__init__()
        self.num_tasks = num_tasks

        # Body (Convolutional Layers)
        self.conv1 = nn.Conv1d(4, 64, kernel_size=25, padding="same")
        self.relu = nn.ReLU()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(64, 64, kernel_size=3, padding="same", dilation=2**i)
                for i in range(1, 10)
            ]
        )

        # Heads (Output Layers)
        self.profile_shape_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(64, 2, kernel_size=(25, 1), padding=(12, 0)),
                    # nn.Flatten(), # check doing without this layer
                )
                for _ in range(num_tasks)
            ]
        )

        self.total_counts_heads = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 2))
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x):
        # Body
        x = self.relu(self.conv1(x))
        for conv_layer in self.conv_layers:
            x = self.relu(conv_layer(x))

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
