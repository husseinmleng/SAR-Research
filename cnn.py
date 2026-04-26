import torch
import torch.nn as nn


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name, map_location=lambda storage, loc: storage))

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (paper Fig. 2).

    Global AvgPool -> FC(32->32) -> ReLU -> Sigmoid -> channel-wise multiply.
    """

    def __init__(self, channels=32):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        w = self.pool(x).view(x.size(0), -1)   # (B, C)
        w = self.fc(w).view(x.size(0), -1, 1, 1)  # (B, C, 1, 1)
        return x * w


class EmbeddingCNN(myModel):
    """4-layer uniform Conv.32 feature extractor with per-layer SE attention.

    Architecture (paper Fig. 2):
      - 4 x [Conv2d(in,32,3,pad=1) + BN + ReLU + SEBlock + MaxPool2d(2)]
      - Final Conv2d(32,64,kernel_size=6) -> AdaptiveAvgPool2d(1) -> flatten -> 64-D
    """

    def __init__(self, image_size=100, cnn_feature_size=64,
                 cnn_hidden_dim=32, cnn_num_layers=4, use_gradient_checkpointing=False):
        super(EmbeddingCNN, self).__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.layers = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        in_ch = 1
        for _ in range(cnn_num_layers):
            block = nn.Sequential(
                nn.Conv2d(in_ch, cnn_hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cnn_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.layers.append(block)
            self.se_blocks.append(SEBlock(cnn_hidden_dim))
            in_ch = cnn_hidden_dim

        self.pool = nn.MaxPool2d(2)

        # Global projection: after 4 x MaxPool2d(2) on 100x100 -> 6x6
        spatial = image_size // (2 ** cnn_num_layers)  # 100 // 16 = 6
        self.global_proj = nn.Sequential(
            nn.Conv2d(cnn_hidden_dim, cnn_feature_size,
                      kernel_size=spatial, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        for layer, se in zip(self.layers, self.se_blocks):
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda inp: self.pool(se(layer(inp))), x, use_reentrant=False
                )
            else:
                x = layer(x)
                x = se(x)
                x = self.pool(x)

        x = self.global_proj(x)          # (B, 64, 1, 1)
        return x.view(x.size(0), -1)     # (B, 64)

    def freeze_weight(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_weight(self):
        for p in self.parameters():
            p.requires_grad = True


class Linear_model(myModel):

    def __init__(self, nway):
        super(Linear_model, self).__init__()
        self.out = nn.Linear(64, nway)

    def forward(self, x):
        return self.out(x)
