import torch
import torch.nn as nn

class ConformerConvolutionModule(nn.Module):
    """
    Standard Conformer Convolution Module.
    Based on Macaron-Net architecture principles.
    """
    def __init__(self, channels, kernel_size=7, dropout=0.1):
        super().__init__()
        # Pointwise -> Depthwise -> Pointwise
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1, 
            padding=(kernel_size - 1) // 2, groups=channels
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: [Batch, Time, Channels] -> Conv1d expects [Batch, Channels, Time]
        x = x.transpose(1, 2)
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Back to [Batch, Time, Channels]
        x = x.transpose(1, 2)
        return x
