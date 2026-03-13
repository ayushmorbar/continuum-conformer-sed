import torch
import torch.nn as nn
from src.models.memory import NeuralMemoryModule
from src.models.convolution import ConformerConvolutionModule

class MultiFrequencyConformer(nn.Module):
    """
    Replaces static Macaron FFNs with Nested Memory Systems.
    Operates on Gamma (Frame), Theta (Event), and Delta (Scene) timescales.
    """
    def __init__(self, cfg):
        super().__init__()
        m_cfg = cfg['model']
        d_model = m_cfg['d_model']
        
        # Spectrogram frontend
        self.input_proj = nn.Linear(cfg['dataset']['mel_bins'], d_model)
        
        # Level 1: Gamma (High-Frequency / Frame Level)
        self.gamma_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=m_cfg['num_heads'], batch_first=True
        )
        self.gamma_conv = ConformerConvolutionModule(channels=d_model)
        
        # Level 2: Theta (Medium-Frequency / Event Level)
        self.theta_memory = NeuralMemoryModule(d_model, chunk_size=m_cfg['c_event'])
        
        # Level 3: Delta (Low-Frequency / Scene Level)
        self.delta_memory = NeuralMemoryModule(d_model, chunk_size=m_cfg['c_scene'])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, m_cfg['num_classes'])

    def forward(self, x):
        x = self.input_proj(x)
        
        M_theta, S_theta = None, None
        M_delta, S_delta = None, None
        
        seq_len = x.size(1)
        outputs = []
        
        # Simulate continuous audio stream
        for step in range(seq_len):
            x_step = x[:, step:step+1, :]
            
            # Level 3: Delta consolidation (Replaces 1st Macaron FFN)
            d_out, M_delta, S_delta = self.delta_memory(x_step, step, M_delta, S_delta)
            x_step = x_step + 0.5 * d_out
            
            # Level 1: Gamma Attention & Local Convolution
            attn_out, _ = self.gamma_attention(x_step, x_step, x_step)
            x_step = x_step + attn_out
            x_step = x_step + self.gamma_conv(x_step)
            
            # Level 2: Theta binding (Replaces 2nd Macaron FFN)
            t_out, M_theta, S_theta = self.theta_memory(x_step, step, M_theta, S_theta)
            x_step = x_step + 0.5 * t_out
            
            outputs.append(self.norm(x_step))
            
        x = torch.cat(outputs, dim=1)
        
        # Mean pooling for the baseline classification test
        pooled = x.mean(dim=1)
        return self.classifier(pooled)
