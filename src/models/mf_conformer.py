import torch
import torch.nn as nn
from src.models.memory import NeuralMemoryModule
from src.models.convolution import ConformerConvolutionModule

class MultiFrequencyConformer(nn.Module):
    """
    Replaces static Macaron FFNs with Nested Memory Systems.
    Operates on Gamma (Frame), Theta (Event), and Delta (Scene) timescales.

    Output: [B, T, n_classes] frame-level probabilities for true SED.
    """
    def __init__(self, cfg):
        super().__init__()
        m_cfg = cfg['model']
        d_model = m_cfg['d_model']
        self.c_event = m_cfg['c_event']
        self.c_scene = m_cfg['c_scene']
        
        # Spectrogram frontend
        self.input_proj = nn.Linear(cfg['dataset']['mel_bins'], d_model)
        
        # Level 1: Gamma (High-Frequency / Frame Level)
        self.gamma_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=m_cfg['num_heads'], batch_first=True
        )
        self.gamma_conv = ConformerConvolutionModule(channels=d_model)
        
        # Level 2: Theta (Medium-Frequency / Event Level, C_event frames)
        self.theta_memory = NeuralMemoryModule(d_model, chunk_size=m_cfg['c_event'])
        
        # Level 3: Delta (Low-Frequency / Scene Level, C_scene frames)
        self.delta_memory = NeuralMemoryModule(d_model, chunk_size=m_cfg['c_scene'])
        
        self.norm = nn.LayerNorm(d_model)
        # Frame-level classifier: outputs logits per timestep for SED
        self.classifier = nn.Linear(d_model, m_cfg['num_classes'])

    def forward(self, x):
        """
        Args:
            x: [B, T, mel_bins] log-Mel spectrogram
        Returns:
            logits: [B, T, num_classes] frame-level event logits for SED
        """
        x = self.input_proj(x)   # [B, T, d_model]

        M_theta, S_theta = None, None
        M_delta, S_delta = None, None

        outputs = []
        T = x.size(1)

        # Chunked processing: iterate over C_scene-sized chunks.
        # Memory updates trigger at biological chunk boundaries inside each module.
        # This avoids the frame-by-frame Python loop while preserving
        # the correct Theta/Delta update semantics.
        chunk_size = self.c_scene  # largest stride; inner modules use c_event
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            x_chunk = x[:, start:end, :]   # [B, chunk, d_model]

            # Level 3: Delta consolidation (replaces 1st Macaron FFN)
            d_out, M_delta, S_delta = self.delta_memory(
                x_chunk, start, M_delta, S_delta
            )
            x_chunk = x_chunk + 0.5 * d_out

            # Level 1: Gamma attention + local convolution
            attn_out, _ = self.gamma_attention(x_chunk, x_chunk, x_chunk)
            x_chunk = x_chunk + attn_out
            x_chunk = x_chunk + self.gamma_conv(x_chunk)

            # Level 2: Theta event binding (replaces 2nd Macaron FFN)
            t_out, M_theta, S_theta = self.theta_memory(
                x_chunk, start, M_theta, S_theta
            )
            x_chunk = x_chunk + 0.5 * t_out

            outputs.append(self.norm(x_chunk))

        x = torch.cat(outputs, dim=1)   # [B, T, d_model]

        # Frame-level output — preserves time axis for PSDS evaluation
        return self.classifier(x)        # [B, T, num_classes]
