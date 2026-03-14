import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralMemoryModule(nn.Module):
    """
    Continuum Memory System (CMS) block.
    Implements momentum-based test-time weight updates (Delta Rule).
    """
    def __init__(self, d_model, chunk_size):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        # Projections for associative key-value mapping
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        
        self.momentum_factor = 0.9
        self.alpha_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid() # Forgetting gate
        )

        # Output normalisation: ensures retrieved features stay on the
        # same activation scale as the incoming residual stream.
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, current_step, M_prev=None, S_prev=None):
        B, T, D = x.shape
        device = x.device
        
        if M_prev is None:
            # Initialize identity memory state
            M_prev = torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        if S_prev is None:
            S_prev = torch.zeros_like(M_prev)

        # Update Memory if we hit the biological chunk boundary
        if current_step % self.chunk_size == 0:
            # Isolate graph for local surprise calculation
            with torch.enable_grad():
                M_local = M_prev.detach().clone().requires_grad_(True)
                k_t = self.proj_k(x.detach())
                v_t = self.proj_v(x.detach())
                
                # Associative memory retrieval and loss
                retrieved_v = torch.matmul(k_t, M_local)
                surprise_loss = F.mse_loss(retrieved_v, v_t)
                
                # Momentary surprise (Gradient of loss w.r.t memory)
                grad_surprise = torch.autograd.grad(surprise_loss, M_local)[0]
            
            alpha_t = self.alpha_gate(x.mean(dim=1)).unsqueeze(-1)
            S_t = self.momentum_factor * S_prev - (1 - self.momentum_factor) * grad_surprise
            M_t = (1 - alpha_t) * M_prev + S_t
        else:
            M_t, S_t = M_prev, S_prev
            
        # Retrieval + output normalisation for residual stability
        out = self.out_norm(torch.matmul(x, M_t))
        return out, M_t, S_t
