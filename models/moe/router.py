import torch
import torch.nn as nn
import torch.nn.functional as F

# --- The Advanced HeteroRouter ---
class HeteroRouter(nn.Module):
    """
    Cost- and entropy-aware router for heterogeneous experts.
    """
    def __init__(self, embed_dim, num_experts, top_k=2, temperature=1.0, beta=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.beta = beta # Cost penalty coefficient
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.register_buffer("costs", torch.ones(num_experts))
        self.last_active_experts = []

    def forward(self, x, compute_losses=False):
        logits = self.gate(x)
        logits = logits - self.beta * self.costs # Penalize high-cost experts
        
        if self.training: # Add Gumbel noise only during training
            gumbel_noise = -torch.empty_like(logits).exponential_().log()
            logits = (logits + gumbel_noise) / self.temperature

        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)
        with torch.no_grad():
            flat_idx = topk_idx.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.num_experts)
    
            print("=== Router Expert Token Counts ===")
            for i, c in enumerate(counts.tolist()):
                print(f"Expert {i}: {c} tokens")
            print("=================================\n")
        # Store active experts for analysis
        if not self.training: # Or on a specific logging step
            self.last_active_experts = torch.unique(topk_idx).cpu().tolist()

        router_losses = {}
        if compute_losses and self.training:
            # Auxiliary losses
            entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
            expert_load = probs.mean(dim=0) # Average probability for each expert
            load_balance_loss = torch.var(expert_load) * self.num_experts
            
            router_losses = {
                "entropy_loss": -entropy, # We want to maximize entropy
                "load_balance_loss": load_balance_loss,
            }

        return topk_idx, topk_probs, probs, router_losses