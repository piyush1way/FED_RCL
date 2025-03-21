# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def triplate_mean_lr(prev, pres, serv, lr):
#     xm = torch.mean(prev)
#     ym = torch.mean(pres)
#     zm = torch.mean(serv)
#     device = prev.device  # Ensure we use the same device as the input tensors
#     return lr * (torch.mean(torch.tensor([xm, ym, zm], device=device)) / (xm + ym + zm + 1e-7))


# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.5, margin=0.2, lr=1e-3):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         self.margin = margin
#         self.lr = lr
    
#     def forward(self, z_prev, z_present, z_serv):
#         """Computes the contrastive loss."""
#         # Normalize the embeddings for stability
#         z_prev = F.normalize(z_prev, dim=-1)
#         z_present = F.normalize(z_present, dim=-1)
        
#         # Compute the squared components for the loss terms
#         sqpres = torch.square(z_present)
#         # Compute margin term: for each element, compute (margin - value) and clamp below 0, then square it.
#         sqMar = torch.square(torch.clamp_min(self.margin - z_present, 0))
        
#         # Compute a combined loss term.
#         # Here, the formulation is: mean(z_prev * sqpres + (1 - z_prev) * sqMar)
#         loss = torch.mean(z_prev * sqpres + (1 - z_prev) * sqMar)
#         loss = loss - (loss * self.lr)  # Scale loss by (1 - lr)
        
#         # Update the learning rate in a detached manner to avoid interfering with gradients.
#         new_lr = triplate_mean_lr(z_prev, z_present, z_serv, self.lr)
#         self.lr = new_lr.detach()
        
#         # Compute a regularization term using the server feature:
#         srv = torch.sqrt(torch.sum(torch.square(z_serv)))
#         # Clamp srv to avoid log(0) issues
#         srv = torch.log(torch.clamp_min(srv, 1e-7))
        
#         return loss / srv


# without dynamic 

import torch
import torch.nn as nn
import torch.nn.functional as F

def triplate_mean_lr(prev, pres, serv, lr):
    """Computes a triplet-based learning rate adjustment."""
    xm = torch.mean(prev)
    ym = torch.mean(pres)
    zm = torch.mean(serv)
    return lr * (torch.mean(torch.tensor([xm, ym, zm], device=prev.device)) / (xm + ym + zm + 1e-7))

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, margin=0.2, lr=1e-3):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.lr = lr  # Keep learning rate fixed, as in TensorFlow

    def forward(self, z_prev, z_present, z_serv):
        """Computes the contrastive loss (TensorFlow equivalent)."""
        # Normalize the feature embeddings
        z_prev = F.normalize(z_prev, dim=-1)
        z_present = F.normalize(z_present, dim=-1)

        # Compute squared terms for loss
        sqpres = torch.square(z_present)
        sqMar = torch.square(torch.maximum(self.margin - z_present, torch.tensor(0.0, device=z_present.device)))

        # Compute contrastive loss
        loss = torch.mean(z_prev * sqpres + (1 - z_prev) * sqMar)
        loss = loss - (loss * self.lr)  # Scale loss using (1 - lr)

        # Compute triplet mean learning rate (but keep `self.lr` fixed)
        lr = triplate_mean_lr(z_prev, z_present, z_serv, self.lr)

        # Regularization term using log of server feature norm
        srv = torch.sqrt(torch.sum(torch.square(z_serv)))
        srv = torch.log(torch.maximum(srv, torch.tensor(1e-7, device=z_serv.device)))  # Avoid log(0)

        return loss / srv


