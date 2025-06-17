import torch
import torch.nn as nn
from torch import Tensor


class BatchPPRFeatures(nn.Module):
    '''
    Batch wide PPR Feature
    '''
    def __init__(self, alpha=0.15, topk=10):
        super().__init__()
        self.alpha = alpha
        self.topk = topk

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        used_nodes = torch.unique(edge_index)
        g2l = {g.item(): l for l, g in enumerate(used_nodes)}
        num_nodes = len(used_nodes)
        
        # Remap edge index for local graph
        src, dst = edge_index
        src = torch.tensor([g2l[n.item()] for n in src])
        dst = torch.tensor([g2l[n.item()] for n in dst])
        local_edge_index = torch.stack([src, dst])
        x_local = x[used_nodes]

        # Build batch-local PPR index
        ppr_index = build_ppr_index_monte_carlo(
            local_edge_index, num_nodes=num_nodes, alpha=self.alpha, topk=self.topk
        )

        # Create PPR feature vectors for each node
        ppr_feat = torch.zeros_like(x_local)
        for i_local, i_global in enumerate(used_nodes):
            if i_local not in ppr_index:
                continue
            for j_local, score in ppr_index[i_local]:
                ppr_feat[i_local] += score * x_local[j_local]

        # Final: project back to global shape
        out = torch.zeros_like(x)
        for i_local, i_global in enumerate(used_nodes):
            out[i_global] = ppr_feat[i_local]

        return out
