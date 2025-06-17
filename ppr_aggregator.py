import torch
from torch_geometric.nn.aggr import Aggregation
from torch import Tensor
import torch_geometric.nn.aggr as pyg_aggr
import heapq
from collections import defaultdict
import random


class TopKPPRAggregation(Aggregation):
    def __init__(self, ppr_index: dict):
        super().__init__()
        self.ppr_index = ppr_index  # Now only batch-local!
        self.local_x = None         # No more global_x!
    
    def forward(self, x: Tensor, index: Tensor, ptr: Tensor = None,
                dim_size: int = None, dim: int = 0) -> Tensor:
        
        # x is already batch-local: [num_nodes, time_steps, dim]
        N = x.size(0)
        _, T, F = x.size()
        out = x.new_zeros((N, T, F))

        for u, neighbors in self.ppr_index.items():  # u is local node id
            if u >= N:
                continue
            loc_idxs, ws = zip(*neighbors)
            feats = x[list(loc_idxs)]               # [k, T, F]
            wts = x.new_tensor(ws).unsqueeze(1).unsqueeze(2)  # [k,1,1]
            out[u] = (feats * wts).sum(dim=0)

        return out


def approx_ppr_push(seed: int,
                    edge_index: torch.LongTensor,
                    num_nodes: int,
                    alpha: float = 0.15,
                    eps: float = 1e-4,
                    topk: int = 50):
    """
    Approximate the Personalized PageRank vector for `seed` using 
    the push algorithm.  Returns the top-k (node, ppr_score) pairs.

    Args:
      seed    : int, the node whose PPR we want.
      edge_index: [2, E] LongTensor of your graph’s edges (undirected or directed).
      num_nodes : int, number of nodes N.
      alpha   : teleport probability (≈0.15).
      eps     : tolerance threshold.
      topk    : how many largest entries to return.

    Returns:
      List of (node, score) sorted by score descending, length ≤ topk.
    """

    # 1) Build a sparse neighbor lookup from edge_index
    nbrs = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        nbrs[u].append(v)
        # if undirected graph:
        nbrs[v].append(u)

    # 2) Initialize residual (r) and estimate (p) vectors as dicts
    r = defaultdict(float)
    p = defaultdict(float)
    r[seed] = 1.0

    # 3) While there exists u with r[u] > eps * deg(u):
    queue = [seed]
    while queue:
        u = queue.pop()
        ru = r[u]
        deg_u = len(nbrs[u]) or 1
        threshold = eps * deg_u
        if ru < threshold:
            continue

        # push mass to p[u]
        delta = alpha * ru
        p[u] += delta

        # distribute the rest to neighbors
        leftover = (ru * (1 - alpha)) / 2.0
        r[u] = leftover

        inc = leftover / deg_u
        for v in nbrs[u]:
            prev = r[v]
            r[v] += inc
            # if neighbor now exceeds threshold, schedule a push
            if prev < eps * (len(nbrs[v]) or 1) <= r[v]:
                queue.append(v)

    # 4) Extract top-k entries from p
    #    (heap of size topk)
    heap = []
    for node, score in p.items():
        if len(heap) < topk:
            heapq.heappush(heap, (score, node))
        else:
            heapq.heappushpop(heap, (score, node))

    # sort descending
    topk_list = sorted([(node, score) for score, node in heap],
                       key=lambda x: -x[1])
    return topk_list


def build_split_ppr(edge_index, x):
    used_nodes = torch.unique(edge_index)
    id_map = {old.item(): new for new, old in enumerate(used_nodes)}
    remapped_src = torch.tensor([id_map[n.item()] for n in edge_index[0]])
    remapped_dst = torch.tensor([id_map[n.item()] for n in edge_index[1]])
    remapped_edge_index = torch.stack([remapped_src, remapped_dst])
    
    # ppr_index = build_ppr_index(
    #     edge_index=remapped_edge_index,
    #     num_nodes=len(used_nodes),
    #     alpha=0.15,
    #     eps=1e-4,
    #     topk=2
    # )
    ppr_index = build_ppr_index_monte_carlo(
        edge_index=remapped_edge_index,
        num_nodes=len(used_nodes),
        alpha=0.15,
        topk=2
    )
    
    # remap x to used nodes
    x_remapped = x[used_nodes]
    
    return ppr_index, x_remapped, remapped_edge_index, used_nodes



def monte_carlo_ppr(seed, nbrs, alpha=0.15, num_walks=50, max_steps=20, topk=20):
    count = defaultdict(int)
    for _ in range(num_walks):
        node = seed
        for _ in range(max_steps):
            if random.random() < alpha:
                break
            neighbors = nbrs[node]
            if not neighbors:
                break
            node = random.choice(neighbors)
        count[node] += 1
    total = sum(count.values())
    return sorted([(n, c / total) for n, c in count.items()], key=lambda x: -x[1])[:topk]


def build_ppr_index(edge_index, num_nodes, alpha=0.15, eps=1e-4, topk=50):
    ppr_index = {}
    for seed in range(num_nodes):
        print(f"---Building PPR index for seed {seed}")
        ppr_index[seed] = approx_ppr_push(
            seed, edge_index, num_nodes, alpha=alpha, eps=eps, topk=topk
        )
    return ppr_index


def build_ppr_index_monte_carlo(edge_index, num_nodes, alpha=0.15, topk=50):
    print(f"Using Monte Carlo PPR for {num_nodes} nodes...")

    # Step 1: Build adjacency list
    nbrs = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        nbrs[u].append(v)
        nbrs[v].append(u)

    # Step 2: Build PPR index only for used nodes
    ppr_index = {}
    for seed in range(num_nodes):
        ppr_index[seed] = monte_carlo_ppr(
            seed, nbrs, alpha=alpha, num_walks=20, max_steps=20, topk=topk
        )

    return ppr_index

# 3) “Register” new aggregation under the name TopKPPRAggregation
#    so that PyG’s resolver can find it when we pass 'TopKPPRAggregation' in PNAConv.
pyg_aggr.TopKPPRAggregation = TopKPPRAggregation


