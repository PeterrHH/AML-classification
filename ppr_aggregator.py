import torch
from torch_geometric.nn.aggr import Aggregation
from torch import Tensor
import torch_geometric.nn.aggr as pyg_aggr
import heapq
from collections import defaultdict
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np


class TopKPPRAggregation(Aggregation):
    def __init__(self, ppr_index: dict):
        super().__init__()
        self.ppr_index = ppr_index

    def forward(self,
                x: Tensor,           # all node embeddings [num_messages, feat_dim]
                index: Tensor,       # message→ destination indices
                ptr: Tensor = None,
                dim_size: int = None,
                dim: int = 0) -> Tensor:
        # We'll ignore x/index since we do a custom gather:
        print(f"TopKPPRAggregation: dim_size={dim_size}, x.shape={x.shape}, index.shape={index.shape}")
        print(f"self ppr {len(self.ppr_index)}")
        N, F = dim_size, x.size(1)  # number of nodes, feat‐dim
        out = x.new_zeros((N, F)) 

        # For each target node i, do weighted sum over its top-K neighbours:
        for i, nbr_scores in self.ppr_index.items():
            if not nbr_scores:
                continue
            # unpack neighbors & scores
            nbrs, scores = zip(*nbr_scores)
            nbr_feats = x[nbrs]        # [K, F]
            w = x.new_tensor(scores).unsqueeze(1)  # [K,1]
            out[i] = (nbr_feats * w).sum(dim=0)    # [F]
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
      edge_index: [2, E] LongTensor of your graph's edges (undirected or directed).
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
    #    we use a simple queue; in practice you'd use a priority or worklist.
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
    
    ppr_index = build_ppr_index(
        edge_index=remapped_edge_index,
        num_nodes=len(used_nodes),
        alpha=0.15,
        eps=1e-4,
        topk=2
    )
    # ppr_index = build_ppr_index_monte_carlo(
    #     edge_index=remapped_edge_index,
    #     num_nodes=len(used_nodes),
    #     alpha=0.15,
    #     topk=2
    # )
    
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


def _process_node_ppr(args):
    seed, edge_index, num_nodes, alpha, eps, topk = args
    return seed, approx_ppr_push(seed, edge_index, num_nodes, alpha=alpha, eps=eps, topk=topk)

def build_ppr_index(edge_index, num_nodes, alpha=0.15, eps=1e-4, topk=50):
    # Convert edge_index to CPU if it's on GPU to avoid multiprocessing issues
    edge_index = edge_index.cpu()
    
    # Prepare arguments for parallel processing
    args = [(seed, edge_index, num_nodes, alpha, eps, topk) 
            for seed in range(num_nodes)]
    
    # Use number of CPU cores minus 1 to leave one core free
    n_cores = max(1, cpu_count() - 1)
    
    # Process nodes in parallel
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(_process_node_ppr, args),
            total=num_nodes,
            desc="Building PPR index"
        ))
    
    # Convert results to dictionary
    return dict(results)

def _process_node_monte_carlo(args):
    seed, nbrs, alpha, num_walks, max_steps, topk = args
    return seed, monte_carlo_ppr(seed, nbrs, alpha, num_walks, max_steps, topk)

def build_ppr_index_monte_carlo(edge_index, num_nodes, alpha=0.15, topk=50):
    print(f"Using Monte Carlo PPR for {num_nodes} nodes...")

    # Step 1: Build adjacency list - optimize with numpy for faster processing
    nbrs = [[] for _ in range(num_nodes)]
    src, dst = edge_index.cpu().numpy()
    for u, v in zip(src, dst):
        nbrs[u].append(v)
        nbrs[v].append(u)
    
    # Convert lists to numpy arrays for faster access
    nbrs = [np.array(nbr_list) for nbr_list in nbrs]

    # Prepare arguments for parallel processing
    args = [(seed, nbrs, alpha, 20, 20, topk) 
            for seed in range(num_nodes)]
    
    # Use number of CPU cores minus 1
    n_cores = max(1, cpu_count() - 1)
    
    # Process nodes in parallel
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(_process_node_monte_carlo, args),
            total=num_nodes,
            desc="Building Monte Carlo PPR index"
        ))
    
    return dict(results)

# 3) "Register" new aggregation under the name "FrequencyAggregation"
#    so that PyG's resolver can find it when we pass 'frequency' in PNAConv.
pyg_aggr.TopKPPRAggregation = TopKPPRAggregation


