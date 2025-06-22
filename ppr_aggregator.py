import torch
from torch_geometric.nn.aggr import Aggregation
from torch import Tensor
import torch_geometric.nn.aggr as pyg_aggr
import heapq
from collections import defaultdict
import random
import tqdm


class TopKPPRAggregation(Aggregation):
    """
    A PyG Aggregation subclass that, for each node in the current batch,
    gathers its pre-computed top-K Personalized PageRank neighbors and
    returns a weighted sum of their embeddings.

    After PPR pooling, returns a tensor of shape [B, F] (single “tower”).
    Registered under pyg_aggr.TopKPPRAggregation for PNAConv.
    """
    def __init__(self, ppr_index: dict, tower_size = 5):
        super().__init__()
        self.ppr_index = ppr_index
        self.global_x  = torch.zeros((1,1))
        self.g2l: dict[int,int] = {} 
        self.tower_size = tower_size

    def set_mapping(self, g2l: dict[int,int]):
        """
        Provide global to local index map for each mini-batch.
        """
        self.g2l = g2l

    def forward(self,
                x: Tensor,
                index: Tensor,
                ptr: Tensor = None,
                dim_size: int = None,
                dim: int = 0) -> Tensor:
        
        self.global_x = x
        N = self.global_x.size(0) 
        F = x.shape[-1]
        T = self.tower_size
        out = x.new_zeros((N, F))  # for single tower

        for g_u, nbrs_and_scores in self.ppr_index.items():
            if g_u not in self.g2l:
                continue
            u = self.g2l[g_u]
            # keep only those neighbors in this batch:
            local = [(self.g2l[g_v], w) for (g_v, w) in nbrs_and_scores if g_v in self.g2l]
            if not local:
                continue
            loc_idxs, ws = zip(*local)
            feats = x[list(loc_idxs)]          # [k, T, F]
            wts  = x.new_tensor(ws).unsqueeze(1)  # [k,1]
            # Modify wts to be [k,1,1] for broadcasting with feats [k,T,F]
            # (feats * wts_broadcastable) will have shape [k,T,F]
            # .sum(dim=0) will sum over k, resulting in shape [T,F]
            out[u,:] = (feats * wts.unsqueeze(2)).sum(dim=0) 
            
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

    # Build a sparse neighbor lookup from edge_index
    nbrs = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        nbrs[u].append(v)
        # if undirected graph:
        nbrs[v].append(u)

    # Initialize residual (r) and estimate (p) vectors as dicts
    r = defaultdict(float)
    p = defaultdict(float)
    r[seed] = 1.0

    # While there exists u with r[u] > eps * deg(u):
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

    # Extract top-k entries from p (heap of size topk)
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
    """
    Remap global IDs to [0..B-1] for the current split,
    then build a PPR index for each remapped node either via
    `build_ppr_index` (power-iteration) or `build_ppr_index_monte_carlo`.
    Returns:
      - ppr_index: dict remapped → list[(nbr_remapped, weight)]
      - x_remapped: node features restricted to used nodes
      - remapped_edge_index: [2,E_split]
      - used_nodes: original global IDs → used global IDs
    """
    used_nodes = torch.unique(edge_index)
    id_map = {old.item(): new for new, old in enumerate(used_nodes)}
    remapped_src = torch.tensor([id_map[n.item()] for n in edge_index[0]])
    remapped_dst = torch.tensor([id_map[n.item()] for n in edge_index[1]])
    remapped_edge_index = torch.stack([remapped_src, remapped_dst])
    
    '''
    Switch Monte Carlo or Build PPR Index here
    '''
    # ppr_index = build_ppr_index(
    #     edge_index=remapped_edge_index,
    #     num_nodes=len(used_nodes),
    #     alpha=0.15,
    #     eps=1e-4,
    #     topk=2,
    #     max_iter= 2
    # )

    ppr_index = build_ppr_index_monte_carlo(
        edge_index=remapped_edge_index,
        num_nodes=len(used_nodes),
        alpha=0.15,
        topk=25
    )
    
    # remap x to used nodes
    x_remapped = x[used_nodes]
    
    return ppr_index, x_remapped, remapped_edge_index, used_nodes



def monte_carlo_ppr(seed, nbrs, alpha=0.15, num_walks=50, max_steps=20, topk=20):
    """
    Monte-Carlo estimate of PPR(seed) by doing `num_walks` random walks
    of length ≤ `max_steps` with restart prob. α.  Returns top-k visit frequencies.
    """
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


def build_ppr_index(edge_index, num_nodes, alpha=0.15, eps=1e-4, topk=50, max_iter=20):
    """
    Compute PPR for all nodes using power iteration (tensor-based).
    Returns a dict: node -> list of (neighbor, score) for topk neighbors.
    """
    device = edge_index.device if hasattr(edge_index, 'device') else 'cpu'
    # Build adjacency matrix (sparse)
    src, dst = edge_index
    values = torch.ones(src.size(0), device=device)
    adj = torch.sparse_coo_tensor(
        torch.stack([src, dst]), values, (num_nodes, num_nodes), device=device
    )
    # Make undirected
    adj = adj.coalesce()
    adj_t = torch.sparse_coo_tensor(
        torch.stack([dst, src]), values, (num_nodes, num_nodes), device=device
    )
    adj_t = adj_t.coalesce()
    adj = torch.sparse_coo_tensor(
        torch.cat([adj.indices(), adj_t.indices()], dim=1),
        torch.cat([adj.values(), adj_t.values()]),
        (num_nodes, num_nodes), device=device
    ).coalesce()
    # Row-normalize adjacency to get transition matrix
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    # For each node, run power iteration
    ppr_index = {}
    for seed in tqdm.tqdm(range(num_nodes), desc="Building PPR index (tensor)"):
        # Personalization vector
        e = torch.zeros(num_nodes, device=device)
        e[seed] = 1.0
        p = e.clone()
        for _ in range(max_iter):
            p_last = p
            # p = alpha * e + (1 - alpha) * A^T p / deg
            p = alpha * e + (1 - alpha) * torch.sparse.mm(adj, (p * deg_inv).unsqueeze(1)).squeeze(1)
            if torch.norm(p - p_last, p=1) < eps:
                break
        # Get top-k
        vals, idxs = torch.topk(p, k=min(topk, num_nodes))
        ppr_index[seed] = [(i.item(), v.item()) for i, v in zip(idxs, vals) if v.item() > 0]
    return ppr_index


def build_ppr_index_monte_carlo(edge_index, num_nodes, alpha=0.15, topk=50):
    print(f"Using Monte Carlo PPR for {num_nodes} nodes...")

    # Build adjacency list
    nbrs = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        nbrs[u].append(v)
        nbrs[v].append(u)

    # Build PPR index only for used nodes
    ppr_index = {}
    for seed in range(num_nodes):
        ppr_index[seed] = monte_carlo_ppr(
            seed, nbrs, alpha=alpha, num_walks=20, max_steps=20, topk=topk
        )

    return ppr_index

# "Register" new aggregation under the name TopKPPRAggregation
# so that PyG's resolver can find it when we pass 'TopKPPRAggregation' in PNAConv.
pyg_aggr.TopKPPRAggregation = TopKPPRAggregation
