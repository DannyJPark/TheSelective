import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, radius_graph

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        # offset = torch.linspace(start, stop, num_gaussians)
        # customized offset
        offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class AngleExpansion(nn.Module):
    def __init__(self, start=1.0, stop=5.0, half_expansion=10):
        super(AngleExpansion, self).__init__()
        l_mul = 1. / torch.linspace(stop, start, half_expansion)
        r_mul = torch.linspace(start, stop, half_expansion)
        coeff = torch.cat([l_mul, r_mul], dim=-1)
        self.register_buffer('coeff', coeff)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.coeff.view(1, -1))

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}

class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def outer_product(*vectors):
    # Check for empty tensors
    for vector in vectors:
        if vector.numel() == 0:
            # Return empty tensor with appropriate shape for empty input
            if len(vectors) == 1:
                return vector
            else:
                # For multiple vectors, return empty tensor with product dimensions
                device = vector.device
                dtype = vector.dtype
                return torch.empty((0,), device=device, dtype=dtype)

    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()

def get_h_dist(dist_metric, hi, hj):
    if dist_metric == 'euclidean':
        h_dist = torch.sum((hi - hj) ** 2, -1, keepdim=True)
        return h_dist
    elif dist_metric == 'cos_sim':
        hi_norm = torch.norm(hi, p=2, dim=-1, keepdim=True)
        hj_norm = torch.norm(hj, p=2, dim=-1, keepdim=True)
        h_dist = torch.sum(hi * hj, -1, keepdim=True) / (hi_norm * hj_norm)
        return h_dist, hj_norm

def get_r_feat(r, r_exp_func, node_type=None, edge_index=None, mode='basic'):
    if mode == 'origin':
        r_feat = r
    elif mode == 'basic':
        r_feat = r_exp_func(r)
    elif mode == 'sparse':
        src, dst = edge_index
        nt_src = node_type[src]  # [n_edges, 8]
        nt_dst = node_type[dst]
        r_exp = r_exp_func(r)
        r_feat = outer_product(nt_src, nt_dst, r_exp)
    else:
        raise ValueError(mode)
    return r_feat

def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand, protein_id=None):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    # mask_ligand: True for ligand atoms, False for protein atoms
    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    # mask_protein: True for protein atoms, False for ligand atoms
    mask_protein = ~mask_ligand
    
    # Align protein_id with other tensors
    # Fill ligand atoms with -1 as placeholder ID
    if protein_id is not None:
        ligand_dummy_ids = torch.full_like(batch_ligand, fill_value=-1)
        protein_id_ctx = torch.cat([protein_id, ligand_dummy_ids], dim=0)[sort_idx]
    else:
        # If protein_id is None, set all atoms to same ID
        protein_id_ctx = None

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    # Always return 6 values (expected by forward_sam_pl etc.)
    # If protein_id is None, return protein_id_ctx as None
    return h_ctx, pos_ctx, batch_ctx, mask_ligand, mask_protein, protein_id_ctx

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    # fully-connected for ligand atoms
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    # knn for ligand-protein edges
    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1)
    knn_l_idx = knn_l_idx.repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)
    pl_edge_index = pl_edge_index.view(2, -1)
    return ll_edge_index, pl_edge_index

def mask_cross_protein_edges(edge_index, protein_id, mask_ligand, debug=False):
    """
    Mask edges between atoms from different proteins to prevent cross-protein information flow.
    This is crucial for selectivity learning where on-target and off-target proteins should be isolated.
    
    Args:
        edge_index: [2, num_edges] edge tensor
        protein_id: protein ID for each atom (-1 for ligands)
        mask_ligand: boolean mask for ligand atoms
        debug: If True, print detailed edge type analysis
    Returns:
        Filtered edge_index with cross-protein edges removed
    """
    src, dst = edge_index
    src_protein_id = protein_id[src]
    dst_protein_id = protein_id[dst]
    src_is_ligand = mask_ligand[src]
    dst_is_ligand = mask_ligand[dst]
    
    if debug:
        # Analyze edge types before masking
        ll_edges = (src_is_ligand & dst_is_ligand).sum().item()
        pl_edges = ((src_is_ligand & ~dst_is_ligand) | (~src_is_ligand & dst_is_ligand)).sum().item()
        pp_same_edges = (~src_is_ligand & ~dst_is_ligand & (src_protein_id == dst_protein_id)).sum().item()
        pp_cross_edges = (~src_is_ligand & ~dst_is_ligand & (src_protein_id != dst_protein_id)).sum().item()
        
        # print(f"[DEBUG] Edge type analysis:")
        # print(f"  Ligand-Ligand: {ll_edges}")
        # print(f"  Protein-Ligand: {pl_edges}")
        # print(f"  Protein-Protein (same): {pp_same_edges}")
        # print(f"  Protein-Protein (cross): {pp_cross_edges}")
    
    # Keep edges if:
    # 1. At least one endpoint is a ligand (ligand-protein or ligand-ligand edges)
    # 2. Both endpoints are from the same protein (intra-protein edges)
    keep_mask = (src_is_ligand | dst_is_ligand) | (src_protein_id == dst_protein_id)
    
    return edge_index[:, keep_mask]

def batch_hybrid_edge_connection_selective(x, k, mask_ligand, batch, protein_id, add_p_index=False, debug=False):
    """
    Efficient edge connection that prevents cross-protein edges from being created initially.
    This approach is more efficient than creating all edges and then masking them.
    
    Args:
        x: node positions
        k: number of nearest neighbors
        mask_ligand: boolean mask for ligand atoms
        batch: batch indices
        protein_id: protein ID for each atom (-1 for ligands)
        add_p_index: if True, also create protein-protein edges
        debug: if True, print debugging information
    """
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    
    if debug:
        total_ligand_atoms = (mask_ligand == 1).sum().item()
        total_protein_atoms = (mask_ligand == 0).sum().item()
        unique_proteins = torch.unique(protein_id[protein_id >= 0])
        num_proteins = len(unique_proteins)
        # print(f"[DEBUG] Selective edge creation - Ligand: {total_ligand_atoms}, Protein: {total_protein_atoms}, Unique proteins: {num_proteins}")
        
        # for pid in unique_proteins:
        #     protein_atom_count = (protein_id == pid).sum().item()
        #     print(f"[DEBUG] Protein {pid}: {protein_atom_count} atoms")
    
    with torch.no_grad():
        for i in range(batch_size):
            batch_mask = (batch == i)
            ligand_index = (batch_mask & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = (batch_mask & (mask_ligand == 0)).nonzero()[:, 0]
            
            # 1. Ligand-Ligand edges (fully connected)
            if len(ligand_index) > 1:
                dst = torch.repeat_interleave(ligand_index, len(ligand_index))
                src = ligand_index.repeat(len(ligand_index))
                mask = dst != src
                dst, src = dst[mask], src[mask]
                ll_edge_index = torch.stack([src, dst])
                batch_ll_edge_index.append(ll_edge_index)
            else:
                batch_ll_edge_index.append(torch.zeros((2, 0), dtype=torch.long, device=x.device))
            
            # 2. Protein-Ligand edges (knn for each ligand to all proteins)
            if len(ligand_index) > 0 and len(protein_index) > 0:
                ligand_pos = x[ligand_index]
                protein_pos = x[protein_index]
                
                # Use the same logic as the original hybrid_edge_connection function
                ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
                ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
                knn_p_idx = torch.topk(ligand_protein_pos_dist, k=min(k, len(protein_index)), largest=False, dim=1).indices
                knn_p_idx = protein_index[knn_p_idx]
                knn_l_idx = torch.unsqueeze(ligand_index, 1)
                knn_l_idx = knn_l_idx.repeat(1, min(k, len(protein_index)))
                pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)
                pl_edge_index = pl_edge_index.view(2, -1)
                batch_pl_edge_index.append(pl_edge_index)
            else:
                batch_pl_edge_index.append(torch.zeros((2, 0), dtype=torch.long, device=x.device))
            
            # 3. Additional edges from knn_graph - exactly replicating original method but with filtering
            if add_p_index:
                ligand_pos = x[ligand_index] if len(ligand_index) > 0 else torch.zeros((0, 3), device=x.device)
                protein_pos = x[protein_index] if len(protein_index) > 0 else torch.zeros((0, 3), device=x.device)
                
                if len(protein_index) > 0:
                    all_pos = torch.cat([protein_pos, ligand_pos], 0)
                    p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')
                    p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                    p_src, p_dst = p_edge_index
                    all_index = torch.cat([protein_index, ligand_index], 0)
                    p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                    
                    # Apply selective filtering: keep only edges where both endpoints are from same protein
                    # or one endpoint is ligand (but dst must be protein due to earlier filtering)
                    src_protein_id = protein_id[p_edge_index[0]]
                    dst_protein_id = protein_id[p_edge_index[1]]
                    src_is_ligand = mask_ligand[p_edge_index[0]]
                    
                    # Keep edges if: 1) src is ligand (ligand->protein) or 2) both are same protein
                    keep_mask = src_is_ligand | (src_protein_id == dst_protein_id)
                    p_edge_index = p_edge_index[:, keep_mask]
                    
                    batch_p_edge_index.append(p_edge_index)
                else:
                    batch_p_edge_index.append(torch.zeros((2, 0), dtype=torch.long, device=x.device))
            else:
                batch_p_edge_index.append(torch.zeros((2, 0), dtype=torch.long, device=x.device))

    if debug:
        total_ll_edges = sum([ll.shape[1] for ll in batch_ll_edge_index])
        total_pl_edges = sum([pl.shape[1] for pl in batch_pl_edge_index])
        total_p_edges = sum([p.shape[1] for p in batch_p_edge_index]) if add_p_index else 0
        
        # print(f"[DEBUG] Selective edge creation results:")
        # print(f"  Ligand-Ligand (LL): {total_ll_edges}")
        # print(f"  Protein-Ligand (PL): {total_pl_edges}")
        # print(f"  Protein-Protein (same protein only): {total_p_edges}")
        # print(f"  Total: {total_ll_edges + total_pl_edges + total_p_edges}")

    # Combine all edges
    if add_p_index:
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]
    
    edge_index = torch.cat(edge_index, -1)
    
    return edge_index

def selective_knn_graph(x, k, batch, protein_id=None, mask_ligand=None, flow='source_to_target', debug=False):
    """
    KNN graph generation with protein ID-based filtering to prevent cross-protein edges.
    
    Args:
        x: node positions [N, 3]
        k: number of nearest neighbors
        batch: batch indices [N]
        protein_id: protein ID for each node (-1 for ligands) [N]
        mask_ligand: boolean mask for ligand nodes [N]
        flow: edge direction ('source_to_target' or 'target_to_source')
        debug: if True, print debugging information
    
    Returns:
        edge_index: [2, num_edges] edge tensor with cross-protein edges filtered out
    """
    if protein_id is None or mask_ligand is None:
        # Fallback to original knn_graph if no protein filtering needed
        return knn_graph(x, k=k, batch=batch, flow=flow)
    
    batch_size = batch.max().item() + 1
    edge_list = []
    
    if debug:
        # print(f"[DEBUG] selective_knn_graph: batch_size={batch_size}, k={k}")
        unique_proteins = torch.unique(protein_id[protein_id >= 0])
        # print(f"[DEBUG] Unique proteins: {unique_proteins.tolist()}")
    
    for b_idx in range(batch_size):
        batch_mask = (batch == b_idx)
        batch_x = x[batch_mask]
        batch_protein_id = protein_id[batch_mask]
        batch_mask_ligand = mask_ligand[batch_mask]
        batch_node_indices = torch.where(batch_mask)[0]
        
        if len(batch_x) == 0:
            continue
            
        # Generate KNN edges for this batch
        if len(batch_x) <= k:
            # If fewer nodes than k, connect to all others
            n_nodes = len(batch_x)
            src_indices = torch.arange(n_nodes, device=x.device).repeat(n_nodes)
            dst_indices = torch.arange(n_nodes, device=x.device).repeat_interleave(n_nodes)
            # Remove self-loops
            mask_no_self = src_indices != dst_indices
            src_indices = src_indices[mask_no_self]
            dst_indices = dst_indices[mask_no_self]
        else:
            # Use KNN
            batch_edge_index = knn_graph(batch_x, k=k, flow=flow)
            src_indices, dst_indices = batch_edge_index
        
        if len(src_indices) == 0:
            continue
            
        # Apply protein ID filtering
        src_protein_ids = batch_protein_id[src_indices]
        dst_protein_ids = batch_protein_id[dst_indices]
        src_is_ligand = batch_mask_ligand[src_indices]
        dst_is_ligand = batch_mask_ligand[dst_indices]
        
        # Keep edges if:
        # 1. At least one endpoint is a ligand (ligand can connect to any protein)
        # 2. Both endpoints are from the same protein
        keep_mask = (src_is_ligand | dst_is_ligand) | (src_protein_ids == dst_protein_ids)
        
        if keep_mask.sum() > 0:
            filtered_src = src_indices[keep_mask]
            filtered_dst = dst_indices[keep_mask]
            
            # Map back to global indices
            global_src = batch_node_indices[filtered_src]
            global_dst = batch_node_indices[filtered_dst]
            
            batch_edges = torch.stack([global_src, global_dst], dim=0)
            edge_list.append(batch_edges)
    
    if len(edge_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=x.device)
    
    edge_index = torch.cat(edge_list, dim=1)
    
    if debug:
        # print(f"[DEBUG] selective_knn_graph generated {edge_index.shape[1]} edges")
        pass
        
    return edge_index

def selective_radius_graph(x, r, batch, protein_id=None, mask_ligand=None, flow='source_to_target', max_num_neighbors=32, debug=False):
    """
    Radius graph generation with protein ID-based filtering to prevent cross-protein edges.
    
    Args:
        x: node positions [N, 3]
        r: radius threshold
        batch: batch indices [N]
        protein_id: protein ID for each node (-1 for ligands) [N]
        mask_ligand: boolean mask for ligand nodes [N]
        flow: edge direction ('source_to_target' or 'target_to_source')
        max_num_neighbors: maximum number of neighbors per node
        debug: if True, print debugging information
    
    Returns:
        edge_index: [2, num_edges] edge tensor with cross-protein edges filtered out
    """
    if protein_id is None or mask_ligand is None:
        # Fallback to original radius_graph if no protein filtering needed
        return radius_graph(x, r=r, batch=batch, flow=flow, max_num_neighbors=max_num_neighbors)
    
    batch_size = batch.max().item() + 1
    edge_list = []
    
    if debug:
        # print(f"[DEBUG] selective_radius_graph: batch_size={batch_size}, r={r}")
        unique_proteins = torch.unique(protein_id[protein_id >= 0])
        # print(f"[DEBUG] Unique proteins: {unique_proteins.tolist()}")
    
    for b_idx in range(batch_size):
        batch_mask = (batch == b_idx)
        batch_x = x[batch_mask]
        batch_protein_id = protein_id[batch_mask]
        batch_mask_ligand = mask_ligand[batch_mask]
        batch_node_indices = torch.where(batch_mask)[0]
        
        if len(batch_x) == 0:
            continue
            
        # Generate radius-based edges for this batch
        batch_edge_index = radius_graph(batch_x, r=r, flow=flow, max_num_neighbors=max_num_neighbors)
        
        if batch_edge_index.shape[1] == 0:
            continue
            
        src_indices, dst_indices = batch_edge_index
        
        # Apply protein ID filtering
        src_protein_ids = batch_protein_id[src_indices]
        dst_protein_ids = batch_protein_id[dst_indices]
        src_is_ligand = batch_mask_ligand[src_indices]
        dst_is_ligand = batch_mask_ligand[dst_indices]
        
        # Keep edges if:
        # 1. At least one endpoint is a ligand (ligand can connect to any protein)
        # 2. Both endpoints are from the same protein
        keep_mask = (src_is_ligand | dst_is_ligand) | (src_protein_ids == dst_protein_ids)
        
        if keep_mask.sum() > 0:
            filtered_src = src_indices[keep_mask]
            filtered_dst = dst_indices[keep_mask]
            
            # Map back to global indices
            global_src = batch_node_indices[filtered_src]
            global_dst = batch_node_indices[filtered_dst]
            
            batch_edges = torch.stack([global_src, global_dst], dim=0)
            edge_list.append(batch_edges)
    
    if len(edge_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=x.device)
    
    edge_index = torch.cat(edge_list, dim=1)
    
    if debug:
        # print(f"[DEBUG] selective_radius_graph generated {edge_index.shape[1]} edges")
        pass
        
    return edge_index

def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False, protein_id=None, mask_cross_protein=False, debug=False):
    """
    Create edge connections with optional cross-protein masking for selectivity learning.
    
    Args:
        mask_cross_protein: If True, mask edges between different proteins
        protein_id: Protein ID tensor (needed for cross-protein masking)
        debug: If True, print detailed debugging information
    """
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    
    if debug:
        # Count atoms by type for debugging
        total_ligand_atoms = (mask_ligand == 1).sum().item()
        total_protein_atoms = (mask_ligand == 0).sum().item()
        
        if protein_id is not None:
            unique_proteins = torch.unique(protein_id[protein_id >= 0])
            num_proteins = len(unique_proteins)
            # print(f"[DEBUG] Atom counts - Ligand: {total_ligand_atoms}, Protein: {total_protein_atoms}, Unique proteins: {num_proteins}")
            
            # Count atoms per protein
            for pid in unique_proteins:
                protein_atom_count = (protein_id == pid).sum().item()
                # print(f"[DEBUG] Protein {pid}: {protein_atom_count} atoms")
    
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            batch_ll_edge_index.append(ll_edge_index)
            batch_pl_edge_index.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                batch_p_edge_index.append(p_edge_index)

    if debug:
        # Count edges by type
        total_ll_edges = sum([ll.shape[1] for ll in batch_ll_edge_index])
        total_pl_edges = sum([pl.shape[1] for pl in batch_pl_edge_index])
        total_p_edges = sum([p.shape[1] for p in batch_p_edge_index]) if add_p_index else 0
        
        # print(f"[DEBUG] Edge counts before masking:")
        # print(f"  Ligand-Ligand (LL): {total_ll_edges}")
        # print(f"  Protein-Ligand (PL): {total_pl_edges}")
        # print(f"  Protein-Protein (PP): {total_p_edges}")
        # print(f"  Total: {total_ll_edges + total_pl_edges + total_p_edges}")

    # Combine all edges
    if add_p_index:
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]
    edge_index = torch.cat(edge_index, -1)
    
    if debug:
        # print(f"[DEBUG] Total edges before masking: {edge_index.shape[1]}")
        pass
    
    # Apply cross-protein masking if requested
    if mask_cross_protein and protein_id is not None:
        original_count = edge_index.shape[1]
        edge_index = mask_cross_protein_edges(edge_index, protein_id, mask_ligand, debug)
        masked_count = edge_index.shape[1]
        removed_count = original_count - masked_count
        if debug:
            # print(f"[DEBUG] Cross-protein masking applied:")
            # print(f"  Edges before masking: {original_count}")
            # print(f"  Edges after masking: {masked_count}")
            # print(f"  Edges removed: {removed_count}")
            # print(f"  Reduction: {removed_count/original_count*100:.1f}%")
            pass
    
    return edge_index
