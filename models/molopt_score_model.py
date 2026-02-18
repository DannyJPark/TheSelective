"""
Unified Score Model for KGDiff with simplified dual-head architecture.

Refactored version with:
1. No hidden_dim*2 concatenation - all embeddings stay at hidden_dim
2. Atom-level BA prediction with scatter_mean (like original KGDiff)
3. Clean head2_mode branching: protein_query_atom, ligand_query_atom, bidirectional_query_atom

Config structure:
    use_dual_head_sam_pl: False  -> Original KGDiff (single head)
    use_dual_head_sam_pl: True   -> Dual head with head2_mode
        head2_mode: 'protein_query_atom'     -> Protein atoms query Ligand atoms
        head2_mode: 'ligand_query_atom'      -> Ligand atoms query Protein atoms
        head2_mode: 'bidirectional_query_atom' -> Both directions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral


def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace( ## 1/T, 1/(T-1), 1/(T-2), ..., 1
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# Categorical diffusion utilities
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    return gumbel_noise + logits


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScorePosNet3D(nn.Module):
    """
    Unified Score Model with clean dual-head architecture.

    Head1: Original KGDiff style
        - Protein-ligand complex graph (WITH interaction)
        - Ligand atom embeddings -> expert_pred_head1 -> scatter_mean(batch_ligand)

    Head2: Cross-attention based (NO interaction between protein and ligand)
        - Protein and ligand processed separately
        - Cross-attention between them
        - Atom-level prediction -> scatter_mean
    """

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config

        # === Diffusion schedule ===
        self.model_mean_type = config.model_mean_type # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        self.loss_exp_weight = config.loss_exp_weight
        self.sample_time_method = config.sample_time_method # ['importance', 'symmetric']
        self.use_classifier_guide = config.use_classifier_guide

        # Beta schedule
        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # Computed tensor (will be moved to device when sqrt_* tensors are moved)
        self._pos_classifier_grad_weight = None  # Lazy initialization

         # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # Atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # === Model definition ===
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim

        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # Atom embeddings
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        
        # center pos
        self.center_pos_mode = config.center_pos_mode # ['none', 'protein']

        # Time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        # RefineNet
        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)

        # Ligand type inference (for generation)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

        # === Head configuration ===
        self.use_dual_head_sam_pl = getattr(config, 'use_dual_head_sam_pl', False)
        self.head2_mode = getattr(config, 'head2_mode', 'protein_query_atom')

        # Head1: Original KGDiff style (atom-level prediction from ligand)
        # Input: hidden_dim, Output: 1
        self.expert_pred_head1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        if self.use_dual_head_sam_pl:
            # Head2: Cross-attention based prediction
            # Initialize cross-attention layers based on head2_mode
            self._init_head2_layers()
            print(f"[INFO] Dual-head Sam-PL enabled with head2_mode: {self.head2_mode}")
        else:
            print(f"[INFO] Single head mode (Original KGDiff)")

    @property
    def pos_classifier_grad_weight(self):
        """Compute grad weight on the same device as model parameters."""
        if self._pos_classifier_grad_weight is None or \
           self._pos_classifier_grad_weight.device != self.sqrt_one_minus_alphas_cumprod.device:
            self._pos_classifier_grad_weight = (
                self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
            )
        return self._pos_classifier_grad_weight

    def _init_head2_layers(self):
        """Initialize Head2 layers based on head2_mode."""

        if self.head2_mode == 'protein_query_atom':
            # Protein atoms query Ligand atoms
            # Output: attended protein atom embeddings -> expert_pred -> scatter_mean
            self.head2_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_key = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_value = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_output = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Atom-level BA prediction for attended protein atoms
            self.expert_pred_head2 = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            print(f"  Head2: Protein atoms (Query) attend to Ligand atoms (K,V)")
            print(f"  Output: scatter_mean over protein atoms")

        elif self.head2_mode == 'ligand_query_atom':
            # Ligand atoms query Protein atoms
            # Output: attended ligand atom embeddings -> expert_pred -> scatter_mean
            self.head2_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_key = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_value = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_output = nn.Linear(self.hidden_dim, self.hidden_dim)

            self.expert_pred_head2 = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            print(f"  Head2: Ligand atoms (Query) attend to Protein atoms (K,V)")
            print(f"  Output: scatter_mean over ligand atoms")

        elif self.head2_mode == 'bidirectional_query_atom':
            # Bidirectional: P->L and L->P
            # Protein -> Ligand attention
            self.head2_p2l_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_p2l_key = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_p2l_value = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_p2l_output = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Ligand -> Protein attention
            self.head2_l2p_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_l2p_key = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_l2p_value = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.head2_l2p_output = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Separate expert_pred for each direction
            self.expert_pred_head2_p2l = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            self.expert_pred_head2_l2p = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            print(f"  Head2: Bidirectional (P->L and L->P)")
            print(f"  Output: average of both direction predictions")

        else:
            raise ValueError(f"Unknown head2_mode: {self.head2_mode}")

    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):
        """
        Forward pass - Original KGDiff style (single head).
        Used when use_dual_head_sam_pl=False.
        """
        batch_size = batch_protein.max().item() + 1

        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()

        # Time embedding 
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # Compose protein-ligand complex graph
        h_all, pos_all, batch_all, mask_ligand, mask_protein, _ = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        # RefineNet forward
        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        # Head1: Original KGDiff BA prediction (atom-level ligand)
        atom_affinity = self.expert_pred_head1(final_ligand_h).squeeze(-1)  # (N_ligand,)
        final_exp_pred = scatter_mean(atom_affinity, batch_ligand)  # (batch_size,)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
            'atom_affinity': atom_affinity,
            'final_exp_pred': final_exp_pred,
            'batch_all': batch_all,
            'mask_ligand': mask_ligand
        }

        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    #### Foward Dual HEAD ####
    def forward_dual_head(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                          time_step=None, return_all=False, fix_x=False):
        """
        Forward pass with dual heads.

        Head1: Original KGDiff (ligand from complex graph WITH interaction)
        Head2: Cross-attention based (protein and ligand processed separately, NO interaction)
        """
        batch_size = batch_protein.max().item() + 1

        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()

        
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # === Head1: Protein-ligand complex graph (WITH interaction) ===
        h_complex, pos_complex, batch_complex, mask_ligand_complex, mask_protein_complex, _ = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs_complex = self.refine_net(h_complex, pos_complex, mask_ligand_complex, batch_complex,
                                          return_all=return_all, fix_x=fix_x)

        # Extract ligand from complex (for generation and Head1)
        final_ligand_h_complex = outputs_complex['h'][mask_ligand_complex]
        final_ligand_pos = outputs_complex['x'][mask_ligand_complex]
        final_ligand_v = self.v_inference(final_ligand_h_complex)

        # Head1 BA prediction: atom-level ligand from complex
        atom_affinity_head1 = self.expert_pred_head1(final_ligand_h_complex).squeeze(-1)  # (N_ligand,)
        pred_affinity_head1 = scatter_mean(atom_affinity_head1, batch_ligand)  # (batch_size,)

        # === Head2: Process protein and ligand separately (NO interaction) ===
        # Process protein only
        mask_protein_only = torch.zeros(len(h_protein), dtype=torch.bool, device=h_protein.device)
        outputs_protein = self.refine_net(h_protein, protein_pos, mask_protein_only, batch_protein,
                                          return_all=False, fix_x=fix_x)
        protein_h_sep = outputs_protein['h']  # (N_protein, hidden_dim)

        # Process ligand only
        mask_ligand_only = torch.ones(len(init_ligand_h), dtype=torch.bool, device=init_ligand_h.device)
        outputs_ligand = self.refine_net(init_ligand_h, init_ligand_pos, mask_ligand_only, batch_ligand,
                                         return_all=False, fix_x=fix_x)
        ligand_h_sep = outputs_ligand['h']  # (N_ligand, hidden_dim)

        # Head2 cross-attention BA prediction
        pred_affinity_head2, atom_affinity_head2 = self._compute_head2_prediction(
            protein_h_sep, ligand_h_sep, batch_protein, batch_ligand, batch_size
        )

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': outputs_complex['h'],
            'final_ligand_h': final_ligand_h_complex,
            'mask_ligand': mask_ligand_complex,
            'batch_all': batch_complex,
            # Head1
            'atom_affinity_head1': atom_affinity_head1,
            'pred_affinity_head1': pred_affinity_head1,
            # Head2
            'atom_affinity_head2': atom_affinity_head2,
            'pred_affinity_head2': pred_affinity_head2,
            # Aliases for compatibility
            'v_head1_pred': pred_affinity_head1,
            'v_head2_pred': pred_affinity_head2,
        }

        if return_all:
            final_all_pos, final_all_h = outputs_complex['all_x'], outputs_complex['all_h']
            final_all_ligand_pos = [pos[mask_ligand_complex] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand_complex]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })

        return preds

    def _compute_head2_prediction(self, protein_h, ligand_h, batch_protein, batch_ligand, batch_size):
        """
        Compute Head2 cross-attention based BA prediction.

        Args:
            protein_h: (N_protein, hidden_dim) - protein atom embeddings (no interaction)
            ligand_h: (N_ligand, hidden_dim) - ligand atom embeddings (no interaction)
            batch_protein: (N_protein,) - batch indices for protein atoms
            batch_ligand: (N_ligand,) - batch indices for ligand atoms
            batch_size: int

        Returns:
            pred_affinity: (batch_size,) - final BA prediction
            atom_affinity: atom-level affinities (varies by mode)
        """

        if self.head2_mode == 'protein_query_atom':
            return self._head2_protein_query(protein_h, ligand_h, batch_protein, batch_ligand, batch_size)
        elif self.head2_mode == 'ligand_query_atom':
            return self._head2_ligand_query(protein_h, ligand_h, batch_protein, batch_ligand, batch_size)
        elif self.head2_mode == 'bidirectional_query_atom':
            return self._head2_bidirectional(protein_h, ligand_h, batch_protein, batch_ligand, batch_size)
        else:
            raise ValueError(f"Unknown head2_mode: {self.head2_mode}")

    def _head2_protein_query(self, protein_h, ligand_h, batch_protein, batch_ligand, batch_size):
        """
        Protein atoms query Ligand atoms.

        Flow:
        1. Q = protein atoms (N_prot, 128)
        2. K, V = ligand atoms (N_lig, 128)
        3. Attention: (N_prot, N_lig) per batch
        4. Output: attended protein embeddings (N_prot, 128)
        5. expert_pred: (N_prot,)
        6. scatter_mean(batch_protein): (batch_size,)
        """
        attended_protein = []

        for b in range(batch_size):
            prot_mask = batch_protein == b
            lig_mask = batch_ligand == b

            prot_h_b = protein_h[prot_mask]  # (N_prot_b, hidden_dim)
            lig_h_b = ligand_h[lig_mask]      # (N_lig_b, hidden_dim)

            # Cross-attention
            Q = self.head2_query(prot_h_b)     # (N_prot_b, hidden_dim)
            K = self.head2_key(lig_h_b)        # (N_lig_b, hidden_dim)
            V = self.head2_value(lig_h_b)      # (N_lig_b, hidden_dim)

            # Attention: Q @ K.T / sqrt(d)
            attn_scores = torch.matmul(Q, K.transpose(0, 1)) / np.sqrt(self.hidden_dim)  # (N_prot_b, N_lig_b)
            attn_weights = F.softmax(attn_scores, dim=-1)  # (N_prot_b, N_lig_b)

            # Output: attn @ V
            attended = torch.matmul(attn_weights, V)  # (N_prot_b, hidden_dim)
            attended = self.head2_output(attended)     # (N_prot_b, hidden_dim)

            attended_protein.append(attended)

        # Concatenate all batches
        attended_protein = torch.cat(attended_protein, dim=0)  # (N_protein_total, hidden_dim)

        # Atom-level BA prediction
        atom_affinity = self.expert_pred_head2(attended_protein).squeeze(-1)  # (N_protein,)

        # Aggregate per batch
        pred_affinity = scatter_mean(atom_affinity, batch_protein)  # (batch_size,)

        return pred_affinity, atom_affinity

    def _head2_ligand_query(self, protein_h, ligand_h, batch_protein, batch_ligand, batch_size):
        """
        Ligand atoms query Protein atoms.

        Flow:
        1. Q = ligand atoms (N_lig, 128)
        2. K, V = protein atoms (N_prot, 128)
        3. Attention: (N_lig, N_prot) per batch
        4. Output: attended ligand embeddings (N_lig, 128)
        5. expert_pred: (N_lig,)
        6. scatter_mean(batch_ligand): (batch_size,)
        """
        attended_ligand = []

        for b in range(batch_size):
            prot_mask = batch_protein == b
            lig_mask = batch_ligand == b

            prot_h_b = protein_h[prot_mask]  # (N_prot_b, hidden_dim)
            lig_h_b = ligand_h[lig_mask]      # (N_lig_b, hidden_dim)

            # Cross-attention: ligand queries protein
            Q = self.head2_query(lig_h_b)      # (N_lig_b, hidden_dim)
            K = self.head2_key(prot_h_b)       # (N_prot_b, hidden_dim)
            V = self.head2_value(prot_h_b)     # (N_prot_b, hidden_dim)

            attn_scores = torch.matmul(Q, K.transpose(0, 1)) / np.sqrt(self.hidden_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)

            attended = torch.matmul(attn_weights, V)
            attended = self.head2_output(attended)

            attended_ligand.append(attended)

        attended_ligand = torch.cat(attended_ligand, dim=0)

        atom_affinity = self.expert_pred_head2(attended_ligand).squeeze(-1)
        pred_affinity = scatter_mean(atom_affinity, batch_ligand)

        return pred_affinity, atom_affinity

    def _head2_bidirectional(self, protein_h, ligand_h, batch_protein, batch_ligand, batch_size):
        """
        Bidirectional cross-attention: P->L and L->P.

        Flow:
        1. P->L: protein queries ligand -> attended_protein (N_prot, 128)
        2. L->P: ligand queries protein -> attended_ligand (N_lig, 128)
        3. expert_pred on each -> scatter_mean
        4. Average both predictions
        """
        attended_protein_list = []
        attended_ligand_list = []

        for b in range(batch_size):
            prot_mask = batch_protein == b
            lig_mask = batch_ligand == b

            prot_h_b = protein_h[prot_mask]
            lig_h_b = ligand_h[lig_mask]

            # P->L attention
            Q_p2l = self.head2_p2l_query(prot_h_b)
            K_p2l = self.head2_p2l_key(lig_h_b)
            V_p2l = self.head2_p2l_value(lig_h_b)

            attn_p2l = F.softmax(torch.matmul(Q_p2l, K_p2l.transpose(0, 1)) / np.sqrt(self.hidden_dim), dim=-1)
            attended_prot = self.head2_p2l_output(torch.matmul(attn_p2l, V_p2l))
            attended_protein_list.append(attended_prot)

            # L->P attention
            Q_l2p = self.head2_l2p_query(lig_h_b)
            K_l2p = self.head2_l2p_key(prot_h_b)
            V_l2p = self.head2_l2p_value(prot_h_b)

            attn_l2p = F.softmax(torch.matmul(Q_l2p, K_l2p.transpose(0, 1)) / np.sqrt(self.hidden_dim), dim=-1)
            attended_lig = self.head2_l2p_output(torch.matmul(attn_l2p, V_l2p))
            attended_ligand_list.append(attended_lig)

        attended_protein = torch.cat(attended_protein_list, dim=0)
        attended_ligand = torch.cat(attended_ligand_list, dim=0)

        # P->L direction prediction
        atom_affinity_p2l = self.expert_pred_head2_p2l(attended_protein).squeeze(-1)
        pred_affinity_p2l = scatter_mean(atom_affinity_p2l, batch_protein)

        # L->P direction prediction
        atom_affinity_l2p = self.expert_pred_head2_l2p(attended_ligand).squeeze(-1)
        pred_affinity_l2p = scatter_mean(atom_affinity_l2p, batch_ligand)

        # Average both directions
        pred_affinity = (pred_affinity_p2l + pred_affinity_l2p) / 2

        # Return combined atom affinities (for logging)
        atom_affinity = {
            'p2l': atom_affinity_p2l,
            'l2p': atom_affinity_l2p
        }

        return pred_affinity, atom_affinity

    # === Diffusion methods ===
    # Compute q(vt | v0 )
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)
        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_prob = log_sample_categorical(log_qvt_v0)
        sample_index = sample_prob.argmax(dim=-1)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior
    
    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')
            
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt
        # Uniform Time Sampling
        elif method == 'symmetric':
            time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt
        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)
        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(self, protein_pos, protein_v, affinity, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None):
        """
        Original KGDiff diffusion loss (single head).
        Used when use_dual_head_sam_pl=False.
        """
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)
        
        # Sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)

        # Perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise

        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

         # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        # Atom Position loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # Atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)

        # Affinity loss
        loss_exp = F.mse_loss(preds['final_exp_pred'], affinity)

        if self.use_classifier_guide:
            loss = loss_pos + loss_v * self.loss_v_weight + loss_exp * self.loss_exp_weight
        else:
            loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_exp': preds['final_exp_pred'],
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'final_ligand_h': preds['final_ligand_h']
        }

    def get_diffusion_loss_dual_head(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand,
                                      on_target_affinity, time_step=None):
        """
        Dual-head diffusion loss.
        Used when use_dual_head_sam_pl=True.

        Both heads predict the same on_target_affinity (for now).
        """
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. Sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        # 2. Select alpha values based on timestep
        a = self.alphas_cumprod.index_select(0, time_step)

        # Perturb pos and v
        # Perturb atom positions
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise

        # Perturb atom types
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # Forward pass with dual heads
        # forward-pass NN, feed perturbed pos and v, output noise

        preds = self.forward_dual_head(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError
        
        # Atom Position loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # Atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)

        # Affinity losses (both heads predict same target for now)
        loss_head1 = F.mse_loss(preds['pred_affinity_head1'], on_target_affinity)
        loss_head2 = F.mse_loss(preds['pred_affinity_head2'], on_target_affinity)
        loss_exp = loss_head1 + loss_head2

        # Total loss
        loss = loss_pos + loss_v * self.loss_v_weight + loss_exp * self.loss_exp_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss_head1': loss_head1,
            'loss_head2': loss_head2,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_head1_exp': preds['pred_affinity_head1'],
            'pred_head2_exp': preds['pred_affinity_head2'],
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'final_ligand_h': preds['final_ligand_h']
        }

    def get_diffusion_loss_generation_only(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None):
        """
        TargetDiff-style generation-only loss (no binding affinity).
        Only computes position and atom type reconstruction losses.

        Used when train_mode='generation_only'.
        """
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # Sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)

        # Perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise

        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # Forward pass (only generation, skip affinity prediction)
        preds = self._forward_generation_only(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        # Position loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # Atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)

        # No affinity loss - this is pure generation training
        loss_exp = torch.tensor(0.0, device=protein_pos.device)

        # Total loss (only position and atom type)
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'final_ligand_h': preds['final_ligand_h']
        }

    def _forward_generation_only(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand, time_step):
        """
        Forward pass for generation-only mode.
        Only computes position and atom type, skips affinity prediction.
        """
        batch_size = batch_protein.max().item() + 1

        # Convert to one-hot if necessary (same as _forward)
        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()

        # Embed protein atoms
        h_protein = self.protein_atom_emb(protein_v)

        # Time conditioning for ligand (same as _forward)
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat[batch_ligand]], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_ligand = self.ligand_atom_emb(input_ligand_feat)

        # Add node indicator
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1, device=h_protein.device)], dim=-1)
            h_ligand = torch.cat([h_ligand, torch.ones(len(h_ligand), 1, device=h_ligand.device)], dim=-1)

        # Compose protein-ligand complex (same as _forward)
        h_all, pos_all, batch_all, mask_ligand, mask_protein, _ = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand
        )

        # RefineNet forward
        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all)

        final_pos = outputs['x']
        final_h = outputs['h']
        final_ligand_pos = final_pos[mask_ligand]
        final_ligand_h = final_h[mask_ligand]

        # Ligand type inference
        pred_ligand_v = self.v_inference(final_ligand_h)

        return {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'final_ligand_h': final_ligand_h
        }

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None):
        """
        Sample from the diffusion model.
        Uses single head (original KGDiff) forward.
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj, exp_traj = [], [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)

            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )

            # Compute posterior
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)

            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)

            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
            ligand_pos = ligand_pos_next

            log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            ligand_v_next = log_sample_categorical(log_model_prob).argmax(dim=-1)
            ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
            exp_traj.append(preds['final_exp_pred'].clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'exp': exp_traj[-1] if exp_traj else None,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'exp_traj': exp_traj,
        }

    def _compute_head2_off_target_prediction(self, off_protein_pos, off_protein_v, off_batch_protein,
                                              ligand_pos, ligand_v_onehot, batch_ligand, time_step):
        """
        Compute Head2 prediction for OFF-TARGET protein (NO interaction, cross-attention based).

        This processes protein and ligand separately, then uses cross-attention for BA prediction.

        Args:
            off_protein_pos: Off-target protein positions
            off_protein_v: Off-target protein features
            off_batch_protein: Off-target protein batch indices
            ligand_pos: Ligand positions
            ligand_v_onehot: Ligand atom type one-hot encoding
            batch_ligand: Ligand batch indices
            time_step: Current timestep

        Returns:
            pred_affinity_head2: (batch_size,) - Head2 BA prediction for off-target
        """
        batch_size = batch_ligand.max().item() + 1

        # Time embedding for ligand
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    ligand_v_onehot,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([ligand_v_onehot, time_feat[batch_ligand]], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = ligand_v_onehot

        # Embed protein and ligand
        h_protein = self.protein_atom_emb(off_protein_v)
        h_ligand = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            h_ligand = torch.cat([h_ligand, torch.ones(len(h_ligand), 1).to(h_ligand)], -1)

        # === Process protein SEPARATELY (NO interaction with ligand) ===
        mask_protein_only = torch.zeros(len(h_protein), dtype=torch.bool, device=h_protein.device)
        outputs_protein = self.refine_net(h_protein, off_protein_pos, mask_protein_only, off_batch_protein,
                                          return_all=False, fix_x=False)
        protein_h_sep = outputs_protein['h']  # (N_protein, hidden_dim)

        # === Process ligand SEPARATELY (NO interaction with protein) ===
        mask_ligand_only = torch.ones(len(h_ligand), dtype=torch.bool, device=h_ligand.device)
        outputs_ligand = self.refine_net(h_ligand, ligand_pos, mask_ligand_only, batch_ligand,
                                         return_all=False, fix_x=False)
        ligand_h_sep = outputs_ligand['h']  # (N_ligand, hidden_dim)

        # === Head2 cross-attention BA prediction ===
        pred_affinity_head2, _ = self._compute_head2_prediction(
            protein_h_sep, ligand_h_sep, off_batch_protein, batch_ligand, batch_size
        )

        return pred_affinity_head2

    def sample_diffusion_with_guidance(
        self, protein_pos, protein_v, batch_protein,
        init_ligand_pos, init_ligand_v, batch_ligand,
        off_target_data=None, num_steps=None, center_pos_mode=None,
        guide_mode='no_guide',
        head1_type_grad_weight=0., head1_pos_grad_weight=0.,
        head2_type_grad_weight=0., head2_pos_grad_weight=0.,
        w_on=1.0, w_off=1.0
    ):
        """
        Sample from the diffusion model with dual-head selectivity guidance.

        Supports three guidance modes:
        1. 'dual_head_guidance': Original mode - both heads use same forward with different proteins
        2. 'head1_head2_sequential': 2-step sequential guidance (both heads applied at all timesteps)
           - Step 1: Head1 (WITH interaction) for on-target BA → apply guidance
           - Step 2: Head2 (NO interaction, cross-attention) for off-target BA → apply repulsive guidance
        3. 'head1_head2_staged': Staged guidance (Head2 deactivated after t<500)
           - Early Stage (t>=500): Head1 + Head2 (on-target + off-target) guidance
           - Late Stage (t<500): Head1 (on-target) guidance ONLY
           - Rationale: Apply selectivity guidance early when exploring structure space,
             then focus on on-target binding refinement in later steps.

        Args:
            protein_pos: On-target protein positions
            protein_v: On-target protein features
            batch_protein: Protein batch indices
            init_ligand_pos: Initial ligand positions (noise)
            init_ligand_v: Initial ligand atom types
            batch_ligand: Ligand batch indices
            off_target_data: Off-target protein data object (optional)
            num_steps: Number of diffusion steps
            center_pos_mode: Position centering mode
            guide_mode: 'no_guide', 'dual_head_guidance', 'head1_head2_sequential', or 'head1_head2_staged'
            head1/2_type/pos_grad_weight: Gradient weights for each head
            w_on/w_off: Weights for on-target/off-target affinity

        Returns:
            Dictionary with sampled ligands and trajectories
        """
        import gc

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        device = protein_pos.device

        # Center on-target protein
        protein_pos_on, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        # Center off-target protein if provided
        if off_target_data is not None and guide_mode != 'no_guide':
            # Create dummy ligand positions for centering
            off_protein_pos = off_target_data.protein_pos.to(device)
            off_protein_v = off_target_data.protein_atom_feature.float().to(device)
            off_batch_protein = torch.zeros(len(off_protein_pos), dtype=torch.long, device=device)
            # Repeat for batch
            if num_graphs > 1:
                off_protein_pos_list = [off_protein_pos.clone() for _ in range(num_graphs)]
                off_protein_v_list = [off_protein_v.clone() for _ in range(num_graphs)]
                off_batch_protein = torch.repeat_interleave(
                    torch.arange(num_graphs, device=device),
                    torch.tensor([len(off_protein_pos)] * num_graphs, device=device)
                )
                off_protein_pos = torch.cat(off_protein_pos_list, dim=0)
                off_protein_v = torch.cat(off_protein_v_list, dim=0)

            # Center off-target using same mode
            off_protein_pos, _, _ = center_pos(
                off_protein_pos, torch.zeros(len(init_ligand_pos), 3, device=device),
                off_batch_protein, batch_ligand, mode=center_pos_mode)
        else:
            off_protein_pos = None
            off_protein_v = None
            off_batch_protein = None

        pos_traj, v_traj = [], []
        exp_on_traj, exp_off_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))

        for iter_idx, i in enumerate(tqdm(time_seq, desc='sampling', total=len(time_seq))):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=device)

            if guide_mode == 'no_guide':
                # No guidance - standard sampling with no_grad
                with torch.no_grad():
                    preds = self(
                        protein_pos=protein_pos_on,
                        protein_v=protein_v,
                        batch_protein=batch_protein,
                        init_ligand_pos=ligand_pos,
                        init_ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        time_step=t
                    )
                exp_on = preds['final_exp_pred']
                exp_off = torch.zeros_like(exp_on)
                head1_pos_grad = torch.zeros_like(ligand_pos)
                head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                head2_pos_grad = torch.zeros_like(ligand_pos)
                head2_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)

            elif guide_mode == 'head1_only':
                # ====================================================================
                # HEAD1_ONLY MODE (Original KGDiff style)
                # - Only Head1: On-target binding affinity prediction
                # - Uses standard forward() method (no dual head needed)
                # - No off-target guidance
                # ====================================================================
                if iter_idx == 0:
                    print(f"\n*** ENTERING head1_only MODE (Original KGDiff) ***")
                    print(f"*** Head1: On-target binding affinity only ***\n")

                # Compute Head1 guidance with standard forward
                if head1_type_grad_weight > 0 or head1_pos_grad_weight > 0:
                    with torch.enable_grad():
                        ligand_pos_h1 = ligand_pos.detach().requires_grad_(True)
                        ligand_v_onehot = F.one_hot(ligand_v, self.num_classes).float().requires_grad_(True)

                        # Use standard forward (no dual head)
                        preds = self(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos_h1,
                            init_ligand_v=ligand_v_onehot,
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                        exp_on = preds['final_exp_pred']

                        # Compute log(affinity) for position gradient
                        exp_on_log = exp_on.log()

                        # Type gradient
                        head1_v_grad = torch.autograd.grad(
                            exp_on.sum(), ligand_v_onehot,
                            grad_outputs=None,
                            retain_graph=True, create_graph=False, allow_unused=True
                        )[0]

                        # Position gradient
                        head1_pos_grad = torch.autograd.grad(
                            exp_on_log.sum(), ligand_pos_h1,
                            grad_outputs=None,
                            retain_graph=False, create_graph=False, allow_unused=True
                        )[0]

                        if head1_v_grad is None:
                            head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                        if head1_pos_grad is None:
                            head1_pos_grad = torch.zeros_like(ligand_pos)

                    del ligand_pos_h1, ligand_v_onehot
                    torch.cuda.empty_cache()
                else:
                    with torch.no_grad():
                        preds = self(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos,
                            init_ligand_v=F.one_hot(ligand_v, self.num_classes).float(),
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                    exp_on = preds['final_exp_pred']
                    head1_pos_grad = torch.zeros_like(ligand_pos)
                    head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)

                # No off-target in head1_only mode
                exp_off = torch.zeros_like(exp_on)
                head2_pos_grad = torch.zeros_like(ligand_pos)
                head2_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)

            elif guide_mode == 'head1_head2_sequential':
                # ====================================================================
                # NEW: HEAD1_HEAD2_SEQUENTIAL MODE
                # - Head1: On-target + WITH interaction (complex graph) → BA prediction
                # - Head2: Off-target + NO interaction (cross-attention) → BA prediction
                # - 2-step sequential gradient application
                # ====================================================================
                if iter_idx == 0:
                    print(f"\n*** ENTERING head1_head2_sequential MODE ***")
                    print(f"*** Head1: On-target (WITH interaction) ***")
                    print(f"*** Head2: Off-target (NO interaction, cross-attention) ***\n")

                # === STEP 1: Head1 - On-target with interaction ===
                if head1_type_grad_weight > 0 or head1_pos_grad_weight > 0:
                    with torch.enable_grad():
                        ligand_pos_h1 = ligand_pos.detach().requires_grad_(True)
                        ligand_v_onehot = F.one_hot(ligand_v, self.num_classes).float().requires_grad_(True)

                        # Use forward_dual_head to get Head1 prediction (WITH interaction)
                        preds_on = self.forward_dual_head(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos_h1,
                            init_ligand_v=ligand_v_onehot,
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                        # Head1: pred_affinity_head1 (WITH interaction)
                        exp_on = preds_on['pred_affinity_head1']

                        # Compute log(affinity) for position gradient
                        exp_on_log = exp_on.log()

                        # Type gradient: use affinity directly
                        head1_v_grad = torch.autograd.grad(
                            exp_on.sum(), ligand_v_onehot,
                            grad_outputs=None,
                            retain_graph=True, create_graph=False, allow_unused=True
                        )[0]

                        # Position gradient: use log(affinity)
                        head1_pos_grad = torch.autograd.grad(
                            exp_on_log.sum(), ligand_pos_h1,
                            grad_outputs=None,
                            retain_graph=False, create_graph=False, allow_unused=True
                        )[0]

                        if head1_v_grad is None:
                            head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                        if head1_pos_grad is None:
                            head1_pos_grad = torch.zeros_like(ligand_pos)

                    # Clean up
                    del ligand_pos_h1, ligand_v_onehot, preds_on
                    torch.cuda.empty_cache()
                else:
                    with torch.no_grad():
                        preds_on = self.forward_dual_head(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos,
                            init_ligand_v=F.one_hot(ligand_v, self.num_classes).float(),
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                    exp_on = preds_on['pred_affinity_head1']
                    head1_pos_grad = torch.zeros_like(ligand_pos)
                    head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                    del preds_on
                    torch.cuda.empty_cache()

                # === STEP 2: Head2 - Off-target with NO interaction (cross-attention) ===
                if off_protein_pos is not None and (head2_type_grad_weight > 0 or head2_pos_grad_weight > 0):
                    with torch.enable_grad():
                        ligand_pos_h2 = ligand_pos.detach().requires_grad_(True)
                        ligand_v_onehot2 = F.one_hot(ligand_v, self.num_classes).float().requires_grad_(True)

                        # Use Head2 (NO interaction, cross-attention) for off-target
                        exp_off = self._compute_head2_off_target_prediction(
                            off_protein_pos=off_protein_pos,
                            off_protein_v=off_protein_v,
                            off_batch_protein=off_batch_protein,
                            ligand_pos=ligand_pos_h2,
                            ligand_v_onehot=ligand_v_onehot2,
                            batch_ligand=batch_ligand,
                            time_step=t
                        )

                        if iter_idx == 0:
                            print(f"[DEBUG] Head2 (cross-attention) exp_off: {exp_off}")

                        # Compute log(affinity) for position gradient
                        exp_off_log = exp_off.log()

                        # Type gradient: use affinity directly
                        head2_v_grad = torch.autograd.grad(
                            exp_off.sum(), ligand_v_onehot2,
                            grad_outputs=None,
                            retain_graph=True, create_graph=False, allow_unused=True
                        )[0]

                        # Position gradient: use log(affinity)
                        head2_pos_grad = torch.autograd.grad(
                            exp_off_log.sum(), ligand_pos_h2,
                            grad_outputs=None,
                            retain_graph=False, create_graph=False, allow_unused=True
                        )[0]

                        if head2_v_grad is None:
                            head2_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                        if head2_pos_grad is None:
                            head2_pos_grad = torch.zeros_like(ligand_pos)

                    # Clean up
                    del ligand_pos_h2, ligand_v_onehot2
                    torch.cuda.empty_cache()
                else:
                    exp_off = torch.zeros(num_graphs, device=device)
                    head2_pos_grad = torch.zeros_like(ligand_pos)
                    head2_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)

            elif guide_mode == 'head1_head2_staged':
                # ====================================================================
                # NEW: HEAD1_HEAD2_STAGED MODE (REVERSED)
                # - First 500 steps (t=999~500): On-target + Off-target (Head1 + Head2) gradients
                # - Last 500 steps (t=499~0): On-target (Head1) gradient ONLY
                #
                # Rationale: Apply selectivity guidance early when exploring structure space,
                # then focus on on-target binding refinement in later steps.
                # ====================================================================
                STAGE_THRESHOLD = 500  # Switch from Head1+Head2 to Head1-only at t=500

                if iter_idx == 0:
                    print(f"\n*** ENTERING head1_head2_staged MODE (REVERSED) ***")
                    print(f"*** Stage 1 (t>=500): On-target + Off-target (Head1+Head2) guidance ***")
                    print(f"*** Stage 2 (t<500): On-target (Head1) guidance ONLY ***\n")

                # Determine current stage based on timestep
                current_t = i  # Current timestep value (999 -> 0)
                in_early_stage = current_t >= STAGE_THRESHOLD  # t>=500: Head1+Head2 (both)

                # === STEP 1: Head1 - On-target with interaction (ALWAYS applied) ===
                if head1_type_grad_weight > 0 or head1_pos_grad_weight > 0:
                    with torch.enable_grad():
                        ligand_pos_h1 = ligand_pos.detach().requires_grad_(True)
                        ligand_v_onehot = F.one_hot(ligand_v, self.num_classes).float().requires_grad_(True)

                        # Use forward_dual_head to get Head1 prediction (WITH interaction)
                        preds_on = self.forward_dual_head(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos_h1,
                            init_ligand_v=ligand_v_onehot,
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                        # Head1: pred_affinity_head1 (WITH interaction)
                        exp_on = preds_on['pred_affinity_head1']

                        # Compute log(affinity) for position gradient
                        exp_on_log = exp_on.log()

                        # Type gradient: use affinity directly
                        head1_v_grad = torch.autograd.grad(
                            exp_on.sum(), ligand_v_onehot,
                            grad_outputs=None,
                            retain_graph=True, create_graph=False, allow_unused=True
                        )[0]

                        # Position gradient: use log(affinity)
                        head1_pos_grad = torch.autograd.grad(
                            exp_on_log.sum(), ligand_pos_h1,
                            grad_outputs=None,
                            retain_graph=False, create_graph=False, allow_unused=True
                        )[0]

                        if head1_v_grad is None:
                            head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                        if head1_pos_grad is None:
                            head1_pos_grad = torch.zeros_like(ligand_pos)

                    # Clean up
                    del ligand_pos_h1, ligand_v_onehot, preds_on
                    torch.cuda.empty_cache()
                else:
                    with torch.no_grad():
                        preds_on = self.forward_dual_head(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos,
                            init_ligand_v=F.one_hot(ligand_v, self.num_classes).float(),
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                    exp_on = preds_on['pred_affinity_head1']
                    head1_pos_grad = torch.zeros_like(ligand_pos)
                    head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                    del preds_on
                    torch.cuda.empty_cache()

                # === STEP 2: Head2 - Off-target (ONLY in Early Stage, t >= 500) ===
                if in_early_stage and off_protein_pos is not None and (head2_type_grad_weight > 0 or head2_pos_grad_weight > 0):
                    # Stage 2: Apply off-target guidance
                    with torch.enable_grad():
                        ligand_pos_h2 = ligand_pos.detach().requires_grad_(True)
                        ligand_v_onehot2 = F.one_hot(ligand_v, self.num_classes).float().requires_grad_(True)

                        # Use Head2 (NO interaction, cross-attention) for off-target
                        exp_off = self._compute_head2_off_target_prediction(
                            off_protein_pos=off_protein_pos,
                            off_protein_v=off_protein_v,
                            off_batch_protein=off_batch_protein,
                            ligand_pos=ligand_pos_h2,
                            ligand_v_onehot=ligand_v_onehot2,
                            batch_ligand=batch_ligand,
                            time_step=t
                        )

                        # Compute log(affinity) for position gradient
                        exp_off_log = exp_off.log()

                        # Type gradient: use affinity directly
                        head2_v_grad = torch.autograd.grad(
                            exp_off.sum(), ligand_v_onehot2,
                            grad_outputs=None,
                            retain_graph=True, create_graph=False, allow_unused=True
                        )[0]

                        # Position gradient: use log(affinity)
                        head2_pos_grad = torch.autograd.grad(
                            exp_off_log.sum(), ligand_pos_h2,
                            grad_outputs=None,
                            retain_graph=False, create_graph=False, allow_unused=True
                        )[0]

                        if head2_v_grad is None:
                            head2_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                        if head2_pos_grad is None:
                            head2_pos_grad = torch.zeros_like(ligand_pos)

                    # Clean up
                    del ligand_pos_h2, ligand_v_onehot2
                    torch.cuda.empty_cache()
                else:
                    # Late Stage (t<500): No off-target gradient, but still measure BA for trajectory
                    head2_pos_grad = torch.zeros_like(ligand_pos)
                    head2_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)

                    if off_protein_pos is not None:
                        # Forward only (no grad) to measure Head2 BA for trajectory visualization
                        with torch.no_grad():
                            exp_off = self._compute_head2_off_target_prediction(
                                off_protein_pos=off_protein_pos,
                                off_protein_v=off_protein_v,
                                off_batch_protein=off_batch_protein,
                                ligand_pos=ligand_pos,
                                ligand_v_onehot=F.one_hot(ligand_v, self.num_classes).float(),
                                batch_ligand=batch_ligand,
                                time_step=t
                            )
                    else:
                        exp_off = torch.zeros(num_graphs, device=device)

            else:  # dual_head_guidance (original mode)
                # ====================================================================
                # ORIGINAL: Both heads use same forward() with different proteins
                # HEAD1: On-target affinity (maximize)
                # HEAD2: Off-target affinity (minimize) - same model, different protein
                # ====================================================================
                if iter_idx == 0:
                    print(f"\n*** ENTERING dual_head_guidance MODE (original) ***")
                    print(f"*** Both heads use same forward() with different proteins ***\n")

                if head1_type_grad_weight > 0 or head1_pos_grad_weight > 0:
                    with torch.enable_grad():
                        ligand_pos_h1 = ligand_pos.detach().requires_grad_(True)
                        ligand_v_onehot = F.one_hot(ligand_v, self.num_classes).float().requires_grad_(True)

                        preds_on = self(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos_h1,
                            init_ligand_v=ligand_v_onehot,
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                        exp_on = preds_on['final_exp_pred']

                        # Compute log(affinity) for position gradient
                        exp_on_log = exp_on.log()

                        # Type gradient: use affinity directly
                        head1_v_grad = torch.autograd.grad(
                            exp_on.sum(), ligand_v_onehot,
                            grad_outputs=None,
                            retain_graph=True, create_graph=False, allow_unused=True
                        )[0]

                        # Position gradient: use log(affinity)
                        head1_pos_grad = torch.autograd.grad(
                            exp_on_log.sum(), ligand_pos_h1,
                            grad_outputs=None,
                            retain_graph=False, create_graph=False, allow_unused=True
                        )[0]

                        if head1_v_grad is None:
                            head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                        if head1_pos_grad is None:
                            head1_pos_grad = torch.zeros_like(ligand_pos)

                    # Clean up
                    del ligand_pos_h1, ligand_v_onehot, preds_on
                else:
                    with torch.no_grad():
                        preds_on = self(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos,
                            init_ligand_v=F.one_hot(ligand_v, self.num_classes).float(),
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                    exp_on = preds_on['final_exp_pred']
                    head1_pos_grad = torch.zeros_like(ligand_pos)
                    head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                    del preds_on

                # HEAD2: Off-target (same model, different protein)
                if off_protein_pos is not None and (head2_type_grad_weight > 0 or head2_pos_grad_weight > 0):
                    with torch.enable_grad():
                        ligand_pos_h2 = ligand_pos.detach().requires_grad_(True)
                        ligand_v_onehot2 = F.one_hot(ligand_v, self.num_classes).float().requires_grad_(True)

                        preds_off = self(
                            protein_pos=off_protein_pos,
                            protein_v=off_protein_v,
                            batch_protein=off_batch_protein,
                            init_ligand_pos=ligand_pos_h2,
                            init_ligand_v=ligand_v_onehot2,
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                        exp_off = preds_off['final_exp_pred']

                        # Compute log(affinity) for position gradient
                        exp_off_log = exp_off.log()

                        # Type gradient: use affinity directly
                        head2_v_grad = torch.autograd.grad(
                            exp_off.sum(), ligand_v_onehot2,
                            grad_outputs=None,
                            retain_graph=True, create_graph=False, allow_unused=True
                        )[0]

                        # Position gradient: use log(affinity)
                        head2_pos_grad = torch.autograd.grad(
                            exp_off_log.sum(), ligand_pos_h2,
                            grad_outputs=None,
                            retain_graph=False, create_graph=False, allow_unused=True
                        )[0]

                        if head2_v_grad is None:
                            head2_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                        if head2_pos_grad is None:
                            head2_pos_grad = torch.zeros_like(ligand_pos)

                    # Clean up
                    del ligand_pos_h2, ligand_v_onehot2, preds_off
                else:
                    exp_off = torch.zeros(num_graphs, device=device)
                    head2_pos_grad = torch.zeros_like(ligand_pos)
                    head2_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)

            # === Compute posterior (always with no_grad) ===
            with torch.no_grad():
                # Get denoising prediction (use on-target complex for position/type prediction)
                preds = self(
                    protein_pos=protein_pos_on,
                    protein_v=protein_v,
                    batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )

                if self.model_mean_type == 'noise':
                    pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                    pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                    v0_from_e = preds['pred_ligand_v']
                elif self.model_mean_type == 'C0':
                    pos0_from_e = preds['pred_ligand_pos']
                    v0_from_e = preds['pred_ligand_v']
                else:
                    raise ValueError

                # Compute posterior
                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)

                # type posterior
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)

                # ====================================================================
                # 2-STEP SEQUENTIAL GRADIENT APPLICATION
                # Step 1: Apply Head1 on-target guidance (positive direction)
                # Step 2: Apply Head2 off-target guidance (negative/repulsive direction)
                # ====================================================================

                # Step 1: Head1 on-target guidance (+)
                pos_model_mean_h1 = pos_model_mean + w_on * head1_pos_grad_weight * (0.5 * pos_log_variance).exp() * head1_pos_grad
                log_ligand_v_h1 = log_ligand_v + w_on * head1_type_grad_weight * head1_v_grad

                # Step 2: Head2 off-target guidance (-) - repulsive
                pos_model_mean_final = pos_model_mean_h1 - w_off * head2_pos_grad_weight * (0.5 * pos_log_variance).exp() * head2_pos_grad
                log_ligand_v_final = log_ligand_v_h1 - w_off * head2_type_grad_weight * head2_v_grad

                # no noise when t == 0
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

                # Sample next step
                ligand_pos_next = pos_model_mean_final + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph

                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v_final, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob).argmax(dim=-1)
                ligand_v = ligand_v_next.detach()  # Break computation graph

                # Store trajectory
                ori_ligand_pos = ligand_pos + offset[batch_ligand]
                pos_traj.append(ori_ligand_pos.clone().cpu())
                v_traj.append(ligand_v.clone().cpu())
                exp_on_traj.append(exp_on.detach().clone().cpu())
                exp_off_traj.append(exp_off.detach().clone().cpu())

            # Clean up iteration variables
            del preds, head1_pos_grad, head1_v_grad, head2_pos_grad, head2_v_grad

            # Clear CUDA cache periodically
            if iter_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

        # Final positions with offset
        ligand_pos = ligand_pos + offset[batch_ligand]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'exp_on': exp_on_traj[-1] if exp_on_traj else None,
            'exp_off': exp_off_traj[-1] if exp_off_traj else None,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'exp_on_traj': exp_on_traj,
            'exp_off_traj': exp_off_traj,
            'v0_traj': [],  # Compatibility
            'vt_traj': [],  # Compatibility
        }


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
