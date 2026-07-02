# PIDiff van der Waals loss + single-head KGDiff affinity guidance — applied patch

**Date:** 2026-07-02
**Env:** conda `theselective` (torch 1.12.1) · **Data:** `./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` (train=99990)

## Goal / hypothesis

Test whether **(a)** a PIDiff-style van der Waals (Lennard-Jones) **training loss** combined with **(b)** a trained binding-affinity predictor used for **guidance at sampling** (KGDiff / TheSelective style) yields molecules with **higher binding affinity**. All three code bases (PIDiff, KGDiff, TheSelective) are TargetDiff forks trained/tested on CrossDock2020.

## Approach (minimal change)

Add PIDiff's **self-contained** vdW loss *into* TheSelective (which already provides the affinity head + guidance sampler + local data/checkpoints). Per decision:

- **Single-head** KGDiff-style `loss_exp` — `use_dual_head_sam_pl: False`. The training script already branches to `get_diffusion_loss(...)` (single head) in this case, so **no branch code change** was needed.
- **From-scratch** training (not fine-tune). Rationale: (1) the physics prior should shape the whole learned distribution; (2) `train_diffusion.py:97-100` — passing `--ckpt` **overrides the CLI config with the checkpoint's embedded config**, so fine-tuning the old 844k dual-head checkpoint would silently *not* see the new vdW keys and would be dual-head. From-scratch reads `training.yml` directly.
- **PIDiff-style curriculum ramp**: the vdW weight grows as `it / vdw_ramp_iters`. Applied in the train loop (the model only returns the raw vdW energy) because the iteration `it` is not known inside the model's loss method — same structure as PIDiff's `energy * (it/200000)`.

All changes are **backward-compatible** (new config keys read via `getattr` defaults; `results.get('loss_vdw', ...)`), so the existing dual-head config/checkpoints keep working unchanged. The existing `configs/training.yml` was **not** modified — a dedicated config file was added instead.

## Files changed

| File | Change |
|---|---|
| `models/molopt_score_model.py` | `__init__` reads `vdw_loss_mode` / `vdw_dm_min`; new `compute_vdw_loss()`; single-head `get_diffusion_loss` returns `'loss_vdw'` |
| `scripts/train_diffusion.py` | single-head branch adds the ramped vdW term to `loss` before `backward()` |
| `configs/training_pidiff_vdw_singlehead.yml` | **new** experiment config (single head + vdW), `training.yml` left untouched |

### `compute_vdw_loss` (Lennard-Jones 12-6)

Pure function of geometry — ligand↔protein pairwise distances only:
- `dm = sqrt(sum((lig-pro)^2) + 1e-10)` (manual, finite gradient as `d→0`)
- guard hard clashes: `dm < vdw_dm_min (0.5)` → `1e10`
- `energy = (1/dm)^12 - 2*(1/dm)^6`, then `clamp(max=100)`
- **`vdw_loss_mode`**:
  - `'lj'` → loss = `energy.sum()` (clean LJ energy, single backward) — cheaper.
  - `'pidiff_exact'` → loss = `sum_ij dE/dd` via `autograd.grad(energy.sum(), dm, create_graph=True)` (PIDiff's original double-backward). Falls back to plain energy under `torch.no_grad()` so validation never crashes.
- returns a scalar (summed over all pairs, averaged over graphs).

## Unified diff

```diff
diff --git a/models/molopt_score_model.py b/models/molopt_score_model.py
@@ class ScorePosNet3D.__init__ (after self.use_classifier_guide) @@
+        # === Physics-informed van der Waals loss (PIDiff) ===
+        # Raw vdW energy is computed in get_diffusion_loss; the base weight and the
+        # PIDiff-style curriculum ramp (it / vdw_ramp_iters) are applied in the train loop.
+        self.vdw_loss_mode = getattr(config, 'vdw_loss_mode', 'lj')  # 'lj' | 'pidiff_exact'
+        self.vdw_dm_min = getattr(config, 'vdw_dm_min', 0.5)

@@ new method inserted before def get_diffusion_loss @@
+    def compute_vdw_loss(self, pred_ligand_pos, protein_pos, batch_ligand, batch_protein, num_graphs):
+        dm_min = self.vdw_dm_min
+        total = pred_ligand_pos.new_zeros(())
+        for i in range(num_graphs):
+            lig = pred_ligand_pos[batch_ligand == i]
+            pro = protein_pos[batch_protein == i]
+            if lig.shape[0] == 0 or pro.shape[0] == 0:
+                continue
+            diff = lig.unsqueeze(1) - pro.unsqueeze(0)            # (L, P, 3)
+            dm = torch.sqrt((diff ** 2).sum(-1) + 1e-10)          # (L, P)
+            dm = torch.where(dm < dm_min, torch.full_like(dm, 1e10), dm)
+            inv = 1.0 / dm
+            energy = torch.pow(inv, 12) - 2.0 * torch.pow(inv, 6)
+            energy = energy.clamp(max=100)
+            if self.vdw_loss_mode == 'pidiff_exact' and torch.is_grad_enabled() and dm.requires_grad:
+                der = torch.autograd.grad(energy.sum(), dm, retain_graph=True, create_graph=True)[0]
+                total = total + der.sum()
+            else:
+                total = total + energy.sum()
+        return total / max(num_graphs, 1)

@@ get_diffusion_loss (after loss is assembled, before return) @@
+        # Physics-informed van der Waals loss (raw energy; weight + PIDiff ramp applied in train loop)
+        loss_vdw = self.compute_vdw_loss(pred_ligand_pos, protein_pos, batch_ligand, batch_protein, num_graphs)
         return {
             'loss_pos': loss_pos,
             'loss_v': loss_v,
             'loss_exp': loss_exp,
+            'loss_vdw': loss_vdw,
             'loss': loss,
             ...

diff --git a/scripts/train_diffusion.py b/scripts/train_diffusion.py
@@ def main() train(it): single-head branch, before `loss = loss / n_acc_batch` @@
+            # === Physics-informed vdW loss with PIDiff-style curriculum ramp ===
+            # total vdW weight = loss_vdw_weight * (it / vdw_ramp_iters)   [uncapped, PIDiff-style]
+            loss_vdw = results.get('loss_vdw', None)
+            vdw_weight = getattr(config.model, 'loss_vdw_weight', 0.)
+            if loss_vdw is not None and vdw_weight > 0:
+                ramp_iters = getattr(config.train, 'vdw_ramp_iters', 200000)
+                loss = loss + vdw_weight * (it / ramp_iters) * loss_vdw
+
             loss = loss / config.train.n_acc_batch
             loss.backward()
```

### New config keys (`configs/training_pidiff_vdw_singlehead.yml`)

```yaml
model:
  use_dual_head_sam_pl: False   # single-head KGDiff-style loss_exp
  loss_vdw_weight: 1.0          # effective weight = loss_vdw_weight * (it / vdw_ramp_iters); 0. disables
  vdw_loss_mode: pidiff_exact   # 'pidiff_exact' (PIDiff dE/dd) | 'lj' (clean LJ12-6 energy)
  vdw_dm_min: 0.5
train:
  vdw_ramp_iters: 200000        # PIDiff-style linear ramp (uncapped)
```

## How to run

```bash
cd /home/phj/TheSelective
conda activate theselective

# Model D (physics ON):
python scripts/train_diffusion.py --config configs/training_pidiff_vdw_singlehead.yml

# Baseline B (single-head, physics OFF) — same config with loss_vdw_weight: 0.
#   (copy the config, set loss_vdw_weight: 0., train)

# Sampling (guidance UNCHANGED — single head uses Head1 on-target guidance):
python scripts/sample_diffusion.py <sampling.yml> \
  --guide_mode head1_only \
  --head1_pos_grad_weight <w> --head1_type_grad_weight <w> --w_on 1.0
```

## Verification (done 2026-07-02)

- `py_compile` clean on both edited files; module imports fine in `theselective`.
- **vdW unit test** (real `compute_vdw_loss`): both modes finite loss + nonzero finite grad; clash guard (overlapping atoms) → no inf/nan; `no_grad` validation path does not crash.
- **End-to-end smoke** on real lmdb batches (single head, 2.84M params):

  | it | ramp | loss_pos | loss_exp | vdw_raw | vdw_eff | total | grad |
  |---|---|---|---|---|---|---|---|
  | 1 | 0.000 | 1.585 | 0.048 | 1.1e-1 | 5.6e-7 | 1.785 | 2.23 |
  | 100000 | 0.500 | 2.516 | 0.027 | 3.07 | 1.53 | 4.237 | 27.6 |
  | 200000 | 1.000 | 0.227 | 0.049 | 0.33 | 0.33 | 0.978 | 1.54 |

  (rows are independent random batches at different ramp weights, not a training curve — they confirm the full path data→single-head loss→vdW double-backward→ramp→backward→optimizer step runs finite.)

## To decide / tune before a long run

1. **Ramp cap** — the ramp is *uncapped* (`it/vdw_ramp_iters`). With `max_iters: 10000000` the weight reaches ~5× by 1M iters. Set `vdw_ramp_iters` ≈ intended training horizon, or cap the multiplier at 1.0.
2. **Baseline** — for a clean A/B, train the *same* single-head config with `loss_vdw_weight: 0.`. The existing 844k checkpoint is dual-head, so it is **not** apples-to-apples.
3. **vdW formula** — default is faithful `pidiff_exact`; `vdw_loss_mode: lj` is a cheaper/cleaner one-line swap (avoids the double backward).
4. **Grad norm** — the vdW term can spike (see it=100000 → grad 27.6, clipped by `max_grad_norm: 8.0`); this is expected and handled by the existing clip.

## Ablation design

| model | vdW loss | affinity guidance | note |
|---|---|---|---|
| A | ✗ | ✗ | pure baseline |
| B | ✗ | ✓ | affinity guidance only |
| C | ✓ | ✗ | physics only |
| **D** | ✓ | ✓ | **hypothesis** — expect higher Vina affinity vs A/B/C |

Metrics: Vina dock score, steric-clash rate, QED/SA, validity.
