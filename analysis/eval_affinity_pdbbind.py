"""
Zero-shot evaluation of TheSelective dual-head binding-affinity predictor
on PDBbind test sets: CASF-2013 (195), CASF-2016 (285), test2019 (EHIGN holdout).

The affinity heads were trained on CrossDocked2020 Vina scores, normalized to
[0, 1] via (0 - clip(vina, -16, 0)) / 16  ->  1.0 == strong binding (very
negative Vina), 0.0 == weak binding. PDBbind labels are experimental pK
(-logKd/Ki). Because the two quantities live on different scales/units, the
meaningful zero-shot metric is RANK CORRELATION between head output and pK
(higher head output should mean higher pK). We report Pearson / Spearman /
Kendall (and RMSE/MAE after a per-set linear fit, for reference only).

Model is run on the CRYSTAL pose at t=0 (no diffusion noise). With
time_emb_dim=0 the network has no time conditioning, so the clean pose is the
natural inference input.
"""
import os
import sys
import argparse
import csv
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.transforms as trans
from utils.data import PDBProtein, parse_sdf_file
from datasets.pl_data import ProteinLigandData, torchify_dict, FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D

PROTEIN_DATA = '/home/phj/ProteinData'
EHIGN = '/home/phj/EHIGN_PLA/data'
POCKET_RADIUS = 10


# --------------------------------------------------------------------------- #
# Test-set definitions
# --------------------------------------------------------------------------- #
def _read_coreset_dat(path):
    """CASF CoreSet.dat -> {pdbid: logKa}. col0=code, col3=logKa."""
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                labels[parts[0].lower()] = float(parts[3])
            except ValueError:
                continue
    return labels


def _read_csv_labels(path):
    labels = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                labels[row[0].lower()] = float(row[1])
            except ValueError:
                continue
    return labels


def build_testsets():
    sets = {}

    # CASF-2013 core set (195)
    lab = _read_coreset_dat(f'{PROTEIN_DATA}/CASF/CASF-2013/power_scoring/CoreSet.dat')
    root = f'{PROTEIN_DATA}/CASF/CASF-2013/coreset'
    items = []
    for pid in sorted(os.listdir(root)):
        d = os.path.join(root, pid)
        if not os.path.isdir(d) or pid not in lab:
            continue
        items.append(dict(pdbid=pid, label=lab[pid],
                          protein=os.path.join(d, f'{pid}_protein.pdb'),
                          ligands=[os.path.join(d, f'{pid}_ligand.sdf'),
                                   os.path.join(d, f'{pid}_ligand.mol2')]))
    sets['CASF-2013'] = items

    # CASF-2016 core set (285)
    lab = _read_coreset_dat(f'{PROTEIN_DATA}/CASF/CASF-2016/power_scoring/CoreSet.dat')
    root = f'{PROTEIN_DATA}/CASF/CASF-2016/coreset'
    items = []
    for pid in sorted(os.listdir(root)):
        d = os.path.join(root, pid)
        if not os.path.isdir(d) or pid not in lab:
            continue
        items.append(dict(pdbid=pid, label=lab[pid],
                          protein=os.path.join(d, f'{pid}_protein.pdb'),
                          ligands=[os.path.join(d, f'{pid}_ligand.sdf'),
                                   os.path.join(d, f'{pid}_ligand.mol2')]))
    sets['CASF-2016'] = items

    # test2019 (EHIGN holdout) - raw files live in ProteinData/PDBbind/v2019
    lab = _read_csv_labels(f'{EHIGN}/test2019.csv')
    root = f'{PROTEIN_DATA}/PDBbind/v2019'
    items = []
    for pid in sorted(lab.keys()):
        d = os.path.join(root, pid)
        if not os.path.isdir(d):
            continue
        items.append(dict(pdbid=pid, label=lab[pid],
                          protein=os.path.join(d, f'{pid}_protein.pdb'),
                          ligands=[os.path.join(d, f'{pid}_ligand.sdf'),
                                   os.path.join(d, f'{pid}_ligand.mol2')]))
    sets['test2019'] = items

    return sets


# --------------------------------------------------------------------------- #
# Per-complex featurization
# --------------------------------------------------------------------------- #
def make_transform(ligand_atom_mode):
    return [trans.FeaturizeProteinAtom(),
            trans.FeaturizeLigandAtom(ligand_atom_mode),
            trans.FeaturizeLigandBond()]


def load_ligand(paths):
    last_err = None
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            return parse_sdf_file(p)
        except Exception as e:  # noqa
            last_err = e
            continue
    raise RuntimeError(f'ligand parse failed: {last_err}')


class _LigShim:
    """Minimal RDKit-mol-like shim so PDBProtein.query_residues_ligand works
    from a parsed positions array (it only calls .GetConformer().GetPositions())."""
    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)
    def GetConformer(self):
        return self
    def GetPositions(self):
        return self._pos


def fix_pdb_elements(block):
    """Many CASF / PDBbind raw protein.pdb files leave the element column (77-78)
    blank, which makes PDBProtein fall back to the wrong atom-name character
    (e.g. 'D' from CD1, 'G' from CG). Protein ATOM records only contain
    H/C/N/O/S(/Se), so the element is the first alphabetic char of the atom name.
    Inject it into cols 77-78 so PDBProtein parses correctly."""
    out = []
    for line in block.splitlines():
        if line[0:6].strip() in ('ATOM', 'HETATM'):
            line = line.ljust(78)
            if line[76:78].strip() == '':
                name = line[12:16].strip()
                el = ''
                for ch in name:
                    if ch.isalpha():
                        el = ch
                        break
                line = line[:76] + el.rjust(2) + line[78:]
        out.append(line)
    return '\n'.join(out)


def build_data(item, transforms):
    ligand_dict = load_ligand(item['ligands'])
    with open(item['protein']) as f:
        pdb_block = f.read()
    protein = PDBProtein(fix_pdb_elements(pdb_block))
    # 10A pocket: residues whose center_of_mass is within radius of any ligand atom
    sel = protein.query_residues_ligand(_LigShim(ligand_dict['pos']), POCKET_RADIUS)
    pocket_block = protein.residues_to_pdb_block(sel)
    pocket_dict = PDBProtein(pocket_block).to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = item['protein']
    data.ligand_filename = item['ligands'][0]
    for t in transforms:
        data = t(data)
    assert data.protein_pos.size(0) > 0 and data.ligand_pos.size(0) > 0
    return data


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_batch(model, datas, device):
    batch = Batch.from_data_list([d.clone() for d in datas],
                                 follow_batch=FOLLOW_BATCH).to(device)
    batch_protein = batch.protein_element_batch
    batch_ligand = batch.ligand_element_batch

    protein_pos = batch.protein_pos
    ligand_pos = batch.ligand_pos
    # center on protein CoM (matches training center_pos_mode='protein').
    offset = scatter_mean(protein_pos, batch_protein, dim=0)
    protein_pos = protein_pos - offset[batch_protein]
    ligand_pos = ligand_pos - offset[batch_ligand]

    n = batch_protein.max().item() + 1
    time_step = torch.zeros(n, dtype=torch.long, device=device)

    preds = model.forward_dual_head(
        protein_pos=protein_pos,
        protein_v=batch.protein_atom_feature.float(),
        batch_protein=batch_protein,
        init_ligand_pos=ligand_pos,
        init_ligand_v=batch.ligand_atom_feature_full,
        batch_ligand=batch_ligand,
        time_step=time_step,
    )
    h1 = preds['pred_affinity_head1'].detach().cpu().numpy()
    h2 = preds['pred_affinity_head2'].detach().cpu().numpy()
    return h1, h2


def metrics(pred, label):
    from scipy.stats import pearsonr, spearmanr, kendalltau
    pred = np.asarray(pred, float)
    label = np.asarray(label, float)
    out = {}
    out['pearson'] = float(pearsonr(pred, label)[0])
    out['spearman'] = float(spearmanr(pred, label)[0])
    out['kendall'] = float(kendalltau(pred, label)[0])
    # RMSE/MAE after per-set linear fit pred->label (units differ; reference only)
    a, b = np.polyfit(pred, label, 1)
    fit = a * pred + b
    out['rmse'] = float(np.sqrt(np.mean((fit - label) ** 2)))
    out['mae'] = float(np.mean(np.abs(fit - label)))
    out['n'] = int(len(pred))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='./checkpoints/theselective.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--out_dir', default='./analysis/affinity_eval')
    ap.add_argument('--only', default=None, help='comma list to restrict sets')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt['config']
    ligand_atom_mode = cfg.data.transform.ligand_atom_mode
    transforms = make_transform(ligand_atom_mode)
    prot_feat = trans.FeaturizeProteinAtom()
    lig_feat = trans.FeaturizeLigandAtom(ligand_atom_mode)

    model = ScorePosNet3D(cfg.model,
                          protein_atom_feature_dim=prot_feat.feature_dim,
                          ligand_atom_feature_dim=lig_feat.feature_dim).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[INFO] model loaded from {args.ckpt} (iter {ckpt.get("iteration")})')
    print(f'[INFO] dual_head={getattr(cfg.model, "use_dual_head_sam_pl", False)} '
          f'head2_mode={getattr(cfg.model, "head2_mode", None)} '
          f'ligand_atom_mode={ligand_atom_mode}')

    sets = build_testsets()
    if args.only:
        keep = set(args.only.split(','))
        sets = {k: v for k, v in sets.items() if k in keep}

    summary = {}
    for name, items in sets.items():
        print(f'\n==== {name}: {len(items)} complexes ====')
        rows = []  # (pdbid, label, h1, h2)
        skipped = []
        buf, meta = [], []

        def flush():
            if not buf:
                return
            try:
                h1, h2 = predict_batch(model, buf, device)
            except Exception as e:  # fall back to per-item on batch failure
                for d, m in zip(buf, meta):
                    try:
                        a, b = predict_batch(model, [d], device)
                        rows.append((m['pdbid'], m['label'], float(a[0]), float(b[0])))
                    except Exception as e2:
                        skipped.append((m['pdbid'], f'infer:{e2}'))
                buf.clear(); meta.clear(); return
            for m, a, b in zip(meta, h1, h2):
                rows.append((m['pdbid'], m['label'], float(a), float(b)))
            buf.clear(); meta.clear()

        for item in tqdm(items, desc=name):
            try:
                d = build_data(item, transforms)
            except Exception as e:
                skipped.append((item['pdbid'], f'build:{e}'))
                continue
            buf.append(d); meta.append(item)
            if len(buf) >= args.batch_size:
                flush()
        flush()

        # write per-complex predictions
        csv_path = os.path.join(args.out_dir, f'{name}_predictions.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pdbid', 'label_pK', 'head1', 'head2'])
            for r in rows:
                w.writerow(r)

        if len(rows) < 3:
            print(f'  [WARN] too few successful ({len(rows)}); skipped={len(skipped)}')
            continue

        labels = [r[1] for r in rows]
        m1 = metrics([r[2] for r in rows], labels)
        m2 = metrics([r[3] for r in rows], labels)
        summary[name] = dict(n=len(rows), skipped=len(skipped), head1=m1, head2=m2)
        print(f'  n={len(rows)}  skipped={len(skipped)}')
        print(f'  Head1: Pearson={m1["pearson"]:.4f}  Spearman={m1["spearman"]:.4f}  '
              f'Kendall={m1["kendall"]:.4f}  RMSE={m1["rmse"]:.3f}')
        print(f'  Head2: Pearson={m2["pearson"]:.4f}  Spearman={m2["spearman"]:.4f}  '
              f'Kendall={m2["kendall"]:.4f}  RMSE={m2["rmse"]:.3f}')
        if skipped[:5]:
            print('  first skipped:', skipped[:5])

    # final summary table
    print('\n' + '=' * 78)
    print(f'{"Set":<12}{"N":>5}{"Head":>7}{"Pearson":>10}{"Spearman":>10}'
          f'{"Kendall":>9}{"RMSE":>8}')
    print('-' * 78)
    for name, s in summary.items():
        for hk in ('head1', 'head2'):
            m = s[hk]
            print(f'{name:<12}{s["n"]:>5}{hk[-1]:>7}{m["pearson"]:>10.4f}'
                  f'{m["spearman"]:>10.4f}{m["kendall"]:>9.4f}{m["rmse"]:>8.3f}')
    print('=' * 78)

    import json
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[INFO] wrote summary + per-complex CSVs to {args.out_dir}')


if __name__ == '__main__':
    main()
