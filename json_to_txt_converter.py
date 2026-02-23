#!/usr/bin/env python3
"""
JSON Docking Results to TXT Converter
Converts docking results from JSON format to readable TXT format

Usage:
    python json_to_txt_converter.py docking_results.json -o output.txt -n "EXPERIMENT_NAME"
"""

import json
import argparse
import os
from pathlib import Path


def load_json_data(json_path):
    """Load JSON data from file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_protein_names(data):
    """Extract protein names from the data"""
    if not data:
        return {}

    first_entry = data[0]
    proteins = {}

    # On-target protein
    proteins['on_target'] = first_entry['docking_results']['on_target']['protein_info']['protein_name']

    # Off-target proteins - dynamically detect available off-targets
    off_targets = {}
    for key, value in first_entry['docking_results'].items():
        if key.startswith('off_target_'):
            off_targets[key] = value['protein_info']['protein_name']

    proteins['off_targets'] = off_targets
    proteins['off_target_keys'] = sorted(off_targets.keys())
    return proteins


def calculate_stats(affinities):
    """Calculate basic statistics for affinity values"""
    valid_affinities = [x for x in affinities if x is not None and isinstance(x, (int, float)) and x < 50]
    if not valid_affinities:
        return {"best": "N/A", "worst": "N/A", "average": "N/A"}

    return {
        "best": min(valid_affinities),
        "worst": max(valid_affinities),
        "average": sum(valid_affinities) / len(valid_affinities)
    }


def convert_json_to_txt(json_path, output_path=None, experiment_name=None):
    """Convert JSON docking results to readable TXT format"""

    # Load JSON data
    data = load_json_data(json_path)

    # Generate output path if not provided
    if output_path is None:
        json_file = Path(json_path)
        output_path = json_file.parent / f"{json_file.stem}_readable.txt"

    # Extract experiment name from path if not provided
    if experiment_name is None:
        experiment_name = Path(json_path).parent.parent.name

    # Get protein information
    proteins = get_protein_names(data)

    # Collect data for analysis
    ligands_data = []
    on_target_affinities = []
    off_target_data = {f'off_target_{i+1}': [] for i in range(3)}

    for entry in data:
        ligand_name = entry['ligand_name']
        on_target_aff = entry['on_target_affinity']
        off_target_affs = entry['off_target_affinities']
        selectivity = entry['selectivity_score']

        # Extract chemical properties from on_target docking results
        chem_props = {}
        if 'docking_results' in entry and 'on_target' in entry['docking_results']:
            on_target_results = entry['docking_results']['on_target']
            if 'chem_results' in on_target_results and on_target_results['chem_results'] is not None:
                chem_results = on_target_results['chem_results']
                chem_props = {
                    'qed': chem_results.get('qed', 'N/A'),
                    'sa': chem_results.get('sa', 'N/A'),
                    'logp': chem_results.get('logp', 'N/A'),
                    'lipinski': chem_results.get('lipinski', 'N/A'),
                    'ring_size': chem_results.get('ring_size', {})
                }

        ligands_data.append({
            'name': ligand_name,
            'on_target': on_target_aff,
            'off_targets': off_target_affs,
            'selectivity': selectivity,
            'chem_props': chem_props
        })

        on_target_affinities.append(on_target_aff)
        for i, aff in enumerate(off_target_affs):
            off_target_data[f'off_target_{i+1}'].append(aff)

    # Calculate statistics
    on_target_stats = calculate_stats(on_target_affinities)
    off_target_stats = {}
    for key, values in off_target_data.items():
        off_target_stats[key] = calculate_stats(values)

    # Generate TXT content
    content = []
    content.append("=" * 65)
    content.append(f"         {experiment_name.upper()} - Docking Results")
    content.append("=" * 65)
    content.append("")

    # Protein Information
    content.append("Protein Information:")
    content.append(f"- On-target: {proteins['on_target']}")
    for i, (key, name) in enumerate(proteins['off_targets'].items(), 1):
        content.append(f"- Off-target {i}: {name}")
    content.append("")

    # Summary table helper
    def create_summary_table_with_median(all_vina_data):
        """Create summary table with average AND median scores for all ligands"""
        summary_content = []
        summary_content.append("=" * 170)
        summary_content.append("                    SUMMARY - AVERAGE & MEDIAN SCORES FOR ALL LIGANDS")
        summary_content.append("=" * 170)

        def safe_float_convert(val_str):
            if val_str == "N/A" or val_str is None:
                return None
            try:
                return float(val_str)
            except (ValueError, TypeError):
                return None

        def calc_stats(values):
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                return None, None
            avg = sum(valid_values) / len(valid_values)
            sorted_vals = sorted(valid_values)
            n = len(sorted_vals)
            median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
            return avg, median

        # Collect all values
        metrics = {
            'on_target_score': [], 'on_target_min': [], 'on_target_dock': [],
            'dock_selectivity': [], 'qed': [], 'sa': []
        }

        off_target_keys = proteins.get('off_target_keys', [])
        for i, off_key in enumerate(off_target_keys, 1):
            metrics[f'off_target_{i}_score'] = []
            metrics[f'off_target_{i}_min'] = []
            metrics[f'off_target_{i}_dock'] = []

        for vina_data in all_vina_data:
            metrics['on_target_score'].append(safe_float_convert(vina_data['on_target']['score']))
            metrics['on_target_min'].append(safe_float_convert(vina_data['on_target']['min']))
            metrics['on_target_dock'].append(safe_float_convert(vina_data['on_target']['dock']))

            num_off_targets = len(off_target_keys)
            for i in range(num_off_targets):
                if i < len(vina_data['off_targets']):
                    metrics[f'off_target_{i+1}_score'].append(safe_float_convert(vina_data['off_targets'][i]['score']))
                    metrics[f'off_target_{i+1}_min'].append(safe_float_convert(vina_data['off_targets'][i]['min']))
                    metrics[f'off_target_{i+1}_dock'].append(safe_float_convert(vina_data['off_targets'][i]['dock']))
                else:
                    metrics[f'off_target_{i+1}_score'].append(None)
                    metrics[f'off_target_{i+1}_min'].append(None)
                    metrics[f'off_target_{i+1}_dock'].append(None)

            dock_selectivity = vina_data['selectivity']['dock']
            metrics['dock_selectivity'].append(safe_float_convert(dock_selectivity) if dock_selectivity != "N/A" else None)
            qed_val = vina_data['chem_props'].get('qed')
            sa_val = vina_data['chem_props'].get('sa')
            metrics['qed'].append(qed_val if isinstance(qed_val, (int, float)) else None)
            metrics['sa'].append(sa_val if isinstance(sa_val, (int, float)) else None)

        stats = {}
        for metric, values in metrics.items():
            avg, median = calc_stats(values)
            stats[metric] = {
                'avg': f"{avg:.3f}" if avg is not None else "N/A",
                'median': f"{median:.3f}" if median is not None else "N/A"
            }

        # Create table
        header_parts = [f"{'Metric':<20}", f"{'On-Target':<30}"]
        for i in range(len(off_target_keys)):
            header_parts.append(f"{'Off-Target ' + str(i+1):<30}")
        header_parts.extend([f"{'Dock Sel.':<12}", f"{'QED':<8}", f"{'SA':<8}"])
        header = "| ".join(header_parts)
        summary_content.append(header)
        summary_content.append("-" * len(header))

        row_types = [
            ('VINA Score (Avg)', 'score', 'avg'), ('VINA Score (Med)', 'score', 'median'),
            ('VINA Min (Avg)', 'min', 'avg'), ('VINA Min (Med)', 'min', 'median'),
            ('VINA Dock (Avg)', 'dock', 'avg'), ('VINA Dock (Med)', 'dock', 'median')
        ]

        for row_name, vina_type, stat_type in row_types:
            row_parts = [f"{row_name:<20}", f"{stats[f'on_target_{vina_type}'][stat_type]:>28}"]
            for i in range(len(off_target_keys)):
                row_parts.append(f"{stats[f'off_target_{i+1}_{vina_type}'][stat_type]:>28}")
            if vina_type == 'dock':
                row_parts.extend([
                    f"{stats['dock_selectivity'][stat_type]:>10}",
                    f"{stats['qed'][stat_type]:>6}",
                    f"{stats['sa'][stat_type]:>6}"
                ])
            else:
                row_parts.extend([f"{'':>10}", f"{'':>6}", f"{'':>6}"])
            summary_content.append(" | ".join(row_parts))

        summary_content.append("")
        summary_content.append("Note: Avg = Average, Med = Median. Statistics calculated from valid (non-N/A) values only.")
        summary_content.append("")
        return summary_content

    # Extract all vina results
    all_vina_data = []
    for entry in data:
        name = entry['ligand_name'].replace('ligand_', '').replace('_from_result_0.pt', '')
        vina_data = {
            'name': name,
            'on_target': {'score': "N/A", 'min': "N/A", 'dock': "N/A"},
            'off_targets': [{'score': "N/A", 'min': "N/A", 'dock': "N/A"} for _ in range(3)],
            'selectivity': {'score': "N/A", 'min': "N/A", 'dock': entry['selectivity_score']},
            'chem_props': {}
        }

        if 'selectivity_scores_by_type' in entry and entry['selectivity_scores_by_type']:
            selectivity_by_type = entry['selectivity_scores_by_type']
            vina_data['selectivity']['score'] = selectivity_by_type.get('score', "N/A")
            vina_data['selectivity']['min'] = selectivity_by_type.get('min', "N/A")
            vina_data['selectivity']['dock'] = selectivity_by_type.get('dock', entry['selectivity_score'])

        if 'docking_results' in entry and 'on_target' in entry['docking_results']:
            on_target_data_entry = entry['docking_results']['on_target']
            if 'vina_results' in on_target_data_entry:
                vr = on_target_data_entry['vina_results']
                if 'score_only' in vr and vr['score_only']:
                    vina_data['on_target']['score'] = f"{vr['score_only'][0]['affinity']:.3f}"
                if 'minimize' in vr and vr['minimize']:
                    vina_data['on_target']['min'] = f"{vr['minimize'][0]['affinity']:.3f}"
                if 'dock' in vr and vr['dock']:
                    vina_data['on_target']['dock'] = f"{vr['dock'][0]['affinity']:.3f}"
            if 'chem_results' in on_target_data_entry and on_target_data_entry['chem_results'] is not None:
                cr = on_target_data_entry['chem_results']
                vina_data['chem_props'] = {'qed': cr.get('qed', 'N/A'), 'sa': cr.get('sa', 'N/A')}

        off_target_keys = proteins.get('off_target_keys', [])
        for i, off_key in enumerate(off_target_keys):
            if 'docking_results' in entry and off_key in entry['docking_results']:
                while len(vina_data['off_targets']) <= i:
                    vina_data['off_targets'].append({'score': 'N/A', 'min': 'N/A', 'dock': 'N/A'})
                off_data = entry['docking_results'][off_key]
                if 'vina_results' in off_data:
                    vr = off_data['vina_results']
                    if 'score_only' in vr and vr['score_only']:
                        vina_data['off_targets'][i]['score'] = f"{vr['score_only'][0]['affinity']:.3f}"
                    if 'minimize' in vr and vr['minimize']:
                        vina_data['off_targets'][i]['min'] = f"{vr['minimize'][0]['affinity']:.3f}"
                    if 'dock' in vr and vr['dock']:
                        vina_data['off_targets'][i]['dock'] = f"{vr['dock'][0]['affinity']:.3f}"

        all_vina_data.append(vina_data)

    # Add summary table
    content.extend(create_summary_table_with_median(all_vina_data))

    # Vina table helper
    def create_vina_table(title, vina_type, all_vina_data_inner, proteins_inner):
        table_content = []
        off_target_keys_inner = proteins_inner.get('off_target_keys', [])
        num_off_targets = len(off_target_keys_inner)
        base_width = 12 + 12 + 12 + 7 + 6 + 5
        off_target_width = num_off_targets * 16
        total_width = base_width + off_target_width

        table_content.append("=" * total_width)
        table_content.append(f"{title:^{total_width}}")
        table_content.append("=" * total_width)

        header_parts = [f"{'Ligand':<12}", f"{'On-Target':<12}"]
        for i in range(num_off_targets):
            header_parts.append(f"{'Off-Target ' + str(i+1):<14}")
        header_parts.extend([f"{'Selectivity':<12}", f"{'QED':<7}", f"{'SA':<6}"])
        header = "| ".join(header_parts)
        table_content.append(header)
        table_content.append("-" * len(header))

        for vd in all_vina_data_inner:
            row_parts = [f"{vd['name']:<12}", f"{vd['on_target'][vina_type]:>10}"]
            for i in range(num_off_targets):
                if i < len(vd['off_targets']):
                    row_parts.append(f"{vd['off_targets'][i][vina_type]:>12}")
                else:
                    row_parts.append(f"{'N/A':>12}")
            sel_val = vd['selectivity'][vina_type]
            sel_str = f"{sel_val:.3f}" if sel_val is not None and sel_val != "N/A" else "N/A"
            qed_v = f"{vd['chem_props'].get('qed', 'N/A'):.3f}" if isinstance(vd['chem_props'].get('qed'), (int, float)) else 'N/A'
            sa_v = f"{vd['chem_props'].get('sa', 'N/A'):.2f}" if isinstance(vd['chem_props'].get('sa'), (int, float)) else 'N/A'
            row_parts.extend([f"{sel_str:>10}", f"{qed_v:>5}", f"{sa_v:>4}"])
            table_content.append("| ".join(row_parts))

        table_content.append("")
        return table_content

    content.append("")
    content.append("=" * 85)
    content.append("                 DETAILED RESULTS BY PROTEIN")
    content.append("=" * 85)
    content.append("")

    # On-target statistics
    content.append(f"ON-TARGET ({proteins['on_target']}):")
    if on_target_stats['best'] != "N/A":
        content.append(f"Best affinity: {on_target_stats['best']:.3f} kcal/mol")
        content.append(f"Worst affinity: {on_target_stats['worst']:.3f} kcal/mol")
        content.append(f"Average affinity: {on_target_stats['average']:.3f} kcal/mol")
    content.append("")

    # Off-target statistics
    off_target_keys = proteins.get('off_target_keys', [])
    for i, off_key in enumerate(off_target_keys, 1):
        if off_key in proteins['off_targets']:
            protein_name = proteins['off_targets'][off_key]
            stats = off_target_stats.get(f'off_target_{i}', {'best': 'N/A', 'worst': 'N/A', 'average': 'N/A'})
            content.append(f"OFF-TARGET {i} ({protein_name}):")
            if stats['best'] != "N/A":
                content.append(f"Best affinity: {stats['best']:.3f} kcal/mol")
                content.append(f"Worst affinity: {stats['worst']:.3f} kcal/mol")
                content.append(f"Average affinity: {stats['average']:.3f} kcal/mol")
            content.append("")

    content.append("=" * 85)
    content.append("                 CHEMICAL PROPERTIES ANALYSIS")
    content.append("=" * 85)
    content.append("")

    qed_values = [l['chem_props'].get('qed') for l in ligands_data if isinstance(l['chem_props'].get('qed'), (int, float))]
    sa_values = [l['chem_props'].get('sa') for l in ligands_data if isinstance(l['chem_props'].get('sa'), (int, float))]
    logp_values = [l['chem_props'].get('logp') for l in ligands_data if isinstance(l['chem_props'].get('logp'), (int, float))]
    lipinski_values = [l['chem_props'].get('lipinski') for l in ligands_data if l['chem_props'].get('lipinski') not in ['N/A', None]]

    if qed_values:
        qed_stats = calculate_stats(qed_values)
        content.append(f"QED (Drug-likeness, 0-1, higher is better):")
        content.append(f"  Best: {qed_stats['best']:.3f}, Worst: {qed_stats['worst']:.3f}, Average: {qed_stats['average']:.3f}")
        content.append("")

    if sa_values:
        sa_stats = calculate_stats(sa_values)
        content.append(f"SA Score (Synthetic Accessibility, 0-1, lower is better):")
        content.append(f"  Best: {sa_stats['best']:.3f}, Worst: {sa_stats['worst']:.3f}, Average: {sa_stats['average']:.3f}")
        content.append("")

    if logp_values:
        logp_stats = calculate_stats(logp_values)
        content.append(f"LogP (Lipophilicity, optimal range 1-3):")
        content.append(f"  Best: {logp_stats['best']:.3f}, Worst: {logp_stats['worst']:.3f}, Average: {logp_stats['average']:.3f}")
        content.append("")

    if lipinski_values:
        lipinski_counts = {}
        for val in lipinski_values:
            lipinski_counts[str(val)] = lipinski_counts.get(str(val), 0) + 1
        content.append("Lipinski Rule Violations (0 is best):")
        for violations, count in sorted(lipinski_counts.items()):
            content.append(f"  {violations} violations: {count} ligands")
        content.append("")

    # Ring size analysis
    all_ring_sizes = {}
    for ligand in ligands_data:
        ring_size_data = ligand['chem_props'].get('ring_size', {})
        for size, count in ring_size_data.items():
            if size not in all_ring_sizes:
                all_ring_sizes[size] = []
            all_ring_sizes[size].append(count)

    if all_ring_sizes:
        content.append("Ring Size Distribution (most common):")
        for size in sorted(all_ring_sizes.keys(), key=int):
            avg_count = sum(all_ring_sizes[size]) / len(all_ring_sizes[size])
            content.append(f"  {size}-membered rings: {avg_count:.1f} average per molecule")
        content.append("")

    content.append("=" * 85)
    content.append("                 DOCKING SUCCESS/FAILURE ANALYSIS")
    content.append("=" * 85)
    content.append("")

    total_ligands = len(ligands_data)
    successful_ligands = [l for l in ligands_data if l['on_target'] is not None]
    failed_ligands = [l for l in ligands_data if l['on_target'] is None]

    fragmented_molecules = []
    reconstruction_failures = []
    docking_failures = []

    for entry in data:
        if entry['on_target_affinity'] is None:
            ligand_name = entry['ligand_name'].replace('ligand_', '').replace('_from_result_0.pt', '')
            smiles = entry.get('smiles', '')
            if smiles and '.' in smiles:
                fragmented_molecules.append((ligand_name, smiles))
            if 'docking_results' in entry and 'on_target' in entry['docking_results']:
                on_target_result = entry['docking_results']['on_target']
                success = on_target_result.get('success', False)
                if not success:
                    if smiles and '.' in smiles:
                        pass
                    elif smiles is None or smiles == "":
                        reconstruction_failures.append(ligand_name)
                    else:
                        docking_failures.append(ligand_name)

    content.append(f"DOCKING SUCCESS STATISTICS:")
    content.append(f"  Total ligands processed: {total_ligands}")
    content.append(f"  Successful docking: {len(successful_ligands)} ({len(successful_ligands)/total_ligands*100:.1f}%)")
    content.append(f"  Failed docking: {len(failed_ligands)} ({len(failed_ligands)/total_ligands*100:.1f}%)")
    content.append("")

    if failed_ligands:
        content.append(f"FAILURE BREAKDOWN:")
        content.append(f"  Fragmented molecules: {len(fragmented_molecules)} ({len(fragmented_molecules)/total_ligands*100:.1f}%)")
        content.append(f"  Molecule reconstruction failures: {len(reconstruction_failures)} ({len(reconstruction_failures)/total_ligands*100:.1f}%)")
        content.append(f"  Other docking failures: {len(docking_failures)} ({len(docking_failures)/total_ligands*100:.1f}%)")
        content.append("")

    content.append("=" * 85)
    content.append("                    SELECTIVITY ANALYSIS")
    content.append("=" * 85)
    content.append("")

    sorted_by_selectivity = sorted(ligands_data, key=lambda x: x['selectivity'] if x['selectivity'] is not None else float('inf'))

    content.append("Most Selective (highest negative selectivity score):")
    for ligand in sorted_by_selectivity[:4]:
        name = ligand['name'].replace('ligand_', '').replace('_from_result_0.pt', '')
        sel_val = f"{ligand['selectivity']:.3f}" if ligand['selectivity'] is not None else 'N/A'
        content.append(f"- {name}: {sel_val}")
    content.append("")

    content.append("Least Selective (closest to 0):")
    for ligand in sorted_by_selectivity[-4:]:
        name = ligand['name'].replace('ligand_', '').replace('_from_result_0.pt', '')
        sel_val = f"{ligand['selectivity']:.3f}" if ligand['selectivity'] is not None else 'N/A'
        content.append(f"- {name}: {sel_val}")
    content.append("")

    content.append("=" * 85)
    content.append("                  EXPERIMENT PARAMETERS")
    content.append("=" * 85)

    if data:
        params = data[0]['parameters']
        content.append(f"- Docking mode: {params.get('docking_mode', 'N/A')}")
        content.append(f"- Exhaustiveness: {params.get('exhaustiveness', 'N/A')}")
        content.append(f"- Mode: {params.get('mode', 'N/A')}")
        content.append(f"- Auto-translate: {params.get('auto_translate', 'N/A')}")

    content.append("")
    content.append("Note: Negative affinity values indicate stronger binding.")
    content.append("Lower (more negative) selectivity scores indicate better selectivity for on-target.")
    content.append("")

    # Detailed vina tables
    content.append("=" * 120)
    content.append("                           DETAILED VINA RESULTS BY LIGAND")
    content.append("=" * 120)
    content.append("")

    content.extend(create_vina_table("VINA SCORE RESULTS (kcal/mol)", "score", all_vina_data, proteins))
    content.extend(create_vina_table("VINA MINIMIZE RESULTS (kcal/mol)", "min", all_vina_data, proteins))
    content.extend(create_vina_table("VINA DOCK RESULTS (kcal/mol)", "dock", all_vina_data, proteins))

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(content))

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert JSON docking results to readable TXT format')
    parser.add_argument('json_path', help='Path to JSON file containing docking results')
    parser.add_argument('-o', '--output', help='Output TXT file path (optional)')
    parser.add_argument('-n', '--name', help='Experiment name (optional)')

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: JSON file '{args.json_path}' not found!")
        return

    try:
        output_path = convert_json_to_txt(args.json_path, args.output, args.name)
        print(f"Successfully converted JSON to TXT!")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    main()
