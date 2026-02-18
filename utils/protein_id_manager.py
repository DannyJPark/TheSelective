"""
Protein ID Manager - Unified protein ID management system.

Resolves protein duplication issues in validation sets and
provides consistent protein ID mapping across scripts.
"""

import json
import collections
from typing import Dict, List, Tuple, Set
import warnings

class ProteinIDManager:
    def __init__(self, validation_info_path: str = './data/multipro_validation_info.json'):
        """
        Initialize protein ID manager.

        Args:
            validation_info_path: Path to validation info JSON file
        """
        self.validation_info_path = validation_info_path
        self.validation_info = None
        self.protein_to_validation_ids = {}  # protein_name -> [validation_id1, validation_id2, ...]
        self.validation_id_to_protein = {}   # validation_id -> protein_name
        self.unique_proteins = []
        self.unique_protein_to_representative_id = {}

        self._load_and_analyze()

    def _load_and_analyze(self):
        """Load validation info and analyze duplicates"""
        with open(self.validation_info_path, 'r') as f:
            self.validation_info = json.load(f)

        # Build mapping
        for i, entry in enumerate(self.validation_info):
            protein_name = entry['protein_dir']
            self.validation_id_to_protein[i] = protein_name

            if protein_name not in self.protein_to_validation_ids:
                self.protein_to_validation_ids[protein_name] = []
            self.protein_to_validation_ids[protein_name].append(i)

        # Create unique protein list (representative ID = smallest validation_id)
        self.unique_proteins = list(self.protein_to_validation_ids.keys())
        for protein_name in self.unique_proteins:
            validation_ids = self.protein_to_validation_ids[protein_name]
            representative_id = min(validation_ids)
            self.unique_protein_to_representative_id[protein_name] = representative_id

        print(f"[ProteinIDManager] Loaded {len(self.validation_info)} validation entries")
        print(f"[ProteinIDManager] Found {len(self.unique_proteins)} unique proteins")

        # Print duplicate info
        duplicates = {name: ids for name, ids in self.protein_to_validation_ids.items() if len(ids) > 1}
        if duplicates:
            print(f"[ProteinIDManager] WARNING: {len(duplicates)} proteins have duplicates:")
            for name, ids in list(duplicates.items())[:5]:
                print(f"  {name}: validation_ids {ids}")
            if len(duplicates) > 5:
                print(f"  ... and {len(duplicates) - 5} more")

    def get_protein_name(self, validation_id: int) -> str:
        """Return protein name from validation_id"""
        return self.validation_id_to_protein.get(validation_id, f"unknown_{validation_id}")

    def get_all_validation_ids_for_protein(self, protein_name: str) -> List[int]:
        """Return all validation_ids for protein name"""
        return self.protein_to_validation_ids.get(protein_name, [])

    def get_representative_id(self, protein_name: str) -> int:
        """Return representative validation_id for protein name"""
        return self.unique_protein_to_representative_id.get(protein_name, -1)

    def validate_id_selection(self, on_target_id: int, off_target_ids: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate on-target and off-target ID selection

        Args:
            on_target_id: On-target validation ID
            off_target_ids: List of off-target validation IDs

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # On-target protein name
        on_target_protein = self.get_protein_name(on_target_id)

        # Duplicate check
        all_ids = [on_target_id] + off_target_ids
        used_proteins = set()

        for val_id in all_ids:
            protein_name = self.get_protein_name(val_id)

            if protein_name in used_proteins:
                errors.append(f"Protein '{protein_name}' is used multiple times")
            else:
                used_proteins.add(protein_name)

        # Off-target ID validation
        for off_id in off_target_ids:
            if off_id == on_target_id:
                errors.append(f"Off-target ID {off_id} is same as on-target ID")

            off_protein = self.get_protein_name(off_id)
            if off_protein == on_target_protein:
                errors.append(f"Off-target ID {off_id} ({off_protein}) is same protein as on-target ID {on_target_id}")

        return len(errors) == 0, errors

    def get_safe_off_target_ids(self, on_target_id: int, requested_off_target_ids: List[int]) -> List[int]:
        """
        Return safe off-target ID list (auto-replace duplicate proteins)

        Args:
            on_target_id: On-target validation ID
            requested_off_target_ids: Requested off-target IDs

        Returns:
            Safe off-target ID list with duplicates removed
        """
        on_target_protein = self.get_protein_name(on_target_id)
        safe_off_target_ids = []
        used_proteins = {on_target_protein}

        for off_id in requested_off_target_ids:
            off_protein = self.get_protein_name(off_id)

            if off_protein not in used_proteins:
                # Safe ID
                safe_off_target_ids.append(off_id)
                used_proteins.add(off_protein)
            else:
                # Duplicate protein - find replacement ID
                print(f"[ProteinIDManager] WARNING: Off-target ID {off_id} ({off_protein}) conflicts with used protein")
                alternative_id = self._find_alternative_protein_id(used_proteins)
                if alternative_id is not None:
                    alternative_protein = self.get_protein_name(alternative_id)
                    print(f"[ProteinIDManager] Replacing with ID {alternative_id} ({alternative_protein})")
                    safe_off_target_ids.append(alternative_id)
                    used_proteins.add(alternative_protein)
                else:
                    print(f"[ProteinIDManager] WARNING: Could not find alternative for {off_protein}")

        return safe_off_target_ids

    def _find_alternative_protein_id(self, used_proteins: Set[str]) -> int:
        """Find representative ID for unused protein"""
        for protein_name in self.unique_proteins:
            if protein_name not in used_proteins:
                return self.get_representative_id(protein_name)
        return None

    def get_protein_summary(self) -> Dict:
        """Return protein info summary"""
        duplicates = {name: ids for name, ids in self.protein_to_validation_ids.items() if len(ids) > 1}

        return {
            'total_validation_entries': len(self.validation_info),
            'unique_proteins': len(self.unique_proteins),
            'duplicate_proteins': len(duplicates),
            'duplicate_details': duplicates
        }

    def print_summary(self):
        """Print protein info summary"""
        summary = self.get_protein_summary()
        print(f"=== Protein ID Manager Summary ===")
        print(f"Total validation entries: {summary['total_validation_entries']}")
        print(f"Unique proteins: {summary['unique_proteins']}")
        print(f"Proteins with duplicates: {summary['duplicate_proteins']}")

        if summary['duplicate_proteins'] > 0:
            print(f"\nDuplicate proteins (first 5):")
            for name, ids in list(summary['duplicate_details'].items())[:5]:
                print(f"  {name}: {ids}")


# Global singleton instance
_protein_manager = None

def get_protein_manager() -> ProteinIDManager:
    """Return global ProteinIDManager singleton instance"""
    global _protein_manager
    if _protein_manager is None:
        _protein_manager = ProteinIDManager()
    return _protein_manager

# Convenience functions
def validate_protein_ids(on_target_id: int, off_target_ids: List[int]) -> Tuple[bool, List[str]]:
    """Validate protein ID selection (convenience function)"""
    return get_protein_manager().validate_id_selection(on_target_id, off_target_ids)

def get_safe_protein_ids(on_target_id: int, off_target_ids: List[int]) -> Tuple[int, List[int]]:
    """Return safe protein ID combination (convenience function)"""
    manager = get_protein_manager()
    safe_off_targets = manager.get_safe_off_target_ids(on_target_id, off_target_ids)
    return on_target_id, safe_off_targets

def get_protein_name_by_id(validation_id: int) -> str:
    """Return protein name for validation_id (convenience function)"""
    return get_protein_manager().get_protein_name(validation_id)


if __name__ == "__main__":
    # Test
    manager = ProteinIDManager()
    manager.print_summary()

    # Duplicate test
    print("\n=== Testing Duplicate Detection ===")
    on_target = 0  # BACE1_HUMAN_49_451_0
    off_targets = [12, 62]  # Both are also BACE1_HUMAN_49_451_0

    is_valid, errors = manager.validate_id_selection(on_target, off_targets)
    print(f"Validation result: {is_valid}")
    for error in errors:
        print(f"ERROR: {error}")

    # Safe ID generation test
    print(f"\n=== Testing Safe ID Generation ===")
    safe_on, safe_off = get_safe_protein_ids(on_target, off_targets)
    print(f"Original: on_target={on_target}, off_targets={off_targets}")
    print(f"Safe: on_target={safe_on}, off_targets={safe_off}")

    print(f"On-target protein: {get_protein_name_by_id(safe_on)}")
    for i, off_id in enumerate(safe_off):
        print(f"Off-target {i}: ID {off_id} ({get_protein_name_by_id(off_id)})")