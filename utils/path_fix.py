"""
Utility function to fix data paths in checkpoint configs.
Normalizes any training-time paths to ./data/ for portability.
"""

import os


def fix_checkpoint_data_paths(ckpt, logger=None):
    """
    Fix data paths in checkpoint config to use ./data/ location.

    Handles checkpoints that were trained with various path prefixes
    (e.g., ./scratch2/data/, absolute paths) and normalizes them.

    Args:
        ckpt: Checkpoint dictionary containing 'config'
        logger: Optional logger for info messages

    Returns:
        Modified checkpoint (modifies in-place and returns)
    """
    if not hasattr(ckpt, 'get'):
        return ckpt

    if 'config' not in ckpt:
        return ckpt

    config = ckpt['config']

    if not hasattr(config, 'data'):
        return ckpt

    def normalize_path(p):
        """Normalize a data path to ./data/..."""
        if p is None:
            return p
        basename = os.path.basename(p)
        # Check common prefixes that should be replaced
        for prefix in ['./scratch2/data/', 'scratch2/data/', './data/']:
            if prefix in p:
                return './data/' + p.split(prefix)[-1]
        # Absolute paths containing known patterns
        if '/data/' in p:
            return './data/' + p.split('/data/')[-1]
        return p

    # Fix data.path
    if hasattr(config.data, 'path'):
        old_path = config.data.path
        config.data.path = normalize_path(old_path)
        if config.data.path != old_path and logger:
            logger.info(f"Updated data path: {old_path} -> {config.data.path}")

    # Fix data.split
    if hasattr(config.data, 'split'):
        old_split = config.data.split
        config.data.split = normalize_path(old_split)
        if config.data.split != old_split and logger:
            logger.info(f"Updated split path: {old_split} -> {config.data.split}")

    return ckpt
