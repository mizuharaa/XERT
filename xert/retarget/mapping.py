"""Map neutral-relative motion offsets to robot site positions using only the spec."""
import numpy as np


def map_positions(spec, offsets, neutral_sites):
    targets = {}
    for source, site in spec.mapping.items():
        if source not in offsets or site not in neutral_sites:
            raise ValueError(f"Missing mapped motion source or neutral site: {source} -> {site}")
        offset = np.asarray(offsets[source], dtype=float)
        neutral = np.asarray(neutral_sites[site], dtype=float)
        if offset.shape != (3,) or neutral.shape != (3,) or not np.isfinite([offset, neutral]).all():
            raise ValueError("Motion offsets and neutral sites must be finite XYZ vectors")
        if site in targets:
            raise ValueError("Each site needs one unambiguous source mapping")
        targets[site] = neutral + offset
    return targets
