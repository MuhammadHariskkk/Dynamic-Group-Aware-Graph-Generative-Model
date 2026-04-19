"""Deterministic graph partitioning for group extraction."""

from __future__ import annotations

from collections import deque

import numpy as np


def connected_components_from_adjacency(adjacency: np.ndarray) -> list[list[int]]:
    """
    Return connected components from undirected adjacency matrix.

    Components and member indices are sorted for deterministic IDs.
    """
    n = int(adjacency.shape[0])
    visited = np.zeros((n,), dtype=bool)
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        queue: deque[int] = deque([start])
        visited[start] = True
        comp: list[int] = []

        while queue:
            node = queue.popleft()
            comp.append(node)
            neighbors = np.where(adjacency[node])[0]
            for nb in neighbors.tolist():
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        components.append(sorted(comp))

    components.sort(key=lambda c: (c[0], len(c)))
    return components


def membership_from_components(
    components: list[list[int]],
    valid_mask_t: np.ndarray,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """
    Build per-agent group IDs and reverse mapping for valid agents.

    Invalid agents get group id = -1.
    """
    n = int(valid_mask_t.shape[0])
    group_ids = np.full((n,), -1, dtype=np.int64)
    group_to_members: dict[int, list[int]] = {}
    gid = 0

    for comp in components:
        valid_members = [idx for idx in comp if bool(valid_mask_t[idx])]
        if not valid_members:
            continue
        for idx in valid_members:
            group_ids[idx] = gid
        group_to_members[gid] = valid_members
        gid += 1

    return group_ids, group_to_members
