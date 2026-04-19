"""Scene split helpers for ETH/UCY workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneSplit:
    """Train/validation/test scene names."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


ETH_UCY_SCENES: tuple[str, ...] = ("eth", "hotel", "univ", "zara1", "zara2")


def normalize_scene_name(scene: str) -> str:
    """Normalize scene aliases to canonical names."""
    scene_norm = scene.strip().lower()
    aliases = {
        "students001": "univ",
        "students003": "univ",
        "zara01": "zara1",
        "zara02": "zara2",
    }
    return aliases.get(scene_norm, scene_norm)


def get_leave_one_out_split(test_scene: str) -> SceneSplit:
    """
    Return leave-one-out split commonly used for ETH/UCY.

    Engineering assumption:
    - Validation is set to one non-test scene (alphabetical first among remaining)
      because the paper specifies early stopping but does not prescribe val-scene policy.
    """
    test_scene_norm = normalize_scene_name(test_scene)
    if test_scene_norm not in ETH_UCY_SCENES:
        raise ValueError(f"Unknown ETH/UCY scene: {test_scene}")

    remaining = [scene for scene in ETH_UCY_SCENES if scene != test_scene_norm]
    val_scene = remaining[0]
    train_scenes = tuple(scene for scene in remaining if scene != val_scene)
    return SceneSplit(train=train_scenes, val=(val_scene,), test=(test_scene_norm,))
