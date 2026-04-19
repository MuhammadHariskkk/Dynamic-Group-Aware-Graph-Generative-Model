"""Dynamic grouping subsystem."""

from groupaware.grouping.group_features import (
    GroupFeatureConfig,
    compute_dynamic_group_features,
    compute_group_features_per_timestep,
    detect_groups_per_timestep,
)
from groupaware.grouping.rules import GroupRuleConfig, build_grouping_adjacency, pairwise_rule_satisfied

__all__ = [
    "GroupFeatureConfig",
    "GroupRuleConfig",
    "build_grouping_adjacency",
    "compute_dynamic_group_features",
    "compute_group_features_per_timestep",
    "detect_groups_per_timestep",
    "pairwise_rule_satisfied",
]
