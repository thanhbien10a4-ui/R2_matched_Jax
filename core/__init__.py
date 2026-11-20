"""
core package initialization.
"""

from .ntbn_primitives import ntbn_clamp, ntbn_bounds_center_width, ntbn_action
from .objectives import utility, compute_var_cvar, UTILITIES
from .model import (
    AgentConfigOriginal, GymConfigOriginal,
    AgentConfigNTBN, GymConfigNTBN,
    init_params_original, init_params_ntbn,
    rollout_original, rollout_ntbn,
    loss_original, loss_ntbn,
    count_parameters
)
from .metrics import evaluate_rollout_metrics, compute_paired_comparison
from .train import TrainConfig, TrainResult, train_paired, as_batch
