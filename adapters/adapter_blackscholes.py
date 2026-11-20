"""
adapters/adapter_blackscholes.py: Adapter for Group 1 (blackscholes.ipynb).

Contains 5 variants:
1. trainer_bs_delta: drift=0.1, cvar@1.0, 400 epochs
2. trainer_bs_delta_exp2@10: drift=0.0, exp2@10.0, 400 epochs
3. trainer_bs_delta_nodrift: drift=0.0, cvar@1.0, 600 epochs, clipvalue=0.5, global_clipnorm=1.0
4. trainer_bs_delta_nodrift_exp2@10: drift=0.0, exp2@10.0, 600 epochs, lr=0.001, clipvalue=0.5, global_clipnorm=1.0
5. trainer_bs_small: samples=10000, drift=0.1, cvar@2.0, 10 epochs

Theoretical Status: SUPPORTED_EMPIRICAL_ZERO_COST
- Transaction cost cost_s = 0.0 in the original notebook.
- NTBN runs technically with band collapsing towards point delta.
"""

from typing import Tuple, List, Optional, Any
import numpy as np

import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3] / "deephedging-jax"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deephedging.world import WorldConfig, simulate

from .base_adapter import BaseNotebookAdapter
try:
    from core.model import AgentConfigOriginal, GymConfigOriginal, AgentConfigNTBN, GymConfigNTBN
    from core.train import TrainConfig
except ImportError:
    from ..core.model import AgentConfigOriginal, GymConfigOriginal, AgentConfigNTBN, GymConfigNTBN
    from ..core.train import TrainConfig


class BlackScholesAdapter(BaseNotebookAdapter):
    
    @property
    def group_id(self) -> str:
        return "blackscholes"

    @property
    def notebook_path(self) -> str:
        return "notebooks/blackscholes.ipynb"

    @property
    def variants(self) -> List[str]:
        return [
            "trainer_bs_delta",
            "trainer_bs_delta_exp2@10",
            "trainer_bs_delta_nodrift",
            "trainer_bs_delta_nodrift_exp2@10",
            "trainer_bs_small"
        ]

    @property
    def theoretical_ntbn_status(self) -> str:
        return "SUPPORTED_EMPIRICAL_ZERO_COST"

    @property
    def description(self) -> str:
        return (
            "Standard European ATM Call under Black-Scholes dynamics. Single hedging instrument (spot). "
            "Zero transaction cost (cost_s = 0.0). Demonstrates frictionless delta hedging approximation."
        )

    def build_worlds(self, seed: int = 42, train_samples: Optional[int] = None, val_samples: Optional[int] = None, test_samples: Optional[int] = None, variant: str = "trainer_bs_delta") -> Tuple[Any, Any, Any]:
        is_small = (variant == "trainer_bs_small")
        default_samples = 10000 if is_small else 50000
        samples = train_samples if train_samples is not None else default_samples
        v_samples = val_samples if val_samples is not None else samples // 10
        t_samples = test_samples if test_samples is not None else samples // 5
        
        drift = 0.1 if variant in ("trainer_bs_delta", "trainer_bs_small") else 0.0
        
        world_cfg = WorldConfig(
            samples=samples,
            steps=20,
            black_scholes=True,
            rvol=0.2,
            drift=drift,
            cost_s=0.0,
            seed=seed
        )
        
        train_world = simulate(world_cfg)
        val_world = train_world.clone(samples=v_samples, seed=seed + 1)
        test_world = train_world.clone(samples=t_samples, seed=seed + 999)
        return train_world, val_world, test_world

    def get_variant_configs(self, variant: str) -> Tuple[GymConfigOriginal, GymConfigNTBN, TrainConfig]:
        if variant == "trainer_bs_delta":
            utility_name, lmbda = "cvar", 1.0
            train_cfg = TrainConfig(epochs=400, batch_size=32, learning_rate=0.001)
        elif variant == "trainer_bs_delta_exp2@10":
            utility_name, lmbda = "exp2", 10.0
            train_cfg = TrainConfig(epochs=400, batch_size=32, learning_rate=0.001)
        elif variant == "trainer_bs_delta_nodrift":
            utility_name, lmbda = "cvar", 1.0
            train_cfg = TrainConfig(epochs=600, batch_size=32, learning_rate=0.001, clipvalue=0.5, global_clipnorm=1.0)
        elif variant == "trainer_bs_delta_nodrift_exp2@10":
            utility_name, lmbda = "exp2", 10.0
            train_cfg = TrainConfig(epochs=600, batch_size=32, learning_rate=0.001, clipvalue=0.5, global_clipnorm=1.0)
        elif variant == "trainer_bs_small":
            utility_name, lmbda = "cvar", 2.0
            train_cfg = TrainConfig(epochs=10, batch_size=32, learning_rate=0.001)
        else:
            raise ValueError(f"Unknown variant '{variant}' for group {self.group_id}")

        agent_orig = AgentConfigOriginal(
            features=("price", "delta", "time_left"),
            width=40,
            depth=5,
            activation="softplus"
        )
        gym_orig = GymConfigOriginal(agent=agent_orig, utility=utility_name, lmbda=lmbda)

        agent_ntbn = AgentConfigNTBN(
            features=("price", "time_left"),  # Excludes delta
            width=40,
            depth=5,
            activation="softplus",
            band_mode="center_width",
            action_clipping="none",
            init_delta=False
        )
        gym_ntbn = GymConfigNTBN(agent=agent_ntbn, utility=utility_name, lmbda=lmbda)

        return gym_orig, gym_ntbn, train_cfg
