"""
adapters/adapter_trainer_bs.py: Adapter for Group 2 (trainer-bs.ipynb).

Contains 2 variants:
1. big: steps=20, dt=5/250 (0.02), epochs=20000, batch_size=2000
2. small: steps=100, dt=1/250 (0.004), epochs=20000, batch_size=2000

Theoretical Status: SUPPORTED_EMPIRICAL_ZERO_COST
- Large-epoch high-batch benchmark (20.000 epochs, batch 2.000). Zero transaction costs.
"""

from typing import Tuple, List, Optional, Any
import sys
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


class TrainerBSAdapter(BaseNotebookAdapter):

    @property
    def group_id(self) -> str:
        return "trainer-bs"

    @property
    def notebook_path(self) -> str:
        return "notebooks/trainer-bs.ipynb"

    @property
    def variants(self) -> List[str]:
        return ["big", "small"]

    @property
    def theoretical_ntbn_status(self) -> str:
        return "SUPPORTED_EMPIRICAL_ZERO_COST"

    @property
    def description(self) -> str:
        return (
            "Deep Hedging in Black-Scholes world with big steps (dt=0.02, 20 steps) vs small steps "
            "(dt=0.004, 100 steps). Stress tests high-epoch training (20.000 epochs) with large batch size (2.000)."
        )

    def build_worlds(self, seed: int = 42, train_samples: Optional[int] = None, val_samples: Optional[int] = None, test_samples: Optional[int] = None, variant: Optional[str] = "big") -> Tuple[Any, Any, Any]:
        samples = train_samples if train_samples is not None else 10000
        v_samples = val_samples if val_samples is not None else samples // 2
        t_samples = test_samples if test_samples is not None else samples // 2
        
        v = variant if variant is not None else "big"
        if v == "big":
            steps = 20
            dt = 5.0 / 250.0
        elif v == "small":
            steps = 20 * 5  # 100
            dt = 1.0 / 250.0
        else:
            raise ValueError(f"Unknown variant '{variant}' for group {self.group_id}")

        world_cfg = WorldConfig(
            samples=samples,
            steps=steps,
            dt=dt,
            drift=0.0,
            cost_s=0.0,
            black_scholes=True,
            seed=seed
        )

        train_world = simulate(world_cfg)
        val_world = train_world.clone(samples=v_samples, seed=seed + 1)
        test_world = train_world.clone(samples=t_samples, seed=seed + 999)
        return train_world, val_world, test_world

    def get_variant_configs(self, variant: str) -> Tuple[GymConfigOriginal, GymConfigNTBN, TrainConfig]:
        agent_orig = AgentConfigOriginal(
            features=("price", "delta", "time_left"),
            width=20,
            depth=3,
            activation="softplus"
        )
        gym_orig = GymConfigOriginal(agent=agent_orig, utility="cvar", lmbda=1.0)

        agent_ntbn = AgentConfigNTBN(
            features=("price", "time_left"),
            width=20,
            depth=3,
            activation="softplus",
            band_mode="center_width",
            action_clipping="none",
            init_delta=False
        )
        gym_ntbn = GymConfigNTBN(agent=agent_ntbn, utility="cvar", lmbda=1.0)

        train_cfg = TrainConfig(
            epochs=20000,
            batch_size=2000,
            learning_rate=0.001
        )
        return gym_orig, gym_ntbn, train_cfg
