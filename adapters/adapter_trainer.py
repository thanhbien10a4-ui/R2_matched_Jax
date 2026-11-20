"""
adapters/adapter_trainer.py: Adapter for Group 3 (trainer.ipynb).

Contains 2 variants:
1. bs: Black-Scholes world with positive transaction costs (cost_s = 0.0002), nInst = 1 (spot), 800 epochs.
2. sv: Stochastic volatility world (Heston-like), positive transaction costs, nInst = 2 (spot + ATM call option), 800 epochs.

Theoretical Status:
- bs: SUPPORTED_THEORETICAL (cost > 0, European smooth ATM call payoff, convex utility).
- sv: SUPPORTED_THEORETICAL (multidimensional extension: 2 hedging instruments with transaction costs).
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


class TrainerAdapter(BaseNotebookAdapter):

    @property
    def group_id(self) -> str:
        return "trainer"

    @property
    def notebook_path(self) -> str:
        return "notebooks/trainer.ipynb"

    @property
    def variants(self) -> List[str]:
        return ["bs", "sv"]

    @property
    def theoretical_ntbn_status(self) -> str:
        return "SUPPORTED_THEORETICAL"

    @property
    def description(self) -> str:
        return (
            "Foundational Deep Hedging with non-zero transaction costs. Variant 'bs' uses spot only (nInst=1). "
            "Variant 'sv' operates in a stochastic volatility world with two hedging instruments (nInst=2: spot and ATM call), "
            "exercising multidimensional NTBN band clamping."
        )

    def build_worlds(self, seed: int = 42, train_samples: Optional[int] = None, val_samples: Optional[int] = None, test_samples: Optional[int] = None, variant: Optional[str] = "bs") -> Tuple[Any, Any, Any]:
        samples = train_samples if train_samples is not None else 10000
        v_samples = val_samples if val_samples is not None else samples // 10
        t_samples = test_samples if test_samples is not None else samples // 5

        v = variant if variant is not None else "bs"
        is_bs = (v == "bs")
        world_cfg = WorldConfig(
            samples=samples,
            steps=20,
            black_scholes=is_bs,
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

        if variant == "bs":
            train_cfg = TrainConfig(epochs=800, batch_size=32, learning_rate=0.001)
        elif variant == "sv":
            train_cfg = TrainConfig(epochs=800, batch_size=32, learning_rate=0.001, clipvalue=1.0, global_clipnorm=1.0)
        else:
            raise ValueError(f"Unknown variant '{variant}' for group {self.group_id}")

        return gym_orig, gym_ntbn, train_cfg
