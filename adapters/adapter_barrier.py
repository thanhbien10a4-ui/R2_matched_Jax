"""
adapters/adapter_barrier.py: Adapter for Group 5 (trainer-recurrent-barrier.ipynb).

Contains 2 variants:
1. none: Feed-forward baseline without recurrence (width=50, depth=5, softplus)
2. digital: Recurrent agent with past_repr=5, event=5 (width=50, depth=5, softplus)

Payoff: Down-and-out Barrier Put with K = 1.0, B = 0.95.
Scale in notebook: steps = 100 (20*5), dt = 0.004 (1/250), samples = 20000.
Utility = exp@1.01. Epochs = 250, batch_size = 32. Cost = 0.0.

Note on Naming Asymmetry:
In deephedging-jax, 'trainer-recurrent-barrier.ipynb' sets steps=100, dt=0.004 (small dt),
while 'trainer-recurrent-barrier_smalldt.ipynb' sets steps=20, dt=0.02.
This adapter adheres strictly to the exact code in 'trainer-recurrent-barrier.ipynb'.

Theoretical Status: SUPPORTED_EMPIRICAL_NONSMOOTH
"""

from typing import Tuple, List, Optional, Any
import numpy as np
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


def make_barrier_put_payoff(strike: float = 1.0, barrier: float = 0.95):
    def payoff_def(spots):
        notko = np.cumprod(np.where(spots > barrier, 1.0, 0.0), axis=1)
        terminal = np.maximum(strike - spots[:, -1], 0.0)
        return dict(
            payoff=-terminal * notko[:, -1],
            features=notko[:, :-1]
        )
    payoff_def.name = f"KO Put k={strike}, b={barrier}"
    return payoff_def


class BarrierAdapter(BaseNotebookAdapter):

    @property
    def group_id(self) -> str:
        return "trainer-recurrent-barrier"

    @property
    def notebook_path(self) -> str:
        return "notebooks/trainer-recurrent-barrier.ipynb"

    @property
    def variants(self) -> List[str]:
        return ["none", "digital"]

    @property
    def theoretical_ntbn_status(self) -> str:
        return "SUPPORTED_EMPIRICAL_NONSMOOTH"

    @property
    def description(self) -> str:
        return (
            "Down-and-out Put (KO Put k=1.0, b=0.95) with fine time steps (100 steps, dt=0.004). "
            "Tests recurrent memory on a path-dependent barrier option under exponential utility (exp@1.01)."
        )

    def build_worlds(self, seed: int = 42, train_samples: Optional[int] = None, val_samples: Optional[int] = None, test_samples: Optional[int] = None, variant: str = "none") -> Tuple[Any, Any, Any]:
        samples = train_samples if train_samples is not None else 20000
        v_samples = val_samples if val_samples is not None else samples // 2
        t_samples = test_samples if test_samples is not None else samples // 2

        world_cfg = WorldConfig(
            steps=20 * 5,  # 100
            dt=1.0 / 50.0 / 5.0,  # 0.004
            samples=samples,
            drift=0.0,
            cost_s=0.0,
            black_scholes=True,
            payoff=make_barrier_put_payoff(strike=1.0, barrier=0.95),
            seed=seed
        )

        train_world = simulate(world_cfg)
        val_world = train_world.clone(samples=v_samples, seed=seed + 1)
        test_world = train_world.clone(samples=t_samples, seed=seed + 999)
        return train_world, val_world, test_world

    def get_variant_configs(self, variant: str) -> Tuple[GymConfigOriginal, GymConfigNTBN, TrainConfig]:
        if variant == "none":
            past, event = 0, 0
        elif variant == "digital":
            past, event = 5, 5
        else:
            raise ValueError(f"Unknown variant '{variant}' for group {self.group_id}")

        agent_orig = AgentConfigOriginal(
            features=("price", "delta", "time_left", "payoff_features"),
            width=50,
            depth=5,
            activation="softplus",
            classic=0,
            aggregate=0,
            past_repr=past,
            event=event
        )
        gym_orig = GymConfigOriginal(agent=agent_orig, utility="exp", lmbda=1.01)

        agent_ntbn = AgentConfigNTBN(
            features=("price", "time_left", "payoff_features"),
            width=50,
            depth=5,
            activation="softplus",
            band_mode="center_width",
            classic=0,
            aggregate=0,
            past_repr=past,
            event=event,
            action_clipping="none",
            init_delta=False
        )
        gym_ntbn = GymConfigNTBN(agent=agent_ntbn, utility="exp", lmbda=1.01)

        train_cfg = TrainConfig(
            epochs=250,
            batch_size=32,
            learning_rate=0.001,
            clipvalue=0.9,
            global_clipnorm=1.0
        )
        return gym_orig, gym_ntbn, train_cfg
