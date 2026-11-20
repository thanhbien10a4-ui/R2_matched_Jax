"""
adapters/adapter_american_digital.py: Adapter for Group 4 (trainer-american_digital.ipynb).

Contains 5 variants:
1. none: Feed-forward baseline without recurrent memory (width=50, depth=5, softplus)
2. digital: Recurrent agent with past_repr=5, event=5 (width=50, depth=5, softplus)
3. standard: Recurrent agent with aggregate=5 (width=50, depth=5, softplus)
4. mix_100_relu: Recurrent agent with aggregate=2, past_repr=2, width=100, depth=5, relu
5. mix_100_softplus: Recurrent agent with aggregate=2, past_repr=2, width=100, depth=5, softplus

Payoff: American Digital Option with barrier B = 0.95.
Features include 'payoff_features' representing the alive indicator notko = cumprod(S_t > B).
Cost: cost_s = 0.0. Steps = 100, dt = 0.02. Utility = exp2@1.0. Epochs = 500.

Theoretical Status: SUPPORTED_EMPIRICAL_NONSMOOTH
- Payoff is discontinuous (step function at barrier) and cost is zero.
- Fully supported and tested technically in code, but optimality of the no-transaction band is not guaranteed by classical smooth-payoff theory.
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


def make_american_digital_payoff(barrier: float = 0.95):
    def payoff_def(spots):
        # spots has shape (nSamples, nSteps + 2)
        notko = np.cumprod(np.where(spots > barrier, 1.0, 0.0), axis=1)
        terminal = 1.0
        return dict(
            payoff=-terminal * notko[:, -1],
            features=notko[:, :-1]  # (nSamples, nSteps) alive indicator
        )
    payoff_def.name = f"American Digital @ {barrier}"
    return payoff_def


class AmericanDigitalAdapter(BaseNotebookAdapter):

    @property
    def group_id(self) -> str:
        return "trainer-american_digital"

    @property
    def notebook_path(self) -> str:
        return "notebooks/trainer-american_digital.ipynb"

    @property
    def variants(self) -> List[str]:
        return [
            "none",
            "digital",
            "standard",
            "mix_100_relu",
            "mix_100_softplus"
        ]

    @property
    def theoretical_ntbn_status(self) -> str:
        return "SUPPORTED_EMPIRICAL_NONSMOOTH"

    @property
    def description(self) -> str:
        return (
            "American Digital option with knock-out barrier at 0.95. Features include the path-dependent "
            "alive indicator. Tests recurrent memory gates against standard feed-forward networks on a discontinuous payoff."
        )

    def build_worlds(self, seed: int = 42, train_samples: Optional[int] = None, val_samples: Optional[int] = None, test_samples: Optional[int] = None, variant: str = "none") -> Tuple[Any, Any, Any]:
        samples = train_samples if train_samples is not None else 20000
        v_samples = val_samples if val_samples is not None else samples // 2
        t_samples = test_samples if test_samples is not None else samples // 2
        
        world_cfg = WorldConfig(
            steps=20 * 5,  # 100
            dt=1.0 / 50.0,  # 0.02
            samples=samples,
            drift=0.0,
            cost_s=0.0,
            black_scholes=True,
            payoff=make_american_digital_payoff(barrier=0.95),
            seed=seed
        )

        train_world = simulate(world_cfg)
        val_world = train_world.clone(samples=v_samples, seed=seed + 1)
        test_world = train_world.clone(samples=t_samples, seed=seed + 999)
        return train_world, val_world, test_world

    def get_variant_configs(self, variant: str) -> Tuple[GymConfigOriginal, GymConfigNTBN, TrainConfig]:
        if variant == "none":
            classic, aggr, past, event = 0, 0, 0, 0
            width, depth, act = 50, 5, "softplus"
        elif variant == "digital":
            classic, aggr, past, event = 0, 0, 5, 5
            width, depth, act = 50, 5, "softplus"
        elif variant == "standard":
            classic, aggr, past, event = 0, 5, 0, 0
            width, depth, act = 50, 5, "softplus"
        elif variant == "mix_100_relu":
            classic, aggr, past, event = 0, 2, 2, 0
            width, depth, act = 100, 5, "relu"
        elif variant == "mix_100_softplus":
            classic, aggr, past, event = 0, 2, 2, 0
            width, depth, act = 100, 5, "softplus"
        else:
            raise ValueError(f"Unknown variant '{variant}' for group {self.group_id}")

        agent_orig = AgentConfigOriginal(
            features=("price", "delta", "time_left", "payoff_features"),
            width=width,
            depth=depth,
            activation=act,
            classic=classic,
            aggregate=aggr,
            past_repr=past,
            event=event
        )
        gym_orig = GymConfigOriginal(agent=agent_orig, utility="exp2", lmbda=1.0)

        agent_ntbn = AgentConfigNTBN(
            features=("price", "time_left", "payoff_features"),  # Excludes delta
            width=width,
            depth=depth,
            activation=act,
            band_mode="center_width",
            classic=classic,
            aggregate=aggr,
            past_repr=past,
            event=event,
            action_clipping="none",
            init_delta=False
        )
        gym_ntbn = GymConfigNTBN(agent=agent_ntbn, utility="exp2", lmbda=1.0)

        train_cfg = TrainConfig(
            epochs=500,
            batch_size=32,
            learning_rate=0.0001,
            clipvalue=0.9,
            global_clipnorm=1.0
        )
        return gym_orig, gym_ntbn, train_cfg
