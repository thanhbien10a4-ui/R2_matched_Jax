"""
adapters/adapter_fwdstart.py: Adapter for Group 7 (trainer-recurrent-fwdstart.ipynb).

Contains 6 variants:
1. none: Feed-forward baseline without memory (classic=0, aggregate=0, past_repr=0, event=0)
2. all_modes: All recurrence types active (classic=2, aggregate=2, past_repr=2, event=2)
3. aggregate_2: aggregate=2 only
4. aggregate_5bnd: aggregate=5, past_repr=1, bound_aggr_states=True
5. aggregate_5: aggregate=5 only, unbounded
6. past_repr: past_repr=2 only

Payoff: Forward-Start Put option where the strike resets to S_{10} at time step 10 (iReset=10).
terminal = max(S_{10} - S_T, 0).
Notice: Payoff does NOT expose an explicit exogenous feature for S_{10}.
The recurrent network must autonomously learn to remember the strike across time!

Scale in notebook: steps = 20, dt = 0.02, samples = 20000.
Utility = cvar@1.01. Epochs = 2000, batch_size = 32. Cost = 0.0.

Theoretical Status: SUPPORTED_EMPIRICAL_PATH_DEPENDENT
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


def make_fwdstart_payoff(i_reset: int = 10):
    def fwd_start(spots):
        # spots has shape (nSamples, nSteps + 2)
        terminal = np.maximum(1.0 * spots[:, i_reset] - spots[:, -1], 0.0)
        return -terminal
    fwd_start.name = f"Forward-Start Put (iReset={i_reset})"
    return fwd_start


class ForwardStartAdapter(BaseNotebookAdapter):

    @property
    def group_id(self) -> str:
        return "trainer-recurrent-fwdstart"

    @property
    def notebook_path(self) -> str:
        return "notebooks/trainer-recurrent-fwdstart.ipynb"

    @property
    def variants(self) -> List[str]:
        return [
            "none",
            "all_modes",
            "aggregate_2",
            "aggregate_5bnd",
            "aggregate_5",
            "past_repr"
        ]

    @property
    def theoretical_ntbn_status(self) -> str:
        return "SUPPORTED_EMPIRICAL_PATH_DEPENDENT"

    @property
    def description(self) -> str:
        return (
            "Forward-Start Put option where the strike is determined at step 10 (S_10). "
            "Examines the ability of recurrent cells (classic, aggregate, past_repr, event) to store "
            "path memory without explicit feature hints."
        )

    def build_worlds(self, seed: int = 42, train_samples: Optional[int] = None, val_samples: Optional[int] = None, test_samples: Optional[int] = None, variant: str = "none") -> Tuple[Any, Any, Any]:
        samples = train_samples if train_samples is not None else 20000
        v_samples = val_samples if val_samples is not None else samples // 2
        t_samples = test_samples if test_samples is not None else samples // 2

        world_cfg = WorldConfig(
            steps=20,
            samples=samples,
            drift=0.0,
            cost_s=0.0,
            black_scholes=True,
            payoff=make_fwdstart_payoff(i_reset=10),
            seed=seed
        )

        train_world = simulate(world_cfg)
        val_world = train_world.clone(samples=v_samples, seed=seed + 1)
        test_world = train_world.clone(samples=t_samples, seed=seed + 999)
        return train_world, val_world, test_world

    def get_variant_configs(self, variant: str) -> Tuple[GymConfigOriginal, GymConfigNTBN, TrainConfig]:
        classic, aggr, past, event = 0, 0, 0, 0
        bound_aggr = False

        if variant == "none":
            pass
        elif variant == "all_modes":
            classic, aggr, past, event = 2, 2, 2, 2
        elif variant == "aggregate_2":
            classic, aggr, past, event = 0, 2, 0, 0
        elif variant == "aggregate_5bnd":
            classic, aggr, past, event = 0, 5, 1, 0
            bound_aggr = True
        elif variant == "aggregate_5":
            classic, aggr, past, event = 0, 5, 0, 0
        elif variant == "past_repr":
            classic, aggr, past, event = 0, 0, 2, 0
        else:
            raise ValueError(f"Unknown variant '{variant}' for group {self.group_id}")

        agent_orig = AgentConfigOriginal(
            features=("price", "delta", "time_left"),
            width=20,
            depth=5,
            activation="softplus",
            classic=classic,
            aggregate=aggr,
            past_repr=past,
            event=event,
            bound_aggr_states=bound_aggr
        )
        gym_orig = GymConfigOriginal(agent=agent_orig, utility="cvar", lmbda=1.01)

        agent_ntbn = AgentConfigNTBN(
            features=("price", "time_left"),
            width=20,
            depth=5,
            activation="softplus",
            band_mode="center_width",
            classic=classic,
            aggregate=aggr,
            past_repr=past,
            event=event,
            bound_aggr_states=bound_aggr,
            action_clipping="none",
            init_delta=False
        )
        gym_ntbn = GymConfigNTBN(agent=agent_ntbn, utility="cvar", lmbda=1.01)

        train_cfg = TrainConfig(
            epochs=2000,
            batch_size=32,
            learning_rate=0.001,
            clipvalue=0.9,
            global_clipnorm=1.0
        )
        return gym_orig, gym_ntbn, train_cfg
