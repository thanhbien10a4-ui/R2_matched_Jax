"""
configs/catalog.py: Master Catalog of all 24 Configurations across the 7 Notebook Groups.

Every configuration preserves 100% of the original notebook scale:
- samples, steps, dt, epochs, batch_size, learning_rate, clipvalue, global_clipnorm
- utility, lmbda, cost, n_inst, network depth, width, activations, recurrent modes
- theoretical NTBN classification status
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

try:
    from adapters import ADAPTERS
except ImportError:
    from ..adapters import ADAPTERS


@dataclass(frozen=True)
class ConfigSpec:
    config_id: str
    group_id: str
    variant: str
    notebook_path: str
    theoretical_status: str
    train_samples: int
    val_samples: int
    test_samples: int
    steps: int
    dt: float
    epochs: int
    batch_size: int
    learning_rate: float
    utility: str
    lmbda: float
    cost_s: float
    n_inst: int
    depth: int
    width: int
    activation: str
    recurrent_states: Dict[str, int]
    description: str


def build_catalog() -> Dict[str, ConfigSpec]:
    catalog = {}
    
    for group_id, adapter in ADAPTERS.items():
        for variant in adapter.variants:
            config_id = f"{group_id}__{variant}"
            
            # Obtain configs from adapter
            train_world, val_world, test_world = adapter.build_worlds(seed=42, test_samples=5000, variant=variant)
            gym_orig, gym_ntbn, train_cfg = adapter.get_variant_configs(variant)
            
            recurrent_states = {
                "classic": gym_orig.agent.classic,
                "aggregate": gym_orig.agent.aggregate,
                "past_repr": gym_orig.agent.past_repr,
                "event": gym_orig.agent.event,
            }
            
            spec = ConfigSpec(
                config_id=config_id,
                group_id=group_id,
                variant=variant,
                notebook_path=adapter.notebook_path,
                theoretical_status=adapter.theoretical_ntbn_status,
                train_samples=train_world.nSamples,
                val_samples=val_world.nSamples,
                test_samples=test_world.nSamples,
                steps=train_world.nSteps,
                dt=train_world.dt,
                epochs=train_cfg.epochs,
                batch_size=train_cfg.batch_size,
                learning_rate=train_cfg.learning_rate,
                utility=gym_orig.utility,
                lmbda=gym_orig.lmbda,
                cost_s=train_world.config.cost_s,
                n_inst=train_world.nInst,
                depth=gym_orig.agent.depth,
                width=gym_orig.agent.width,
                activation=gym_orig.agent.activation,
                recurrent_states=recurrent_states,
                description=f"{adapter.description} [Variant: {variant}]"
            )
            catalog[config_id] = spec
            
    return catalog


CATALOG = build_catalog()


def get_config_spec(config_id: str) -> ConfigSpec:
    if config_id not in CATALOG:
        raise KeyError(f"Unknown config_id '{config_id}'. Supported ({len(CATALOG)} total): {list(CATALOG.keys())}")
    return CATALOG[config_id]


def list_configs_by_group(group_id: str) -> List[ConfigSpec]:
    return [spec for spec in CATALOG.values() if spec.group_id == group_id]
