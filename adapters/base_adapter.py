"""
adapters/base_adapter.py: Abstract Base Adapter for Deep Hedging Notebook Groups.

Defines the contract for creating worlds, datasets, configurations, and metadata for each
of the 7 notebook groups in the original JAX repository.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import replace
import numpy as np
import jax.numpy as jnp

try:
    from core.model import AgentConfigOriginal, GymConfigOriginal, AgentConfigNTBN, GymConfigNTBN
    from core.train import TrainConfig, as_batch
except ImportError:
    from ..core.model import AgentConfigOriginal, GymConfigOriginal, AgentConfigNTBN, GymConfigNTBN
    from ..core.train import TrainConfig, as_batch


class BaseNotebookAdapter(ABC):
    """
    Standard interface for all 7 Deep Hedging notebook adapters.
    """
    
    @property
    @abstractmethod
    def group_id(self) -> str:
        """Unique group identifier matching notebook name."""
        pass
    
    @property
    @abstractmethod
    def notebook_path(self) -> str:
        """Relative path to source notebook in deephedging-jax."""
        pass

    @property
    @abstractmethod
    def variants(self) -> List[str]:
        """List of all configuration variants defined in this notebook."""
        pass

    @property
    @abstractmethod
    def theoretical_ntbn_status(self) -> str:
        """
        Classification:
          - SUPPORTED_THEORETICAL: Cost > 0, smooth payoff, convex utility
          - SUPPORTED_EMPIRICAL_ZERO_COST: Cost = 0 (band collapses to point delta)
          - SUPPORTED_EMPIRICAL_NONSMOOTH: Discontinuous / Barrier / Knock-out payoff
          - SUPPORTED_EMPIRICAL_PATH_DEPENDENT: Path-dependent strike (Forward-Start)
          - BLOCKED: Impossible without modifying underlying problem formulation
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Technical description of payoff, features, and financial dynamics."""
        pass

    @abstractmethod
    def build_worlds(self, seed: int = 42, train_samples: Optional[int] = None, val_samples: Optional[int] = None, test_samples: Optional[int] = None, variant: str = None) -> Tuple[Any, Any, Any]:
        """
        Simulates and returns (train_world, val_world, test_world).
        """
        pass

    @abstractmethod
    def get_variant_configs(self, variant: str) -> Tuple[GymConfigOriginal, GymConfigNTBN, TrainConfig]:
        """
        Returns (gym_cfg_original, gym_cfg_ntbn, train_cfg) for the specified variant.
        """
        pass

    def build_datasets(self, seed: int = 42, train_samples: Optional[int] = None, val_samples: Optional[int] = None, test_samples: Optional[int] = None, variant: Optional[str] = None) -> Tuple[dict, dict, dict, dict, int]:
        """
        Converts worlds into JAX batches and returns:
          (train_batch, val_batch, test_batch, feature_dims, n_inst)
        """
        v = variant if variant is not None else self.variants[0]
        train_world, val_world, test_world = self.build_worlds(seed=seed, train_samples=train_samples, val_samples=val_samples, test_samples=test_samples, variant=v)
        train_batch = as_batch(train_world)
        val_batch = as_batch(val_world)
        test_batch = as_batch(test_world)
        feature_dims = {k: v.shape[2] for k, v in train_world.features.items()}
        n_inst = train_world.nInst
        return train_batch, val_batch, test_batch, feature_dims, n_inst
