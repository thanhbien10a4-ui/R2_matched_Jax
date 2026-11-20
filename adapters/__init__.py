"""
adapters package initialization.
"""

from .base_adapter import BaseNotebookAdapter
from .adapter_blackscholes import BlackScholesAdapter
from .adapter_trainer_bs import TrainerBSAdapter
from .adapter_trainer import TrainerAdapter
from .adapter_american_digital import AmericanDigitalAdapter
from .adapter_barrier import BarrierAdapter
from .adapter_barrier_smalldt import BarrierSmallDTAdapter
from .adapter_fwdstart import ForwardStartAdapter


ADAPTERS = {
    "blackscholes": BlackScholesAdapter(),
    "trainer-bs": TrainerBSAdapter(),
    "trainer": TrainerAdapter(),
    "trainer-american_digital": AmericanDigitalAdapter(),
    "trainer-recurrent-barrier": BarrierAdapter(),
    "trainer-recurrent-barrier_smalldt": BarrierSmallDTAdapter(),
    "trainer-recurrent-fwdstart": ForwardStartAdapter(),
}


def get_adapter(group_id: str) -> BaseNotebookAdapter:
    if group_id not in ADAPTERS:
        raise KeyError(f"Unknown group_id '{group_id}'. Supported: {list(ADAPTERS.keys())}")
    return ADAPTERS[group_id]
