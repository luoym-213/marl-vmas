"""CommSpread VMAS scenario and BenchMARL helpers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comm_spread.sar_scenario import SarScenario
    from comm_spread.scenario import CommSpreadScenario

__all__ = ["CommSpreadScenario", "SarScenario"]


def __getattr__(name: str):
    if name == "CommSpreadScenario":
        from comm_spread.scenario import CommSpreadScenario

        return CommSpreadScenario
    if name == "SarScenario":
        from comm_spread.sar_scenario import SarScenario

        return SarScenario
    raise AttributeError(name)
