"""Milestones 9-10: analysis service, FastAPI backend and the dashboard it serves.

`disc` needs only numpy/scipy. `service` imports the triage model and so
needs PyTorch; it is loaded on first use, so `drpinn.api.disc` (used by
scripts/m9_check_disc.py) works without PyTorch. `app` (FastAPI) and `colab`
are imported on demand.
"""

from .disc import DISC_TO_FOV, estimate_disc

__all__ = ["DISC_TO_FOV", "Analyser", "ServiceConfig", "estimate_disc"]


def __getattr__(name: str):
    if name in ("Analyser", "ServiceConfig"):
        from . import service
        return getattr(service, name)
    raise AttributeError(f"module 'drpinn.api' has no attribute {name!r}")
