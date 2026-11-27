"""Load the standalone Mask2Former head from a pinned external REIN checkout."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_MODULE_NAME = "causalq._upstream_rein_mask2former"


def load_rein_mask2former_head(rein_root: Path) -> type:
    """Register the upstream head without initializing REIN's backbone package."""
    source = (rein_root / "rein/models/heads/rein_mask2former.py").resolve(strict=True)
    loaded = sys.modules.get(_MODULE_NAME)
    if loaded is not None:
        if Path(loaded.__file__) != source:
            raise RuntimeError(f"The REIN head is already loaded from {loaded.__file__}")
        return loaded.ReinMask2FormerHead

    spec = spec_from_file_location(_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the REIN head at {source}")
    module = module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        return module.ReinMask2FormerHead
    except BaseException:
        del sys.modules[_MODULE_NAME]
        raise
