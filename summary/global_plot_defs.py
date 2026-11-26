"""Compatibility import; paper/plot_config.py owns the shared plot settings."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_source = Path(__file__).resolve().parents[1] / "paper/plot_config.py"
_spec = spec_from_file_location("_jvmc_plot_config", _source)
_config = module_from_spec(_spec)
_spec.loader.exec_module(_config)


def global_plot_defs():
    """Apply the consolidated style for plots that configure their own grids."""
    _config.setup(grid=False)


figure_size = _config.figure_size
TEXT_WIDTH_MM = _config.TEXT_WIDTH_MM
MM = _config.MM
FONTSIZE = _config.FONTSIZE
TICK_FONTSIZE = _config.TICK_FONTSIZE
