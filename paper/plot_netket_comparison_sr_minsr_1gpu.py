"""Peak memory (a, b) and time (c, d) of one SR step (a, c) and one MinSR step (b, d), jVMC_exp vs NetKet, on 1 GPU.
Columns share x (samples, parameters) and rows share y (memory, time), so the panels touch.
Data from Scripts/compare_netket_sr.py and Scripts/compare_netket_minsr.py."""
import json

import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

SR_DATA = cfg.DATA_DIR / "compare_netket_sr"
MINSR_DATA = cfg.DATA_DIR / "compare_netket_minsr"
SERIES = {
    "jvmc-dense": ("jVMC, full Jacobian", cfg.COLORS["jvmc_dense"], "o"),
    "jvmc-batched": ("jVMC, batched Jacobian", cfg.COLORS["jvmc_batched"], "s"),
    "netket": ("NetKet, VMC_SR", cfg.COLORS["netket"], "^"),
}

def load_points(data, x):
    # Points of every series, sorted by x
    return {
        series: sorted((json.loads(p.read_text()) for p in data.glob(f"{series}_*.json")), key=lambda p: p[x])
        for series in SERIES
    }

def plot_step(ax_mem, ax_time, points, x, step):
    measured = [p for pts in points.values() for p in pts if "times" in p]
    limit = measured[0]["bytes_limit"] / 1e9

    for series, (label, color, marker) in SERIES.items():
        ok = [p for p in points[series] if "times" in p]
        n = [p[x] for p in ok]
        ax_mem.plot(n, [p["peak_bytes"] / 1e9 for p in ok], marker=marker, color=color, label=label)
        ax_time.plot(n, [np.median(p["times"][step]) for p in ok], marker=marker, color=color, label=label)

    # Series that run out of memory end with a marker at the memory available to JAX
    oom = [p[x] for pts in points.values() for p in pts if p.get("oom")]
    if oom:
        ax_mem.plot(oom, [limit] * len(oom), "x", color="k", ms=6, mew=1.2, label="out of memory")

cfg.setup()

fig, axes = plt.subplots(2, 2, figsize=cfg.figure_size(width_fraction=0.8, height_mm=145),
                         sharex="col", sharey="row")
(ax_sr_mem, ax_minsr_mem), (ax_sr_time, ax_minsr_time) = axes

plot_step(ax_sr_mem, ax_sr_time, load_points(SR_DATA, "n_samples"), "n_samples", "sr_step")
plot_step(ax_minsr_mem, ax_minsr_time, load_points(MINSR_DATA, "n_parameters"), "n_parameters", "minsr_step")

# With more samples than the batch size, the batched Jacobian is built in batches
batch_size = json.loads((SR_DATA / "config.json").read_text())["batch_size"]
ax_sr_mem.axvline(batch_size, color=cfg.COLORS["guide"], ls="--")
ax_sr_mem.annotate("batch size", (batch_size, 0.7), xycoords=("data", "axes fraction"),
                   xytext=(-4, 0), textcoords="offset points", rotation=90, ha="right", va="center", color=cfg.COLORS["guide"])

for ax, letter in zip(axes.flat, "abcd"):
    # Inside the panel, there is no space between panels for a title
    ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(2.0 ** np.arange(10, 21, 2))
    ax.set_yscale("log")
    cfg.plain_log_labels(ax.yaxis)
# Keep the tick labels next to where the panels meet apart: no time label at the top edge,
# and room between 2^20 of the samples and 2^10 of the parameters
ax_sr_time.set_ylim(top=2.9)
ax_sr_time.set_xlim(right=2**20.6)
ax_minsr_time.set_xlim(left=2**9.6)
# Ticks on the inner spines point inward, not into the neighboring panel
for ax in axes[0]:
    ax.tick_params(axis="x", which="both", direction="in")
for ax in axes[:, 1]:
    ax.tick_params(axis="y", which="both", direction="in")
# Shared axes, so labels only on the outer panels
ax_sr_time.set_xlabel("Number of samples")
ax_minsr_time.set_xlabel("Number of parameters")
ax_sr_mem.set_ylabel("Peak device memory [GB]")
ax_sr_time.set_ylabel("Time [s]")

# One entry per series over the four panels
legend = {}
for ax in axes.flat:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        legend.setdefault(label, handle)
# Two columns filled column by column, so jVMC in the first row, NetKet and out of memory in the second
order = ["jVMC, full Jacobian", "NetKet, VMC_SR", "jVMC, batched Jacobian", "out of memory"]
fig_legend = fig.legend([legend[label] for label in order], order, loc="upper center", ncol=2, frameon=False)
# Panels right below the legend
fig.canvas.draw()
fig.tight_layout(rect=(0, 0, 1, fig_legend.get_window_extent().transformed(fig.transFigure.inverted()).y0))
# The panels share their axes, so they touch
fig.subplots_adjust(wspace=0, hspace=0)
cfg.save(fig, "netket_comparison_sr_minsr_1gpu")
