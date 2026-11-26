"""Mixed-precision VMC benchmark on the RTX 5090 and A100.

Three panels show mirrored RTX 5090 time/batch bars, 10,000-step accuracy
estimated from 10 million samples, and a full-width process breakdown.
Precision sets the hue; process stage sets color intensity. Run from any directory;
the default input is Data/precision_benchmark.h5 and output is
Figures/mixed_precision_benchmark.pdf. No Jupyter or GPU runtime is required.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
from matplotlib.colors import to_rgb

import plot_config as cfg

DATA_FILE = cfg.DATA_DIR / "precision_benchmark.h5"
ESTIMATE_GROUP = "alex_10000_10m"
PRECISION_COLORS = {
    "FP64 / FP64": "#4477AA",
    "FP32 / FP64": "#44AA99",
    "FP32 / FP32": "#CCAA55",
    "FP16 / FP64": "#8877AA",
    "FP16 / FP32": "#CC6677",
}
SYMMETRY_HATCHES = ["", "///", "..."]
SHORT_VARIANTS = ["Identity", r"$D_4$", r"$D_4+T$"]
# Fraction of the precision color retained in each process stage.
# The grayscale legend uses the same mixture with black as its base color.
PROCESS_INTENSITIES = np.array([1.0, .72, .44, .16])
PROCESS_COLORS = [str(1.0-strength) for strength in PROCESS_INTENSITIES]
PROCESS_LABELS = ["Sampling", "Objective", "Solve", "Other"]
COMPONENT_LABELS = [
    "MCMC sampling", "Objective / local energies", "Neural tangent kernel + RHS",
    "Direct linear solve", "Parameter backprojection", "Other solve work", "Other step work",
]


def arrays(group):
    return {
        name: dataset.asstr()[()] if h5py.check_string_dtype(dataset.dtype) else dataset[()]
        for name, dataset in group.items() if isinstance(dataset, h5py.Dataset)
    }


def load_data(path=DATA_FILE):
    """Load the archived measurements and check the plotted statistics."""
    with h5py.File(path, "r") as source:
        if source.attrs["schema"] != "jvmc-precision-benchmark-v1":
            raise ValueError("Unsupported precision benchmark archive schema")
        if not source.attrs["complete"] or source.attrs["completed_local_variants"] != 15:
            raise ValueError("Precision benchmark archive is incomplete")
        methods = json.loads(source["metadata/methods_json"][:].tobytes())
        if ESTIMATE_GROUP not in source or not source[ESTIMATE_GROUP].attrs["complete"]:
            raise ValueError(f"Archive needs complete /{ESTIMATE_GROUP} estimates")
        precision, rtx, alex = (arrays(source[name]) for name in ("precision", "rtx5090", ESTIMATE_GROUP))
    assert rtx["native_hot_seconds"].shape == (5, 3, 3)
    np.testing.assert_allclose(rtx["native_median_seconds"],
                               np.median(rtx["native_hot_seconds"], axis=-1), rtol=1e-11)
    np.testing.assert_allclose(rtx["exclusive_mean_seconds"],
                               rtx["exclusive_hot_seconds"].mean(axis=2), rtol=1e-12)
    np.testing.assert_allclose(rtx["exclusive_hot_seconds"].sum(axis=-1),
                               rtx["wall_seconds"][:, :, 1:], rtol=1e-12)
    reference = methods["ed_reference_energy"]
    np.testing.assert_allclose(alex["relative_error"],
                               (alex["energy"]-reference)/abs(reference), rtol=1e-12)
    assert np.all(rtx["exclusive_hot_seconds"] >= 0)
    assert np.all(alex["validation_samples"] == 10_000_000)
    assert np.all(alex["updates"] == 10000) and np.all(alex["validation_passed"])
    np.testing.assert_allclose(alex["relative_error_mcse"], alex["energy_mcse"]/abs(reference), rtol=1e-12)
    assert np.all(alex["variance_mcse"] > 0)
    for i, record in enumerate(alex["endpoint_evidence_json"]):
        evidence = json.loads(record)
        result = evidence["result"]
        assert evidence["configuration"]["training_checkpoint_step"] == 10000
        assert result["samples"] == 10_000_000 and result["passed"]
        for name, field in [("energy", "mean"), ("energy_mcse", "mcse"),
                            ("variance", "variance"), ("variance_mcse", "variance_mcse")]:
            np.testing.assert_allclose(alex[name][i], result[field], rtol=1e-12)
    return precision, rtx, alex


def validate_layout(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    offscale = set()
    for axis in figure.axes:
        for positions, ticklabels, limits in ((axis.get_xticks(), axis.get_xticklabels(), axis.get_xlim()),
                                              (axis.get_yticks(), axis.get_yticklabels(), axis.get_ylim())):
            low, high = sorted(limits)
            offscale.update(label for position, label in zip(positions, ticklabels) if not low <= position <= high)
    texts = [text for text in figure.findobj(Text)
             if text not in offscale and text.get_visible() and text.get_text().strip()]
    outside = []
    for text in texts:
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > figure.bbox.x1+1 or bounds.y1 > figure.bbox.y1+1:
            outside.append(text.get_text())
    if outside:
        raise ValueError(f'Text outside the canvas: {outside}')
    # Legends must not cover any axes text, figure annotation, or each other.
    legend_bounds = [legend.get_window_extent(renderer) for legend in figure.legends]
    legend_texts = {text for legend in figure.legends for text in legend.findobj(Text)}
    for i, bounds in enumerate(legend_bounds):
        assert not any(bounds.overlaps(other) for other in legend_bounds[i+1:])
        for text in texts:
            if text not in legend_texts:
                assert not bounds.overlaps(text.get_window_extent(renderer)), text.get_text()
    assert min(text.get_fontsize() for text in texts) >= 9
    np.testing.assert_allclose(figure.get_figwidth()/cfg.MM, cfg.TEXT_WIDTH_MM)
    return dict(width_mm=figure.get_figwidth()/cfg.MM, height_mm=figure.get_figheight()/cfg.MM,
        minimum_font_pt=min(text.get_fontsize() for text in texts), outside_canvas=outside, legend_text_overlaps=False)



def grid(axis, which='y'):
    axis.set_axisbelow(True)
    axis.grid(axis=which, color='#E9EBED', linewidth=.5)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def runtime_and_batches(time_axis, batch_axis, rtx, colors, pair_labels):
    # Two aligned log axes meet at a visual divider, not a common zero.
    # Every upper/lower bar pair represents the same precision and symmetry.
    hot = rtx['native_hot_seconds']
    width = .24
    time_floor, batch_floor = .045, 512
    for j, hatch in enumerate(SYMMETRY_HATCHES):
        x = np.arange(5) + (j-1)*width
        median = rtx['native_median_seconds'][:, j]
        time_bars = time_axis.bar(x, median-time_floor, bottom=time_floor,
                                  width=width*.9, color=colors, hatch=hatch,
                                  edgecolor='#38424C', linewidth=.25, zorder=3)
        time_axis.errorbar(x, median,
            yerr=[median-hot[:, j].min(axis=-1), hot[:, j].max(axis=-1)-median],
            fmt='none', color='#333333', elinewidth=.7, capsize=1.5, zorder=4)
        batch = rtx['evaluation_batch_size'][:, j]
        batch_bars = batch_axis.bar(x, batch-batch_floor, bottom=batch_floor,
                                    width=width*.9, color=colors, hatch=hatch,
                                  edgecolor='#38424C', linewidth=.25, zorder=3)
        np.testing.assert_allclose([bar.get_y()+bar.get_height() for bar in time_bars], median)
        np.testing.assert_array_equal([bar.get_y()+bar.get_height() for bar in batch_bars], batch)
    time_axis.set_yscale('log')
    time_axis.set_ylim(time_floor, 35)
    time_axis.set_yticks([.1, 1, 10], ['0.1', '1', '10'])
    time_axis.yaxis.set_minor_locator(NullLocator())
    time_axis.set_ylabel('Time [s]', labelpad=6)
    time_axis.tick_params(axis='x', labelbottom=False, length=0)
    grid(time_axis)
    # Larger batches extend farther down. Positive tick labels show counts.
    batch_axis.set_yscale('log', base=2)
    batch_axis.set_ylim(32768, batch_floor)
    batch_axis.set_yticks([1024, 4096, 16384], [r'$2^{10}$', r'$2^{12}$', r'$2^{14}$'])
    batch_axis.yaxis.set_minor_locator(NullLocator())
    batch_axis.set_ylabel('Batch size', labelpad=6)
    batch_axis.set_xlabel('Precision (network / VMC)')
    batch_axis.tick_params(axis='x', length=0, pad=5)
    for axis in (time_axis, batch_axis):
        axis.set_xlim(-.55, 4.55)
        axis.set_xticks(np.arange(5), pair_labels)
    grid(batch_axis)
    # Draw the horizontal axis at the seam; leave the tick-label positions below.
    time_axis.spines['bottom'].set_visible(False)
    batch_axis.spines['bottom'].set_visible(False)
    batch_axis.spines['top'].set_visible(False)
    batch_axis.axhline(batch_floor, color='black', linewidth=.8, zorder=5, clip_on=False)


def process_times(axis, rtx, colors, pair_labels):
    # Five precision blocks, each with identity, D4 and D4+T bars.
    # Exclusive solve components remain grouped without dropping any time.
    means = rtx['exclusive_mean_seconds']
    grouped = np.stack([means[..., 0], means[..., 1], means[..., 2:6].sum(axis=-1),
                        means[..., 6]], axis=-1)
    np.testing.assert_allclose(grouped.sum(axis=-1), means.sum(axis=-1), rtol=1e-14)
    fractions = grouped / grouped.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(fractions.sum(axis=-1), 1, atol=1e-14)
    positions = (4*np.arange(5)[:, None] + np.arange(3)).ravel()
    values = fractions.reshape(15, 4)
    base_colors = np.repeat(np.array([to_rgb(color) for color in colors]), 3, axis=0)
    bottom = np.zeros(15)
    for k, strength in enumerate(PROCESS_INTENSITIES):
        # Adding white varies intensity while preserving each precision hue.
        facecolors = 1.0-strength*(1.0-base_colors)
        axis.bar(positions, 100*values[:, k], bottom=bottom, width=.76,
                 color=facecolors, edgecolor='white', linewidth=.35, zorder=3)
        bottom += 100*values[:, k]
    np.testing.assert_allclose(bottom, 100, atol=1e-12)
    for j, hatch in enumerate(SYMMETRY_HATCHES):
        if hatch:
            axis.bar(4*np.arange(5)+j, np.full(5, 100), width=.76,
                     facecolor='none', edgecolor='#38424C', linewidth=.25,
                     hatch=hatch, zorder=4)
    axis.set_xlim(-.7, 18.7)
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel(r'Step wall time [\%]')
    axis.set_xticks(4*np.arange(5)+1, pair_labels)
    axis.tick_params(axis='x', length=0, pad=5)
    axis.set_xlabel('Precision (network / VMC)')
    grid(axis)


def variance_error(axis, alex, colors, markers):
    for i, (color, marker) in enumerate(zip(colors, markers)):
        axis.errorbar(alex['relative_error'][i], alex['variance'][i],
            xerr=alex['relative_error_mcse'][i], yerr=alex['variance_mcse'][i],
            marker=marker, color=color, markeredgecolor='white', markeredgewidth=.5,
            elinewidth=.8, capsize=2, linestyle='none')
    axis.set_xscale('log')
    axis.set_yscale('log')
    # Include the full error bars of the selected cohort, with space around them.
    for values, errors, setter in [(alex['relative_error'], alex['relative_error_mcse'], axis.set_xlim),
                                    (alex['variance'], alex['variance_mcse'], axis.set_ylim)]:
        low, high = (values-errors).min(), (values+errors).max()
        assert low > 0
        pad = .18*(high-low)
        setter(max(low-pad, low*.8), high+pad)
    for coordinate, nbins in [(axis.xaxis, 4), (axis.yaxis, 5)]:
        coordinate.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]))
        coordinate.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        coordinate.set_minor_locator(NullLocator())
    axis.set_xlabel('Relative energy error\n' + r'$(E-E_0)/|E_0|$')
    axis.set_ylabel(r'Energy variance $\sigma_H^2/J_1^2$')
    grid(axis, 'both')


def make_figure(precision, rtx, alex):
    """Build the publication figure; the notebook calls this same function."""
    cfg.setup(grid=False)
    plt.rcParams["hatch.linewidth"] = .6
    labels = precision["label"].tolist()
    pair_labels = [label.replace("FP", "").replace(" / ", "/") for label in labels]
    colors = [PRECISION_COLORS[label] for label in labels]
    markers = precision["marker"].tolist()
    height_mm = 166
    fig = plt.figure(figsize=cfg.figure_size(height_mm=height_mm))

    def panel_rect(left, bottom_mm, width, panel_height_mm):
        return [left, bottom_mm/height_mm, width, panel_height_mm/height_mm]

    a = fig.add_axes(panel_rect(.10, 122, .35, 26))
    a_batch = fig.add_axes(panel_rect(.10, 103, .35, 19), sharex=a)
    b = fig.add_axes(panel_rect(.61, 103, .35, 45))
    c = fig.add_axes(panel_rect(.10, 22, .86, 43))
    runtime_and_batches(a, a_batch, rtx, colors, pair_labels)
    variance_error(b, alex, colors, markers)
    process_times(c, rtx, colors, pair_labels)
    for axis, title in [(a, 'a) Time and batch size'), (b, 'b) Ground-state accuracy'),
                         (c, 'c) Process fractions')]:
        axis.set_title(title, loc='left', pad=8)
    fig.legend([Patch(facecolor='#E4E7EA', edgecolor='#38424C', hatch=hatch, linewidth=.25)
                for hatch in SYMMETRY_HATCHES], SHORT_VARIANTS,
               loc='upper left', bbox_to_anchor=(.09, 163/height_mm), ncol=3, frameon=False,
               borderpad=0, handlelength=1, handletextpad=.4, columnspacing=.9)
    fig.legend([Line2D([], [], marker=marker, color=color, linestyle='none')
                for color, marker in zip(colors, markers)], pair_labels,
               loc='upper center', bbox_to_anchor=(.53, 84/height_mm), ncol=5, frameon=False,
               borderpad=0, handletextpad=.4, columnspacing=1)
    fig.legend([Patch(facecolor=color) for color in PROCESS_COLORS], PROCESS_LABELS,
               loc='lower center', bbox_to_anchor=(.53, 2/height_mm), ncol=4, frameon=False,
               borderpad=0, handlelength=1, handletextpad=.5, columnspacing=1.2)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Benchmark HDF5 archive")
    args = parser.parse_args()
    fig = make_figure(*load_data(args.data))
    validate_layout(fig)
    cfg.save(fig, "mixed_precision_benchmark")
    plt.close(fig)


if __name__ == "__main__":
    main()
