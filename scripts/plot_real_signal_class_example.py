"""Plot HARD15 and INT15 classes on one real BOAS wearable EEG event."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paired_enriched_feature_pruning_v0_1 as prior


# Section 1: select one frozen primary event

SUBJECT = "sub-3"
TRANSITION_ID = 15
PLOT_WINDOW_SEC = 30.0

root = Path(__file__).resolve().parents[1]
labels = pd.read_csv(
    root / "labels/transition_labels_v0.1/transition_labels_v0.1.tsv", sep="\t"
)
event = labels.loc[labels.transition_id.eq(TRANSITION_ID)].iloc[0]
if event.subject != SUBJECT or event.transition_type != "REM_to_Wake":
    raise ValueError("The selected event identity changed")
boundary_sec = float(event.nominal_boundary_sec)


# Section 2: load the real signal using the project preprocessing path

raw_uv = prior.read_uv(SUBJECT, "headband", ["HB_1", "HB_2"])
filtered_uv = prior.filter_resample(raw_uv, prior.filter_sos())
sampling_frequency = prior.INPUT_SFREQ / 2.0
start = int(round((boundary_sec - PLOT_WINDOW_SEC) * sampling_frequency))
stop = int(round((boundary_sec + PLOT_WINDOW_SEC) * sampling_frequency))
signal = filtered_uv[:, start:stop]
time = np.arange(signal.shape[1]) / sampling_frequency - PLOT_WINDOW_SEC


# Section 3: draw the class map and real wearable traces

colors = {
    "negative": "#007C83",
    "excluded": "#9AA0A6",
    "hard": "#B3261E",
    "interval": "#E07A00",
    "rem": "#7257A8",
    "wake": "#3B78B5",
    "signal": "#202124",
}

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
grid = fig.add_gridspec(3, 1, height_ratios=[1.15, 1, 1])
timeline = fig.add_subplot(grid[0])
hb1 = fig.add_subplot(grid[1])
hb2 = fig.add_subplot(grid[2], sharex=hb1)

timeline.axvspan(-150, -135, color=colors["negative"], alpha=0.12)
timeline.axvspan(135, 150, color=colors["negative"], alpha=0.12)
timeline.axvspan(-135, 135, color=colors["excluded"], alpha=0.10)
timeline.fill_between([-30, 0], 2.45, 2.75, color=colors["rem"], alpha=0.75)
timeline.fill_between([0, 30], 2.45, 2.75, color=colors["wake"], alpha=0.75)
timeline.text(-15, 2.60, "PSG REM", color="white", ha="center", va="center")
timeline.text(15, 2.60, "PSG Wake", color="white", ha="center", va="center")

candidate_times = np.arange(-150, 151, 15, dtype=float)
for row_y, mode in [(1.75, "HARD15"), (0.75, "INT15")]:
    for candidate in candidate_times:
        if abs(candidate) >= 135:
            timeline.scatter(candidate, row_y, s=54, color=colors["negative"], marker="o", zorder=3)
        elif mode == "HARD15" and candidate == 0:
            timeline.scatter(candidate, row_y, s=90, color=colors["hard"], marker="D", zorder=4)
        elif mode == "INT15" and candidate in {-15, 0, 15}:
            timeline.scatter(candidate, row_y, s=90, color=colors["interval"], marker="D", zorder=4)
        else:
            timeline.scatter(candidate, row_y, s=42, color=colors["excluded"], marker="x", zorder=3)

timeline.axvline(0, color=colors["hard"], linewidth=2.2)
timeline.axvline(-15, color=colors["interval"], linewidth=1.5, linestyle="--")
timeline.axvline(15, color=colors["interval"], linewidth=1.5, linestyle="--")
timeline.set_xlim(-150, 150)
timeline.set_ylim(0.25, 2.95)
timeline.set_yticks([0.75, 1.75, 2.60])
timeline.set_yticklabels(["INT15 labels", "HARD15 labels", "PSG stage"])
timeline.set_xticks([-150, -135, -30, -15, 0, 15, 30, 135, 150])
timeline.set_xlabel("Seconds relative to the PSG-derived boundary")
timeline.grid(axis="x", alpha=0.18)
timeline.spines[["top", "right", "left"]].set_visible(False)
timeline.tick_params(axis="y", length=0)

legend = [
    Line2D([0], [0], marker="D", color="none", markerfacecolor=colors["hard"], markeredgecolor=colors["hard"], label="HARD15 positive: weight 1"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=colors["interval"], markeredgecolor=colors["interval"], label="INT15 positive: weight 1/3 each"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["negative"], markeredgecolor=colors["negative"], label="Negative: distance >=135 s"),
    Line2D([0], [0], marker="x", color=colors["excluded"], linestyle="none", label="Excluded: ambiguous near-event point"),
]
timeline.legend(
    handles=legend,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
)

for axis, values, channel in [(hb1, signal[0], "HB_1"), (hb2, signal[1], "HB_2")]:
    axis.axvspan(-30, 0, color=colors["rem"], alpha=0.08)
    axis.axvspan(0, 30, color=colors["wake"], alpha=0.08)
    axis.plot(time, values, color=colors["signal"], linewidth=0.45, rasterized=True)
    axis.axvline(-15, color=colors["interval"], linewidth=1.4, linestyle="--")
    axis.axvline(0, color=colors["hard"], linewidth=2.0)
    axis.axvline(15, color=colors["interval"], linewidth=1.4, linestyle="--")
    lower, upper = np.percentile(values, [0.5, 99.5])
    padding = max((upper - lower) * 0.08, 1.0)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_ylabel(f"{channel} (microvolts)")
    axis.grid(alpha=0.16)
    axis.spines[["top", "right"]].set_visible(False)

hb1.text(-15, hb1.get_ylim()[1], "PSG REM", color=colors["rem"], ha="center", va="top")
hb1.text(15, hb1.get_ylim()[1], "PSG Wake", color=colors["wake"], ha="center", va="top")
hb2.set_xlabel("Seconds relative to the PSG-derived REM-to-Wake boundary")

fig.suptitle(
    "Positive and negative classes on a real BOAS wearable EEG signal\n"
    f"{SUBJECT}, transition {TRANSITION_ID}, nominal boundary {boundary_sec:.0f} s",
    fontsize=16,
)
output = (
    root
    / "docs/figures"
    / "interval_label_example_sub-3_transition-15.png"
)
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)
