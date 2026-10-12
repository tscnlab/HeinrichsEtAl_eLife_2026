"""Abstract figure: protocol timeline showing chromatic stimulation conditions."""

import numpy as np
import matplotlib.pyplot as plt
from civibeplr.vis_io import apply_font_style

apply_font_style()
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from pathlib import Path

OUTPUT_PATH = Path("results/figures/abstract_protocol.svg")

CONDITIONS = ["mel", "s", "l-m", "lms", "flux"]
CONDITION_LABELS = {
    "mel": "Melanopsin",
    "s": "S-cone",
    "l-m": "L-M",
    "lms": "Luminance",
    "flux": "Broadband flux",
}

FILL_COLORS_POS = {
    "mel":  "#00f7ff",
    "s":    "#6200ff",
    "l-m":  "#ff00c8",
    "lms":  "#fff6e3",
    "flux": "#e6fcfc",
}

SINE_CYCLES = 2
SINE_AMPLITUDE = 0.7
GAP_WIDTH = 1.5
SINE_DURATION = 2.0
ADAPTATION_WIDTH = 2.0
BASELINE_END = 0.5

GREY_MID = "#c9c9c9"
LINE_COLOR = "#000000"
BRACKET_COLOR = "#888888"
BRACKET_LABEL_COLOR = "#555555"


def _complementary(hex_color: str) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((1 - r, 1 - g, 1 - b))


def _sine_burst(t_start, duration, amplitude, cycles, n=500):
    t = np.linspace(t_start, t_start + duration, n)
    y = amplitude * np.sin(2 * np.pi * cycles * (t - t_start) / duration)
    return t, y


def _gradient_fill(ax, t_arr, y_arr, pos_color: str, amplitude: float):
    comp_color = _complementary(pos_color)
    grey = np.array(mcolors.to_rgb(GREY_MID))
    pos_rgb = np.array(mcolors.to_rgb(pos_color))
    neg_rgb = np.array(mcolors.to_rgb(comp_color))

    n_y = 256
    y_range = np.linspace(-amplitude, amplitude, n_y)

    img_col = np.zeros((n_y, 1, 4))
    for i, yv in enumerate(y_range):
        frac = abs(yv) / amplitude
        rgb = grey * (1 - frac) + (pos_rgb if yv >= 0 else neg_rgb) * frac
        img_col[i, 0, :3] = np.clip(rgb, 0, 1)
        img_col[i, 0, 3] = 1.0

    img = np.tile(img_col, (1, len(t_arr), 1))
    im = ax.imshow(
        img,
        extent=[t_arr[0], t_arr[-1], -amplitude, amplitude],
        origin="lower", aspect="auto", zorder=0,
    )

    verts = list(zip(t_arr, y_arr)) + list(zip(t_arr[::-1], np.zeros(len(t_arr))))
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(t_arr) - 1) + [MPath.LINETO] * len(t_arr)
    patch = PathPatch(MPath(verts, codes), transform=ax.transData,
                      facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    im.set_clip_path(patch)


def _draw_bracket(ax, x_start, x_end, y, tick_h, label, fontsize=8):
    ax.plot([x_start, x_end], [y, y], color=BRACKET_COLOR, linewidth=1.0, zorder=5)
    ax.plot([x_start, x_start], [y - tick_h, y], color=BRACKET_COLOR, linewidth=1.0, zorder=5)
    ax.plot([x_end, x_end], [y - tick_h, y], color=BRACKET_COLOR, linewidth=1.0, zorder=5)
    ax.text((x_start + x_end) / 2, y + 0.04, label,
            ha="center", va="bottom", fontsize=fontsize,
            color=BRACKET_LABEL_COLOR, style="italic")


def make_protocol_figure():
    t = 0.0
    adapt_start = t
    adapt_end = t + ADAPTATION_WIDTH
    t = adapt_end

    segments = []
    for i, cond in enumerate(CONDITIONS):
        if i > 0:
            segments.append(("gap", t, GAP_WIDTH))
            t += GAP_WIDTH
        segments.append(("sine", t, SINE_DURATION, cond))
        t += SINE_DURATION
    total_time = t + GAP_WIDTH

    sine_segs = [(s[1], s[1] + s[2]) for s in segments if s[0] == "sine"]
    gap_segs  = [(s[1], s[1] + s[2]) for s in segments if s[0] == "gap"]

    fig, ax = plt.subplots(figsize=(12, 4.4))

    # Shade white-light regions: adaptation, inter-trial gaps, and post-last-sine tail
    white_regions = [(adapt_start, adapt_end)] + gap_segs + [(sine_segs[-1][1], sine_segs[-1][1] + GAP_WIDTH)]
    for g_start, g_end in white_regions:
        ax.axvspan(g_start, g_end, ymin=0, ymax=1,
                   color="#f5f5f5", zorder=0, linewidth=0)

    # Draw baseline only through the flat (white-light) sections, not under sine bursts
    for g_start, g_end in white_regions:
        ax.plot([g_start, g_end], [0, 0], color=LINE_COLOR, linewidth=1.2, zorder=3)

    for seg in segments:
        if seg[0] != "sine":
            continue
        _, t_start, duration, cond = seg
        t_arr, y_arr = _sine_burst(t_start, duration, SINE_AMPLITUDE, SINE_CYCLES)
        pos_color = FILL_COLORS_POS[cond]

        _gradient_fill(ax, t_arr, y_arr, pos_color, SINE_AMPLITUDE)
        ax.plot(t_arr, y_arr, color=LINE_COLOR, linewidth=1.5, zorder=4)

        label_x = t_start + duration / 2
        ax.text(label_x, -SINE_AMPLITUDE - 0.18, CONDITION_LABELS[cond],
                ha="center", va="top", fontsize=7.5,
                color=FILL_COLORS_POS[cond], fontweight="bold")

    # --- Brackets ---
    # Two bracket rows: row 1 (closer) for stimulation + baseline, row 2 for adaptation + trial
    bracket_y1 = SINE_AMPLITUDE + 0.22   # stimulation / baseline / adaptation
    bracket_y2 = SINE_AMPLITUDE + 0.62   # trial duration
    tick_h = 0.06

    # Adaptation phase (row 1)
    _draw_bracket(ax, adapt_start, adapt_end, bracket_y1, tick_h,
                  "Adaptation\n(4 min, 60 cd/m2)", fontsize=7.5)

    # Stimulation bracket -- first sine (row 1)
    _draw_bracket(ax, sine_segs[0][0], sine_segs[0][1], bracket_y1, tick_h,
                  "Stimulation\n5 s, 0.4 Hz", fontsize=7.5)

    # Baseline bracket -- first gap (row 1)
    if gap_segs:
        _draw_bracket(ax, gap_segs[0][0], gap_segs[0][1], bracket_y1, tick_h,
                      "White baseline\n13 s", fontsize=7.5)

    # Trial duration bracket: first sine through end of first gap (row 2)
    if gap_segs:
        trial_start = sine_segs[0][0]
        trial_end   = gap_segs[0][1]
        _draw_bracket(ax, trial_start, trial_end, bracket_y2, tick_h,
                      "1 trial = 18 s  (25 trials per test, 5x each condition)", fontsize=7.5)

    # Annotation: 25 trials, m-sequence counterbalancing
    ax.text(total_time + 0.1, bracket_y2 + 0.05,
            "m-sequence\ncounterbalancing",
            ha="left", va="center", fontsize=6.5, color=BRACKET_LABEL_COLOR,
            style="italic", linespacing=1.4)

    # x-axis spine well below the trough so it reads as a true time axis
    y_bottom = -SINE_AMPLITUDE - 0.55
    ax.set_xlim(-0.2, total_time + 1.6)
    ax.set_ylim(y_bottom - 0.1, SINE_AMPLITUDE + 1.05)

    ax.set_ylabel("Photoreceptor contrast", fontsize=9)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_yticks([-SINE_AMPLITUDE, 0, SINE_AMPLITUDE])
    ax.set_yticklabels(["-", "0", "+"], fontsize=8)
    ax.set_xticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", y_bottom))
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(LINE_COLOR)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_position(("data", adapt_start - 0.05))
    ax.spines["left"].set_bounds(-SINE_AMPLITUDE, SINE_AMPLITUDE)

    # Legend with black edge
    patches = [
        mpatches.Patch(facecolor=FILL_COLORS_POS[c], edgecolor="black",
                       linewidth=0.8, label=CONDITION_LABELS[c])
        for c in CONDITIONS
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5,
              frameon=False, ncol=len(CONDITIONS))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = make_protocol_figure()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")
    plt.show()
