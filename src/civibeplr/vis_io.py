from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Change to ".pdf" or ".svg" to switch all saved figures.
# DPI is ignored for vector formats (pdf, svg, eps).
FIGURE_FORMAT = ".png"
FIGURE_DPI = 300

_VECTOR_FORMATS = {".pdf", ".svg", ".eps"}


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = FIGURE_DPI,
    bbox_inches: str | None = "tight",
) -> None:
    """Save *fig* to *path*, replacing its suffix with FIGURE_FORMAT.

    DPI and bbox_inches are ignored for vector formats (pdf, svg, eps) since
    they are resolution-independent and matplotlib handles layout automatically.
    Prints the saved path and closes the figure when not in Jupyter.
    """
    path = Path(path).with_suffix(FIGURE_FORMAT)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if FIGURE_FORMAT not in _VECTOR_FORMATS:
        kwargs["dpi"] = dpi
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
    fig.savefig(path, **kwargs)
    print(f"Saved: {path}")
    if not _in_jupyter():
        plt.close(fig)


TRIAL_TYPE_COLORS = {
    "mel":   "#2BA6B6",
    "s":     "#9749FD",
    "l-m":   "#4CD597",
    "lms":   "#565656",
    "flux":  "#A1A1A1",
}

# Full condition names for figure titles and panel headings.
CONDITION_LABELS = {
    "flux": "Broadband flux (LMS+Mel)",
    "mel":  "Melanopsin (Mel)",
    "s":    "S-cone (S)",
    "l-m":  "L-M cone-opponent (L-M)",
    "lms":  "Luminance (LMS)",
}

# Order used whenever conditions are laid out side by side.
CONDITION_ORDER = ["mel", "s", "l-m", "lms", "flux"]


def readable(color: str, max_luminance: float = 0.40) -> tuple[float, float, float]:
    """Darken *color* until it is legible as text on white.

    The condition palette is tuned for marks, where LMS (#565656) and flux
    (#A1A1A1) read fine; as title text the lighter entries fall below usable
    contrast, so scale them down to a luminance ceiling. Marks keep the
    original palette colour.
    """
    r, g, b = mcolors.to_rgb(color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance <= max_luminance:
        return r, g, b
    factor = max_luminance / luminance
    return r * factor, g * factor, b * factor


def apply_font_style() -> None:
    """Apply project-wide matplotlib style (Arial font)."""
    plt.rcParams["font.family"] = "Arial"


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter kernel (notebook or lab)."""
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False