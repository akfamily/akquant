"""Utility functions for plotting."""

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    go = None
    make_subplots = None


def check_plotly() -> bool:
    """Check if plotly is installed."""
    if not _HAS_PLOTLY:
        print(
            "Plotly is not installed. Please install it using `pip install plotly` "
            "or `pip install akquant[plot]`."
        )
        return False
    return True


THEMES = {
    "light": {
        "up_color": "#d32f2f",  # China Red (Rise)
        "down_color": "#2e7d32",  # China Green (Fall)
        "bg_color": "#ffffff",
        "grid_color": "#f0f0f0",
        "text_color": "#333333",
    },
    "dark": {
        "up_color": "#ff5252",  # Bright Red
        "down_color": "#69f0ae",  # Bright Green
        "bg_color": "#1e1e1e",
        "grid_color": "#2b2b2b",
        "text_color": "#e0e0e0",
    },
}


def get_color(theme: str, key: str) -> str:
    """Get color from theme."""
    return THEMES.get(theme, THEMES["light"]).get(key, "#000000")
