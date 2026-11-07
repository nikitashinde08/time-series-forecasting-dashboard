"""Central configuration: model names, palette, default hyper-parameters, widget key map."""
from __future__ import annotations

# ---- Author credit / watermark: edit ONLY these two lines to change it everywhere ----------------------
AUTHOR_NAME = "Nikita Shinde"
AUTHOR_TAGLINE = "MSc Data Science & AI"
AUTHOR_FULL = f"{AUTHOR_NAME} · {AUTHOR_TAGLINE}"

MODEL_NAMES = ["ARIMA", "SARIMA", "Prophet", "LSTM (MLP)"]

# Executive light palette - saturated, high-contrast colours chosen to read clearly on white.
COLORS = {
    "actual": "#0f172a",
    "ARIMA": "#2563eb",
    "SARIMA": "#16a34a",
    "Prophet": "#ea580c",
    "LSTM (MLP)": "#9333ea",
    "anomaly": "#dc2626",
    "muted": "#64748b",
    "accent": "#0284c7",
    "stage1": "#0284c7",
    "stage2": "#d97706",
}
# Colour swaps applied when rendering figures to the white PDF page (palette is already print-safe).
PRINT_COLOR_MAP: dict = {}

CPS_OPTIONS = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
UNITS_OPTIONS = [16, 32, 64, 128, 256]
EPOCH_OPTIONS = [100, 200, 300, 400, 500]

# Sidebar widget keys for each tunable hyper-parameter (used to "apply tuned params" to the UI).
KEYMAP = {
    "ARIMA": {"p": "a_p", "d": "a_d", "q": "a_q"},
    "SARIMA": {"p": "s_p", "d": "s_d", "q": "s_q", "P": "s_P", "D": "s_D", "Q": "s_Q"},
    "Prophet": {"mode": "pr_mode", "cps": "pr_cps", "yearly": "pr_year", "weekly": "pr_week"},
    "LSTM (MLP)": {"lookback": "nn_lb", "units": "nn_units", "epochs": "nn_ep"},
}

# Optuna search spaces: name -> ("int", lo, hi) | ("cat", [choices])
SEARCH_SPACES = {
    "ARIMA": {"p": ("int", 0, 6), "d": ("int", 0, 2), "q": ("int", 0, 6)},
    "SARIMA": {"p": ("int", 0, 3), "d": ("int", 0, 2), "q": ("int", 0, 3),
               "P": ("int", 0, 2), "D": ("int", 0, 1), "Q": ("int", 0, 2)},
    "Prophet": {"mode": ("cat", ["additive", "multiplicative"]), "cps": ("cat", CPS_OPTIONS),
                "yearly": ("cat", [True, False]), "weekly": ("cat", [True, False])},
    "LSTM (MLP)": {"lookback": ("int", 3, 60), "units": ("cat", UNITS_OPTIONS), "epochs": ("cat", EPOCH_OPTIONS)},
}
