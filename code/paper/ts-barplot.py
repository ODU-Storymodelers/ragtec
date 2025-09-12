#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicate Mozambique bar charts (same square dimensions as the network)
Reads:
  - mozambique_topic_roles_analysis.csv
Outputs:
  - mozambique_bar_distribution.png
  - mozambique_bar_importance.png
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ts_style.py
import matplotlib as mpl

def apply_ts_style():
    mpl.rcParams.update({
        # Fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # Lines / axes
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.5,
        # Savefig: keep consistent canvas (no tight bounding-box auto-cropping)
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.02,
    })


# ---------- Config ----------
ROLES_PATH = "../../output/mozambique/ragtec/storytelling/mozambique_topic_roles_analysis.csv"
FIGSIZE = (10, 10)  # square, to match the network figure
TITLE_PREFIX = ""

# ts_barplots_export.py
import pandas as pd
import matplotlib.pyplot as plt

FIGSIZE = (3.2, 3.2)   # square; matches network
LEFT, RIGHT, BOTTOM, TOP = 0.12, 0.98, 0.14, 0.98  # consistent margins

LETTER_MAP = {
    "Election Violence": "A",
    "Climate Challenges": "B",
    "Insurgency Crisis": "C",
    "Social Media Restrictions": "D",
    "Economic Disparities": "E",
    "Humanitarian Response": "F",
}

PRIMARY_COLOR = "#4E4E4E"   # dark gray (primary)
SECONDARY_COLOR = "#2CA02C" # green (secondary) -- pick your final accent

def export_distribution(csv_path, out_png):
    apply_ts_style()
    roles = pd.read_csv(csv_path)
    roles["letter"] = roles["topic"].map(LETTER_MAP)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars1 = ax.bar(roles["letter"], roles["primary_count"], color="#6A51A3", label="Primary Topic")
    bars2 = ax.bar(roles["letter"], roles["secondary_count"],
                  bottom=roles["primary_count"], color="#4CAE4C", label="Secondary Topic")

    ax.set_ylabel("Number of articles")
    ax.legend(frameon=False, loc="best")
    # No title; legend moved to caption; keep labels short
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)
    fig.savefig(out_png)   # No bbox_inches='tight'
    plt.close(fig)

def export_importance(csv_path, out_png):
    apply_ts_style()
    roles = pd.read_csv(csv_path)
    roles["letter"] = roles["topic"].map(LETTER_MAP)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Use Blues colormap for bar colors
    cmap = plt.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    importance_pct = roles["primary_ratio"] * 100
    bar_colors = [cmap(norm(v)) for v in importance_pct]
    bars = ax.bar(roles["letter"], importance_pct, color=bar_colors)

    # Optional reference lines (subtle, so they don't blow margins)
    for y, ls in [(20, (2,2)), (50, (3,2)), (80, (4,2))]:
        ax.axhline(y, color="0.5", linestyle=(0, ls), linewidth=0.7)
    
    
    ax.set_ylabel("Topic importance (%)")
    ax.set_ylim(0, 100)

    # Remove y-axis tick labels entirely
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)


    # # Format y-axis ticks as percentages and ensure labels are visible
    # ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0, symbol='%'))
    # ax.tick_params(axis='y', which='both', labelleft=True, left=True)

    # Tight layout to help with label visibility
    plt.tight_layout()

    # Add colorbar legend for the Blues colormap
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    # cbar.set_label('Topic importance (%)')
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0, symbol='%'))

    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)
    fig.savefig(out_png)
    plt.close(fig)

if __name__ == "__main__":
    export_distribution(ROLES_PATH, "../../image/mozambique_bar_distribution.png")
    export_importance(ROLES_PATH, "../../image/mozambique_bar_importance.png")