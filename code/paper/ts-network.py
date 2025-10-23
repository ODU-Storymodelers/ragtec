# ts_network_square.py
# Generate a square network figure that fits alongside square barplots

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

# ---------- Config ----------
COOCC_PATH = "../../output/mozambique/ragtec/storytelling/mozambique_topic_cooccurrence_matrix.csv"
ROLES_PATH = "../../output/mozambique/ragtec/storytelling/mozambique_topic_roles_analysis.csv"
OUT_PATH   = "../../image/mozambique_ts_network.png"

NODE_SIZE_SCALE = 95
EDGE_WIDTH_MIN = 0.8
EDGE_WIDTH_MAX = 5.0

# ---------- Shared style (match your barplots) ----------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "savefig.bbox": None,          # keep identical canvas size
    "savefig.pad_inches": 0.02,
})

FIGSIZE = (3.2, 3.2)  # same as barplots

LETTER_MAP = {
    "Election Violence": "EV",
    "Climate Challenges": "CC",
    "Insurgency Crisis": "IC",
    "Social Media Restrictions": "SMR",
    "Economic Disparities": "ED",
    "Humanitarian Response": "HR",
}

def export_network_square():
    coocc = pd.read_csv(COOCC_PATH).set_index("Unnamed: 0")
    roles = pd.read_csv(ROLES_PATH)

    # Build graph
    G = nx.Graph()
    for _, r in roles.iterrows():
        G.add_node(r["topic"], size=float(r["total_count"]), imp=float(r["primary_ratio"]))
    for i, t1 in enumerate(coocc.index):
        for j, t2 in enumerate(coocc.index):
            if j > i:
                w = float(coocc.iloc[i, j])
                if w > 0:
                    G.add_edge(t1, t2, w=w)

    # Layout + encodings
    pos = nx.spring_layout(G, seed=42, k=0.7)
    # ensure some padding around layout so nodes/labels are not clipped
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    dx = (max(xs) - min(xs)) * 0.12 or 0.1
    dy = (max(ys) - min(ys)) * 0.12 or 0.1
    x_min, x_max = min(xs) - dx, max(xs) + dx
    y_min, y_max = min(ys) - dy, max(ys) + dy
    node_sizes = [G.nodes[n]["size"] * NODE_SIZE_SCALE for n in G.nodes]  # scale as needed, slightly smaller
    node_colors = [G.nodes[n]["imp"] for n in G.nodes]

    # Normalize edge widths
    raw_w = [G.edges[e]["w"] for e in G.edges] or [1.0]
    w_min, w_max = min(raw_w), max(raw_w)
    def _scale(w, lo=EDGE_WIDTH_MIN, hi=EDGE_WIDTH_MAX):
        return lo if w_max == w_min else lo + (w - w_min) * (hi - lo) / (w_max - w_min)
    edge_widths = [_scale(G.edges[e]["w"]) for e in G.edges]

    # Figure/axes
    fig = plt.figure(figsize=FIGSIZE)
    # more left/bottom padding; slightly smaller drawing area to ensure nodes/labels stay inside
    ax = fig.add_axes([0.06, 0.28, 0.88, 0.66])

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="0.65", alpha=0.8)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
        cmap="Blues", vmin=0, vmax=1, linewidths=1.0, edgecolors="black"
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={node: LETTER_MAP.get(node, node) for node in G.nodes},
        font_size=9,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()

    # Horizontal colorbar
    sm = mpl.cm.ScalarMappable(cmap="Blues", norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.20, 0.10, 0.60, 0.05])  # slightly narrower/shorter
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Topic Importance (%)", fontsize=6)
    cb.ax.tick_params(labelsize=6, pad=1)
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    export_network_square()