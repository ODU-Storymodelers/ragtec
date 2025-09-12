import matplotlib.pyplot as plt
import numpy as np

# Data
collections = ["Burundi", "DRC", "Mozambique", "Sudan"]
topic_counts = [1, 2, 3, 4, 5, 6, 7]

# Percentages per collection
burundi = [2.8, 19.3, 34.9, 27.5, 13.8, 0.9, 0.9]
drc = [56.9, 29.2, 10.8, 3.1, 0, 0, 0]
mozambique = [50.0, 28.3, 15.0, 5.0, 1.7, 0, 0]
sudan = [60.0, 25.0, 10.0, 5.0, 0, 0, 0]

data = np.array([burundi, drc, mozambique, sudan])

# Bar settings
x = np.arange(len(topic_counts))  # positions for topic counts
width = 0.2  # width of each bar
colors = plt.get_cmap("Set2").colors  # consistent palette

# Plot
fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)  # one-column size

for i, collection in enumerate(collections):
    ax.bar(x + i*width, data[i], width, label=collection, color=colors[i])

# Formatting
ax.set_xlabel("Number of Topics per Document", fontsize=7)
ax.set_ylabel("Percentage of Documents (%)", fontsize=7)
ax.set_xticks(x + width * (len(collections)-1) / 2)
ax.set_xticklabels(topic_counts, fontsize=6)
ax.tick_params(axis="y", labelsize=6)
ax.legend(fontsize=6, title="Collection", title_fontsize=6, frameon=False, loc="upper right")
ax.set_title("Multi-Topic Assignment Distribution by Collection", fontsize=8)

plt.tight_layout()
plt.savefig("../../image/docs-topics.png", dpi=300, bbox_inches="tight")
plt.show()