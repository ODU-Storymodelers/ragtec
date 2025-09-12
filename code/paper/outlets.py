import json
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import re

# -----------------------
# CONFIG
# -----------------------
schema_files = {
    "Burundi": "../../data/burundi/burundi_schema.json",
    "DRC": "../../data/drc/drc_schema.json",
    "Mozambique": "../../data/mozambique/mozambique_schema.json",
    "Sudan": "../../data/sudan/sudan_schema.json"
}

# -----------------------
# FUNCTIONS
# -----------------------

def get_domain(url):
    """Extract domain from a URL, handling archive.org and similar prefixes.
    Returns a lowercase netloc like 'bbc.com' or 'bbc.co.uk'.
    """
    if not url:
        return None

    # Normalize whitespace
    url = str(url).strip()

    # Handle Wayback Machine formats like
    #   https://web.archive.org/web/<ts>/http://example.com/...
    #   https://web.archive.org/web/<ts>id_/https://example.com/...
    if "web.archive.org" in url:
        # Find the last occurrence of 'http://' or 'https://' in the string and take from there
        m = re.search(r"(https?://[^\s]+)", url.split("web.archive.org")[-1])
        if m:
            url = m.group(1)

    # Handle archive.today / archive.ph / archive.is formats where the original URL follows the last 'http'
    if any(host in url for host in ["archive.today/", "archive.ph/", "archive.is/"]):
        m = re.search(r"(https?://[^\s]+)", url[url.rfind("http"):])
        if m:
            url = m.group(1)

    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        # Remove common prefixes
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return None

def domain_to_name(domain):
    """Map a domain (e.g., 'bbc.com', 'bbc.co.uk') to a clean publisher label.
    Falls back to a sensible name using the second-level domain.
    """
    if not domain:
        return "Unknown"

    # Known mappings (both bare and with www. already stripped in get_domain)
    mapping = {
        "bbc.com": "BBC News",
        "bbc.co.uk": "BBC News",
        "africanews.com": "Africanews",
        "allafrica.com": "AllAfrica",
        "dw.com": "Deutsche Welle",
        "aljazeera.com": "Al Jazeera",
        "apnews.com": "AP News",
        "reuters.com": "Reuters",
        "cnn.com": "CNN",
        "reliefweb.int": "ReliefWeb",
        "sosmediasburundi.org": "SOS Médias Burundi",
        "theeastafrican.co.ke": "The East African",
    }

    if domain in mapping:
        return mapping[domain]

    # Fallbacks using the registrable (second-level) domain
    parts = domain.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "gov", "ac", "edu"}:
        base = parts[-3]  # e.g., 'bbc' from 'bbc.co.uk'
    elif len(parts) >= 2:
        base = parts[-2]
    else:
        base = parts[0]

    base = base.lower()
    fallback_map = {
        "dw": "Deutsche Welle",
        "apnews": "AP News",
        "aljazeera": "Al Jazeera",
        "bbc": "BBC News",
        "allafrica": "AllAfrica",
        "africanews": "Africanews",
        "reuters": "Reuters",
        "cnn": "CNN",
    }
    if base in fallback_map:
        return fallback_map[base]

    # Generic formatting
    return base.capitalize()

def extract_publishers_from_entry(entry):
    """Extract publisher name(s) from a schema entry.
    Supports top-level 'publisher' and nested items in entry['schema'].
    Returns a list of cleaned names or None if not available.
    """
    names = []

    def _extract_from_pub_obj(pub_obj):
        if isinstance(pub_obj, dict):
            n = pub_obj.get("name")
            if n:
                names.append(str(n).strip())
        elif isinstance(pub_obj, list):
            for p in pub_obj:
                if isinstance(p, dict) and p.get("name"):
                    names.append(str(p["name"]).strip())

    # Top-level publisher (if present in some datasets)
    if isinstance(entry, dict) and "publisher" in entry:
        _extract_from_pub_obj(entry.get("publisher"))

    # Nested schema objects (common in your JSON files)
    sch = entry.get("schema")
    if isinstance(sch, list):
        for obj in sch:
            if isinstance(obj, dict) and "publisher" in obj:
                _extract_from_pub_obj(obj.get("publisher"))

    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(n)

    return deduped or None

# -----------------------
# PROCESS DATA
# -----------------------

rows_all = []
for coll, path in schema_files.items():
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        pubs = extract_publishers_from_entry(entry)
        if pubs:
            for n in pubs:
                rows_all.append({
                    "Collection": coll,
                    "Publisher": n.strip()
                })
        else:
            domain = get_domain(entry.get("url", ""))
            rows_all.append({
                "Collection": coll,
                "Publisher": domain_to_name(domain)
            })

df_all = pd.DataFrame(rows_all)

# Normalize publisher labels (case/whitespace) for consistent grouping
if not df_all.empty:
    df_all["Publisher"] = df_all["Publisher"].astype(str).str.strip()

# Aggregate counts
counts_all = df_all.groupby(["Publisher", "Collection"]).size().reset_index(name="Articles")

# Top publishers by total articles (stable sorting with tie-breaker by name)
_totals = counts_all.groupby("Publisher", as_index=False)["Articles"].sum()
_totals = _totals.sort_values(["Articles", "Publisher"], ascending=[False, True])
top_publishers_all = _totals.head(7)["Publisher"].tolist()
counts_top_all = counts_all[counts_all["Publisher"].isin(top_publishers_all)]

# Pivot for plotting
pivot_all = counts_top_all.pivot(index="Publisher", columns="Collection", values="Articles").fillna(0).astype(int)

# Add total and sort by total descending
pivot_with_total_all = pivot_all.copy()
pivot_with_total_all["Total"] = pivot_with_total_all.sum(axis=1)
pivot_sorted_desc = pivot_with_total_all.sort_values("Total", ascending=False)
pivot_sorted_desc_no_total = pivot_sorted_desc.drop(columns=["Total"])

# -----------------------
# PLOT
# -----------------------

fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)  # one-column size
pivot_sorted_desc_no_total.plot(kind="barh", ax=ax, color=plt.get_cmap("Set2").colors)

# Invert y-axis so highest total is on top
ax.invert_yaxis()

ax.set_xlabel("Articles", fontsize=7)
ax.set_ylabel("Publisher", fontsize=7)
ax.tick_params(axis="both", labelsize=6)
ax.legend(title="Collection", fontsize=6, title_fontsize=6, loc="best", frameon=False)
ax.set_title("Top Publishers Across Four Collections", fontsize=8)
plt.tight_layout()
plt.savefig("../../image/publisher.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------
# FINAL TABLE
# -----------------------

latex_df_sorted_desc = pivot_sorted_desc.astype(int).reset_index()

# Export to LaTeX
latex_code = latex_df_sorted_desc.to_latex(index=False, header=True, caption="Top 7 publishers by number of articles in each collection, sorted by total coverage.", label="tab:top_publishers", bold_rows=True)
print(latex_code)