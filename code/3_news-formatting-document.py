import json
from pathlib import Path

# Paths (update as needed)
dataset_name = "burundi"
SCHEMA_PATH = Path(f"../data/{dataset_name}/{dataset_name}_schema.json")
GNEWS_PATH = Path(f"../data/{dataset_name}/{dataset_name}_articles_gnews.json")
OUTPUT_PATH = Path(f"../data/{dataset_name}/{dataset_name}_articles_formatted.txt")
SEPARATOR = "\n" + "="*80 + "\n"

# Load the JSON files
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    mozambique_schema = json.load(f)

with open(GNEWS_PATH, "r", encoding="utf-8") as f:
    mozambique_gnews = json.load(f)

# Create a map of gnews articles by URL for quick lookup
gnews_map = {article.get("url"): article for article in mozambique_gnews}

# Build the text document
output_text = ""

for entry in mozambique_schema:
    schema_url = entry.get("url")
    gnews_article = gnews_map.get(schema_url)
    if not gnews_article or gnews_article.get("status_code") != 200:
        continue

    # Extract metadata from the schema field (list of dicts) if present
    schema_data = None
    if isinstance(entry.get("schema"), list) and entry["schema"]:
        schema_data = entry["schema"][0]
    elif isinstance(entry.get("schema"), dict):
        schema_data = entry["schema"]
    else:
        schema_data = {}

    # Use schema_data for metadata extraction, fallback to entry if missing
    title = schema_data.get("name") or schema_data.get("headline") or entry.get("name") or entry.get("headline") or "No Title"
    publication_date = schema_data.get("datePublished") or entry.get("datePublished") or "Unknown Date"
    section = schema_data.get("articleSection") or entry.get("articleSection") or "Unknown Section"
    image = ""
    # Image can be a string or a list
    img = schema_data.get("image") or entry.get("image")
    if isinstance(img, list) and img:
        image = img[0]
    elif isinstance(img, dict) and "url" in img:
        image = img["url"]
    elif isinstance(img, str):
        image = img

    # Publisher/source
    publisher = ""
    pub = schema_data.get("publisher") or entry.get("publisher")
    if isinstance(pub, dict):
        publisher = pub.get("name", "")
    elif isinstance(pub, str):
        publisher = pub

    # Extract content from gnews
    data = gnews_article.get("data", {})
    content = data.get("content", "").strip()
    if not content:
        content = schema_data.get("articleBody", "").strip() if schema_data else ""
    if not content:
        content = title

    # Format the article
    formatted_article = (
        f"Title: {title}\n"
        f"Source: {publisher}\n"
        f"Publication Date: {publication_date}\n"
        f"URL: {schema_url}\n"
        f"Section: {section}\n"
        f"Image: {image}\n"
        f"\n{content}"
    )

    output_text += formatted_article + SEPARATOR

# Save to file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Saved {OUTPUT_PATH} with all formatted articles.")
