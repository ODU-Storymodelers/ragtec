import os
import dotenv
from utils.ragtec_topic_storytelling import run_ragtec_storytelling_analysis

# Load environment variables
dotenv.load_dotenv()

# Define file paths dynamically based on dataset
dataset_name = "sudan"  # Change to your dataset name
articles_file = f"../output/{dataset_name}/ragtec/classification/{dataset_name}_ragtec_classification_results.json"
output_dir = f"../output/{dataset_name}/ragtec/storytelling"

# Storytelling visualization parameters
SIZE_MULTIPLIER = 50.0  # Node size = mentions × multiplier
EDGE_MULTIPLIER = 20.0   # Edge width = co-occurrence × multiplier
COLLECTION_NAME = f"{dataset_name.capitalize()} Crisis News"  # Collection name for visualization titles

# Color scheme based on primary/secondary ratio gradient:
# - Darker blue: Topics that are primarily main topics (higher importance)
# - Lighter blue: Topics that are predominantly secondary (lower importance)
# Uses a smooth gradient from light to dark based on exact ratio values

# User decides whether to save results
SAVE_RESULTS = True  # Change to False to run without saving files

# Run the complete RAGTEC storytelling analysis pipeline with user control
print("RAGTEC: Topic Co-occurrence Storytelling Analysis")
print("=" * 70)

results = run_ragtec_storytelling_analysis(
    articles_file=articles_file,
    output_dir=output_dir if SAVE_RESULTS else None,
    size_multiplier=SIZE_MULTIPLIER,
    edge_multiplier=EDGE_MULTIPLIER,
    collection_name=COLLECTION_NAME,
    save_results=SAVE_RESULTS
)

print("\nRAGTEC Storytelling Analysis Pipeline completed successfully!")
if SAVE_RESULTS:
    print(f"Results saved to: {output_dir}")
else:
    print("Results available in memory only (not saved to files)")

# Print summary statistics (always available regardless of save setting)
if 'loading' in results:
    loading = results['loading']
    print(f"\nSummary:")
    print(f"  Total articles analyzed: {loading['total_articles']} articles")
    print(f"  Unique topics found: {loading['unique_topics']}")
    print(f"  Articles with multiple topics: {loading['articles_with_multiple_topics']}")

if 'storytelling_analysis' in results:
    storytelling = results['storytelling_analysis']
    print(f"  Topic connections: {len(storytelling.get('primary_secondary_pairs', []))} primary-secondary pairs")
    print(f"  Co-occurrence stories: {len(storytelling['networks'])} threshold levels")

# Save summary statistics to a txt file
summary_lines = []
total_articles = None

if 'loading' in results:
    loading = results['loading']
    total_articles = loading.get('total_articles')

    summary_lines.append("Summary:")
    summary_lines.append(f"  Total articles analyzed: {total_articles} articles")
    summary_lines.append(f"  Unique topics found: {loading['unique_topics']}")
    summary_lines.append(f"  Articles with multiple topics: {loading['articles_with_multiple_topics']}")

if 'storytelling_analysis' in results:
    storytelling = results['storytelling_analysis']
    summary_lines.append(f"  Topic connections: {len(storytelling.get('primary_secondary_pairs', []))} primary-secondary pairs")
    summary_lines.append(f"  Co-occurrence stories: {len(storytelling['networks'])} threshold levels")

if summary_lines:
    summary_txt_path = os.path.join(output_dir, f"{dataset_name}_storytelling_summary.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_txt_path, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print(f"\nSummary statistics saved to: {summary_txt_path}")

if 'visualizations' in results:
    visualizations = results['visualizations']
    print(f"  Visualizations created: {visualizations.get('visualizations_created', 0)}")
    print(f"  Node size multiplier: {SIZE_MULTIPLIER}")
    print(f"  Edge width multiplier: {EDGE_MULTIPLIER}")
    print(f"  Collection name: '{COLLECTION_NAME}'")
    print(f"  Node colors: Gradient-based (Dark blue=Primary/Important, Light blue=Secondary/Less important)")

# Example: Access results programmatically
if 'analysis_object' in results:
    print(f"  Network analysis object: Available in results['analysis_object']")
