import os
import dotenv
import json
from datetime import datetime
from utils.ragtec_topic_classification import run_ragtec_topic_classification

def generate_classification_summary(results, dataset_name, output_dir):
    """Generate a comprehensive text summary of the classification results."""
    
    if 'parsed_output' not in results or 'classification' not in results:
        print("Warning: No classification results available for summary generation")
        return None
    
    # Get classification data
    classification_data = results['parsed_output']
    classification_stats = results['classification']
    
    # Load topics data to get topic names
    topics_file = f"../output/{dataset_name}/ragtec/extraction/{dataset_name}_ragtec_topic_results.json"
    try:
        with open(topics_file, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        topic_names = topics_data.get('collection_topics', [])
    except:
        topic_names = []
    
    # Count topic assignments
    total_articles = len(classification_data)
    topic_counts = {}
    primary_topic_counts = {}
    
    # Count all topic assignments and primary topics
    for article in classification_data:
        if 'assigned_topics' in article and article['assigned_topics']:
            primary_topic = article['assigned_topics'][0]  # First topic is primary
            primary_topic_counts[primary_topic] = primary_topic_counts.get(primary_topic, 0) + 1
            
            # Count all assigned topics
            for topic in article['assigned_topics']:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Generate summary text
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_text = f"""{dataset_name.upper()} TOPIC ASSIGNMENT SUMMARY
Generated on: {timestamp}
===================================================

DATASET OVERVIEW:
- Total articles processed: {total_articles}
- Predefined topics available: {len(topic_names)}
- Topic names: {', '.join(topic_names) if topic_names else 'N/A'}

ALL TOPIC ASSIGNMENTS:
"""
    
    # Add topic assignment counts (sorted by count, descending)
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        summary_text += f"- {topic}: {count} articles\n"
    
    summary_text += f"\nPRIMARY TOPIC DISTRIBUTION:\n"
    
    # Add primary topic counts (sorted by count, descending)
    for topic, count in sorted(primary_topic_counts.items(), key=lambda x: x[1], reverse=True):
        summary_text += f"- {topic}: {count} articles\n"
    
    # Add token usage and cost information
    if 'token_usage' in classification_stats:
        tokens = classification_stats['token_usage']
        summary_text += f"""
TOKEN USAGE AND COST SUMMARY:
- Input tokens: {tokens.get('total_input_tokens', 0):,}
- Output tokens: {tokens.get('total_output_tokens', 0):,}
- Total tokens: {tokens.get('total_tokens', 0):,}
- Input cost: ${tokens.get('total_input_cost_usd', 0):.6f}
- Output cost: ${tokens.get('total_output_cost_usd', 0):.6f}
- Total cost: ${tokens.get('total_cost_usd', 0):.6f}
- Model: {tokens.get('model', 'N/A')}
"""
    
    # Add processing statistics
    summary_text += f"""
PROCESSING STATISTICS:
- Successful classifications: {classification_stats.get('successful_classifications', 0)}
- Failed classifications: {classification_stats.get('failed_classifications', 0)}
- Success rate: {(classification_stats.get('successful_classifications', 0) / max(total_articles, 1) * 100):.1f}%
"""
    
    # Save to file
    summary_filename = f"{output_dir}/{dataset_name}_classification_summary.txt"
    try:
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"Classification summary saved to: {summary_filename}")
        return summary_filename
    except Exception as e:
        print(f"Error saving summary: {e}")
        return None

# Load environment variables
dotenv.load_dotenv()

# Define file paths
dataset_name = "drc"
articles_file = f"../data/{dataset_name}/{dataset_name}_articles_gnews.json"
topics_file = f"../output/{dataset_name}/ragtec/extraction/{dataset_name}_ragtec_topic_results.json"
prompt_path = "../prompts/topic_classification_prompt.txt"
output_dir = f"../output/{dataset_name}/ragtec/classification"

# User decides whether to save results
SAVE_RESULTS = True  # Change to False to run without saving files
BATCH_SAVE_INTERVAL = 10  # Save progress every 10 articles

# Create output directory if it doesn't exist and we're saving results
if SAVE_RESULTS:
    os.makedirs(output_dir, exist_ok=True)

# Run the complete RAGTEC classification pipeline with user control
print("RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification")
print("=" * 70)
print(f"Batch saving enabled: Every {BATCH_SAVE_INTERVAL} articles")

results = run_ragtec_topic_classification(
    articles_file=articles_file,
    topics_file=topics_file,
    prompt_path=prompt_path,
    output_dir=output_dir if SAVE_RESULTS else None,
    model="gpt-4o-mini",
    save_results=SAVE_RESULTS,
    batch_save_interval=BATCH_SAVE_INTERVAL,
    dataset_name=dataset_name
)

print("\nRAGTEC Classification Pipeline completed successfully!")
if SAVE_RESULTS:
    print(f"Results saved to: {output_dir}")
    
    # Generate and save comprehensive text summary
    summary_file = generate_classification_summary(results, dataset_name, output_dir)
    if summary_file:
        print(f"Summary report saved to: {summary_file}")
else:
    print("Results available in memory only (not saved to files)")

# Print summary statistics (always available regardless of save setting)
if 'classification' in results:
    classification = results['classification']
    print(f"\nSummary:")
    print(f"  Total articles processed: {classification['total_articles']} articles")
    print(f"  Successful classifications: {classification['successful_classifications']}")
    print(f"  Failed classifications: {classification['failed_classifications']}")

if 'classification' in results and 'token_usage' in results['classification']:
    tokens = results['classification']['token_usage']
    print(f"  Total cost: ${tokens['total_cost_usd']:.6f}")
    print(f"  Total tokens: {tokens['total_tokens']:,}")

# Example: Access results programmatically
if 'parsed_output' in results:
    print(f"  Classification results: Available in results['parsed_output']")
