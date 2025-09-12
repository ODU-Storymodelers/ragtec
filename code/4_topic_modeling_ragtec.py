import os
import dotenv
from datetime import datetime
from utils.ragtec_topic_extraction import run_ragtec_topic_modeling

# Load environment variables
dotenv.load_dotenv()

# Define file paths dynamically based on dataset
dataset_name = "drc"
articles_path = f"../data/{dataset_name}/{dataset_name}_articles_formatted.txt"
prompt_path = "../prompts/topic_extraction_prompt.txt"
query_path = "../prompts/docs_retrieve_query.txt"
output_dir = f"../output/{dataset_name}/ragtec/extraction"

# User decides whether to save results
SAVE_RESULTS = True  # Change to False to run without saving files

# User decides whether to use custom k or optimal k formula
USE_CUSTOM_K = False  # Change to True to use custom k value
CUSTOM_K_VALUE = 120   # Set your desired k value here (only used if USE_CUSTOM_K=True)

# Create summary log list to capture all output
summary_log = []

def log_and_print(message):
    """Helper function to both print and log messages"""
    print(message)
    summary_log.append(message)

# Run the complete RAGTEC pipeline with user control
log_and_print("RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification")
log_and_print("=" * 70)

# Determine k value to use
custom_k = CUSTOM_K_VALUE if USE_CUSTOM_K else None
if USE_CUSTOM_K:
    log_and_print(f"Using custom k value: {CUSTOM_K_VALUE}")
else:
    log_and_print("Using optimal k formula")

results = run_ragtec_topic_modeling(
    articles_path=articles_path,
    prompt_path=prompt_path,
    query_path=query_path,
    output_dir=output_dir if SAVE_RESULTS else None,
    model="gpt-4o-mini",
    save_results=SAVE_RESULTS,
    dataset_name=dataset_name,  # Optional: Use "Sudan" instead of auto-detected "SUDAN"
    custom_k=custom_k  # Pass custom k value
)

log_and_print("\nRAGTEC Pipeline completed successfully!")
if SAVE_RESULTS:
    log_and_print(f"Results saved to: {output_dir}")
else:
    log_and_print("Results available in memory only (not saved to files)")

# Print summary statistics (always available regardless of save setting)
if 'analysis' in results:
    analysis = results['analysis']
    log_and_print(f"\nSummary:")
    log_and_print(f"  Collection size: {analysis['collection_size']} articles")
    log_and_print(f"  Optimal k used: {analysis['optimal_k']} articles")
    log_and_print(f"  Selection ratio: {analysis['selection_strategy']['percentage_selected']:.1f}%")

if 'modeling' in results:
    tokens = results['modeling']['token_usage']
    log_and_print(f"  Total cost: ${tokens['total_cost_usd']:.6f}")
    log_and_print(f"  Total tokens: {tokens['total_tokens']:,}")

# Example: Access results programmatically
if 'parsed_output' in results:
    log_and_print(f"  Topics extracted: Available in results['parsed_output']")

# Save the complete summary log to a text file
if SAVE_RESULTS and output_dir:
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for the summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"{dataset_name}_ragtec_execution_summary_{timestamp}.txt"
    summary_path = os.path.join(output_dir, summary_filename)
    
    # Write summary to file
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"RAGTEC EXECUTION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write all logged messages
        for message in summary_log:
            f.write(message + "\n")
    
    print(f"\nExecution summary saved to: {summary_path}")
else:
    print("\nNote: Execution summary not saved (SAVE_RESULTS=False or no output_dir)")

