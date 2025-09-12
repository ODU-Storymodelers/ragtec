import json
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

# Download NLTK resources if not already downloaded
nltk_resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
for resource in nltk_resources:
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif resource == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        elif resource == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif resource == 'wordnet':
            nltk.data.find('corpora/wordnet')
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource)

def clean_text(text):
    """Clean the text content for preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_for_lda(text):
    """Additional preprocessing specific to LDA"""
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

# Add the new Unique Word Proportion diversity function
def calculate_unique_word_proportion(topic_keywords, top_n=10):
    """Compute topic diversity as the proportion of unique words across topics."""
    unique_words = set()
    total_words = 0
    
    for keywords in topic_keywords.values():
        words = keywords[:top_n]  # Take only top_n keywords per topic
        unique_words.update(words)
        total_words += len(words)
    
    diversity_score = len(unique_words) / total_words if total_words > 0 else 0
    return diversity_score

# Modify the calculate_coherence_values function to include likelihood
def calculate_coherence_values(dictionary, corpus, texts, start=2, stop=40, step=2):
    coherence_values = []
    cv_coherence_values = []
    likelihood_values = []  # Add likelihood values
    perplexity_values = []  # Add perplexity values
    jaccard_diversity_values = []
    uwp_diversity_values = []  # New Unique Word Proportion diversity values
    model_list = []
    
    # Create CSV file to save metrics during optimization
    optimization_csv_path = f"{output_dir}/{dataset_name}_document_topic_optimization_lda.csv"
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(optimization_csv_path), exist_ok=True)
    
    with open(optimization_csv_path, "w", encoding="utf-8") as f:
        f.write("Number of Topics,Log-Likelihood,Perplexity,UMass Coherence,CV Coherence,Jaccard Diversity,UWP Diversity\n")
    
    print("Evaluating topic models from", start, "to", stop, "topics...")
    
    for num_topics in range(start, stop, step):
        # Train the model
        print(f"\nTraining model with {num_topics} topics...")
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=20,
            alpha='auto',
            eta='auto'
        )
        
        # Save the model
        model_list.append(model)
        
        # Calculate log-likelihood and perplexity
        log_likelihood = None
        perplexity = None
        try:
            log_likelihood = model.log_perplexity(corpus)
            likelihood_values.append(round(log_likelihood, 3))
            
            # Calculate perplexity (exp(-log_likelihood))
            perplexity = np.exp(-log_likelihood)
            perplexity_values.append(round(perplexity, 3))
            
            print(f"Log-Likelihood: {log_likelihood:.3f} (higher is better)")
            print(f"Perplexity: {perplexity:.3f} (lower is better)")
        except Exception as e:
            print(f"Error calculating likelihood/perplexity: {e}")
            likelihood_values.append(None)
            perplexity_values.append(None)
        
        # Extract top 10 keywords for consistency with final model
        topic_word_lists = []
        topic_keywords = {}  # Dictionary for UWP calculation
        for topic_idx in range(num_topics):
            top_words = [term for term, _ in model.show_topic(topic_idx, topn=10)]
            topic_word_lists.append(top_words)
            topic_keywords[topic_idx] = top_words
        
        # Calculate UMass coherence
        umass_coherence = None
        try:
            coherence_umass = CoherenceModel(
                topics=topic_word_lists,
                texts=texts,
                dictionary=dictionary,
                coherence='u_mass',
                processes=1
            )
            umass_coherence = coherence_umass.get_coherence()
            coherence_values.append(round(umass_coherence, 3))
            print(f"UMass Coherence: {umass_coherence:.3f} (closer to 0 is better)")
        except Exception as e:
            print(f"Error calculating UMass coherence: {e}")
            coherence_values.append(None)
            
        # Calculate CV coherence
        cv_coherence = None
        try:
            coherence_cv = CoherenceModel(
                topics=topic_word_lists,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v',
                processes=1
            )
            cv_coherence = coherence_cv.get_coherence()
            cv_coherence_values.append(round(cv_coherence, 3))
            print(f"CV Coherence: {cv_coherence:.3f} (higher is better)")
        except Exception as e:
            print(f"Error calculating CV coherence: {e}")
            cv_coherence_values.append(None)
        
        # Calculate Jaccard diversity
        jaccard_diversity = None
        try:
            topic_terms = [set(words) for words in topic_word_lists]
            
            jaccard_distances = []
            for i in range(len(topic_terms)):
                for j in range(i+1, len(topic_terms)):
                    intersection = len(topic_terms[i].intersection(topic_terms[j]))
                    union = len(topic_terms[i].union(topic_terms[j]))
                    distance = 1 - (intersection / union if union > 0 else 0)
                    jaccard_distances.append(distance)
                    
            jaccard_diversity = sum(jaccard_distances) / len(jaccard_distances) if jaccard_distances else 0
            jaccard_diversity = round(jaccard_diversity, 3)
            jaccard_diversity_values.append(jaccard_diversity)
            print(f"Jaccard Diversity: {jaccard_diversity:.3f} (higher is better)")
        except Exception as e:
            print(f"Error calculating Jaccard diversity: {e}")
            jaccard_diversity_values.append(None)
            
        # Calculate Unique Word Proportion diversity
        uwp_diversity = None
        try:
            uwp_diversity = calculate_unique_word_proportion(topic_keywords, top_n=10)
            uwp_diversity = round(uwp_diversity, 3)
            uwp_diversity_values.append(uwp_diversity)
            print(f"Unique Word Proportion: {uwp_diversity:.3f} (higher is better)")
        except Exception as e:
            print(f"Error calculating Unique Word Proportion: {e}")
            uwp_diversity_values.append(None)
        
        # Save this iteration's metrics to CSV
        with open(optimization_csv_path, "a", encoding="utf-8") as f:
            likelihood_val = f"{log_likelihood:.3f}" if log_likelihood is not None else "N/A"
            perplexity_val = f"{perplexity:.3f}" if perplexity is not None else "N/A"
            umass_val = f"{umass_coherence:.3f}" if umass_coherence is not None else "N/A"
            cv_val = f"{cv_coherence:.3f}" if cv_coherence is not None else "N/A"
            jaccard_val = f"{jaccard_diversity:.3f}" if jaccard_diversity is not None else "N/A"
            uwp_val = f"{uwp_diversity:.3f}" if uwp_diversity is not None else "N/A"
            f.write(f"{num_topics},{likelihood_val},{perplexity_val},{umass_val},{cv_val},{jaccard_val},{uwp_val}\n")
    
    print(f"Saved detailed optimization process to {optimization_csv_path}")
    return model_list, likelihood_values, perplexity_values, coherence_values, cv_coherence_values, jaccard_diversity_values, uwp_diversity_values

# -------------------- MAIN SCRIPT --------------------

# Define file paths
dataset_name = "burundi"
input_file = f"../data/{dataset_name}/{dataset_name}_articles_gnews.json"
output_dir = f"../output/{dataset_name}/lda"

output_file = f"{output_dir}/{dataset_name}_articles_topic_document_lda.json"
topics_file = f"{output_dir}/{dataset_name}_topics_document_lda.json"
metrics_csv_path = f"{output_dir}/{dataset_name}_document_topic_lda_metrics.csv"

# Load articles from the input file
articles = []
if os.path.exists(input_file):
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            articles = json.load(file)
    except json.JSONDecodeError:
        print("Error reading input articles file, exiting.")
        exit(1)
else:
    print("Input articles file not found, exiting.")
    exit(1)

# Filter articles by status code and minimum words
filtered_articles = [
    entry for entry in articles
    if entry["status_code"] == 200 and len(entry["data"]["content"].split()) > 30
]

# Process articles as whole documents
documents = []
processed_documents = []
document_ids = []  # To track original article indices

print("Processing articles...")
for article_idx, entry in enumerate(filtered_articles):
    # Clean the text first
    cleaned_content = clean_text(entry["data"]["content"])
    
    # Store the full document
    documents.append(cleaned_content)
    document_ids.append(article_idx)
    
    # Process document for LDA
    processed = preprocess_for_lda(cleaned_content)
    processed_documents.append(processed)

# Print summary statistics
print(f"Total articles/documents: {len(filtered_articles)}")
if documents:
    print("Sample document (first 100 chars):", documents[0][:100])
    print("Processed tokens (first 10):", processed_documents[0][:10])

# Create a dictionary and corpus for LDA
print("Creating dictionary and corpus...")
dictionary = Dictionary(processed_documents)
# Filter out extremely rare and extremely common terms
dictionary.filter_extremes(no_below=2, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in processed_documents]

# Find optimal number of topics
print("Finding optimal number of topics...")
try:
    # Determine range of topics to test
    start_topics, stop_topics, step_topics = 5, 41, 1
    
    # Calculate coherence scores with both diversity metrics and likelihood
    model_list, likelihood_values, perplexity_values, coherence_values, cv_coherence_values, jaccard_diversity_values, uwp_diversity_values = calculate_coherence_values(
        dictionary=dictionary, 
        corpus=corpus, 
        texts=processed_documents,
        start=start_topics, 
        stop=stop_topics, 
        step=step_topics
    )
    
    # Find optimal number of topics based on highest log-likelihood (primary metric)
    optimal_idx = None
    if likelihood_values and not all(v is None for v in likelihood_values):
        # Find the model with the highest log-likelihood
        valid_likelihood = [(i, v) for i, v in enumerate(likelihood_values) if v is not None]
        if valid_likelihood:
            # Select model with highest log-likelihood
            optimal_idx = max(valid_likelihood, key=lambda x: x[1])[0]
            best_likelihood = likelihood_values[optimal_idx]
            print(f"Best log-likelihood: {best_likelihood:.3f} at {start_topics + optimal_idx*step_topics} topics")

    # Find optimal number of topics based on closest to zero UMass coherence (secondary metric)
    coherence_optimal_idx = None
    if coherence_values and not all(v is None for v in coherence_values):
        # Find the model with the coherence score closest to zero (best UMass coherence)
        valid_coherence = [(i, v) for i, v in enumerate(coherence_values) if v is not None]
        if valid_coherence:
            # Select model with highest coherence (closest to zero)
            coherence_optimal_idx = min(valid_coherence, key=lambda x: abs(x[1]))[0]
            best_coherence = coherence_values[coherence_optimal_idx]
            print(f"Best UMass coherence: {best_coherence:.3f} at {start_topics + coherence_optimal_idx*step_topics} topics")

    # Find optimal number of topics based on highest CV coherence
    cv_optimal_idx = None
    if cv_coherence_values and not all(v is None for v in cv_coherence_values):
        valid_cv = [(i, v) for i, v in enumerate(cv_coherence_values) if v is not None]
        if valid_cv:
            cv_optimal_idx = max(valid_cv, key=lambda x: x[1])[0]
            best_cv = cv_coherence_values[cv_optimal_idx]
            print(f"Best CV coherence: {best_cv:.3f} at {start_topics + cv_optimal_idx*step_topics} topics")

    # Calculate Jaccard diversity-optimal number of topics (for visualization only)
    jaccard_optimal_idx = None
    if jaccard_diversity_values and not all(v is None for v in jaccard_diversity_values):
        valid_diversity = [(i, v) for i, v in enumerate(jaccard_diversity_values) if v is not None]
        if valid_diversity:
            jaccard_optimal_idx = max(valid_diversity, key=lambda x: x[1])[0]
            best_jaccard = jaccard_diversity_values[jaccard_optimal_idx]
            print(f"Best Jaccard diversity: {best_jaccard:.3f} at {start_topics + jaccard_optimal_idx*step_topics} topics")

    # Calculate UWP diversity-optimal number of topics (for visualization only)
    uwp_diversity_optimal_idx = None
    if uwp_diversity_values and not all(v is None for v in uwp_diversity_values):
        valid_uwp_diversity = [(i, v) for i, v in enumerate(uwp_diversity_values) if v is not None]
        if valid_uwp_diversity:
            uwp_diversity_optimal_idx = max(valid_uwp_diversity, key=lambda x: x[1])[0]
            best_uwp_diversity = uwp_diversity_values[uwp_diversity_optimal_idx]
            print(f"Best UWP diversity: {best_uwp_diversity:.3f} at {start_topics + uwp_diversity_optimal_idx*step_topics} topics")

    # Visualize the results
    x = list(range(start_topics, stop_topics, step_topics))
    
    # Create CSV with results summary
    results_csv_path = f"{output_dir}/{dataset_name}_document_topic_optimization_summary_lda.csv"
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    
    with open(results_csv_path, "w", encoding="utf-8") as f:
        f.write("Number of Topics,Log-Likelihood,Perplexity,Coherence (UMass),Diversity\n")
        for i, (topics, likelihood, perplexity, coh, div) in enumerate(zip(x, likelihood_values, perplexity_values, coherence_values, jaccard_diversity_values)):
            likelihood_val = f"{likelihood:.3f}" if likelihood is not None else "N/A"
            perplexity_val = f"{perplexity:.3f}" if perplexity is not None else "N/A"
            coherence_val = f"{coh:.3f}" if coh is not None else "N/A"
            diversity_val = f"{div:.3f}" if div is not None else "N/A"
            f.write(f"{topics},{likelihood_val},{perplexity_val},{coherence_val},{diversity_val}\n")
    
    # Update visualization to include likelihood and perplexity (6 subplots total)
    try:
        plt.figure(figsize=(12, 18))  # Make figure even taller for 6 subplots
        
        # Log-likelihood plot (primary optimization metric)
        plt.subplot(6, 1, 1)
        plt.plot(x, likelihood_values, 'purple', marker='o', linewidth=2)
        plt.title('Log-Likelihood by Number of Topics (Primary Optimization Metric)')
        plt.xlabel('Number of Topics')
        plt.ylabel('Log-Likelihood (higher is better)')
        if optimal_idx is not None:
            plt.axvline(x=x[optimal_idx], color='red', linestyle='--', linewidth=2,
                      label=f'Optimal: {x[optimal_idx]} topics')
            plt.scatter(x[optimal_idx], likelihood_values[optimal_idx], color='red', s=150, zorder=5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Perplexity plot 
        plt.subplot(6, 1, 2)
        plt.plot(x, perplexity_values, 'brown', marker='o')
        plt.title('Perplexity by Number of Topics')
        plt.xlabel('Number of Topics')
        plt.ylabel('Perplexity (lower is better)')
        if optimal_idx is not None:
            plt.axvline(x=x[optimal_idx], color='red', linestyle='--', 
                      label=f'Optimal: {x[optimal_idx]} topics')
            plt.scatter(x[optimal_idx], perplexity_values[optimal_idx], color='red', s=100, zorder=5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # UMass coherence plot
        plt.subplot(6, 1, 3)
        plt.plot(x, coherence_values, 'b-o')
        plt.title('UMass Coherence by Number of Topics')
        plt.xlabel('Number of Topics')
        plt.ylabel('UMass Coherence (closer to 0 is better)')
        if coherence_optimal_idx is not None:
            plt.axvline(x=x[coherence_optimal_idx], color='blue', linestyle='--', 
                      label=f'Coherence Optimal: {x[coherence_optimal_idx]} topics')
            plt.scatter(x[coherence_optimal_idx], coherence_values[coherence_optimal_idx], color='blue', s=100, zorder=5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # CV coherence plot
        plt.subplot(6, 1, 4)
        plt.plot(x, cv_coherence_values, 'r-o')
        plt.title('CV Coherence by Number of Topics')
        plt.xlabel('Number of Topics')
        plt.ylabel('CV Coherence (higher is better)')
        if cv_optimal_idx is not None:
            plt.axvline(x=x[cv_optimal_idx], color='purple', linestyle='--', 
                      label=f'Optimal: {x[cv_optimal_idx]} topics')
            plt.scatter(x[cv_optimal_idx], cv_coherence_values[cv_optimal_idx], color='purple', s=100, zorder=5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Jaccard diversity plot
        plt.subplot(6, 1, 5)
        plt.plot(x, jaccard_diversity_values, 'g-o')
        plt.title('Jaccard Diversity by Number of Topics')
        plt.xlabel('Number of Topics')
        plt.ylabel('Jaccard Diversity (higher is better)')
        if jaccard_optimal_idx is not None:
            plt.axvline(x=x[jaccard_optimal_idx], color='purple', linestyle='--', 
                      label=f'Optimal: {x[jaccard_optimal_idx]} topics')
            plt.scatter(x[jaccard_optimal_idx], jaccard_diversity_values[jaccard_optimal_idx], color='purple', s=100, zorder=5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # UWP diversity plot
        plt.subplot(6, 1, 6)
        plt.plot(x, uwp_diversity_values, 'orange', marker='o')
        plt.title('Unique Word Proportion by Number of Topics')
        plt.xlabel('Number of Topics')
        plt.ylabel('UWP (higher is better)')
        if uwp_diversity_optimal_idx is not None:
            plt.axvline(x=x[uwp_diversity_optimal_idx], color='purple', linestyle='--', 
                      label=f'Optimal: {x[uwp_diversity_optimal_idx]} topics')
            plt.scatter(x[uwp_diversity_optimal_idx], uwp_diversity_values[uwp_diversity_optimal_idx], 
                       color='purple', s=100, zorder=5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create LDA output directory if it doesn't exist and save plot there
        image_path = f"{output_dir}/{dataset_name}_document_topic_optimization_lda.png"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {image_path}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Determine optimal number of topics (use likelihood as primary metric)
    if optimal_idx is not None:
        optimal_num_topics = x[optimal_idx]
        print(f"Optimal number of topics (based on log-likelihood): {optimal_num_topics}")
        num_topics = optimal_num_topics
    elif coherence_optimal_idx is not None:
        # Fallback to coherence if likelihood failed
        optimal_num_topics = x[coherence_optimal_idx]
        print(f"Optimal number of topics (fallback to coherence): {optimal_num_topics}")
        num_topics = optimal_num_topics
    else:
        print("Could not determine optimal number of topics, using default value of 10")
        num_topics = 10
        
except Exception as e:
    print(f"Error in topic optimization: {e}")
    print("Using default number of topics of 10")
    num_topics = 10

# Train final LDA model
print(f"Training final LDA model with {num_topics} topics...")
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=20,
    alpha='auto',
    eta='auto'
)

# Generate topic names
print("Generating topic names...")
topic_names = {}
for topic_id in range(num_topics):
    top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=5)]
    topic_names[topic_id] = f"Topic {topic_id}: {', '.join(top_words)}"

# Get topic distributions for all documents
print("Assigning topics to documents...")
document_topics = []
for bow in corpus:
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    # Convert to a full distribution over all topics
    topic_dist_full = np.zeros(num_topics)
    for topic_id, prob in topic_dist:
        topic_dist_full[topic_id] = prob
    document_topics.append(topic_dist_full)

# Calculate coherence and diversity for final model
print("Calculating final model metrics...")

# Calculate likelihood and perplexity for final model
final_log_likelihood = None
final_perplexity = None

try:
    final_log_likelihood = lda_model.log_perplexity(corpus)
    final_perplexity = np.exp(-final_log_likelihood)
    final_log_likelihood = round(final_log_likelihood, 3)
    final_perplexity = round(final_perplexity, 3)
    print(f"Final model log-likelihood: {final_log_likelihood:.3f}")
    print(f"Final model perplexity: {final_perplexity:.3f}")
except Exception as e:
    print(f"Error calculating final model likelihood/perplexity: {e}")

# Extract top 10 keywords per topic for consistent metrics
topic_word_lists = []
for topic_id in range(num_topics):
    # Get exactly 10 top words for each topic
    top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=10)]
    topic_word_lists.append(top_words)
    print(f"Topic {topic_id} keywords: {', '.join(top_words)}")

# Calculate coherence using both UMass and CV with consistent keywords
coherence_umass = None
coherence_cv = None

try:
    # UMass coherence using topic_word_lists
    coherence_umass_model = CoherenceModel(
        topics=topic_word_lists,
        texts=processed_documents,
        dictionary=dictionary,
        coherence='u_mass',
        processes=1
    )
    coherence_umass = coherence_umass_model.get_coherence()
    coherence_umass = round(coherence_umass, 3)  # Round to 3 decimal places
    print(f"Final model UMass coherence: {coherence_umass:.3f}")
    
    # CV coherence using same topic_word_lists
    coherence_cv_model = CoherenceModel(
        topics=topic_word_lists,
        texts=processed_documents,
        dictionary=dictionary,
        coherence='c_v',
        processes=1
    )
    coherence_cv = coherence_cv_model.get_coherence()
    coherence_cv = round(coherence_cv, 3)  # Round to 3 decimal places
    print(f"Final model CV coherence: {coherence_cv:.3f}")
    
except Exception as e:
    print(f"Error calculating coherence: {e}")

# Calculate both diversity metrics using the same top 10 keywords
jaccard_diversity = None
uwp_diversity = None

try:
    # Use the topic_word_lists for both diversity calculations
    topic_terms = [set(words) for words in topic_word_lists]
    topic_keywords = {i: words for i, words in enumerate(topic_word_lists)}
    
    # Jaccard diversity
    jaccard_distances = []
    for i in range(len(topic_terms)):
        for j in range(i+1, len(topic_terms)):
            intersection = len(topic_terms[i].intersection(topic_terms[j]))
            union = len(topic_terms[i].union(topic_terms[j]))
            distance = 1 - (intersection / union if union > 0 else 0)
            jaccard_distances.append(distance)
    
    jaccard_diversity = sum(jaccard_distances) / len(jaccard_distances) if jaccard_distances else 0
    jaccard_diversity = round(jaccard_diversity, 3)
    print(f"Final model Jaccard diversity: {jaccard_diversity:.3f}")
    
    # Unique Word Proportion diversity
    uwp_diversity = calculate_unique_word_proportion(topic_keywords, top_n=10)
    uwp_diversity = round(uwp_diversity, 3)
    print(f"Final model Unique Word Proportion: {uwp_diversity:.3f}")
    
except Exception as e:
    print(f"Error calculating diversity: {e}")

# Create topic information dictionary
topics_info = []
for topic_id in range(num_topics):
    # Count how many documents have this as their dominant topic
    count = sum(1 for dist in document_topics if np.argmax(dist) == topic_id)
    
    # Get top words and their weights
    top_words = lda_model.show_topic(topic_id, topn=10)
    
    # Get representative documents (those most aligned with this topic)
    topic_doc_scores = [(idx, dist[topic_id]) 
                         for idx, dist in enumerate(document_topics)]
    top_docs = sorted(topic_doc_scores, key=lambda x: x[1], reverse=True)[:3]
    representative_docs = [{"doc_idx": int(idx), 
                            "text": documents[idx][:200] + "...", 
                            "score": round(float(score), 3)}  # Round representative doc scores
                           for idx, score in top_docs if score > 0.2]
    
    topics_info.append({
        "Topic": int(topic_id),
        "Name": topic_names[topic_id],
        "Count": int(count),
        "TopWords": [{"word": word, "weight": round(float(weight), 3)} for word, weight in top_words],  # Round weights
        "Representative_Docs": representative_docs
    })

# Prepare output data
print("Preparing output data...")
output_data = []
for doc_idx, (article_idx, topic_dist) in enumerate(zip(document_ids, document_topics)):
    # Get the original article
    entry = filtered_articles[article_idx]
    
    # Get dominant topic with rounded probability
    dominant_topic_id = np.argmax(topic_dist)
    dominant_topic_prob = round(float(topic_dist[dominant_topic_id]), 3)
    
    # Get topic distribution formatted for output with rounded probabilities
    topic_dist_formatted = [{"topic": int(i), 
                             "name": topic_names[i], 
                             "probability": round(float(prob), 3)}  # Round probabilities 
                           for i, prob in enumerate(topic_dist) 
                           if prob > 0.05]  # Filter very low probabilities
    
    # Sort by probability
    topic_dist_formatted.sort(key=lambda x: x["probability"], reverse=True)
    
    # Create output entry
    output_data.append({
        "url": entry["url"],
        "content": entry["data"]["content"],
        "dominant_topic": {
            "topic": int(dominant_topic_id),
            "topic_name": topic_names[dominant_topic_id],
            "probability": dominant_topic_prob
        },
        "topic_distribution": topic_dist_formatted
    })

# Save output
print("Saving results...")
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)
print(f"Saved {len(output_data)} articles to {output_file}")

# Save topic information with metrics
print("Saving topic information...")
topics_meta = {
    "model_info": {
        "num_topics": num_topics,
        "log_likelihood": float(final_log_likelihood) if final_log_likelihood is not None else None,
        "perplexity": float(final_perplexity) if final_perplexity is not None else None,
        "umass_coherence": float(coherence_umass) if coherence_umass is not None else None,
        "cv_coherence": float(coherence_cv) if coherence_cv is not None else None,
        "jaccard_diversity": float(jaccard_diversity) if jaccard_diversity is not None else None,
        "uwp_diversity": float(uwp_diversity) if uwp_diversity is not None else None
    },
    "topics": topics_info
}

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(topics_file), exist_ok=True)

with open(topics_file, "w", encoding="utf-8") as f:
    json.dump(topics_meta, f, indent=2, ensure_ascii=False)
print(f"Saved topic information to {topics_file}")

# Save metrics to CSV
metrics_csv_path = f"{output_dir}/{dataset_name}_document_topic_lda_metrics.csv"
# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)

with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
    f.write("OVERALL METRICS\n")
    f.write(f"Number of Topics,{num_topics}\n")
    
    # Likelihood metrics (primary)
    likelihood_str = f"{final_log_likelihood:.3f}" if final_log_likelihood is not None else "N/A"
    f.write(f"Log-Likelihood,{likelihood_str},Higher is better (primary optimization metric)\n")
    
    perplexity_str = f"{final_perplexity:.3f}" if final_perplexity is not None else "N/A"
    f.write(f"Perplexity,{perplexity_str},Lower is better\n")
    
    # Coherence metrics
    umass_str = f"{coherence_umass:.3f}" if coherence_umass is not None else "N/A"
    f.write(f"UMass Coherence,{umass_str},Closer to 0 is better\n")
    
    cv_str = f"{coherence_cv:.3f}" if coherence_cv is not None else "N/A"
    f.write(f"CV Coherence,{cv_str},Higher is better\n")
    
    # Diversity metrics
    jaccard_str = f"{jaccard_diversity:.3f}" if jaccard_diversity is not None else "N/A"
    f.write(f"Jaccard Diversity,{jaccard_str},Higher is better\n")
    
    uwp_str = f"{uwp_diversity:.3f}" if uwp_diversity is not None else "N/A"
    f.write(f"Unique Word Proportion,{uwp_str},Higher is better\n\n")
    
    f.write("INDIVIDUAL TOPIC METRICS\n")
    f.write("Topic ID,Topic Name,Document Count,Top Words\n")
    for info in topics_info:
        top_words = ", ".join(item["word"] for item in info["TopWords"])
        # Safely format the topic name for CSV
        safe_name = info['Name'].replace(",", ";")
        f.write(f"{info['Topic']},{safe_name},{info['Count']},{top_words}\n")

print(f"Saved detailed metrics to {metrics_csv_path}")
print(f"Number of topics: {num_topics}")
print(f"Analysis complete!")