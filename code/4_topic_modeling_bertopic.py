import json
import os
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from bs4 import BeautifulSoup
import re
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

# Disable parallel tokenization to avoid multiprocessing errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_text(text):
    """Clean the text content for preprocessing"""
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

if __name__ == "__main__":
    # Define file paths
    dataset_name = "sudan"
    input_file = f"../data/{dataset_name}/{dataset_name}_articles_gnews.json"

    output_dir = f"../output/{dataset_name}/bertopic"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/{dataset_name}_articles_topic_document_bertopic.json"
    topics_file = f"{output_dir}/{dataset_name}_topics_document_bertopic.json"
    scores_file = f"{output_dir}/{dataset_name}_document_topics_bertopic.json"
    metrics_csv_path = f"{output_dir}/{dataset_name}_document_bertopic_metrics.csv"

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

    # Ensure clean content, filtering by status code and minimum words
    content = [
        clean_text(entry["data"]["content"]) for entry in articles
        if entry["status_code"] == 200 and len(entry["data"]["content"].split()) > 30
    ]

    print(f"Total articles: {len(articles)} | Articles with unique content: {len(content)}")
    if content:
        print("Sample content:", content[0])

    # Train BERTopic model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.1, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer()
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model
    )

    topics, probs = topic_model.fit_transform(content)

    # Save url, content, and topic data
    output_data = []
    document_info = topic_model.get_document_info(content)
    filtered_index = 0  # Counter for filtered articles that match document_info
    
    for i, entry in enumerate(articles):
        if entry["status_code"] == 200 and len(entry["data"]["content"].split()) > 30:
            output_data.append({
                "url": entry["url"],
                "content": entry["data"]["content"],
                "topic_info": document_info.iloc[filtered_index][["Topic", "Name", "Top_n_words", "Probability", "Representative_document"]].to_dict()
            })
            filtered_index += 1  # Increment only for valid articles
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(output_data)} articles to {output_file}")

    # Save topic information
    topics_info = topic_model.get_topic_info()
    topics_info_json = topics_info.to_dict(orient='records')
    with open(topics_file, "w", encoding="utf-8") as f:
        json.dump(topics_info_json, f, indent=2, ensure_ascii=False)
    print(f"Saved topic information to {topics_file}")

    # Compute Coherence Scores - both UMass and CV
    print("Computing coherence scores...")
    dictionary = Dictionary([doc.split() for doc in content])
    topic_words = [topic["Representation"] for topic in topics_info_json if "Representation" in topic]

    # Calculate UMass coherence
    coherence_umass = CoherenceModel(
        topics=topic_words, 
        texts=[doc.split() for doc in content], 
        dictionary=dictionary, 
        coherence='u_mass', 
        processes=1
    )
    umass_score = coherence_umass.get_coherence()

    # Calculate CV coherence
    coherence_cv = CoherenceModel(
        topics=topic_words, 
        texts=[doc.split() for doc in content], 
        dictionary=dictionary, 
        coherence='c_v',  # CV coherence
        processes=1
    )
    cv_score = coherence_cv.get_coherence()

    print(f"UMass Coherence Score: {umass_score}")
    print(f"CV Coherence Score: {cv_score}")

    # Compute Topic Diversity
    def topic_diversity(topics, top_n=10):
        unique_words = set()
        total_words = 0
        for topic in topics:
            words = topic[:top_n]
            unique_words.update(words)
            total_words += len(words)
        return len(unique_words) / total_words

    diversity_score = topic_diversity(topic_words)
    print(f"Topic Diversity Score: {diversity_score}")
    
    # Round the scores to 3 decimal places
    umass_score = round(umass_score, 3)
    cv_score = round(cv_score, 3)
    diversity_score = round(diversity_score, 3)

    print(f"UMass Coherence Score: {umass_score:.3f} (closer to 0 is better)")
    print(f"CV Coherence Score: {cv_score:.3f} (higher is better)")
    print(f"Topic Diversity Score: {diversity_score:.3f} (higher is better)")

    # Save metrics to CSV
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write("OVERALL METRICS\n")
        f.write(f"Number of Topics,{len(topics_info['Topic'])}\n")  # Include all topics including -1
        f.write(f"UMass Coherence,{umass_score:.3f},Closer to 0 is better\n")
        f.write(f"CV Coherence,{cv_score:.3f},Higher is better\n")
        f.write(f"Topic Diversity,{diversity_score:.3f},Higher is better\n\n")
        
        # Write individual topic metrics
        f.write("INDIVIDUAL TOPIC METRICS\n")
        f.write("Topic ID,Topic Name,Document Count,Top Words\n")
        
        # Write each topic's details (including Topic -1)
        for _, row in topics_info.iterrows():
            topic_id = row["Topic"]
            name = str(row["Name"]).replace(",", ";")  # Escape commas for CSV
            count = row["Count"]
            # Get top words for this topic (adapt this based on your data structure)
            if "Representation" in topics_info_json[_]:
                top_words = ", ".join(topics_info_json[_]["Representation"][:10])
            else:
                words = topic_model.get_topic(topic_id)
                top_words = ", ".join([word for word, _ in words][:10])
                
            f.write(f"{topic_id},{name},{count},{top_words}\n")

    print(f"Saved detailed metrics to {metrics_csv_path}")

    # Update the JSON output to include both coherence scores
    scores_data = {
        "umass_coherence_score": umass_score,
        "cv_coherence_score": cv_score,
        "diversity_score": diversity_score
    }

    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, indent=2, ensure_ascii=False)
    print(f"Saved rounded scores to {scores_file}")

    # Also update the topics JSON to include both coherence metrics
    topics_metrics = {
        "model_info": {
            "num_topics": len(topics_info['Topic']),  # Include all topics including -1
            "umass_coherence": umass_score,
            "cv_coherence": cv_score,
            "diversity": diversity_score
        },
        "topics": topics_info_json
    }

    with open(topics_file, "w", encoding="utf-8") as f:
        json.dump(topics_metrics, f, indent=2, ensure_ascii=False)
    print(f"Updated topic information with metrics to {topics_file}")




