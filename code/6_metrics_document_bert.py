import json
import os
import numpy as np
from collections import defaultdict
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
import re
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return []
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def load_data(file_path):
    """Load the JSON file containing LLM-generated topics."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} articles from {file_path}")
    return data

def load_standardized_topics(file_path):
    """Load the standardized topics and mappings from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        std_topics = json.load(f)
    print(f"Loaded standardized topics: {len(std_topics['main_topics'])} main topics, {len(std_topics['sub_topics'])} sub topics")
    return std_topics

def normalize_topic_name(name):
    """Normalize topic name by removing apostrophes and making lowercase."""
    return name.lower().replace("'s", "").replace("'", "").strip()

def is_similar_topic(topic1, topic2, threshold=0.8):
    """Check if two topic names are similar using sequence matching."""
    norm1 = normalize_topic_name(topic1)
    norm2 = normalize_topic_name(topic2)
    
    # Direct matching after normalization
    if norm1 == norm2:
        return True
    
    # Check for singular/plural variations
    if norm1.rstrip('s') == norm2.rstrip('s'):
        return True
    
    # Use sequence matcher for fuzzy matching
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity > threshold

def extract_standardized_topics_and_texts(data, std_topics):
    """Extract main topics, their keywords, and associated texts using standardized topic names."""
    # Get the mapping dictionaries from the standardized topics
    main_topic_mapping = std_topics.get('main_topics_mapping', {})
    sub_topic_mapping = std_topics.get('sub_topics_mapping', {})
    
    # Create dictionaries of standard topics with their details
    std_main_topics_dict = {topic['name']: topic for topic in std_topics['main_topics']}
    std_sub_topics_dict = {topic['name']: topic for topic in std_topics['sub_topics']}
    
    topic_keywords = defaultdict(list)
    topic_texts = defaultdict(list)
    all_texts = []
    
    # Count the total number of main topics across all articles
    total_main_topics = 0
    standardized_topics_count = 0
    
    for article in data:
        # Always add text to all_texts regardless of topics
        if 'text' in article:
            clean_text = article['text'].lower()
            all_texts.append(clean_text)
        
        if 'main_topic' not in article or not article['main_topic']:
            continue
            
        main_topics = article['main_topic']
        
        # Handle both object and list formats
        if isinstance(main_topics, dict):
            if 'main_topic' in main_topics:
                total_main_topics += 1
                original_topic_name = main_topics['main_topic']
                
                # Get standardized topic name from mapping
                std_topic_name = main_topic_mapping.get(original_topic_name)
                if std_topic_name:
                    standardized_topics_count += 1
                    
                    # Get keywords from standardized topic definition instead of article
                    if std_topic_name in std_main_topics_dict:
                        keywords = std_main_topics_dict[std_topic_name].get('keywords', [])
                        topic_keywords[std_topic_name].extend(keywords)
                    
                    if 'text' in article:
                        topic_texts[std_topic_name].append(clean_text)
                else:
                    # Fallback to original topic if not in mapping
                    print(f"Warning: Topic '{original_topic_name}' not found in mapping")
                    if 'keywords' in main_topics:
                        keywords = [k.lower().strip() for k in main_topics['keywords'] if k and k.strip()]
                        topic_keywords[original_topic_name].extend(keywords)
                    if 'text' in article:
                        topic_texts[original_topic_name].append(clean_text)
                        
        elif isinstance(main_topics, list):
            total_main_topics += len(main_topics)
            
            for main in main_topics:
                if 'main_topic' not in main:
                    continue
                
                original_topic_name = main['main_topic']
                
                # Get standardized topic name from mapping
                std_topic_name = main_topic_mapping.get(original_topic_name)
                if std_topic_name:
                    standardized_topics_count += 1
                    
                    # Get keywords from standardized topic definition
                    if std_topic_name in std_main_topics_dict:
                        keywords = std_main_topics_dict[std_topic_name].get('keywords', [])
                        topic_keywords[std_topic_name].extend(keywords)
                    
                    if 'text' in article:
                        topic_texts[std_topic_name].append(clean_text)
                else:
                    # Fallback to original topic if not in mapping
                    print(f"Warning: Topic '{original_topic_name}' not found in mapping")
                    if 'keywords' in main:
                        keywords = [k.lower().strip() for k in main['keywords'] if k and k.strip()]
                        topic_keywords[original_topic_name].extend(keywords)
                    if 'text' in article:
                        topic_texts[original_topic_name].append(clean_text)
    
    # Remove duplicates in keywords
    for topic in topic_keywords:
        topic_keywords[topic] = list(set(topic_keywords[topic]))
    
    # Handle case when no topics are found
    if not topic_keywords and all_texts:
        print("No topics found in data - using standardized topics instead")
        # Use standardized topics directly
        for topic in std_topics['main_topics']:
            topic_keywords[topic['name']] = topic['keywords']
            topic_texts[topic['name']] = all_texts
    
    print(f"Mapped {standardized_topics_count} of {total_main_topics} topics to standard topics")
    print(f"Extracted {len(topic_keywords)} distinct topics")
    print(f"Collected {len(all_texts)} unique article texts")
    
    return topic_keywords, topic_texts, all_texts

def extract_default_topics(texts, num_topics=6):
    """Extract default topics when none are found in the data."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.decomposition import NMF
    
    # Create a document-term matrix
    print("Generating topics using NMF...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    counts = vectorizer.fit_transform(texts)
    
    # Convert to TF-IDF representation
    tfidf = TfidfTransformer().fit_transform(counts)
    
    # Extract topics using NMF
    nmf = NMF(n_components=num_topics, random_state=42).fit(tfidf)
    
    # Get the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    
    # Create topic names based on the top words
    for i, topic in enumerate(nmf.components_):
        top_words = [feature_names[j] for j in topic.argsort()[:-11:-1]]
        topic_name = f"Topic {i+1}: {', '.join(top_words[:3])}"
        topics[topic_name] = top_words
        print(f"Generated {topic_name}")
    
    return topics

def prepare_data_for_coherence(topic_keywords, all_texts):
    """Prepare data structures needed for coherence calculation."""
    # Tokenize all documents
    print("Tokenizing texts...")
    tokenized_texts = [clean_text(doc) for doc in all_texts]
    
    # Build dictionary and corpus
    print("Building dictionary and corpus...")
    dictionary = Dictionary(tokenized_texts)
    original_dict_size = len(dictionary)
    print(f"Original dictionary size: {original_dict_size}")
    
    # Filter dictionary - parameters matching other methods for consistency
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    filtered_dict_size = len(dictionary)
    print(f"Filtered dictionary size: {filtered_dict_size} (removed {original_dict_size - filtered_dict_size} terms)")
    
    # Build corpus
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    print(f"Corpus size: {len(corpus)} documents")
    
    # Format topic keywords for coherence models - use exactly 10 keywords per topic
    topic_word_lists = []
    total_keywords = 0
    keywords_in_dict = 0
    
    print("\n--- KEYWORD VALIDATION DETAILS ---")
    
    for topic_id, keywords in topic_keywords.items():
        if keywords:  # Only process non-empty keyword lists
            valid_keywords = []
            invalid_keywords = []
            
            for keyword in keywords[:10]:  # Limit to top 10 for consistency
                if keyword in dictionary.token2id:
                    valid_keywords.append(keyword)
                    keywords_in_dict += 1
                else:
                    invalid_keywords.append(keyword)
                total_keywords += 1
            
            if valid_keywords:
                topic_word_lists.append(valid_keywords)
                print(f"Topic {topic_id}:")
                print(f"  - Valid keywords ({len(valid_keywords)}): {', '.join(valid_keywords)}")
                if invalid_keywords:
                    print(f"  - Invalid keywords ({len(invalid_keywords)}): {', '.join(invalid_keywords)}")
            else:
                print(f"Topic {topic_id}: No valid keywords found in dictionary")
                if invalid_keywords:
                    print(f"  - Invalid keywords ({len(invalid_keywords)}): {', '.join(invalid_keywords)}")
    
    print("\n--- END KEYWORD VALIDATION ---")
    
    # Fix the division by zero error
    if total_keywords > 0:
        percentage = (keywords_in_dict/total_keywords)*100
        print(f"Keywords in dictionary: {keywords_in_dict} out of {total_keywords} ({percentage:.1f}%)")
    else:
        print("No keywords found in topics")
    
    return dictionary, corpus, tokenized_texts, topic_word_lists

def calculate_coherence(dictionary, corpus, tokenized_texts, topic_word_lists):
    """Calculate CV and UMass coherence scores for topics."""
    # Ensure we have valid data
    if not topic_word_lists or len(topic_word_lists) < 2:
        print("Warning: Not enough valid topics for coherence calculation.")
        return None, None
    
    if len(dictionary) == 0 or len(corpus) == 0:
        print("Warning: Dictionary or corpus is empty. Cannot compute coherence.")
        return None, None
    
    # Calculate CV coherence (more reliable for this data structure)
    try:
        print("Calculating CV coherence...")
        coherence_cv = CoherenceModel(
            topics=topic_word_lists, 
            texts=tokenized_texts, 
            dictionary=dictionary, 
            coherence='c_v',
            processes=1  # Use single process to avoid multiprocessing issues
        )
        cv_score = coherence_cv.get_coherence()
        print(f"Successfully calculated CV coherence score: {cv_score}")
    except Exception as e:
        print(f"Error calculating CV coherence: {e}")
        cv_score = None
    
    # Calculate UMass coherence
    try:
        print("Calculating UMass coherence...")
        coherence_umass = CoherenceModel(
            topics=topic_word_lists, 
            texts=tokenized_texts,  # Use tokenized texts instead of corpus
            dictionary=dictionary, 
            coherence='u_mass',
            processes=1  # Use single process to avoid multiprocessing issues
        )
        umass_score = coherence_umass.get_coherence()
        print(f"Successfully calculated UMass coherence score: {umass_score}")
        
        # Check if we got a valid coherence score
        if np.isnan(umass_score) or np.isinf(umass_score):
            print("Warning: UMass coherence calculation resulted in invalid value.")
            umass_score = None
    except Exception as e:
        print(f"Error calculating UMass coherence: {e}")
        umass_score = None
    
    return cv_score, umass_score

def calculate_unique_word_proportion(topic_keywords, top_n=10):
    """Compute topic diversity as the proportion of unique words across topics."""
    unique_words = set()
    total_words = 0
    
    for keywords in topic_keywords.values():
        words = keywords[:min(len(keywords), top_n)]  # Take only top_n keywords per topic
        unique_words.update(words)
        total_words += len(words)
    
    diversity_score = len(unique_words) / total_words if total_words > 0 else 0
    return diversity_score

def calculate_jaccard_diversity(topic_keywords, top_n=10):
    """Calculate Jaccard-based topic diversity (average distance between topics)."""
    try:
        # Convert to sets of words for each topic
        topic_terms = []
        for keywords in topic_keywords.values():
            words = keywords[:min(len(keywords), top_n)]  # Take only top_n keywords per topic
            topic_terms.append(set(words))
        
        # Calculate average Jaccard distance between all topic pairs
        jaccard_distances = []
        for i in range(len(topic_terms)):
            for j in range(i+1, len(topic_terms)):
                intersection = len(topic_terms[i].intersection(topic_terms[j]))
                union = len(topic_terms[i].union(topic_terms[j]))
                distance = 1 - (intersection / union if union > 0 else 0)
                jaccard_distances.append(distance)
        
        diversity_score = sum(jaccard_distances) / len(jaccard_distances) if jaccard_distances else 0
        return diversity_score
    
    except Exception as e:
        print(f"Error calculating Jaccard diversity: {e}")
        return None

if __name__ == "__main__":
    # Define file paths
    dataset_name = "sudan"
    input_file = f"../output/{dataset_name}/bertopic/{dataset_name}_articles_topic_document_bertopic.json"
    country = os.path.basename(input_file).split("_")[0]
    output_dir = f"../output/{dataset_name}/bertopic"
    
    output_metrics_file = f"{output_dir}/{country}_document_bertopic_metrics.csv"
    json_output = f"{output_dir}/{country}_document_bertopic_metrics.json"
    
    # Load data files
    data = load_data(input_file)
    
    # Extract topic information from BERTopic model results - FIXED
    topic_keywords = {}
    topic_texts = {}
    all_texts = []

    for article in data:
        if 'content' in article:
            all_texts.append(article['content'])
            
        if 'topic_info' in article and article['topic_info']:
            # Fix: Use 'Topic' instead of 'topic_id'
            topic_id = article['topic_info'].get('Topic')
            
            topic_name = article['topic_info'].get('Name', f"Topic {topic_id}")
            
            # Fix: Parse Top_n_words string into keyword list
            top_n_words = article['topic_info'].get('Top_n_words', '')
            if top_n_words:
                keywords = [word.strip() for word in top_n_words.split('-') if word.strip()]
            else:
                keywords = []
            
            if topic_id is not None:
                if topic_id not in topic_keywords:
                    topic_keywords[topic_id] = keywords
                    topic_texts[topic_id] = []
                
                if 'content' in article:
                    topic_texts[topic_id].append(article['content'])
    
    # Print debug info
    print(f"Extracted {len(topic_keywords)} topics with keywords")
    if topic_keywords:
        sample_topic = next(iter(topic_keywords))
        print(f"Sample topic {sample_topic} keywords: {topic_keywords[sample_topic]}")
    
    # Continue with the rest of your script
    dictionary, corpus, tokenized_texts, topic_word_lists = prepare_data_for_coherence(topic_keywords, all_texts)
    
    # Count topics and print
    num_topics = len(topic_keywords)
    print(f"Number of BERTopic Topics: {num_topics}")
    
    # Calculate metrics
    coherence_cv_score, coherence_umass_score = calculate_coherence(dictionary, corpus, tokenized_texts, topic_word_lists)
    uwp_diversity_score = calculate_unique_word_proportion(topic_keywords, top_n=10)
    jaccard_diversity_score = calculate_jaccard_diversity(topic_keywords, top_n=10)
    
    # Round scores to 3 decimal places
    coherence_cv_score = round(coherence_cv_score, 3) if coherence_cv_score is not None else None
    coherence_umass_score = round(coherence_umass_score, 3) if coherence_umass_score is not None else None
    uwp_diversity_score = round(uwp_diversity_score, 3) if uwp_diversity_score is not None else None
    jaccard_diversity_score = round(jaccard_diversity_score, 3) if jaccard_diversity_score is not None else None
    
    # Output formatted scores
    print(f"C_V Coherence Score: {coherence_cv_score:.3f}" if coherence_cv_score is not None else "C_V Coherence Score: Not calculable")
    print(f"UMass Coherence Score: {coherence_umass_score:.3f}" if coherence_umass_score is not None else "UMass Coherence Score: Not calculable")
    print(f"Jaccard Diversity Score: {jaccard_diversity_score:.3f}" if jaccard_diversity_score is not None else "Jaccard Diversity Score: Not calculable")
    print(f"Unique Word Proportion Score: {uwp_diversity_score:.3f}" if uwp_diversity_score is not None else "UWP Diversity Score: Not calculable")
    
    # Save metrics to CSV in format consistent with LDA
    with open(output_metrics_file, 'w', newline='', encoding='utf-8') as f:
        f.write("OVERALL METRICS\n")
        f.write(f"Number of Topics,{num_topics}\n")
        
        # Write coherence scores
        cv_str = f"{coherence_cv_score:.3f}" if coherence_cv_score is not None else "N/A"
        umass_str = f"{coherence_umass_score:.3f}" if coherence_umass_score is not None else "N/A"
        
        f.write(f"CV Coherence,{cv_str},Higher is better\n")
        f.write(f"UMass Coherence,{umass_str},Closer to 0 is better\n")
        f.write(f"Jaccard Diversity,{jaccard_diversity_score:.3f},Higher is better\n")
        f.write(f"Unique Word Proportion,{uwp_diversity_score:.3f},Higher is better\n\n")
        
        # Add individual topic details
        f.write("INDIVIDUAL TOPIC METRICS\n")
        f.write("Topic ID,Topic Name,Document Count,Top Words\n")
        
        # Write each topic's details (including Topic -1)
        for topic_id, keywords in topic_keywords.items():
            topic_name = f"{topic_id}_" + "_".join(keywords[:3]) if keywords else f"Topic {topic_id}"
            doc_count = len(topic_texts[topic_id]) if topic_id in topic_texts else 0
            top_keywords = ", ".join(keywords[:10])  # Get top 10 keywords
            f.write(f"{topic_id},{topic_name},{doc_count},{top_keywords}\n")
    
    print(f"Saved detailed metrics to {output_metrics_file}")
    
    # Save as JSON in format consistent with LDA
    metrics_json = {
        "model_info": {
            "num_topics": num_topics,
            "cv_coherence": coherence_cv_score,
            "umass_coherence": coherence_umass_score,
            "jaccard_diversity": jaccard_diversity_score,
            "uwp_diversity": uwp_diversity_score
        },
        "topics": [
            {
                "Topic": topic_id,
                "Name": f"{topic_id}_" + "_".join(keywords[:3]),
                "Count": len(topic_texts[topic_id]) if topic_id in topic_texts else 0,
                "TopWords": [
                    {"word": word, "weight": 1.0} for word in keywords[:10]
                ]
            }
            for topic_id, keywords in topic_keywords.items()
        ]
    }
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    
    print(f"Saved JSON metrics to {json_output}")

