import json
import os

# Load classification results
with open('../output/mozambique/ragtec/classification/classification_results.json', 'r') as f:
    data = json.load(f)

# Flatten articles
flattened = []
for article in data:
    flat = {'url': article['url'], 'text_snippet': article['text_snippet']}
    if 'response' in article:
        resp = article['response']
        if 'topics' in resp:
            flat['topics'] = resp['topics']
        if 'primary_topic' in resp:
            flat['primary_topic'] = resp['primary_topic']
        if 'coverage_assessment' in resp:
            flat['coverage_assessment'] = resp['coverage_assessment']
    flattened.append(flat)

# Save flattened articles
with open('../output/mozambique/ragtec/classification/mozambique_articles_rag_assigned_topics.json', 'w') as f:
    json.dump(flattened, f, indent=2)

print(f"Generated flattened file with {len(flattened)} articles")
