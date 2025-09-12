# RAGTEC: RAG-based Topic Extraction and Classification for News Analysis

## Abstract

This repository implements RAGTEC (Retrieval-Augmented Generation for Topic Extraction and Classification), a novel approach for news topic identification and analysis. The system compares multiple topic modeling methodologies including traditional LDA, modern BERTopic, and our proposed RAGTEC approach, providing a comprehensive framework for news content analysis across multiple conflict-affected regions.

## Methodology

### Overview

The RAGTEC methodology follows a systematic pipeline for news topic extraction and classification, implemented through a sequential workflow of numbered scripts that ensure reproducibility and systematic evaluation.

### Workflow Pipeline

The complete methodology is implemented through the following sequential steps:

#### 1. Data Preparation and URL Processing
- **`1_data.py`**: Main data preparation pipeline
- **`1_url-formatting.py`**: URL formatting and validation utilities

#### 2. Content Extraction and Scraping
- **`2_news_extraction.py`**: Advanced news content extraction
- **`2_gnews-content-scrapper.py`**: Google News content scraping utilities

#### 3. Document Preprocessing
- **`3_news-formatting-document.py`**: Document formatting and text preprocessing

#### 4. Topic Modeling Implementation
- **`4_topic_modeling_lda.py`**: Traditional Latent Dirichlet Allocation implementation
- **`4_topic_modeling_bertopic.py`**: BERTopic-based topic modeling
- **`4_topic_modeling_ragtec.py`**: Novel RAGTEC approach implementation

#### 5. Topic Assignment and Classification
- **`5_topic_ragtec_classifcation.py`**: RAGTEC topic classification
- **`5_topic_ragtec_assignation.py`**: Topic assignment utilities

#### 6. Evaluation and Metrics
- **`6_metrics_document_lda.py`**: LDA performance evaluation
- **`6_metrics_document_bert.py`**: BERTopic performance evaluation
- **`6_metrics_document_ragtec.py`**: RAGTEC performance evaluation
- **`6_metrics_document_ragtec_keybert.py`**: Enhanced RAGTEC with KeyBERT integration

#### 7. Advanced Analysis and Storytelling
- **`7_topic_storytelling_ragtec.py`**: Topic narrative generation and storytelling

## Project Structure

```
ragtec/
├── code/                              # Source code and implementation
│   ├── 1_data.py                     # Data preparation pipeline
│   ├── 1_url-formatting.py          # URL processing utilities
│   ├── 2_news_extraction.py         # Content extraction
│   ├── 3_news-formatting-document.py # Document preprocessing
│   ├── 4_topic_modeling_lda.py      # LDA implementation
│   ├── 4_topic_modeling_bertopic.py # BERTopic implementation
│   ├── 4_topic_modeling_ragtec.py   # RAGTEC implementation
│   ├── 5_topic_ragtec_*.py          # Topic assignment and classification
│   ├── 6_metrics_document_*.py      # Evaluation metrics for all methods
│   ├── 7_topic_storytelling_ragtec.py # Topic storytelling generation
│   ├── utils/                        # Utility modules
│   │   ├── news_extraction.py       # News extraction utilities
│   │   ├── ragtec_topic_*.py        # RAGTEC core modules
│   │   └── optimal_k.py             # Optimal topic number detection
│   ├── paper/                       # Paper-specific analysis scripts
│   │   ├── docs-topics.py           # Document-topic analysis
│   │   ├── ts-*.py                  # Time series analysis
│   │   └── outlets.py               # News outlet analysis
│   └── combined_scraper.py          # Integrated scraping solution
├── data/                             # Datasets organized by region
│   ├── burundi/                     # Burundi conflict dataset
│   ├── drc/                         # Democratic Republic of Congo dataset
│   ├── mozambique/                  # Mozambique conflict dataset
│   └── sudan/                       # Sudan conflict dataset
├── output/                          # Generated results by region and method
│   ├── [region]/
│   │   ├── lda/                     # LDA results
│   │   ├── bertopic/                # BERTopic results
│   │   └── ragtec/                  # RAGTEC results
│   └── ...
├── image/                           # Generated visualizations
│   ├── optimal_k_analysis.png       # Optimal topic analysis
│   ├── multi-topics-*.png          # Multi-topic distributions
│   ├── *_ts_network.png            # Topic-temporal networks
│   └── cooccurrence_network.html   # Interactive network visualization
├── prompts/                         # LLM prompt templates
│   ├── topic_extraction_prompt.txt  # Topic extraction prompts
│   ├── topic_classification_prompt.txt # Classification prompts
│   └── docs_retrieve_query.txt      # Document retrieval queries
└── requirements.txt                 # Project dependencies
```

## Datasets

The project analyzes news content from four conflict-affected regions:

- **Burundi**: Post-electoral violence and political crisis
- **Democratic Republic of Congo (DRC)**: Ongoing armed conflicts and humanitarian crisis
- **Mozambique**: Insurgency and conflict in Cabo Delgado province
- **Sudan**: Political transition and civil unrest

Each dataset contains structured news articles with metadata for comprehensive topic analysis.

## Methods Comparison

### Traditional LDA (Latent Dirichlet Allocation)
- Probabilistic topic modeling approach
- Bag-of-words representation
- Automated optimal topic number detection

### BERTopic
- Transformer-based embeddings
- HDBSCAN clustering
- UMAP dimensionality reduction

### RAGTEC (Proposed Method)
- Retrieval-Augmented Generation approach
- LLM-powered topic extraction
- Context-aware classification
- Enhanced with KeyBERT integration

## Environment Setup

### Prerequisites
- **Python Version:** 3.11.13
- **Conda Environment Management**
- **OpenAI API Key** (for RAGTEC implementation)

### Installation

1. **Create Conda Environment:**
```bash
conda create -n ragtec python=3.11 -y
conda activate ragtec
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment Verification:**
```bash
python --version
python -c "from gensim.models.coherencemodel import CoherenceModel; print('Gensim imports successful!')"
```

4. **Configure API Keys:**
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Dependency Resolution

The project resolves known compatibility issues:
- **SciPy/Gensim Compatibility**: Compatible versions (gensim==4.3.3, numpy==1.26.4, scipy==1.13.1)
- **Transformer Integration**: Optimized for BERT and sentence transformers
- **Visualization**: Plotly and matplotlib for interactive and static visualizations

## Reproduction Instructions

### Full Pipeline Execution

To reproduce the complete analysis for a specific dataset:

```bash
conda activate ragtec
cd code

# Set dataset name (burundi, drc, mozambique, or sudan)
export DATASET_NAME="drc"

# 1. Data preparation
python 1_data.py

# 2. Content extraction
python 2_news_extraction.py

# 3. Document preprocessing
python 3_news-formatting-document.py

# 4. Topic modeling (run all approaches)
python 4_topic_modeling_lda.py
python 4_topic_modeling_bertopic.py
python 4_topic_modeling_ragtec.py

# 5. Topic classification
python 5_topic_ragtec_classifcation.py

# 6. Evaluation and metrics
python 6_metrics_document_lda.py
python 6_metrics_document_bert.py
python 6_metrics_document_ragtec.py

# 7. Advanced analysis
python 7_topic_storytelling_ragtec.py
```

### Individual Method Execution

For RAGTEC-specific analysis:
```bash
python 4_topic_modeling_ragtec.py
python 6_metrics_document_ragtec.py
```

### Paper-Specific Analysis

For generating paper visualizations:
```bash
cd paper
python docs-topics.py
python ts-network.py
python outlets.py
```

## Results Structure

### Output Organization

Results are systematically organized by:
- **Dataset**: Regional conflict datasets
- **Method**: LDA, BERTopic, RAGTEC
- **Analysis Type**: Topic extraction, classification, metrics, storytelling

### Key Metrics

The evaluation framework provides:
- **Topic Coherence**: Semantic consistency measures
- **Classification Accuracy**: Topic assignment performance
- **Temporal Analysis**: Topic evolution over time
- **Network Analysis**: Topic co-occurrence patterns

### Visualization Outputs

- **Topic Distribution Plots**: Multi-topic distribution analysis
- **Network Visualizations**: Topic and temporal relationship networks
- **Time Series Analysis**: Topic evolution patterns
- **Interactive Dashboards**: HTML-based exploration tools

## Citation

```bibtex
@article{ragtec2024,
  title={RAGTEC: RAG-based Topic Extraction and Classification for News Analysis},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.
