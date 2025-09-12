# RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification

## Abstract

This repository implements RAG-TEC (Retrieval-Augmented Generation for Topic Extraction and Classification), a novel approach addressing redundancy and interpretability challenges in topic modeling of news corpora. RAG-TEC leverages Large Language Models (LLMs) with retrieval-augmented generation to improve unsupervised topic discovery. The framework demonstrates superior performance compared to traditional methods such as LDA, BERTopic, and the enhanced RAG-TEC+KeyBERT hybrid. Additionally, it produces rich storytelling outputs including Topic Dictionary, Collection Context, Topic Distribution, and Topic Co-occurrence Network, facilitating deeper insights into news collections.

## Research Objectives

- **Primary**: Introduce and evaluate RAG-TEC for unsupervised topic discovery in news collections  
- **Secondary**: Compare against LDA, BERTopic, and RAG-TEC+KeyBERT in coherence and diversity  
- **Application**: Case studies on four African crisis-related news collections

## Methodology Overview

The research methodology follows a systematic 5-stage pipeline:

![RAGTEC Pipeline Overview](image/ragtec.png)

### Collection Generation

Curate news collections by manually selecting URLs from diverse outlets (e.g., Reuters, DW), archiving each page via the Internet Archive’s Wayback Machine to ensure persistence. Validate URLs by checking for successful responses and exclude inaccessible or paywalled links. For each valid URL, extract metadata (title, author, images) using schema.org standards and retrieve full-text content with tools like Gnews Web Scraper. Apply random delays and retry logic to minimize blocking and maximize completeness. Merge metadata and content into a standardized format, providing rich context for topic discovery via retrieval systems.

### Topic Extraction

Perform context-aware topic extraction using retrieval-augmented LLMs and utility functions (`utils/ragtec_topic_extraction.py`) to identify meaningful topics with reduced redundancy.

### Topic Classification

Assign multiple topics per document with multi-topic classification techniques and supporting utilities (`utils/ragtec_topic_classification.py`) to capture complex thematic structures.

### Topic Quality

Evaluate topics with a focus on coherence and diversity tradeoffs, integrating metrics and utility modules (`utils/ragtec_topic_quality.py`) to assess quality comprehensively.

### Topic Storytelling

Produce storytelling outputs including Topic Dictionary, Collection Context, Topic Distribution, and Topic Co-occurrence Network to enhance interpretability, supported by utility scripts for data formatting and visualization.


## Repository Structure

```
ragtec/
├── code/                                    # Source code implementation
│   ├── 1_data.py                           # Stage 1: Data collection and URL processing
│   ├── 2_news_extraction.py               # Stage 2: News content extraction
│   ├── 3_news-formatting-document.py      # Stage 3: Document preprocessing
│   ├── 4_topic_modeling_*.py              # Stage 4: Topic modeling implementations
│   │   ├── 4_topic_modeling_ragtec.py     #   - RAGTEC implementation
│   │   ├── 4_topic_modeling_lda.py        #   - LDA baseline
│   │   └── 4_topic_modeling_bertopic.py   #   - BERTopic baseline
│   ├── 5_topic_*_classification.py        # Stage 5: Topic classification
│   ├── 6_metrics_document_*.py            # Stage 6: Evaluation metrics
│   └── 7_topic_storytelling_ragtec.py     # Stage 7: Results visualization
│   ├── utils/                              # Core utility modules
│   │   ├── ragtec_topic_extraction.py     #   - RAGTEC extraction logic
│   │   ├── ragtec_topic_classification.py #   - RAGTEC classification logic
│   │   ├── ragtec_topic_quality.py        #   - Quality assessment metrics
│   │   └── news_extraction.py             #   - News scraping utilities
│   └── paper/                              # Research output scripts
├── data/                                    # Dataset storage
│   ├── burundi/                            # Burundi news corpus
│   ├── drc/                                # Democratic Republic of Congo corpus
│   ├── mozambique/                         # Mozambique news corpus
│   └── sudan/                              # Sudan news corpus
├── output/                                  # Results and generated outputs
├── prompts/                                 # LLM prompt templates
│   ├── topic_extraction_prompt.txt         # RAGTEC extraction prompts
│   ├── topic_classification_prompt.txt     # RAGTEC classification prompts
│   └── docs_retrieve_query.txt             # Retrieval query templates
├── image/                                   # Generated visualizations
└── requirements.txt                         # Dependencies specification
```

## Datasets

The research analyzes curated news collections from Mozambique, Burundi, Democratic Republic of Congo (DRC), and Sudan. These collections are archived via the Internet Archive’s Wayback Machine and focus on political, humanitarian, and climate crises in these regions. The datasets are formatted and processed to support comprehensive topic modeling analysis.

## Implementation Details

### Stage 1: Collection Generation
- URL collection and validation  
- Schema extraction from news sources  
- Multi-source aggregation  

### Stage 2: Topic Extraction (`4_topic_modeling_ragtec.py`)
- Context-aware topic discovery using retrieval-augmented LLMs  
- Redundancy reduction and interpretability enhancements  

### Stage 3: Topic Classification (`5_topic_ragtec_classification.py`)
- Multi-topic assignment per document  
- Confidence scoring and classification logic  

### Stage 4: Topic Quality (`6_metrics_document_ragtec.py`, `6_metrics_document_ragtec_keybert.py`)
- Coherence and diversity tradeoff evaluation  


### Stage 5: Topic Storytelling (`7_topic_storytelling_ragtec.py`)
- Generation of Topic Dictionary, Collection Context, Topic Distribution, and Topic Co-occurrence Network  

## Installation and Setup

### Environment Requirements

- **Python**: 3.11.13  
- **Primary Dependencies**: gensim==4.3.3, numpy==1.26.4, scipy==1.13.1  

### Installation Steps

1. **Create conda environment**:
```bash
conda create -n ragtec python=3.11 -y
conda activate ragtec
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "from gensim.models.coherencemodel import CoherenceModel; print('Installation successful!')"
```

### Environment Variables

Create a `.env` file with required API keys:
```bash
# OpenAI API key for RAGTEC implementation
OPENAI_API_KEY=your_openai_key_here

# Additional API keys as needed
```

## Usage Instructions

### Complete Pipeline Execution

Run the full RAGTEC pipeline for a specific dataset:

```bash
conda activate ragtec
cd code

# Example: Process DRC dataset
python 1_data.py --dataset drc
python 2_news_extraction.py --dataset drc  
python 3_news-formatting-document.py --dataset drc
python 4_topic_modeling_ragtec.py --dataset drc
python 5_topic_ragtec_classification.py --dataset drc
python 6_metrics_document_ragtec.py --dataset drc
python 7_topic_storytelling_ragtec.py --dataset drc
```

### Individual Component Usage

**RAGTEC Topic Extraction**:
```bash
python 4_topic_modeling_ragtec.py
```

**Metrics Calculation**:
```bash
python 6_metrics_document_ragtec.py
```

**Baseline Comparisons**:
```bash
python 4_topic_modeling_lda.py      # LDA baseline
python 4_topic_modeling_bertopic.py # BERTopic baseline
```

## Evaluation Metrics

The framework focuses on evaluating topic coherence (CV) and diversity (Jaccard Diversity), emphasizing the tradeoff between these metrics. While KeyBERT integration improves coherence scores, it slightly reduces topic diversity, highlighting the balance required for optimal topic quality.

## Results and Outputs

Results are organized by dataset and method, aligned with the storytelling stage outputs:
```
output/
├── {dataset}/
│   ├── ragtec/
│   │   ├── extraction/     # Topic extraction results
│   │   ├── classification/ # Classification outputs  
│   │   └── quality/        # Evaluation metrics
│   ├── lda/               # LDA baseline results
│   └── bertopic/          # BERTopic baseline results
```

Storytelling outputs include Topic Dictionary, Collection Context, Topic Distribution, and Topic Co-occurrence Network to facilitate comprehensive understanding.

## Contributing

For research collaboration or technical contributions:

1. Fork the repository  
2. Create feature branches for specific improvements  
3. Ensure all tests pass and code follows project standards  
4. Submit pull requests with detailed descriptions  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions regarding this research or collaboration opportunities, please contact [research team contact information].

---

**Keywords**: Topic Modeling, Retrieval-Augmented Generation, Large Language Models, News Analysis, Computational Journalism, Conflict Analysis
