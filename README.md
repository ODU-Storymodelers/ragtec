# RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification

## Abstract

This repository implements RAGTEC (Retrieval-Augmented Generation for Topic Extraction and Classification), a novel approach to topic modeling that leverages Large Language Models (LLMs) with retrieval-augmented generation for enhanced topic discovery and classification in news corpora. The framework provides a comprehensive pipeline for analyzing news articles from conflict-affected regions, comparing RAGTEC performance against traditional methods like LDA and BERT-based topic modeling.

## Research Objectives

- **Primary**: Develop and evaluate RAGTEC methodology for topic modeling in news corpora
- **Secondary**: Compare performance across multiple topic modeling approaches (LDA, BERTopic, RAGTEC)
- **Application**: Analyze news coverage patterns in conflict-affected African regions

## Methodology Overview

### RAGTEC Framework

RAGTEC combines retrieval-augmented generation with structured prompting to achieve:
1. **Context-Aware Topic Extraction**: Uses retrieval systems to provide relevant context for LLM-based topic discovery
2. **Hierarchical Topic Classification**: Employs predefined topic taxonomies for consistent classification
3. **Quality Metrics Integration**: Implements comprehensive evaluation metrics including coherence, diversity, and coverage

### Workflow Pipeline

The research methodology follows a systematic 7-stage pipeline:

```
Data Collection → Content Extraction → Preprocessing → Topic Modeling → Classification → Evaluation → Visualization
     (1)              (2)               (3)           (4)            (5)           (6)          (7)
```

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

The research analyzes news corpora from four conflict-affected African regions:
- **Burundi**: Political crisis and electoral violence
- **Democratic Republic of Congo (DRC)**: Armed conflict and humanitarian crisis  
- **Mozambique**: Insurgency and climate-related disasters
- **Sudan**: Political transition and ethnic conflicts

Each dataset contains news articles collected from multiple sources, preprocessed and formatted for topic modeling analysis.

## Implementation Details

### Stage 1: Data Collection (`1_data.py`)
- URL collection and validation
- Schema extraction from news sources
- Multi-source aggregation

### Stage 2: Content Extraction (`2_news_extraction.py`)
- Web scraping with robust error handling
- Content cleaning and normalization
- Metadata preservation

### Stage 3: Preprocessing (`3_news-formatting-document.py`)
- Text normalization and tokenization
- Document formatting for downstream processing
- Quality filtering

### Stage 4: Topic Modeling (`4_topic_modeling_*.py`)
- **RAGTEC**: LLM-based extraction with retrieval augmentation
- **LDA**: Traditional statistical approach with optimal hyperparameter tuning
- **BERTopic**: Transformer-based embeddings with UMAP dimensionality reduction

### Stage 5: Classification (`5_topic_*_classification.py`)
- Document-level topic assignment
- Multi-topic classification support
- Confidence scoring

### Stage 6: Evaluation (`6_metrics_document_*.py`)
- Coherence metrics (CV, UMass, C_NPMI)
- Topic diversity and coverage analysis
- Inter-method comparison

### Stage 7: Visualization (`7_topic_storytelling_ragtec.py`)
- Topic distribution analysis
- Temporal pattern visualization
- Comparative performance charts

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

The framework implements comprehensive evaluation metrics:

- **Coherence Measures**: CV, UMass, C_NPMI coherence scores
- **Topic Diversity**: Intra-topic and inter-topic diversity analysis  
- **Coverage Analysis**: Document and vocabulary coverage assessment
- **Classification Quality**: Precision, recall, and F1 scores for topic assignment

## Results and Outputs

Results are organized by dataset and method:
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

## Contributing

For research collaboration or technical contributions:

1. Fork the repository
2. Create feature branches for specific improvements
3. Ensure all tests pass and code follows project standards
4. Submit pull requests with detailed descriptions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{ragtec2024,
  title={RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification in News Corpora},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  publisher={[Publisher]}
}
```

## Contact

For questions regarding this research or collaboration opportunities, please contact [research team contact information].

---

**Keywords**: Topic Modeling, Retrieval-Augmented Generation, Large Language Models, News Analysis, Computational Journalism, Conflict Analysis
