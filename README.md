# News Topic Identification Project

## Environment Setup

### Conda Environment: ragtec

This project uses a dedicated conda environment called `ragtec` to manage dependencies and ensure reproducible results.

**Python Version:** 3.11.13

### Environment Creation

To create the environment:
```bash
conda create -n ragtec python=3.11 -y
```

### Environment Activation

To activate the environment:
```bash
conda activate ragtec
```

### Dependencies Installation

Install all required packages:
```bash
conda run -n ragtec pip install -r requirements.txt
```

Or install core packages individually:
```bash
# Core dependencies (resolved compatibility issues)
conda run -n ragtec pip install numpy pandas scipy gensim

# Utility packages
conda run -n ragtec pip install tqdm requests beautifulsoup4 python-dotenv

# Additional packages as needed
conda run -n ragtec pip install scikit-learn nltk matplotlib seaborn
```

### Environment Verification

To verify the installation:
```bash
conda run -n ragtec python --version
conda run -n ragtec python -c "from gensim.models.coherencemodel import CoherenceModel; print('Gensim imports successful!')"
```

### Known Issues Resolved

- **SciPy/Gensim Compatibility**: The original error with `ImportError: cannot import name 'triu' from 'scipy.linalg'` has been resolved by installing compatible versions:
  - gensim==4.3.3
  - numpy==1.26.4 
  - scipy==1.13.1

## Project Structure

```
clean/
├── code/                           # Source code files
│   ├── 0_url-formatting.py        # URL formatting utilities
│   ├── 1_schema-exctration.py     # Schema extraction
│   ├── 2_gnews-content-scrapper.py # Google News content scraping
│   ├── 3_news-formatting-document.py # Document formatting
│   ├── 4_topic_modeling_*.py      # Topic modeling implementations
│   ├── 5_topic_*_assignation.py   # Topic assignment scripts
│   ├── 6_metrics_document_*.py    # Metrics calculation scripts
│   ├── lib/                       # Library files
│   └── utils/                     # Utility functions
├── data/                          # Data files and outputs
├── docs/                          # Documentation
├── image/                         # Image outputs
├── prompts/                       # Prompt templates
└── test/                          # Test files
```

## Usage

This project implements various topic modeling approaches including:
- BERT-based topic modeling
- LDA (Latent Dirichlet Allocation)
- LLM-based topic modeling
- RAGTEC (RAG Topic Extraction and Classification)

To run the RAGTEC metrics calculation:
```bash
conda activate ragtec
cd code
python 6_metrics_document_ragtec.py
```
