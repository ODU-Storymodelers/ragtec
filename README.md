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

