"""
Optimal K Selection for Representative News Articles

This module implements the heuristic from AlNoamany, Weigle, & Nelson (2017) 
for selecting the optimal number of representative articles from a news collection.

Reference:
AlNoamany, Y., Weigle, M. C., & Nelson, M. L. (2017, June). 
Generating stories from archived collections. In Proceedings of the 2017 
ACM on Web Science Conference (pp. 309-318).
"""

import math
from typing import List, Dict, Tuple, Union


def calculate_optimal_k(collection_size: int) -> int:
    """
    Calculate the optimal number of representative articles (k) to select 
    from a news collection based on its total size.
    
    Formula:
    - If |N| ≤ 28: k = |N| (use all articles)
    - If |N| > 28: k = ⌈28 + log₁₀(|N|)⌉ (fixed base + logarithmic scaling)
    
    Args:
        collection_size (int): Total number of articles in the collection (|N|)
        
    Returns:
        int: Optimal number of representative articles to select (k)
        
    Raises:
        ValueError: If collection_size is not a positive integer
        
    Examples:
        >>> calculate_optimal_k(15)
        15
        >>> calculate_optimal_k(100)
        30
        >>> calculate_optimal_k(1000)
        31
    """
    if not isinstance(collection_size, int) or collection_size <= 0:
        raise ValueError("Collection size must be a positive integer")
    
    if collection_size <= 28:
        return collection_size
    else:
        return math.ceil(28 + math.log10(collection_size))


def get_selection_strategy(collection_size: int) -> Dict[str, Union[int, float, str]]:
    """
    Get detailed information about the selection strategy for a given collection size.
    
    Args:
        collection_size (int): Total number of articles in the collection
        
    Returns:
        dict: Dictionary containing strategy information with keys:
            - collection_size: Original collection size
            - optimal_k: Calculated optimal k
            - selection_ratio: k/N ratio
            - percentage_selected: Percentage of articles selected
            - strategy_type: Type of strategy used
            - formula_used: Mathematical formula applied
            - description: Strategy description
        
    Examples:
        >>> strategy = get_selection_strategy(100)
        >>> print(strategy['optimal_k'])
        30
    """
    if not isinstance(collection_size, int) or collection_size <= 0:
        raise ValueError("Collection size must be a positive integer")
    
    k = calculate_optimal_k(collection_size)
    ratio = k / collection_size
    
    if collection_size <= 28:
        strategy_type = "Complete Coverage"
        description = "Include all articles - collection is small enough for complete analysis"
        formula_used = "k = |N|"
    else:
        strategy_type = "Logarithmic Scaling"
        description = "Fixed base (28) + logarithmic adjustment to balance coverage and efficiency"
        formula_used = "k = ⌈28 + log₁₀(|N|)⌉"
    
    return {
        'collection_size': collection_size,
        'optimal_k': k,
        'selection_ratio': ratio,
        'percentage_selected': ratio * 100,
        'strategy_type': strategy_type,
        'formula_used': formula_used,
        'description': description
    }


def analyze_collection_sizes(sizes: List[int]) -> Dict[str, Union[List[Tuple], Dict[str, float]]]:
    """
    Analyze multiple collection sizes and return optimal k values with statistics.
    
    Args:
        sizes (List[int]): List of collection sizes to analyze
        
    Returns:
        dict: Dictionary containing analysis results with keys:
            - 'results': List of tuples (collection_size, optimal_k, selection_ratio)
            - 'statistics': Dictionary with min, max, mean k values and ratios
            
    Examples:
        >>> sizes = [10, 50, 100, 500, 1000]
        >>> analysis = analyze_collection_sizes(sizes)
        >>> print(analysis['statistics']['mean_k'])
    """
    if not sizes or not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("Sizes must be a non-empty list of positive integers")
    
    results = []
    k_values = []
    ratios = []
    
    for size in sizes:
        k = calculate_optimal_k(size)
        ratio = k / size
        results.append((size, k, ratio))
        k_values.append(k)
        ratios.append(ratio)
    
    statistics = {
        'min_k': min(k_values),
        'max_k': max(k_values),
        'mean_k': sum(k_values) / len(k_values),
        'min_ratio': min(ratios),
        'max_ratio': max(ratios),
        'mean_ratio': sum(ratios) / len(ratios)
    }
    
    return {
        'results': results,
        'statistics': statistics
    }


def calculate_k_range(collection_size: int, variation_percent: float = 0.2) -> Tuple[int, int, int]:
    """
    Calculate a range of k values around the optimal k for experimentation.
    
    Args:
        collection_size (int): Total number of articles in the collection
        variation_percent (float): Percentage variation around optimal k (default: 0.2 = 20%)
        
    Returns:
        Tuple[int, int, int]: (min_k, optimal_k, max_k)
        
    Examples:
        >>> calculate_k_range(100, 0.3)
        (21, 30, 39)
    """
    optimal_k = calculate_optimal_k(collection_size)
    variation = max(1, int(optimal_k * variation_percent))
    
    min_k = max(1, optimal_k - variation)
    max_k = min(collection_size, optimal_k + variation)
    
    return min_k, optimal_k, max_k


def estimate_topic_count(k: int, topic_model_type: str = "lda") -> Dict[str, Union[int, float, str]]:
    """
    Estimate the number of topics that can be effectively discovered given k articles.
    
    Based on empirical research in topic modeling, this function provides estimates
    for different topic modeling approaches.
    
    Args:
        k (int): Number of articles selected for analysis
        topic_model_type (str): Type of topic model ("lda", "bert", "nmf", "general")
        
    Returns:
        dict: Dictionary containing topic estimates with keys:
            - estimated_topics: Primary estimate of topic count
            - min_topics: Conservative minimum estimate
            - max_topics: Optimistic maximum estimate
            - topic_ratio: Topics per article ratio
            - confidence: Confidence level in estimate
            - methodology: Brief description of estimation method
            
    Examples:
        >>> estimate_topic_count(30, "lda")
        {'estimated_topics': 6, 'min_topics': 4, 'max_topics': 8, ...}
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    
    # Topic modeling heuristics based on literature and empirical studies
    if topic_model_type.lower() == "lda":
        # LDA typically needs 10-20 documents per topic for stable results
        # Conservative: 15 docs/topic, Optimistic: 8 docs/topic
        estimated_topics = max(2, int(k / 12))
        min_topics = max(2, int(k / 20))
        max_topics = max(2, int(k / 8))
        confidence = "high" if k >= 50 else "medium" if k >= 20 else "low"
        methodology = "LDA rule: ~12 documents per topic (range: 8-20)"
        
    elif topic_model_type.lower() == "bert":
        # BERT-based models (BERTopic) can work with fewer documents per topic
        # More flexible clustering approach
        estimated_topics = max(2, int(k / 8))
        min_topics = max(2, int(k / 15))
        max_topics = max(2, int(k / 5))
        confidence = "high" if k >= 30 else "medium" if k >= 15 else "low"
        methodology = "BERT-based clustering: ~8 documents per topic (range: 5-15)"
        
    elif topic_model_type.lower() == "nmf":
        # Non-negative Matrix Factorization
        # Similar requirements to LDA but slightly more flexible
        estimated_topics = max(2, int(k / 10))
        min_topics = max(2, int(k / 18))
        max_topics = max(2, int(k / 6))
        confidence = "medium" if k >= 40 else "low"
        methodology = "NMF rule: ~10 documents per topic (range: 6-18)"
        
    else:  # general case
        # General heuristic for various topic modeling approaches
        estimated_topics = max(2, int(k / 10))
        min_topics = max(2, int(k / 20))
        max_topics = max(2, int(k / 6))
        confidence = "medium" if k >= 30 else "low"
        methodology = "General heuristic: ~10 documents per topic (range: 6-20)"
    
    # Apply upper bounds based on collection size constraints
    # Avoid over-segmentation for small collections
    if k <= 10:
        estimated_topics = min(estimated_topics, 3)
        max_topics = min(max_topics, 4)
    elif k <= 20:
        estimated_topics = min(estimated_topics, 5)
        max_topics = min(max_topics, 7)
    
    topic_ratio = estimated_topics / k if k > 0 else 0
    
    return {
        'k_articles': k,
        'estimated_topics': estimated_topics,
        'min_topics': min_topics,
        'max_topics': max_topics,
        'topic_ratio': topic_ratio,
        'confidence': confidence,
        'methodology': methodology,
        'model_type': topic_model_type
    }


def analyze_k_topic_relationship(collection_size: int, 
                               k_range: Tuple[int, int] = None,
                               topic_model_types: List[str] = None) -> Dict[str, List[Dict]]:
    """
    Analyze how topic count varies with different k values for a given collection.
    
    Args:
        collection_size (int): Total number of articles in collection
        k_range (Tuple[int, int], optional): Range of k values to test. 
                                           If None, uses optimal k ± 50%
        topic_model_types (List[str], optional): Topic models to analyze.
                                               If None, uses ["lda", "bert", "nmf"]
                                               
    Returns:
        dict: Dictionary with analysis results for each model type
        
    Examples:
        >>> analysis = analyze_k_topic_relationship(100)
        >>> print(analysis['lda'][0]['estimated_topics'])
    """
    if topic_model_types is None:
        topic_model_types = ["lda", "bert", "nmf"]
    
    optimal_k = calculate_optimal_k(collection_size)
    
    if k_range is None:
        # Use ±50% around optimal k for analysis
        k_min = max(5, int(optimal_k * 0.5))
        k_max = min(collection_size, int(optimal_k * 1.5))
    else:
        k_min, k_max = k_range
        k_min = max(1, k_min)
        k_max = min(collection_size, k_max)
    
    # Generate k values to test
    k_values = []
    if k_max - k_min <= 20:
        k_values = list(range(k_min, k_max + 1))
    else:
        # For large ranges, sample more intelligently
        k_values = list(range(k_min, k_min + 10)) + \
                  list(range(k_min + 10, k_max - 10, 2)) + \
                  list(range(k_max - 10, k_max + 1))
        k_values = sorted(list(set(k_values)))  # Remove duplicates
    
    results = {}
    
    for model_type in topic_model_types:
        model_results = []
        
        for k in k_values:
            topic_analysis = estimate_topic_count(k, model_type)
            topic_analysis.update({
                'collection_size': collection_size,
                'optimal_k': optimal_k,
                'k_deviation_from_optimal': k - optimal_k,
                'k_relative_to_optimal': k / optimal_k if optimal_k > 0 else 0
            })
            model_results.append(topic_analysis)
        
        results[model_type] = model_results
    
    return results


def get_topic_modeling_recommendations(collection_size: int) -> Dict[str, Union[str, int, Dict]]:
    """
    Get comprehensive recommendations for topic modeling based on collection size.
    
    Args:
        collection_size (int): Total number of articles in collection
        
    Returns:
        dict: Comprehensive recommendations including optimal k, topic counts, and strategies
    """
    optimal_k = calculate_optimal_k(collection_size)
    strategy = get_selection_strategy(collection_size)
    
    # Get topic estimates for different models
    topic_estimates = {}
    for model_type in ["lda", "bert", "nmf"]:
        topic_estimates[model_type] = estimate_topic_count(optimal_k, model_type)
    
    # Determine recommended approach based on collection size and optimal k
    if collection_size <= 28:
        approach = "complete_analysis"
        description = "Analyze all articles with multiple topic modeling approaches"
        primary_model = "bert"  # More flexible for small datasets
    elif optimal_k <= 20:
        approach = "focused_analysis"
        description = "Use BERT-based models for better performance with limited articles"
        primary_model = "bert"
    elif optimal_k <= 50:
        approach = "balanced_analysis"
        description = "LDA and BERT models both viable, compare results"
        primary_model = "lda"
    else:
        approach = "comprehensive_analysis"
        description = "Sufficient articles for robust LDA modeling"
        primary_model = "lda"
    
    return {
        'collection_size': collection_size,
        'optimal_k': optimal_k,
        'selection_percentage': strategy['percentage_selected'],
        'recommended_approach': approach,
        'approach_description': description,
        'primary_model_recommendation': primary_model,
        'topic_estimates': topic_estimates,
        'expected_topics_range': f"{topic_estimates[primary_model]['min_topics']}-{topic_estimates[primary_model]['max_topics']}",
        'confidence_level': topic_estimates[primary_model]['confidence']
    }


if __name__ == "__main__":
    # Simple demonstration when run directly
    print("Optimal K Selection Utility")
    print("=" * 30)
    
    # Test with a few examples
    test_cases = [15, 50, 100, 500, 1000]
    
    for size in test_cases:
        k = calculate_optimal_k(size)
        strategy = get_selection_strategy(size)
        print(f"N={size:4d} → k={k:2d} ({strategy['percentage_selected']:.1f}%) - {strategy['strategy_type']}")
    
    print("\nUse this module by importing: from optimal_k import calculate_optimal_k")
