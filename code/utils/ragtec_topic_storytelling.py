"""
RAGTEC Topic Storytelling Analysis Module
Retrieval-Augmented Generation for Topic Extraction and Classification

This module provides advanced co-occurrence storytelling analysis for topics,
visualizing relationships between topics with enhanced node coloring and sizing
based on primary/secondary roles and mention frequencies.

Author: RAGTEC Framework
Version: 1.0.0
"""

import json
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set academic paper style for all plots
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'axes.edgecolor': 'black',
    'axes.grid': False,
    'grid.alpha': 0.3
})


class RAGTECTopicStorytelling:
    """
    RAGTEC Topic Storytelling class for co-occurrence storytelling analysis and visualization.
    
    This class analyzes topic relationships in classified articles and generates
    advanced storytelling visualizations with enhanced node coloring and sizing.
    """
    
    def __init__(self, articles_file: str = None, size_multiplier: float = 50.0):
        """
        Initialize the RAGTEC Topic Storytelling system.
        
        Args:
            articles_file (str, optional): Path to the classified articles JSON file
            size_multiplier (float): Multiplier for node sizes based on mentions (n * multiplier)
        """
        self.size_multiplier = size_multiplier
        
        # Analysis results storage
        self.articles = []
        self.topic_data = []
        self.cooc_matrix = None
        self.topic_names = []
        self.networks = {}
        self.topic_frequencies = {}
        self.topic_roles = {}  # primary vs secondary counts
        
        # Network analysis results
        self.network_metrics = {}
        self.centrality_measures = {}
        
        # Load and process data if file is provided
        if articles_file:
            self.load_classified_articles(articles_file)
        
    def load_classified_articles(self, articles_file: str) -> Dict[str, Any]:
        """
        Load classified articles from JSON file.
        
        Args:
            articles_file (str): Path to the classified articles JSON file
            
        Returns:
            Dict containing loading results and statistics
        """
        print(f"Loading classified articles from: {articles_file}")
        
        try:
            with open(articles_file, "r", encoding="utf-8") as f:
                self.articles = json.load(f)
            
            print(f"Successfully loaded {len(self.articles)} articles")
            
            # Extract topic information
            self._extract_topic_data()
            
            # Calculate topic roles (primary vs secondary)
            self._calculate_topic_roles()
            
            # Analyze co-occurrence network
            self.analyze_cooccurrence_network()
            
            return {
                "status": "success",
                "total_articles": len(self.articles),
                "unique_topics": len(self.topic_names),
                "articles_with_multiple_topics": sum(1 for topics in self._get_article_topics() if len(topics) > 1),
                "topic_roles": self.topic_roles
            }
            
        except Exception as e:
            print(f"Error loading articles: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _extract_topic_data(self):
        """Extract and organize topic data from loaded articles."""
        print("Extracting and organizing topic data...")
        self.topic_data = []
        article_topics = []
        primary_topics = []
        
        print(f"Processing {len(self.articles)} articles...")
        
        for i, article in enumerate(self.articles):
            url = article["url"]
            
            # Handle both direct format and nested response format
            if "response" in article and isinstance(article["response"], dict):
                # New RAGTEC classification format with nested response
                response_data = article["response"]
                primary_topic = response_data.get("primary_topic", "")
                topics_list = response_data.get("topics", [])
                coverage_assessment = response_data.get("coverage_assessment", {})
            else:
                # Direct format (legacy or other formats)
                primary_topic = article.get("primary_topic", "")
                topics_list = article.get("topics", [])
                coverage_assessment = article.get("coverage_assessment", {})
            
            primary_topics.append(primary_topic)
            
            # Get all assigned topics for this article
            assigned_topics = []
            if topics_list:
                for topic_obj in topics_list:
                    if "main_topic" in topic_obj:
                        assigned_topics.append(topic_obj["main_topic"])
            
            # If no topics in the topics array, use primary topic
            if not assigned_topics and primary_topic:
                assigned_topics = [primary_topic]
            
            # Debug first few articles
            if i < 3:
                print(f"  Article {i+1}: Primary='{primary_topic}', All Topics={assigned_topics}")
            
            article_topics.append(assigned_topics)
            
            # Store detailed information
            for topic in assigned_topics:
                self.topic_data.append({
                    "url": url,
                    "topic": topic,
                    "is_primary": topic == primary_topic,
                    "primary_topic": primary_topic,
                    "coverage_quality": coverage_assessment.get("coverage_quality", ""),
                    "confidence_score": coverage_assessment.get("confidence_score", "")
                })
        
        # Calculate topic frequencies (following co-occurence-topic.py pattern)
        self.topic_frequencies = {}
        for topics_list in article_topics:
            for topic in topics_list:
                # Filter out 'no_match' topics from visualizations
                if topic != "no_match":
                    self.topic_frequencies[topic] = self.topic_frequencies.get(topic, 0) + 1
        
        # Get unique topic names from the data
        self.topic_names = sorted(list(self.topic_frequencies.keys()))
        
        # Store for network analysis
        self._article_topics = article_topics
        
        print(f"Found {len(self.topic_names)} unique topics:")
        for topic in self.topic_names:
            print(f"  {topic}: {self.topic_frequencies[topic]} mentions")
    
    def _calculate_topic_roles(self):
        """Calculate primary vs secondary role statistics for each topic."""
        self.topic_roles = {}
        
        for topic in self.topic_names:
            topic_info = [item for item in self.topic_data if item["topic"] == topic]
            primary_count = sum(1 for item in topic_info if item["is_primary"])
            secondary_count = len(topic_info) - primary_count
            total_count = len(topic_info)
            
            self.topic_roles[topic] = {
                "primary_count": primary_count,
                "secondary_count": secondary_count,
                "total_count": total_count,
                "primary_ratio": primary_count / total_count if total_count > 0 else 0
            }
        
        # Calculate topic importance scores
        self._calculate_topic_importance()
    
    def _calculate_topic_importance(self) -> None:
        """
        Calculate topic importance as primary_mentions / all_mentions ratio.
        Higher ratio indicates topics that are more often primary (main focus) vs secondary.
        """
        self.topic_importance = {}
        total_all_mentions = sum(self.topic_frequencies.values())
        total_primary_mentions = sum(role_data["primary_count"] for role_data in self.topic_roles.values())
        
        for topic in self.topic_names:
            primary_count = self.topic_roles[topic]["primary_count"]
            all_mentions = self.topic_frequencies[topic]
            
            # Topic importance: how often this topic appears as primary vs all mentions
            importance_ratio = primary_count / all_mentions if all_mentions > 0 else 0
            
            # Normalized importance (relative to overall dataset pattern)
            overall_primary_ratio = total_primary_mentions / total_all_mentions if total_all_mentions > 0 else 0
            normalized_importance = importance_ratio / overall_primary_ratio if overall_primary_ratio > 0 else 1.0
            
            self.topic_importance[topic] = {
                "primary_count": primary_count,
                "all_mentions": all_mentions,
                "importance_ratio": importance_ratio,
                "normalized_importance": normalized_importance
            }
    
    def print_topic_importance_analysis(self) -> None:
        """Print detailed topic importance analysis."""
        print("\n" + "="*60)
        print("TOPIC IMPORTANCE ANALYSIS")
        print("="*60)
        print(f"Importance Ratio = Primary Count / All Mentions")
        print(f"(Higher ratio = more often primary topic vs secondary)")
        print("-"*60)
        
        # Sort by importance ratio
        sorted_topics = sorted(
            self.topic_importance.items(), 
            key=lambda x: x[1]["importance_ratio"], 
            reverse=True
        )
        
        total_articles = sum(data["primary_count"] for data in self.topic_importance.values())
        total_mentions = sum(data["all_mentions"] for data in self.topic_importance.values())
        
        print(f"Dataset Summary: {total_articles} articles, {total_mentions} total mentions")
        if total_mentions > 0:
            print(f"Overall Primary/Mention Ratio: {total_articles/total_mentions:.3f}")
        else:
            print(f"Overall Primary/Mention Ratio: No mentions found")
        print("-"*60)
        
        for topic, data in sorted_topics:
            primary = data["primary_count"]
            mentions = data["all_mentions"]
            ratio = data["importance_ratio"]
            norm_importance = data["normalized_importance"]
            
            print(f"{topic:20} | Primary: {primary:2d} | Mentions: {mentions:2d} | "
                  f"Ratio: {ratio:.3f} | Norm: {norm_importance:.2f}")
        
        print("="*60)
    
    def _get_article_topics(self) -> List[List[str]]:
        """Get list of topic lists for each article, excluding 'no_match' topics."""
        raw_topics = getattr(self, '_article_topics', [])
        # Filter out 'no_match' topics from each article's topic list
        return [[topic for topic in article_topics if topic != "no_match"] 
                for article_topics in raw_topics]
    
    def analyze_cooccurrence_network(self) -> Dict[str, Any]:
        """
        Analyze topic co-occurrence patterns and create network structures.
        
        Returns:
            Dict containing network analysis results
        """
        print("Analyzing topic co-occurrence networks...")
        
        article_topics = self._get_article_topics()
        
        # Only proceed if we have topics
        if not self.topic_names:
            print("No topics found, skipping network analysis")
            return {
                "cooccurrence_matrix": [],
                "topic_names": [],
                "networks": {},
                "primary_secondary_pairs": [],
                "topic_frequencies": {}
            }
        
        # Create co-occurrence matrix (don't overwrite self.topic_names)
        self.cooc_matrix, _ = self._create_cooccurrence_matrix(article_topics)
        
        # Create networks with different thresholds
        self.networks = {}
        for min_cooc in [1, 2, 3]:
            self.networks[min_cooc] = self._create_topic_network(self.cooc_matrix, self.topic_names, min_cooc)
        
        # Analyze primary-secondary relationships
        primary_secondary_pairs = self._analyze_primary_secondary_relationships()
        
        return {
            "cooccurrence_matrix": self.cooc_matrix.tolist() if self.cooc_matrix is not None else [],
            "topic_names": self.topic_names,
            "networks": {k: {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()} 
                        for k, G in self.networks.items()},
            "primary_secondary_pairs": primary_secondary_pairs[:10],  # Top 10
            "topic_frequencies": self.topic_frequencies
        }
    
    def _create_cooccurrence_matrix(self, article_topics_list: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
        """Create co-occurrence matrix for topics appearing in the same articles."""
        # Get all unique topics
        all_topics = set()
        for topics in article_topics_list:
            all_topics.update(topics)
        
        all_topics = sorted(list(all_topics))
        n_topics = len(all_topics)
        
        # Initialize co-occurrence matrix
        cooc_matrix = np.zeros((n_topics, n_topics))
        
        # Fill the matrix
        for topics in article_topics_list:
            if len(topics) > 1:  # Only consider articles with multiple topics
                # Get all pairs of topics in this article
                for i, topic1 in enumerate(all_topics):
                    for j, topic2 in enumerate(all_topics):
                        if topic1 in topics and topic2 in topics and i != j:
                            cooc_matrix[i][j] += 1
        
        return cooc_matrix, all_topics
    
    def _create_topic_network(self, cooc_matrix: np.ndarray, topic_names: List[str], min_cooccurrence: int = 1) -> nx.Graph:
        """Create a network graph from co-occurrence matrix."""
        G = nx.Graph()
        
        # Add nodes
        for topic in topic_names:
            G.add_node(topic)
        
        # Add edges based on co-occurrence
        n_topics = len(topic_names)
        for i in range(n_topics):
            for j in range(i+1, n_topics):  # Only upper triangle to avoid duplicates
                cooc_count = cooc_matrix[i][j]
                if cooc_count >= min_cooccurrence:
                    G.add_edge(topic_names[i], topic_names[j], weight=cooc_count)
        
        return G
    
    def _analyze_primary_secondary_relationships(self) -> List[Tuple[str, str, int]]:
        """Analyze primary-secondary topic relationships."""
        primary_secondary_pairs = []
        for article in self.articles:
            primary = article.get("primary_topic", "")
            if "topics" in article and article["topics"]:
                assigned_topics = [t.get("main_topic", "") for t in article["topics"] if "main_topic" in t]
                secondary_topics = [t for t in assigned_topics if t != primary and t != ""]
                
                for secondary in secondary_topics:
                    primary_secondary_pairs.append((primary, secondary))
        
        # Count relationships
        ps_counter = Counter(primary_secondary_pairs)
        return [(primary, secondary, count) for (primary, secondary), count in ps_counter.most_common()]
    
    def create_advanced_network_visualizations(self, output_dir: str, country: str = "Mozambique", collection_name: str = "News Articles") -> Dict[str, Any]:
        """
        Create simplified network visualizations with clean layouts.
        
        Args:
            output_dir (str): Directory to save visualizations
            country (str): Country name for file naming
            collection_name (str): Name of the data collection for titles
            
        Returns:
            Dict containing visualization results and saved file paths
        """
        print("Creating simplified storytelling visualizations...")
        
        # Print topic importance analysis
        self.print_topic_importance_analysis()
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # 1. Create simple network plot (no role-based coloring)
        if 1 in self.networks and self.networks[1].number_of_edges() > 0:
            network_file = self._create_simple_network_plot(
                f"{output_dir}/{country}_topic_network.png", collection_name
            )
            saved_files["network"] = network_file
        
        # 2. Create simple heatmap
        heatmap_file = self._create_simple_heatmap(
            f"{output_dir}/{country}_cooccurrence_heatmap.png", collection_name
        )
        saved_files["heatmap"] = heatmap_file
        
        # 3. Create pareto chart
        pareto_file = self._create_pareto_chart(
            f"{output_dir}/{country}_topic_pareto.png", collection_name
        )
        saved_files["pareto"] = pareto_file
        
        # 4. Create topic distribution by role
        role_dist_file = self._create_role_distribution_chart(
            f"{output_dir}/{country}_topic_roles.png", collection_name
        )
        saved_files["roles"] = role_dist_file
        
        return {
            "visualizations_created": len(saved_files),
            "saved_files": saved_files,
            "network_statistics": self._get_network_statistics()
        }
    
    def _create_simple_network_plot(self, save_path: str, collection_name: str = "News Articles") -> str:
        """Create a simple, clean network plot without role-based coloring."""
        G = self.networks[1]  # Use threshold 1 network
        
        # Optimize figure size based on network structure and node count
        if G.number_of_nodes() <= 6:
            figsize = (12, 9)
        elif G.number_of_nodes() <= 10:
            figsize = (14, 10) 
        else:
            figsize = (16, 12)
            
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'No network connections found', ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes)
            ax.set_title(f'Topic Storytelling Network of {collection_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            return save_path
        
        # Calculate layout with optimized spacing and positioning
        if G.number_of_nodes() <= 6:
            # Use circular layout for small networks with tighter spacing
            pos = nx.circular_layout(G, scale=1.8)
        elif G.number_of_nodes() <= 8:
            # Spring layout with reduced spacing parameter for better space utilization
            # pos = nx.spring_layout(G, k=3.5, iterations=200, seed=42)
            pos = nx.circular_layout(G, scale=1.5)
        elif G.number_of_nodes() <= 12:
            # Medium networks with moderate spacing
            pos = nx.spring_layout(G, k=2.8, iterations=150, seed=42)
        else:
            # Large networks with kamada-kawai for optimal positioning
            pos = nx.kamada_kawai_layout(G)
        
        # Node sizes based on frequency with optimized scaling for space efficiency
        max_mentions = max(self.topic_frequencies.values())
        min_mentions = min(self.topic_frequencies.values())
        
        # Calculate adaptive size range based on network size and available space
        if G.number_of_nodes() <= 6:
            base_size, max_size = 1200, 3000  # Larger nodes for small networks
        elif G.number_of_nodes() <= 10:
            base_size, max_size = 800, 2200   # Medium nodes for medium networks
        else:
            base_size, max_size = 600, 1800   # Smaller nodes for large networks
        
        node_sizes = []
        for node in G.nodes():
            mentions = self.topic_frequencies.get(node, 1)
            if max_mentions > min_mentions:
                normalized_size = (mentions - min_mentions) / (max_mentions - min_mentions)
                size = base_size + normalized_size * (max_size - base_size)
            else:
                size = (base_size + max_size) // 2  # Default to middle size
            node_sizes.append(size)
        
        # Optimize node positioning to fill available space better
        pos_values = np.array(list(pos.values()))
        if len(pos_values) > 0:
            # Calculate bounding box of current positions
            min_x, min_y = pos_values.min(axis=0)
            max_x, max_y = pos_values.max(axis=0)
            
            # Scale positions to better use available space (with padding for nodes)
            if max_x > min_x and max_y > min_y:
                # Target bounds for better space utilization
                target_width = 0.85   # Use 85% of available width
                target_height = 0.85  # Use 85% of available height
                
                # Scale and center positions
                current_width = max_x - min_x
                current_height = max_y - min_y
                
                scale_x = target_width / current_width if current_width > 0 else 1
                scale_y = target_height / current_height if current_height > 0 else 1
                
                # Use uniform scaling to maintain proportions
                scale = min(scale_x, scale_y)
                
                # Apply scaling and centering
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                
                for node in pos:
                    x, y = pos[node]
                    # Scale relative to center
                    pos[node] = (
                        (x - center_x) * scale,
                        (y - center_y) * scale
                    )
        
        # Color nodes based on primary/secondary ratio using gradient (darker = more important)
        import matplotlib.colors as mcolors
        
        node_colors = []
        ratios = []
        for node in G.nodes():
            role_info = self.topic_roles.get(node, {})
            primary_ratio = role_info.get("primary_ratio", 0)
            ratios.append(primary_ratio)
        
        # Create a color gradient from light (low importance) to dark (high importance)
        # Using blues: light blue for low ratios, dark blue for high ratios
        cmap = plt.cm.Blues
        norm = mcolors.Normalize(vmin=0, vmax=1)
        
        for ratio in ratios:
            # Ensure minimum visibility by mapping to range [0.3, 1.0] instead of [0, 1]
            adjusted_ratio = 0.3 + ratio * 0.7
            color = cmap(adjusted_ratio)
            node_colors.append(color)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                             alpha=0.8, linewidths=2, edgecolors='black', ax=ax)
        
        # Draw edges with varying thickness based on edge multiplier
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [self.edge_multiplier * (weight / max_weight) + 0.5 for weight in edge_weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                             edge_color='gray', ax=ax)
        
        # Draw labels with smaller font size to better fit larger nodes
        labels = {node: node.replace(' ', '\n') if len(node) > 12 else node 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold',
                              bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9),
                              ax=ax)
        
        # Add title and statistics with clearer subtitle
        ax.set_title(f'Topic Storytelling Network of {collection_name}\n(Node size = total mentions, Edge thickness = co-occurrence frequency, Color = importance ratio)', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Add network statistics
        stats_text = f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Density: {nx.density(G):.3f}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        # Add gradient colorbar legend
        from matplotlib.patches import Rectangle
        
        # Create a colorbar to show the gradient
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label('Primary Topic Ratio\n(Darker = More Important)', 
                      rotation=270, labelpad=20, fontsize=10)
        
        # Add discrete labels on the colorbar
        cbar.set_ticks([0, 0.2, 0.5, 0.8, 1.0])
        cbar.set_ticklabels(['0%\n(Secondary)', '20%', '50%\n(Balanced)', '80%', '100%\n(Primary)'])
        
        # Optimize axis limits to use space more efficiently
        if pos:
            pos_array = np.array(list(pos.values()))
            x_coords = pos_array[:, 0]
            y_coords = pos_array[:, 1]
            
            # Calculate bounds with minimal padding
            x_margin = (x_coords.max() - x_coords.min()) * 0.15
            y_margin = (y_coords.max() - y_coords.min()) * 0.15
            
            ax.set_xlim(x_coords.min() - x_margin, x_coords.max() + x_margin)
            ax.set_ylim(y_coords.min() - y_margin, y_coords.max() + y_margin)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Use tight layout with optimized padding
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.08)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.show()
        
        return save_path
    
    def _create_simple_heatmap(self, save_path: str, collection_name: str = "News Articles") -> str:
        """Create a simple, clean co-occurrence heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # Create co-occurrence matrix
        matrix = self.cooc_matrix  # Fixed: use cooc_matrix instead of cooccurrence_matrix
        topic_names = self.topic_names
        
        if matrix is None or len(topic_names) == 0:
            ax.text(0.5, 0.5, 'No co-occurrence data available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, style='italic', color='gray')
            ax.set_title(f'Topic Co-occurrence Matrix of {collection_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            return save_path
        
        # Create heatmap
        mask = np.triu(np.ones_like(matrix, dtype=bool))  # Show only lower triangle
        
        im = ax.imshow(matrix, cmap='YlOrRd', alpha=0.8)
        
        # Apply mask to show only lower triangle
        for i in range(len(topic_names)):
            for j in range(len(topic_names)):
                if i <= j:  # Upper triangle and diagonal
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                             fill=True, color='white', alpha=0.8))
        
        # Add text annotations for non-zero values
        for i in range(len(topic_names)):
            for j in range(len(topic_names)):
                if i > j and matrix[i, j] > 0:  # Only lower triangle with values
                    text = ax.text(j, i, f'{int(matrix[i, j])}', 
                                 ha='center', va='center', fontweight='bold',
                                 fontsize=10, color='black')
        
        # Set labels
        ax.set_xticks(range(len(topic_names)))
        ax.set_yticks(range(len(topic_names)))
        
        # Format labels with smart word-wrapping instead of truncation
        wrapped_names = []
        for name in topic_names:
            if len(name) <= 12:
                wrapped_names.append(name)
            else:
                # Smart word wrapping for longer names
                words = name.split()
                if len(words) == 1:
                    # Single long word - break at reasonable point
                    wrapped_names.append(name[:8] + '\n' + name[8:])
                else:
                    # Multiple words - wrap at word boundaries
                    line1, line2 = '', ''
                    for word in words:
                        if len(line1) + len(word) <= 10:
                            line1 += word + ' '
                        else:
                            line2 += word + ' '
                    wrapped_names.append(line1.strip() + '\n' + line2.strip())
        
        ax.set_xticklabels(wrapped_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(wrapped_names, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Co-occurrence Frequency', fontsize=12)
        
        # Add title
        max_cooccur = int(np.max(matrix))
        total_connections = int(np.sum(matrix[np.triu_indices_from(matrix, k=1)]))
        
        ax.set_title(f'Topic Co-occurrence Matrix of {collection_name}\n(Lower triangle shows frequencies)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add statistics
        ax.text(0.02, 0.98, f'Max co-occurrence: {max_cooccur}\nTotal connections: {total_connections}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.show()
        
        return save_path
    
    def _create_pareto_chart(self, save_path: str, collection_name: str = "News Articles") -> str:
        """Create a Pareto chart showing topic frequency distribution."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor='white')  # Increased width from 12 to 16
        
        # Prepare data
        topics_sorted = sorted(self.topic_frequencies.items(), key=lambda x: x[1], reverse=True)
        topic_names = [item[0] for item in topics_sorted]
        topic_counts = [item[1] for item in topics_sorted]
        
        if not topic_counts:
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'No topic frequency data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=16, style='italic', color='gray')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            return save_path
        
        # Top plot: Bar chart with purple palette (distinct from other visualizations)
        bars = ax1.bar(range(len(topic_names)), topic_counts, 
                      color='#7C3AED', alpha=0.8, edgecolor='black', linewidth=1)  # Changed to purple
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, topic_counts)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(topic_counts) * 0.01,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Topics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency (Number of Articles)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Topic Storytelling Frequency Distribution of {collection_name}\n(Bar height = total mentions per topic)', fontsize=14, fontweight='bold', pad=20)
        
        # Format x-axis labels with better wrapping instead of truncation
        wrapped_names = []
        for name in topic_names:
            if len(name) > 12:
                # Split long names at natural break points
                words = name.split(' ')
                if len(words) >= 2:
                    mid = len(words) // 2
                    wrapped_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                else:
                    wrapped_name = name  # Keep full name if single word
                wrapped_names.append(wrapped_name)
            else:
                wrapped_names.append(name)
        
        ax1.set_xticks(range(len(topic_names)))
        ax1.set_xticklabels(wrapped_names, rotation=0, ha='center', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Bottom plot: Cumulative percentage (Pareto)
        total = sum(topic_counts)
        cumulative = np.cumsum(topic_counts)
        cumulative_pct = (cumulative / total) * 100
        
        bars2 = ax2.bar(range(len(topic_names)), 
                       [(count / total) * 100 for count in topic_counts], 
                       color='#4F46E5', alpha=0.8, edgecolor='black', linewidth=1)  # Indigo bars for better contrast
        
        # Add cumulative line with distinct orange color for classic Pareto appearance
        ax2_twin = ax2.twinx()
        line = ax2_twin.plot(range(len(topic_names)), cumulative_pct, 
                           color='#EA580C', linewidth=3, marker='o', markersize=8,  # Orange for cumulative line
                           markerfacecolor='white', markeredgewidth=2, markeredgecolor='#EA580C',
                           label='Cumulative %')
        
        # Add 80% reference line with matching orange color
        ax2_twin.axhline(y=80, color='#EA580C', linestyle='--', alpha=0.8, linewidth=2)
        pareto_80_idx = next((i for i, cum in enumerate(cumulative_pct) if cum >= 80), len(cumulative_pct)-1)
        ax2_twin.text(pareto_80_idx + 0.2, 82, f'80% at topic #{pareto_80_idx + 1}', 
                     fontsize=10, color='#EA580C', fontweight='bold')
        
        ax2.set_xlabel('Topics (Ranked by Frequency)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Individual Contribution (%)', fontsize=12, fontweight='bold', color='#4F46E5')  # Match bar color
        ax2_twin.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold', color='#EA580C')  # Match line color
        ax2.set_title(f'Pareto Analysis of {collection_name} (80-20 Rule)', fontsize=14, fontweight='bold', pad=20)
        
        ax2.set_xticks(range(len(topic_names)))
        ax2.set_xticklabels([f'{i+1}' for i in range(len(topic_names))])
        ax2.grid(axis='y', alpha=0.3)
        ax2_twin.set_ylim(0, 105)
        
        # Add legend
        ax2_twin.legend(loc='center right')
        
        # Add statistics
        total_articles = sum(role_data["primary_count"] for role_data in self.topic_roles.values())
        stats_text = f'Topics: {len(topic_names)} | Total mentions: {total} | Total articles: {total_articles} | Top mentions: {max(topic_counts)} ({(max(topic_counts)/total)*100:.1f}%)'
        fig.text(0.02, 0.02, stats_text, fontsize=10, style='italic', color='#333333')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Increased from 0.08 to 0.12 for better label space
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)  # Increased padding
        plt.show()
        
        return save_path
    
    def _create_role_distribution_chart(self, save_path: str, collection_name: str = "News Articles") -> str:
        """Create topic distribution charts grouped by primary/secondary roles."""
        # Optimize figure size based on number of topics
        if not self.topic_roles:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No topic role data available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, style='italic', color='gray')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            return save_path
            
        num_topics = len(self.topic_roles)
        if num_topics <= 6:
            figsize = (14, 8)
        elif num_topics <= 10:
            figsize = (16, 9)  
        else:
            figsize = (18, 10)
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        fig.subplots_adjust(wspace=0.3, left=0.07, right=0.95, top=0.88, bottom=0.15)
        
        # Prepare data
        topics_sorted = sorted(self.topic_roles.items(), key=lambda x: x[1]['total_count'], reverse=True)
        topic_names = [item[0] for item in topics_sorted]
        primary_counts = [item[1]['primary_count'] for item in topics_sorted]
        secondary_counts = [item[1]['secondary_count'] for item in topics_sorted]
        
        # Left plot: Primary vs Secondary counts
        x_pos = np.arange(len(topic_names))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, primary_counts, width, 
                       label='Primary Topic', color='#2E86AB', alpha=0.8, 
                       edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x_pos + width/2, secondary_counts, width,
                       label='Secondary Topic', color='#F18F01', alpha=0.8, 
                       edgecolor='black', linewidth=1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=9)
        
        ax1.set_xlabel('Topics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
        ax1.set_title(f'Primary vs Secondary Topic Distribution of {collection_name}', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        
        # Format labels with smart word-wrapping instead of truncation
        wrapped_names = []
        for name in topic_names:
            if len(name) <= 12:
                wrapped_names.append(name)
            else:
                # Smart word wrapping for longer names
                words = name.split()
                if len(words) == 1:
                    # Single long word - break at reasonable point
                    wrapped_names.append(name[:8] + '\n' + name[8:])
                else:
                    # Multiple words - wrap at word boundaries
                    line1, line2 = '', ''
                    for word in words:
                        if len(line1) + len(word) <= 10:
                            line1 += word + ' '
                        else:
                            line2 += word + ' '
                    wrapped_names.append(line1.strip() + '\n' + line2.strip())
        
        ax1.set_xticklabels(wrapped_names, ha='center', fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Right plot: Topic Importance Ratio (Primary/Total) - Much more insightful than redundant totals
        importance_ratios = []
        for item in topics_sorted:
            topic_name = item[0]
            if topic_name in self.topic_importance:
                importance_ratios.append(self.topic_importance[topic_name]["importance_ratio"])
            else:
                # Fallback calculation if not in topic_importance
                primary = item[1]["primary_count"]
                total = item[1]["total_count"]
                ratio = primary / total if total > 0 else 0
                importance_ratios.append(ratio)
        
        # Color bars based on importance ratio for better visual interpretation
        colors = []
        for ratio in importance_ratios:
            if ratio > 0.8:
                colors.append('#2E3B4E')  # Dark blue for highly important (mostly primary)
            elif ratio > 0.5:
                colors.append('#4A5568')  # Medium blue for moderately important  
            elif ratio > 0.2:
                colors.append('#718096')  # Light blue for contextual topics
            else:
                colors.append('#A0AEC0')  # Very light for secondary-only topics
        
        bars3 = ax2.bar(x_pos, importance_ratios, color=colors, 
                       edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add percentage labels on bars
        for i, (bar, ratio) in enumerate(zip(bars3, importance_ratios)):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{ratio:.1%}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
        
        ax2.set_xlabel('Topics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Topic Importance Ratio', fontsize=12, fontweight='bold')
        ax2.set_title(f'Topic Importance in {collection_name}\n(Primary Mentions ÷ Total Mentions)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(wrapped_names, ha='center', fontsize=9)  # Use same wrapped names
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add reference lines for interpretation
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax2.text(len(topic_names)*0.7, 0.82, 'Highly Important (80%+)', fontsize=9, color='green', alpha=0.8)
        
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax2.text(len(topic_names)*0.7, 0.52, 'Balanced (50%+)', fontsize=9, color='orange', alpha=0.8)
        
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax2.text(len(topic_names)*0.7, 0.22, 'Contextual (20%+)', fontsize=9, color='red', alpha=0.8)
        
        # Add overall statistics
        total_mentions = sum(p + s for p, s in zip(primary_counts, secondary_counts))  # Total topic mentions (not articles)
        total_articles = sum(primary_counts)  # Total articles (primary topics only)
        avg_importance = sum(importance_ratios) / len(importance_ratios) if importance_ratios else 0
        fig.suptitle(f'Topic Analysis: {len(topic_names)} topics, {total_articles} articles, {total_mentions} mentions | Avg Importance: {avg_importance:.1%}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.show()
        
        return save_path
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'No network connections found', ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.show()
            return save_path
        
        # Calculate node sizes based on mentions with multiplier (larger range)
        max_mentions = max(self.topic_frequencies.values())
        min_mentions = min(self.topic_frequencies.values())
        
        node_sizes = []
        for node in G.nodes():
            mentions = self.topic_frequencies.get(node, 1)
            # Use larger size range: 300-2000 instead of simple multiplier
            if max_mentions > min_mentions:
                normalized_size = (mentions - min_mentions) / (max_mentions - min_mentions)
                size = 300 + normalized_size * 1700  # Size range: 300-2000
            else:
                size = 800  # Default size if all nodes have same frequency
            node_sizes.append(size)
        
        # Calculate node colors based on primary/secondary ratio with better contrast
        node_colors = []
        edge_colors = []
        for node in G.nodes():
            role_info = self.topic_roles.get(node, {})
            primary_ratio = role_info.get("primary_ratio", 0)
            
            if primary_ratio >= 0.8:
                node_colors.append('#2E86AB')  # Strong blue for primary-dominant
                edge_colors.append('#1F5F7A')
            elif primary_ratio >= 0.5:
                node_colors.append('#A23B72')  # Purple for balanced topics
                edge_colors.append('#7A2B54')
            elif primary_ratio >= 0.2:
                node_colors.append('#F18F01')  # Orange for secondary-leaning
                edge_colors.append('#C47201')
            else:
                node_colors.append('#C73E1D')  # Red for secondary-dominant
                edge_colors.append('#9A2F16')
        
        # Calculate edge widths based on co-occurrence strength (better scaling)
        if G.number_of_edges() > 0:
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(weights)
            min_weight = min(weights)
            
            edge_widths = []
            for u, v in G.edges():
                weight = G[u][v]['weight']
                if max_weight > min_weight:
                    normalized_weight = (weight - min_weight) / (max_weight - min_weight)
                    width = 1.0 + normalized_weight * self.edge_multiplier  # Use edge_multiplier
                else:
                    width = self.edge_multiplier / 2  # Use half multiplier as default
                edge_widths.append(width)
        else:
            edge_widths = []
        
        # Use optimized layout algorithm with better space utilization
        if len(G.nodes()) <= 4:
            pos = nx.circular_layout(G, scale=1.2)
        elif len(G.nodes()) <= 8:
            pos = nx.spring_layout(G, k=3.0, iterations=200, seed=42)
        elif len(G.nodes()) <= 12:
            pos = nx.spring_layout(G, k=2.5, iterations=150, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)
            
        # Optimize positions to better fill available space
        if pos:
            pos_array = np.array(list(pos.values()))
            if len(pos_array) > 0:
                pos_min = pos_array.min(axis=0)
                pos_max = pos_array.max(axis=0)
                pos_center = (pos_min + pos_max) / 2
                pos_range = pos_max - pos_min
                
                # Scale to use more space efficiently
                target_scale = 0.75
                if max(pos_range) > 0:
                    scale_factor = target_scale / max(pos_range)
                    for node in pos:
                        pos[node] = (pos[node] - pos_center) * scale_factor
        
        # Draw edges first (so they appear behind nodes)
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                                  edge_color='#666666', ax=ax, style='-')
        
        # Draw nodes with enhanced styling
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.9, 
                              edgecolors='black', linewidths=2.0, ax=ax)
        
        # Enhanced labels with better contrast and positioning
        labels = {}
        for node in G.nodes():
            # Smart label formatting
            label = node
            if len(label) > 20:
                words = label.split()
                if len(words) > 2:
                    # Multi-line for long labels
                    mid = len(words) // 2
                    label = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                elif len(words) == 2:
                    label = words[0] + '\n' + words[1]
            labels[node] = label
        
        # Draw labels with white background for better readability
        for node, (x, y) in pos.items():
            ax.text(x, y, labels[node], ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Add edge labels only for small networks and strong connections
        if G.number_of_edges() > 0 and G.number_of_edges() <= 15:
            strong_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] >= max(weights) * 0.6]
            if strong_edges:
                edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in strong_edges}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, 
                                            bbox=dict(boxstyle='round,pad=0.2', 
                                                     facecolor='white', 
                                                     edgecolor='gray', 
                                                     alpha=0.9), ax=ax)
        
        # Create enhanced legend with size reference
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
                      markersize=15, label='Primary-dominant (≥80%)', markeredgecolor='black', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A23B72', 
                      markersize=15, label='Balanced (50-80%)', markeredgecolor='black', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F18F01', 
                      markersize=15, label='Secondary-leaning (20-50%)', markeredgecolor='black', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#C73E1D', 
                      markersize=15, label='Secondary-dominant (<20%)', markeredgecolor='black', markeredgewidth=2)
        ]
        
        # Add size reference to legend
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='gray', markersize=20, 
                                        label=f'Size ∝ mentions (max: {max_mentions})', 
                                        markeredgecolor='black', markeredgewidth=2))
        
        legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                          frameon=True, fancybox=True, shadow=True, fontsize=12,
                          title="Topic Role Distribution & Node Size", title_fontsize=13)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        
        # Enhanced title and subtitle
        ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
        
        # Add network statistics as subtitle
        stats_text = f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}"
        if G.number_of_edges() > 0:
            density = nx.density(G)
            stats_text += f" | Density: {density:.3f}"
        
        ax.text(0.5, 0.02, stats_text, ha='center', va='bottom', transform=ax.transAxes,
               fontsize=12, style='italic', color='#333333')
        
        ax.axis('off')
        ax.set_aspect('equal')
        
        # Optimize axis limits for better space utilization
        if pos:
            pos_array = np.array(list(pos.values()))
            if len(pos_array) > 0:
                x_coords = pos_array[:, 0]
                y_coords = pos_array[:, 1]
                
                # Minimal margins for better space usage
                x_margin = max(0.1, (x_coords.max() - x_coords.min()) * 0.1)
                y_margin = max(0.1, (y_coords.max() - y_coords.min()) * 0.1)
                
                ax.set_xlim(x_coords.min() - x_margin, x_coords.max() + x_margin)
                ax.set_ylim(y_coords.min() - y_margin, y_coords.max() + y_margin)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout(pad=1.5)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.15)
        plt.show()
        
        return save_path
    
    def _create_cooccurrence_heatmap(self, save_path: str) -> str:
        """Create academic-style co-occurrence heatmap."""
        # Calculate optimal figure size based on number of topics
        n_topics = len(self.topic_names)
        fig_width = max(10, min(16, n_topics * 1.2))
        fig_height = max(8, min(14, n_topics * 1.0))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
        
        # Create DataFrame from co-occurrence matrix
        cooc_df = pd.DataFrame(self.cooc_matrix, index=self.topic_names, columns=self.topic_names)
        
        # Create mask for upper triangle to avoid redundancy
        mask = np.zeros_like(cooc_df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Use better color palette with more contrast
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        
        # Create heatmap with enhanced styling
        heatmap = sns.heatmap(cooc_df, 
                             annot=True, 
                             cmap=cmap, 
                             fmt='g',
                             cbar_kws={
                                 'label': 'Co-occurrence Frequency', 
                                 'shrink': 0.8,
                                 'aspect': 20
                             },
                             mask=mask, 
                             square=True, 
                             linewidths=1.0,
                             linecolor='white',
                             annot_kws={
                                 'size': max(8, min(12, 100//n_topics)), 
                                 'weight': 'bold',
                                 'color': 'black'
                             }, 
                             ax=ax)
        
        # Enhance title and labels
        ax.set_title('Topic Co-occurrence Matrix\n(Lower triangle shows co-occurrence frequencies)', 
                    fontsize=16, fontweight='bold', pad=25)
        ax.set_xlabel('Topics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Topics', fontsize=14, fontweight='bold')
        
        # Improve tick labels
        ax.set_xticklabels([label.get_text().replace(' ', '\n') if len(label.get_text()) > 15 
                           else label.get_text() for label in ax.get_xticklabels()], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels([label.get_text().replace(' ', '\n') if len(label.get_text()) > 15 
                           else label.get_text() for label in ax.get_yticklabels()], 
                          rotation=0, ha='right', fontsize=10)
        
        # Add colorbar enhancement
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Co-occurrence Frequency', fontsize=12, fontweight='bold')
        
        # Add statistics annotation
        max_cooc = np.max(cooc_df.values)
        total_cooc = np.sum(cooc_df.values) // 2  # Divide by 2 since matrix is symmetric
        stats_text = f"Max co-occurrence: {max_cooc:.0f} | Total connections: {total_cooc:.0f}"
        ax.text(0.5, -0.15, stats_text, ha='center', va='top', transform=ax.transAxes,
               fontsize=11, style='italic', color='#333333')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        plt.show()
        
        return save_path
    
    def _create_topic_role_charts(self, save_path: str) -> str:
        """Create topic role distribution charts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')
        
        # Prepare data
        topics_sorted = sorted(self.topic_roles.items(), key=lambda x: x[1]['total_count'], reverse=True)
        topic_names = [item[0] for item in topics_sorted]
        primary_counts = [item[1]['primary_count'] for item in topics_sorted]
        secondary_counts = [item[1]['secondary_count'] for item in topics_sorted]
        primary_ratios = [item[1]['primary_ratio'] for item in topics_sorted]
        
        # Enhanced color palette
        primary_color = '#2E86AB'
        secondary_color = '#F18F01'
        
        # Left plot: Stacked bar chart (Primary vs Secondary) with better styling
        x_pos = np.arange(len(topic_names))
        width = 0.8
        
        bars1 = ax1.bar(x_pos, primary_counts, width, color=primary_color, alpha=0.85, 
                        label='Primary Topic', edgecolor='black', linewidth=1.2)
        bars2 = ax1.bar(x_pos, secondary_counts, width, bottom=primary_counts, 
                        color=secondary_color, alpha=0.85,
                        label='Secondary Topic', edgecolor='black', linewidth=1.2)
        
        ax1.set_xlabel('Topics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Frequency (Number of Articles)', fontsize=14, fontweight='bold')
        ax1.set_title('Primary vs Secondary Topic Distribution', fontsize=16, fontweight='bold', pad=25)
        ax1.set_xticks(x_pos)
        
        # Smart label formatting for x-axis
        formatted_labels = []
        for name in topic_names:
            if len(name) > 15:
                words = name.split()
                if len(words) > 2:
                    mid = len(words) // 2
                    formatted_labels.append(' '.join(words[:mid]) + '\n' + ' '.join(words[mid:]))
                else:
                    formatted_labels.append(name.replace(' ', '\n'))
            else:
                formatted_labels.append(name)
        
        ax1.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=10)
        
        # Enhanced value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Primary count label
            height1 = bar1.get_height()
            if height1 > 0:
                ax1.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                        f'{int(height1)}', ha='center', va='center', 
                        fontweight='bold', fontsize=11, color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            # Secondary count label
            height2 = bar2.get_height()
            if height2 > 0:
                ax1.text(bar2.get_x() + bar2.get_width()/2., 
                        height1 + height2/2,
                        f'{int(height2)}', ha='center', va='center', 
                        fontweight='bold', fontsize=11, color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Enhanced legend
        legend1 = ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                            fontsize=12, title='Topic Role', title_fontsize=13)
        legend1.get_frame().set_facecolor('white')
        legend1.get_frame().set_alpha(0.95)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.set_axisbelow(True)
        
        # Right plot: Primary dominance ratios with gradient coloring
        colors = []
        for ratio in primary_ratios:
            if ratio >= 0.8:
                colors.append('#2E86AB')  # Strong blue
            elif ratio >= 0.6:
                colors.append('#5B9BD5')  # Medium blue
            elif ratio >= 0.4:
                colors.append('#A23B72')  # Purple
            elif ratio >= 0.2:
                colors.append('#F18F01')  # Orange
            else:
                colors.append('#C73E1D')  # Red
        
        bars3 = ax2.bar(x_pos, primary_ratios, width, color=colors, 
                        edgecolor='black', linewidth=1.2, alpha=0.85)
        
        ax2.set_xlabel('Topics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Primary Topic Ratio', fontsize=14, fontweight='bold')
        ax2.set_title('Topic Primary Dominance Ratio', fontsize=16, fontweight='bold', pad=25)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=10)
        ax2.set_ylim(0, 1.1)
        
        # Enhanced percentage labels
        for i, (bar, ratio) in enumerate(zip(bars3, primary_ratios)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                    f'{ratio:.1%}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11, color='black')
        
        # Add reference lines with labels
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.axhline(y=0.8, color='blue', linestyle='--', alpha=0.6, linewidth=1)
        
        ax2.text(len(topic_names)-0.3, 0.52, '50% balanced', fontsize=10, color='red', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax2.text(len(topic_names)-0.3, 0.82, '80% primary-dominant', fontsize=10, color='blue',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.set_axisbelow(True)
        
        # Add overall statistics
        total_articles = sum(item[1]['total_count'] for item in topics_sorted)
        avg_primary_ratio = np.mean(primary_ratios)
        
        fig.suptitle(f'Topic Role Analysis: {len(topic_names)} topics, {total_articles} total mentions, {avg_primary_ratio:.1%} avg primary ratio', 
                    fontsize=14, y=0.02, style='italic', color='#333333')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        plt.show()
        
        return save_path
    
    def _create_frequency_distribution_chart(self, save_path: str) -> str:
        """Create enhanced topic frequency distribution chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), facecolor='white')
        
        topics_sorted = sorted(self.topic_frequencies.items(), key=lambda x: x[1], reverse=True)
        topic_names = [item[0] for item in topics_sorted]
        topic_counts = [item[1] for item in topics_sorted]
        
        # Enhanced color gradient based on frequency with distinct color ranges
        max_count = max(topic_counts)
        min_count = min(topic_counts)
        
        colors = []
        for count in topic_counts:
            if count == max_count:
                colors.append('#2E86AB')  # Highest - dark blue
            elif count > max_count * 0.8:
                colors.append('#A23B72')  # Very high - purple
            elif count > max_count * 0.6:
                colors.append('#F18F01')  # High - orange
            elif count > max_count * 0.4:
                colors.append('#40916C')  # Medium - green
            elif count > max_count * 0.2:
                colors.append('#7209B7')  # Low - violet
            else:
                colors.append('#C73E1D')  # Lowest - red
        
        # Top plot: Main frequency distribution
        bars = ax1.bar(range(len(topic_names)), topic_counts, color=colors, 
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        
        ax1.set_xlabel('Topics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Frequency (Number of Articles)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Topic Distribution in News Articles\n(Network node sizes = frequency × {self.size_multiplier:.1f})', 
                    fontsize=16, fontweight='bold', pad=25)
        
        ax1.set_xticks(range(len(topic_names)))
        
        # Smart label formatting
        formatted_labels = []
        for name in topic_names:
            if len(name) > 15:
                words = name.split()
                if len(words) > 2:
                    mid = len(words) // 2
                    formatted_labels.append(' '.join(words[:mid]) + '\n' + ' '.join(words[mid:]))
                else:
                    formatted_labels.append(name.replace(' ', '\n'))
            else:
                formatted_labels.append(name)
        
        ax1.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=11)
        
        # Enhanced value labels on bars
        for i, (bar, count) in enumerate(zip(bars, topic_counts)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max_count * 0.01,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=12, color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Add percentage labels
            percentage = (count / sum(topic_counts)) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{percentage:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=10, color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Add frequency tier lines
        for tier, label, color in [(0.8, '80% tier', '#2E86AB'), 
                                  (0.6, '60% tier', '#A23B72'),
                                  (0.4, '40% tier', '#F18F01'),
                                  (0.2, '20% tier', '#40916C')]:
            y_val = max_count * tier
            ax1.axhline(y=y_val, color=color, linestyle='--', alpha=0.6, linewidth=1)
            ax1.text(len(topic_names)-0.2, y_val + max_count * 0.02, label, 
                    fontsize=9, color=color, fontweight='bold')
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.set_axisbelow(True)
        
        # Bottom plot: Cumulative percentage
        cumulative_counts = np.cumsum(topic_counts)
        cumulative_percentages = (cumulative_counts / sum(topic_counts)) * 100
        
        # Bar chart for individual contributions
        bars2 = ax2.bar(range(len(topic_names)), 
                       [(count / sum(topic_counts)) * 100 for count in topic_counts], 
                       color=colors, alpha=0.6, edgecolor='black', linewidth=1)
        
        # Line plot for cumulative percentage
        ax2_twin = ax2.twinx()
        line = ax2_twin.plot(range(len(topic_names)), cumulative_percentages, 
                           color='red', linewidth=3, marker='o', markersize=6,
                           markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                           label='Cumulative %')
        
        ax2.set_xlabel('Topics (Ranked by Frequency)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Individual Contribution (%)', fontsize=12, fontweight='bold', color='blue')
        ax2_twin.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold', color='red')
        
        ax2.set_title('Topic Contribution Analysis (Pareto Chart)', fontsize=14, fontweight='bold', pad=20)
        
        # Add Pareto markers (80-20 rule)
        pareto_80_idx = next((i for i, cum in enumerate(cumulative_percentages) if cum >= 80), len(cumulative_percentages)-1)
        ax2_twin.axhline(y=80, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2_twin.axvline(x=pareto_80_idx, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2_twin.text(pareto_80_idx + 0.5, 82, f'80% reached at\ntopic #{pareto_80_idx + 1}', 
                     fontsize=10, color='red', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        ax2.set_xticks(range(len(topic_names)))
        ax2.set_xticklabels([f'{i+1}' for i in range(len(topic_names))], fontsize=10)
        ax2.set_ylim(0, max([(count / sum(topic_counts)) * 100 for count in topic_counts]) * 1.1)
        ax2_twin.set_ylim(0, 105)
        
        ax2.spines['top'].set_visible(False)
        ax2_twin.spines['top'].set_visible(False)
        ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.set_axisbelow(True)
        
        # Add legend for cumulative line
        ax2_twin.legend(loc='center right', frameon=True, fancybox=True, shadow=True)
        
        # Add statistics summary
        total_articles = sum(topic_counts)
        unique_topics = len(topic_names)
        avg_frequency = total_articles / unique_topics
        median_frequency = np.median(topic_counts)
        
        stats_text = f'Statistics: {unique_topics} topics, {total_articles} total mentions\nAvg: {avg_frequency:.1f}, Median: {median_frequency:.1f}, Top topic: {max_count} ({(max_count/total_articles)*100:.1f}%)'
        fig.text(0.02, 0.02, stats_text, fontsize=11, style='italic', color='#333333',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        plt.show()
        
        return save_path
    
    def _create_threshold_networks(self, output_dir: str, country: str) -> Dict[str, str]:
        """Create enhanced network visualizations for different co-occurrence thresholds."""
        saved_files = {}
        
        # Create a comparison plot for all thresholds - optimized layout
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='white')  # Slightly smaller height
        fig.subplots_adjust(wspace=0.25, left=0.05, right=0.95, top=0.88, bottom=0.12)
        
        thresholds = [1, 2, 3]
        for i, threshold in enumerate(thresholds):
            ax = axes[i]
            
            if threshold in self.networks and self.networks[threshold].number_of_edges() > 0:
                G = self.networks[threshold]
                
                # Calculate layout - optimized for space utilization
                if G.number_of_nodes() <= 4:
                    pos = nx.circular_layout(G, scale=1.5)
                elif G.number_of_nodes() <= 8:
                    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
                else:
                    pos = nx.kamada_kawai_layout(G)
                
                # Optimize positions to fill subplot area better
                if pos:
                    pos_array = np.array(list(pos.values()))
                    pos_min = pos_array.min(axis=0)
                    pos_max = pos_array.max(axis=0)
                    pos_range = pos_max - pos_min
                    
                    # Scale to use 80% of available space in each subplot
                    target_scale = 0.8
                    if pos_range[0] > 0 and pos_range[1] > 0:
                        scale_factor = target_scale / max(pos_range)
                        center = (pos_min + pos_max) / 2
                        
                        for node in pos:
                            pos[node] = (pos[node] - center) * scale_factor
                
                # Node sizes and colors (same as main network plot)
                max_mentions = max(self.topic_frequencies.values())
                min_mentions = min(self.topic_frequencies.values())
                
                node_sizes = []
                node_colors = []
                for node in G.nodes():
                    mentions = self.topic_frequencies.get(node, 1)
                    # Size calculation
                    if max_mentions > min_mentions:
                        normalized_size = (mentions - min_mentions) / (max_mentions - min_mentions)
                        size = 200 + normalized_size * 800  # Smaller for comparison plot
                    else:
                        size = 400
                    node_sizes.append(size)
                    
                    # Color based on role
                    primary_ratio = self.topic_roles.get(node, {}).get('primary_ratio', 0.5)
                    if primary_ratio >= 0.8:
                        node_colors.append('#2E86AB')  # Primary dominant - blue
                    elif primary_ratio >= 0.6:
                        node_colors.append('#A23B72')  # Mostly primary - purple
                    elif primary_ratio >= 0.4:
                        node_colors.append('#F18F01')  # Balanced - orange
                    else:
                        node_colors.append('#C73E1D')  # Secondary dominant - red
                
                # Draw network
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                     alpha=0.9, linewidths=2, edgecolors='black', ax=ax)
                
                # Draw edges with varying thickness based on edge multiplier
                edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                max_weight = max(edge_weights) if edge_weights else 1
                edge_widths = [self.edge_multiplier * (weight / max_weight) + 0.5 for weight in edge_weights]
                
                nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                                     edge_color='gray', ax=ax)
                
                # Draw labels
                labels = {node: node.replace(' ', '\n') if len(node) > 10 else node 
                         for node in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                                      ax=ax)
                
                # Statistics for this threshold
                stats_text = f"Nodes: {G.number_of_nodes()}\nEdges: {G.number_of_edges()}\nDensity: {nx.density(G):.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='lightblue', alpha=0.8))
                
            else:
                ax.text(0.5, 0.5, f'No connections\nat threshold {threshold}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, style='italic', color='gray')
            
            ax.set_title(f'Threshold ≥ {threshold}', fontsize=14, fontweight='bold', pad=15)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Add overall title and legend
        fig.suptitle(f'Topic Co-occurrence Networks at Different Thresholds\n(Node size = mentions × {self.size_multiplier:.1f}, Color = role dominance)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add color legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
                      markersize=12, label='Primary dominant (≥80%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A23B72', 
                      markersize=12, label='Mostly primary (60-80%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F18F01', 
                      markersize=12, label='Balanced (40-60%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#C73E1D', 
                      markersize=12, label='Secondary dominant (<40%)')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
                  frameon=True, fancybox=True, shadow=True, fontsize=11,
                  bbox_to_anchor=(0.5, 0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.85)
        
        comparison_path = f"{output_dir}/{country}_network_threshold_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        plt.show()
        
        saved_files["network_comparison"] = comparison_path
        
        # Create individual detailed network plots
        for threshold in thresholds:
            if threshold in self.networks and self.networks[threshold].number_of_edges() > 0:
                title = f"Topic Co-occurrence Network (minimum threshold = {threshold})\n(Node size = mentions × {self.size_multiplier:.1f})"
                save_path = f"{output_dir}/{country}_topic_network_threshold_{threshold}.png"
                
                self._create_enhanced_network_plot(self.networks[threshold], save_path, title)
                saved_files[f"network_threshold_{threshold}"] = save_path
        
        return saved_files
    
    def _get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        stats = {}
        
        for threshold, G in self.networks.items():
            if G.number_of_edges() > 0:
                # Calculate centrality measures
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                closeness_centrality = nx.closeness_centrality(G)
                
                stats[f"threshold_{threshold}"] = {
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "density": nx.density(G),
                    "components": nx.number_connected_components(G),
                    "clustering": nx.average_clustering(G),
                    "top_degree_centrality": sorted(degree_centrality.items(), 
                                                   key=lambda x: x[1], reverse=True)[:3],
                    "top_betweenness_centrality": sorted(betweenness_centrality.items(), 
                                                        key=lambda x: x[1], reverse=True)[:3]
                }
        
        return stats
    
    def save_network_analysis_results(self, output_dir: str, country: str = "Mozambique") -> Dict[str, str]:
        """
        Save network analysis results to files.
        
        Args:
            output_dir (str): Directory to save results
            country (str): Country name for file naming
            
        Returns:
            Dict with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save network analysis summary
        analysis_summary = {
            "network_statistics": self._get_network_statistics(),
            "topic_roles": self.topic_roles,
            "topic_frequencies": self.topic_frequencies,
            "co_occurrence_matrix": self.cooc_matrix.tolist() if self.cooc_matrix is not None else [],
            "topic_names": self.topic_names,
            "analysis_metadata": {
                "total_articles": len(self.articles),
                "total_topics": len(self.topic_names),
                "size_multiplier": self.size_multiplier,
                "articles_with_multiple_topics": sum(1 for topics in self._get_article_topics() if len(topics) > 1)
            }
        }
        
        summary_file = f"{output_dir}/{country}_network_analysis_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(analysis_summary, f, indent=2)
        saved_files["analysis_summary"] = summary_file
        
        # Save co-occurrence matrix as CSV
        if self.cooc_matrix is not None:
            cooc_df = pd.DataFrame(self.cooc_matrix, index=self.topic_names, columns=self.topic_names)
            matrix_file = f"{output_dir}/{country}_topic_cooccurrence_matrix.csv"
            cooc_df.to_csv(matrix_file)
            saved_files["cooccurrence_matrix"] = matrix_file
        
        # Save topic roles as CSV
        roles_data = []
        for topic, role_info in self.topic_roles.items():
            roles_data.append({
                "topic": topic,
                "primary_count": role_info["primary_count"],
                "secondary_count": role_info["secondary_count"],
                "total_count": role_info["total_count"],
                "primary_ratio": role_info["primary_ratio"],
                "frequency": self.topic_frequencies.get(topic, 0)
            })
        
        roles_df = pd.DataFrame(roles_data)
        roles_file = f"{output_dir}/{country}_topic_roles_analysis.csv"
        roles_df.to_csv(roles_file, index=False)
        saved_files["topic_roles"] = roles_file
        
        return saved_files


def run_ragtec_storytelling_analysis(
    articles_file: str,
    output_dir: Optional[str] = None,
    size_multiplier: float = 50.0,
    edge_multiplier: float = 20.0,
    collection_name: str = "News Articles",
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run the complete RAGTEC storytelling analysis pipeline.
    
    Args:
        articles_file (str): Path to classified articles JSON file
        output_dir (str, optional): Directory to save results
        size_multiplier (float): Multiplier for node sizes (mentions × multiplier)
        edge_multiplier (float): Multiplier for edge widths (co-occurrence × multiplier)
        collection_name (str): Name of the data collection for titles
        save_results (bool): Whether to save results to files
        
    Returns:
        Dict containing all results from the network analysis pipeline
    """
    print("RAGTEC Topic Storytelling Analysis Pipeline")
    print("=" * 50)
    
    # Initialize storytelling analysis system
    storytelling_analysis = RAGTECTopicStorytelling(size_multiplier=size_multiplier)
    storytelling_analysis.edge_multiplier = edge_multiplier  # Add edge multiplier support
    
    # Load articles
    load_results = storytelling_analysis.load_classified_articles(articles_file)
    if load_results["status"] == "error":
        return {"error": load_results["message"]}
    
    # Analyze co-occurrence storytelling
    storytelling_results = storytelling_analysis.analyze_cooccurrence_network()
    
    # Create visualizations and save results if requested
    visualization_results = {}
    saved_files = {}
    
    if save_results and output_dir:
        country = os.path.basename(articles_file).split("_")[0]
        
        # Create visualizations with collection name
        visualization_results = storytelling_analysis.create_advanced_network_visualizations(
            output_dir, country, collection_name
        )
        
        # Save analysis results
        analysis_files = storytelling_analysis.save_network_analysis_results(output_dir, country)
        saved_files.update(analysis_files)
        saved_files.update(visualization_results.get("saved_files", {}))
        
        print(f"Results saved to: {output_dir}")
    
    return {
        "loading": load_results,
        "storytelling_analysis": storytelling_results,
        "visualizations": visualization_results,
        "saved_files": saved_files,
        "analysis_object": storytelling_analysis
    }


# Backward compatibility alias
def run_ragtec_network_analysis(*args, **kwargs):
    """Backward compatibility alias for run_ragtec_storytelling_analysis."""
    return run_ragtec_storytelling_analysis(*args, **kwargs)


# Backward compatibility alias for class name
RAGTECTopicNetworkAnalysis = RAGTECTopicStorytelling
