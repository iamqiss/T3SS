// T3SS Project
// File: core/graph_core/pagerank_engine.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use petgraph::{Graph, Directed, NodeIndex};
use petgraph::algo::dijkstra;
use petgraph::visit::{Dfs, EdgeRef};
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

/// Represents a web page in the link graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebPage {
    pub id: u64,
    pub url: String,
    pub title: String,
    pub domain: String,
    pub content_length: u64,
    pub last_crawled: u64,
    pub page_rank: f64,
    pub in_links: Vec<u64>,
    pub out_links: Vec<u64>,
    pub quality_score: f64,
}

/// Represents a link between web pages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub from_page_id: u64,
    pub to_page_id: u64,
    pub anchor_text: String,
    pub link_type: LinkType,
    pub weight: f64,
    pub discovered_at: u64,
}

/// Types of links for different weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinkType {
    Internal,    // Same domain
    External,    // Different domain
    Navigational, // Navigation links
    Content,     // Content links
    Sponsored,   // Paid links
}

/// Configuration for PageRank computation
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    pub damping_factor: f64,           // Usually 0.85
    pub convergence_threshold: f64,    // Convergence criteria
    pub max_iterations: usize,         // Maximum iterations
    pub enable_parallel: bool,        // Enable parallel computation
    pub enable_adaptive_damping: bool, // Adaptive damping based on graph structure
    pub enable_topic_sensitive: bool,  // Topic-sensitive PageRank
    pub enable_personalized: bool,     // Personalized PageRank
    pub batch_size: usize,             // Batch size for processing
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            convergence_threshold: 1e-6,
            max_iterations: 100,
            enable_parallel: true,
            enable_adaptive_damping: true,
            enable_topic_sensitive: false,
            enable_personalized: false,
            batch_size: 1000,
        }
    }
}

/// Advanced PageRank engine with multiple algorithms
pub struct PageRankEngine {
    config: PageRankConfig,
    graph: Arc<AsyncRwLock<Graph<WebPage, Link, Directed>>>,
    page_index: Arc<RwLock<HashMap<u64, NodeIndex>>>,
    stats: Arc<Mutex<PageRankStats>>,
}

/// Statistics for PageRank computation
#[derive(Debug, Default)]
pub struct PageRankStats {
    pub total_pages: u64,
    pub total_links: u64,
    pub computation_time: Duration,
    pub iterations_completed: usize,
    pub convergence_achieved: bool,
    pub memory_usage: u64,
    pub graph_density: f64,
}

impl PageRankEngine {
    /// Create a new PageRank engine
    pub fn new(config: PageRankConfig) -> Self {
        Self {
            config,
            graph: Arc::new(AsyncRwLock::new(Graph::new())),
            page_index: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PageRankStats::default())),
        }
    }

    /// Add a web page to the graph
    pub async fn add_page(&self, page: WebPage) -> Result<(), String> {
        let mut graph = self.graph.write().await;
        let mut index = self.page_index.write().unwrap();
        
        let node_idx = graph.add_node(page.clone());
        index.insert(page.id, node_idx);
        
        Ok(())
    }

    /// Add a link between pages
    pub async fn add_link(&self, link: Link) -> Result<(), String> {
        let graph = self.graph.read().await;
        let index = self.page_index.read().unwrap();
        
        if let (Some(&from_idx), Some(&to_idx)) = (
            index.get(&link.from_page_id),
            index.get(&link.to_page_id)
        ) {
            let mut graph = self.graph.write().await;
            graph.add_edge(from_idx, to_idx, link);
        }
        
        Ok(())
    }

    /// Compute PageRank for all pages using the power iteration method
    pub async fn compute_pagerank(&self) -> Result<HashMap<u64, f64>, String> {
        let start_time = Instant::now();
        
        let graph = self.graph.read().await;
        let page_count = graph.node_count();
        
        if page_count == 0 {
            return Ok(HashMap::new());
        }

        // Initialize PageRank values
        let initial_rank = 1.0 / page_count as f64;
        let mut ranks: HashMap<NodeIndex, f64> = graph.node_indices()
            .map(|idx| (idx, initial_rank))
            .collect();

        // Compute PageRank using power iteration
        for iteration in 0..self.config.max_iterations {
            let mut new_ranks = HashMap::new();
            let mut total_change = 0.0;

            // Process each node
            for node_idx in graph.node_indices() {
                let mut rank_sum = 0.0;
                let mut incoming_weight_sum = 0.0;

                // Sum contributions from incoming links
                for edge in graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                    let source_idx = edge.source();
                    let link_weight = edge.weight().weight;
                    
                    // Get out-degree of source node
                    let out_degree = graph.edges_directed(source_idx, petgraph::Direction::Outgoing).count() as f64;
                    
                    if out_degree > 0.0 {
                        let contribution = ranks.get(&source_idx).unwrap_or(&0.0) * link_weight / out_degree;
                        rank_sum += contribution;
                    }
                    
                    incoming_weight_sum += link_weight;
                }

                // Apply damping factor
                let new_rank = (1.0 - self.config.damping_factor) / page_count as f64 + 
                              self.config.damping_factor * rank_sum;
                
                new_ranks.insert(node_idx, new_rank);
                
                // Track convergence
                let old_rank = ranks.get(&node_idx).unwrap_or(&0.0);
                total_change += (new_rank - old_rank).abs();
            }

            // Check convergence
            if total_change < self.config.convergence_threshold {
                self.update_stats(start_time.elapsed(), iteration + 1, true);
                break;
            }

            ranks = new_ranks;
            
            if iteration == self.config.max_iterations - 1 {
                self.update_stats(start_time.elapsed(), iteration + 1, false);
            }
        }

        // Convert to page ID -> rank mapping
        let index = self.page_index.read().unwrap();
        let mut result = HashMap::new();
        
        for (node_idx, rank) in ranks {
            if let Some((&page_id, _)) = index.iter().find(|(_, &idx)| idx == node_idx) {
                result.insert(page_id, rank);
            }
        }

        Ok(result)
    }

    /// Compute Topic-Sensitive PageRank
    pub async fn compute_topic_sensitive_pagerank(&self, topic_pages: &[u64]) -> Result<HashMap<u64, f64>, String> {
        if !self.config.enable_topic_sensitive {
            return Err("Topic-sensitive PageRank not enabled".to_string());
        }

        let start_time = Instant::now();
        let graph = self.graph.read().await;
        let page_count = graph.node_count();
        
        if page_count == 0 {
            return Ok(HashMap::new());
        }

        // Create topic vector
        let mut topic_vector = HashMap::new();
        let topic_weight = 1.0 / topic_pages.len() as f64;
        
        let index = self.page_index.read().unwrap();
        for &page_id in topic_pages {
            if let Some(&node_idx) = index.get(&page_id) {
                topic_vector.insert(node_idx, topic_weight);
            }
        }

        // Initialize ranks
        let initial_rank = 1.0 / page_count as f64;
        let mut ranks: HashMap<NodeIndex, f64> = graph.node_indices()
            .map(|idx| (idx, initial_rank))
            .collect();

        // Power iteration with topic bias
        for iteration in 0..self.config.max_iterations {
            let mut new_ranks = HashMap::new();
            let mut total_change = 0.0;

            for node_idx in graph.node_indices() {
                let mut rank_sum = 0.0;

                // Sum contributions from incoming links
                for edge in graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                    let source_idx = edge.source();
                    let link_weight = edge.weight().weight;
                    let out_degree = graph.edges_directed(source_idx, petgraph::Direction::Outgoing).count() as f64;
                    
                    if out_degree > 0.0 {
                        let contribution = ranks.get(&source_idx).unwrap_or(&0.0) * link_weight / out_degree;
                        rank_sum += contribution;
                    }
                }

                // Apply topic bias
                let topic_bias = topic_vector.get(&node_idx).unwrap_or(&0.0);
                let new_rank = (1.0 - self.config.damping_factor) * topic_bias + 
                              self.config.damping_factor * rank_sum;
                
                new_ranks.insert(node_idx, new_rank);
                
                let old_rank = ranks.get(&node_idx).unwrap_or(&0.0);
                total_change += (new_rank - old_rank).abs();
            }

            if total_change < self.config.convergence_threshold {
                break;
            }

            ranks = new_ranks;
        }

        // Convert to page ID -> rank mapping
        let mut result = HashMap::new();
        for (node_idx, rank) in ranks {
            if let Some((&page_id, _)) = index.iter().find(|(_, &idx)| idx == node_idx) {
                result.insert(page_id, rank);
            }
        }

        Ok(result)
    }

    /// Compute Personalized PageRank for a specific user
    pub async fn compute_personalized_pagerank(&self, user_preferences: &HashMap<u64, f64>) -> Result<HashMap<u64, f64>, String> {
        if !self.config.enable_personalized {
            return Err("Personalized PageRank not enabled".to_string());
        }

        let start_time = Instant::now();
        let graph = self.graph.read().await;
        let page_count = graph.node_count();
        
        if page_count == 0 {
            return Ok(HashMap::new());
        }

        // Normalize user preferences
        let total_preference: f64 = user_preferences.values().sum();
        let mut preference_vector = HashMap::new();
        
        let index = self.page_index.read().unwrap();
        for (&page_id, &pref) in user_preferences {
            if let Some(&node_idx) = index.get(&page_id) {
                preference_vector.insert(node_idx, pref / total_preference);
            }
        }

        // Initialize ranks
        let initial_rank = 1.0 / page_count as f64;
        let mut ranks: HashMap<NodeIndex, f64> = graph.node_indices()
            .map(|idx| (idx, initial_rank))
            .collect();

        // Power iteration with personalization
        for iteration in 0..self.config.max_iterations {
            let mut new_ranks = HashMap::new();
            let mut total_change = 0.0;

            for node_idx in graph.node_indices() {
                let mut rank_sum = 0.0;

                // Sum contributions from incoming links
                for edge in graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                    let source_idx = edge.source();
                    let link_weight = edge.weight().weight;
                    let out_degree = graph.edges_directed(source_idx, petgraph::Direction::Outgoing).count() as f64;
                    
                    if out_degree > 0.0 {
                        let contribution = ranks.get(&source_idx).unwrap_or(&0.0) * link_weight / out_degree;
                        rank_sum += contribution;
                    }
                }

                // Apply personalization bias
                let personal_bias = preference_vector.get(&node_idx).unwrap_or(&0.0);
                let new_rank = (1.0 - self.config.damping_factor) * personal_bias + 
                              self.config.damping_factor * rank_sum;
                
                new_ranks.insert(node_idx, new_rank);
                
                let old_rank = ranks.get(&node_idx).unwrap_or(&0.0);
                total_change += (new_rank - old_rank).abs();
            }

            if total_change < self.config.convergence_threshold {
                break;
            }

            ranks = new_ranks;
        }

        // Convert to page ID -> rank mapping
        let mut result = HashMap::new();
        for (node_idx, rank) in ranks {
            if let Some((&page_id, _)) = index.iter().find(|(_, &idx)| idx == node_idx) {
                result.insert(page_id, rank);
            }
        }

        Ok(result)
    }

    /// Compute HITS algorithm (Hubs and Authorities)
    pub async fn compute_hits(&self) -> Result<(HashMap<u64, f64>, HashMap<u64, f64>), String> {
        let start_time = Instant::now();
        let graph = self.graph.read().await;
        let page_count = graph.node_count();
        
        if page_count == 0 {
            return Ok((HashMap::new(), HashMap::new()));
        }

        // Initialize hub and authority scores
        let mut hub_scores: HashMap<NodeIndex, f64> = graph.node_indices()
            .map(|idx| (idx, 1.0))
            .collect();
        let mut auth_scores: HashMap<NodeIndex, f64> = graph.node_indices()
            .map(|idx| (idx, 1.0))
            .collect();

        // Iterative computation
        for iteration in 0..self.config.max_iterations {
            let mut new_hub_scores = HashMap::new();
            let mut new_auth_scores = HashMap::new();
            let mut total_change = 0.0;

            // Update authority scores
            for node_idx in graph.node_indices() {
                let mut auth_sum = 0.0;
                for edge in graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                    let source_idx = edge.source();
                    auth_sum += hub_scores.get(&source_idx).unwrap_or(&0.0);
                }
                new_auth_scores.insert(node_idx, auth_sum);
            }

            // Update hub scores
            for node_idx in graph.node_indices() {
                let mut hub_sum = 0.0;
                for edge in graph.edges_directed(node_idx, petgraph::Direction::Outgoing) {
                    let target_idx = edge.target();
                    hub_sum += new_auth_scores.get(&target_idx).unwrap_or(&0.0);
                }
                new_hub_scores.insert(node_idx, hub_sum);
            }

            // Normalize scores
            let hub_norm: f64 = new_hub_scores.values().map(|x| x * x).sum::<f64>().sqrt();
            let auth_norm: f64 = new_auth_scores.values().map(|x| x * x).sum::<f64>().sqrt();

            if hub_norm > 0.0 {
                for score in new_hub_scores.values_mut() {
                    *score /= hub_norm;
                }
            }
            if auth_norm > 0.0 {
                for score in new_auth_scores.values_mut() {
                    *score /= auth_norm;
                }
            }

            // Check convergence
            for node_idx in graph.node_indices() {
                let old_hub = hub_scores.get(&node_idx).unwrap_or(&0.0);
                let new_hub = new_hub_scores.get(&node_idx).unwrap_or(&0.0);
                total_change += (new_hub - old_hub).abs();
            }

            if total_change < self.config.convergence_threshold {
                break;
            }

            hub_scores = new_hub_scores;
            auth_scores = new_auth_scores;
        }

        // Convert to page ID -> score mapping
        let index = self.page_index.read().unwrap();
        let mut hub_result = HashMap::new();
        let mut auth_result = HashMap::new();
        
        for (node_idx, score) in hub_scores {
            if let Some((&page_id, _)) = index.iter().find(|(_, &idx)| idx == node_idx) {
                hub_result.insert(page_id, score);
            }
        }
        
        for (node_idx, score) in auth_scores {
            if let Some((&page_id, _)) = index.iter().find(|(_, &idx)| idx == node_idx) {
                auth_result.insert(page_id, score);
            }
        }

        Ok((hub_result, auth_result))
    }

    /// Compute graph metrics for analysis
    pub async fn compute_graph_metrics(&self) -> Result<GraphMetrics, String> {
        let graph = self.graph.read().await;
        let page_count = graph.node_count();
        let edge_count = graph.edge_count();
        
        if page_count == 0 {
            return Ok(GraphMetrics::default());
        }

        // Compute degree statistics
        let mut in_degrees = Vec::new();
        let mut out_degrees = Vec::new();
        
        for node_idx in graph.node_indices() {
            in_degrees.push(graph.edges_directed(node_idx, petgraph::Direction::Incoming).count());
            out_degrees.push(graph.edges_directed(node_idx, petgraph::Direction::Outgoing).count());
        }

        let avg_in_degree = in_degrees.iter().sum::<usize>() as f64 / page_count as f64;
        let avg_out_degree = out_degrees.iter().sum::<usize>() as f64 / page_count as f64;
        
        // Compute clustering coefficient
        let mut clustering_sum = 0.0;
        let mut nodes_with_neighbors = 0;
        
        for node_idx in graph.node_indices() {
            let neighbors: HashSet<NodeIndex> = graph.neighbors(node_idx).collect();
            let neighbor_count = neighbors.len();
            
            if neighbor_count >= 2 {
                let mut edges_between_neighbors = 0;
                for &neighbor1 in &neighbors {
                    for &neighbor2 in &neighbors {
                        if neighbor1 < neighbor2 && graph.find_edge(neighbor1, neighbor2).is_some() {
                            edges_between_neighbors += 1;
                        }
                    }
                }
                
                let possible_edges = neighbor_count * (neighbor_count - 1) / 2;
                if possible_edges > 0 {
                    clustering_sum += edges_between_neighbors as f64 / possible_edges as f64;
                    nodes_with_neighbors += 1;
                }
            }
        }
        
        let clustering_coefficient = if nodes_with_neighbors > 0 {
            clustering_sum / nodes_with_neighbors as f64
        } else {
            0.0
        };

        // Compute graph density
        let max_possible_edges = page_count * (page_count - 1);
        let density = if max_possible_edges > 0 {
            edge_count as f64 / max_possible_edges as f64
        } else {
            0.0
        };

        Ok(GraphMetrics {
            total_pages: page_count as u64,
            total_links: edge_count as u64,
            average_in_degree: avg_in_degree,
            average_out_degree: avg_out_degree,
            clustering_coefficient,
            graph_density: density,
            max_in_degree: in_degrees.iter().max().copied().unwrap_or(0) as u64,
            max_out_degree: out_degrees.iter().max().copied().unwrap_or(0) as u64,
        })
    }

    /// Update statistics
    fn update_stats(&self, computation_time: Duration, iterations: usize, converged: bool) {
        let mut stats = self.stats.lock().unwrap();
        stats.computation_time = computation_time;
        stats.iterations_completed = iterations;
        stats.convergence_achieved = converged;
    }

    /// Get current statistics
    pub fn get_stats(&self) -> PageRankStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get graph size information
    pub async fn get_graph_info(&self) -> (usize, usize) {
        let graph = self.graph.read().await;
        (graph.node_count(), graph.edge_count())
    }
}

/// Graph metrics for analysis
#[derive(Debug, Clone, Default)]
pub struct GraphMetrics {
    pub total_pages: u64,
    pub total_links: u64,
    pub average_in_degree: f64,
    pub average_out_degree: f64,
    pub clustering_coefficient: f64,
    pub graph_density: f64,
    pub max_in_degree: u64,
    pub max_out_degree: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pagerank_computation() {
        let config = PageRankConfig::default();
        let engine = PageRankEngine::new(config);
        
        // Add test pages
        let page1 = WebPage {
            id: 1,
            url: "https://example.com/page1".to_string(),
            title: "Page 1".to_string(),
            domain: "example.com".to_string(),
            content_length: 1000,
            last_crawled: 1234567890,
            page_rank: 0.0,
            in_links: vec![],
            out_links: vec![2],
            quality_score: 0.8,
        };
        
        let page2 = WebPage {
            id: 2,
            url: "https://example.com/page2".to_string(),
            title: "Page 2".to_string(),
            domain: "example.com".to_string(),
            content_length: 2000,
            last_crawled: 1234567890,
            page_rank: 0.0,
            in_links: vec![1],
            out_links: vec![1],
            quality_score: 0.9,
        };
        
        engine.add_page(page1).await.unwrap();
        engine.add_page(page2).await.unwrap();
        
        // Add links
        let link1 = Link {
            from_page_id: 1,
            to_page_id: 2,
            anchor_text: "Link to page 2".to_string(),
            link_type: LinkType::Internal,
            weight: 1.0,
            discovered_at: 1234567890,
        };
        
        let link2 = Link {
            from_page_id: 2,
            to_page_id: 1,
            anchor_text: "Link to page 1".to_string(),
            link_type: LinkType::Internal,
            weight: 1.0,
            discovered_at: 1234567890,
        };
        
        engine.add_link(link1).await.unwrap();
        engine.add_link(link2).await.unwrap();
        
        // Compute PageRank
        let ranks = engine.compute_pagerank().await.unwrap();
        
        assert_eq!(ranks.len(), 2);
        assert!(ranks.contains_key(&1));
        assert!(ranks.contains_key(&2));
        
        // Both pages should have equal rank due to symmetric links
        let rank1 = ranks.get(&1).unwrap();
        let rank2 = ranks.get(&2).unwrap();
        assert!((rank1 - rank2).abs() < 0.01);
    }
}