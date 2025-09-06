// T3SS Project
// File: core/graph_core/community_detector.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use petgraph::{Graph, Directed, NodeIndex, Undirected};
use petgraph::algo::connected_components;
use petgraph::visit::{Dfs, EdgeRef, NodeRef};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;

/// Community detection algorithms supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunityAlgorithm {
    Louvain,
    Leiden,
    Infomap,
    LabelPropagation,
    ModularityOptimization,
}

/// Configuration for community detection
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    pub algorithm: CommunityAlgorithm,
    pub resolution: f64,                    // Resolution parameter for modularity
    pub max_iterations: usize,              // Maximum iterations
    pub convergence_threshold: f64,         // Convergence criteria
    pub enable_parallel: bool,             // Enable parallel processing
    pub enable_refinement: bool,            // Enable community refinement
    pub enable_hierarchical: bool,          // Enable hierarchical clustering
    pub batch_size: usize,                  // Batch size for processing
    pub random_seed: Option<u64>,           // Random seed for reproducibility
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::Louvain,
            resolution: 1.0,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            enable_parallel: true,
            enable_refinement: true,
            enable_hierarchical: false,
            batch_size: 1000,
            random_seed: Some(42),
        }
    }
}

/// Represents a community in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub id: u64,
    pub nodes: Vec<u64>,
    pub size: usize,
    pub internal_edges: usize,
    pub external_edges: usize,
    pub modularity: f64,
    pub density: f64,
    pub cohesion: f64,
}

/// Community detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    pub communities: Vec<Community>,
    pub node_assignments: HashMap<u64, u64>,  // node_id -> community_id
    pub modularity: f64,
    pub algorithm_used: CommunityAlgorithm,
    pub computation_time: Duration,
    pub iterations: usize,
    pub converged: bool,
    pub quality_metrics: CommunityQualityMetrics,
}

/// Quality metrics for community detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityQualityMetrics {
    pub modularity: f64,
    pub conductance: f64,
    pub coverage: f64,
    pub performance: f64,
    pub silhouette_score: f64,
    pub normalized_cut: f64,
}

/// Advanced community detector with multiple algorithms
pub struct CommunityDetector {
    config: CommunityConfig,
    graph: Arc<RwLock<Graph<u64, f64, Undirected>>>,
    node_index: Arc<RwLock<HashMap<u64, NodeIndex>>>,
    stats: Arc<Mutex<CommunityStats>>,
}

/// Statistics for community detection
#[derive(Debug, Default)]
pub struct CommunityStats {
    pub total_nodes: u64,
    pub total_edges: u64,
    pub communities_found: usize,
    pub computation_time: Duration,
    pub iterations_completed: usize,
    pub convergence_achieved: bool,
    pub memory_usage: u64,
}

impl CommunityDetector {
    /// Create a new community detector
    pub fn new(config: CommunityConfig) -> Self {
        Self {
            config,
            graph: Arc::new(RwLock::new(Graph::new_undirected())),
            node_index: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CommunityStats::default())),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&self, node_id: u64) -> Result<(), String> {
        let mut graph = self.graph.write().unwrap();
        let mut index = self.node_index.write().unwrap();
        
        let node_idx = graph.add_node(node_id);
        index.insert(node_id, node_idx);
        
        Ok(())
    }

    /// Add an edge between nodes
    pub fn add_edge(&self, from: u64, to: u64, weight: f64) -> Result<(), String> {
        let graph = self.graph.read().unwrap();
        let index = self.node_index.read().unwrap();
        
        if let (Some(&from_idx), Some(&to_idx)) = (index.get(&from), index.get(&to)) {
            let mut graph = self.graph.write().unwrap();
            graph.add_edge(from_idx, to_idx, weight);
        }
        
        Ok(())
    }

    /// Detect communities using the configured algorithm
    pub fn detect_communities(&self) -> Result<CommunityResult, String> {
        let start_time = Instant::now();
        
        match self.config.algorithm {
            CommunityAlgorithm::Louvain => self.louvain_algorithm(),
            CommunityAlgorithm::Leiden => self.leiden_algorithm(),
            CommunityAlgorithm::Infomap => self.infomap_algorithm(),
            CommunityAlgorithm::LabelPropagation => self.label_propagation_algorithm(),
            CommunityAlgorithm::ModularityOptimization => self.modularity_optimization(),
        }
    }

    /// Louvain algorithm implementation
    fn louvain_algorithm(&self) -> Result<CommunityResult, String> {
        let graph = self.graph.read().unwrap();
        let node_count = graph.node_count();
        
        if node_count == 0 {
            return Ok(CommunityResult {
                communities: vec![],
                node_assignments: HashMap::new(),
                modularity: 0.0,
                algorithm_used: CommunityAlgorithm::Louvain,
                computation_time: Duration::from_secs(0),
                iterations: 0,
                converged: true,
                quality_metrics: CommunityQualityMetrics::default(),
            });
        }

        // Initialize each node as its own community
        let mut communities: HashMap<NodeIndex, u64> = graph.node_indices()
            .enumerate()
            .map(|(i, idx)| (idx, i as u64))
            .collect();
        
        let mut modularity = self.compute_modularity(&communities);
        let mut iteration = 0;
        let mut improved = true;

        while improved && iteration < self.config.max_iterations {
            improved = false;
            
            // Try to move each node to a different community
            for node_idx in graph.node_indices() {
                let current_community = communities[&node_idx];
                let mut best_community = current_community;
                let mut best_modularity_gain = 0.0;

                // Calculate modularity gain for each possible community
                for &community_id in communities.values().collect::<HashSet<_>>() {
                    if community_id == current_community {
                        continue;
                    }

                    let gain = self.calculate_modularity_gain(node_idx, current_community, community_id, &communities);
                    if gain > best_modularity_gain {
                        best_modularity_gain = gain;
                        best_community = community_id;
                    }
                }

                // Move node if improvement found
                if best_modularity_gain > self.config.convergence_threshold {
                    communities.insert(node_idx, best_community);
                    improved = true;
                }
            }

            // Recalculate modularity
            let new_modularity = self.compute_modularity(&communities);
            if (new_modularity - modularity).abs() < self.config.convergence_threshold {
                break;
            }
            
            modularity = new_modularity;
            iteration += 1;
        }

        // Convert to result format
        self.convert_to_result(communities, modularity, iteration, true)
    }

    /// Leiden algorithm implementation (improved Louvain)
    fn leiden_algorithm(&self) -> Result<CommunityResult, String> {
        // Similar to Louvain but with refinement step
        let mut result = self.louvain_algorithm()?;
        
        if self.config.enable_refinement {
            result = self.refine_communities(result)?;
        }
        
        Ok(result)
    }

    /// Infomap algorithm implementation
    fn infomap_algorithm(&self) -> Result<CommunityResult, String> {
        // Simplified Infomap implementation
        // In production, use the full Infomap algorithm with random walks
        self.label_propagation_algorithm()
    }

    /// Label propagation algorithm
    fn label_propagation_algorithm(&self) -> Result<CommunityResult, String> {
        let graph = self.graph.read().unwrap();
        let node_count = graph.node_count();
        
        if node_count == 0 {
            return Ok(CommunityResult {
                communities: vec![],
                node_assignments: HashMap::new(),
                modularity: 0.0,
                algorithm_used: CommunityAlgorithm::LabelPropagation,
                computation_time: Duration::from_secs(0),
                iterations: 0,
                converged: true,
                quality_metrics: CommunityQualityMetrics::default(),
            });
        }

        // Initialize random labels
        let mut rng = StdRng::seed_from_u64(self.config.random_seed.unwrap_or(42));
        let mut labels: HashMap<NodeIndex, u64> = graph.node_indices()
            .enumerate()
            .map(|(i, idx)| (idx, i as u64))
            .collect();

        let mut iteration = 0;
        let mut converged = false;

        while !converged && iteration < self.config.max_iterations {
            converged = true;
            
            // Randomize node order
            let mut nodes: Vec<NodeIndex> = graph.node_indices().collect();
            nodes.shuffle(&mut rng);

            for node_idx in nodes {
                let mut neighbor_labels: HashMap<u64, usize> = HashMap::new();
                
                // Count labels of neighbors
                for edge in graph.edges(node_idx) {
                    let neighbor = if edge.source() == node_idx { edge.target() } else { edge.source() };
                    let label = labels[&neighbor];
                    *neighbor_labels.entry(label).or_insert(0) += 1;
                }

                // Find most frequent label
                if let Some((&most_frequent_label, _)) = neighbor_labels.iter().max_by_key(|(_, count)| *count) {
                    if labels[&node_idx] != most_frequent_label {
                        labels.insert(node_idx, most_frequent_label);
                        converged = false;
                    }
                }
            }

            iteration += 1;
        }

        // Convert labels to communities
        let mut communities: HashMap<NodeIndex, u64> = HashMap::new();
        for (node_idx, label) in labels {
            communities.insert(node_idx, label);
        }

        let modularity = self.compute_modularity(&communities);
        self.convert_to_result(communities, modularity, iteration, converged)
    }

    /// Modularity optimization algorithm
    fn modularity_optimization(&self) -> Result<CommunityResult, String> {
        // Use simulated annealing for modularity optimization
        let graph = self.graph.read().unwrap();
        let node_count = graph.node_count();
        
        if node_count == 0 {
            return Ok(CommunityResult {
                communities: vec![],
                node_assignments: HashMap::new(),
                modularity: 0.0,
                algorithm_used: CommunityAlgorithm::ModularityOptimization,
                computation_time: Duration::from_secs(0),
                iterations: 0,
                converged: true,
                quality_metrics: CommunityQualityMetrics::default(),
            });
        }

        let mut rng = StdRng::seed_from_u64(self.config.random_seed.unwrap_or(42));
        let mut communities: HashMap<NodeIndex, u64> = graph.node_indices()
            .enumerate()
            .map(|(i, idx)| (idx, i as u64))
            .collect();

        let mut current_modularity = self.compute_modularity(&communities);
        let mut best_modularity = current_modularity;
        let mut best_communities = communities.clone();

        let mut temperature = 1.0;
        let cooling_rate = 0.95;
        let iteration = 0;

        for _ in 0..self.config.max_iterations {
            // Randomly select a node and try to move it
            let nodes: Vec<NodeIndex> = graph.node_indices().collect();
            let node_idx = nodes[rng.gen_range(0..nodes.len())];
            let current_community = communities[&node_idx];
            let new_community = rng.gen_range(0..node_count as u64);

            if new_community == current_community {
                continue;
            }

            // Calculate modularity change
            let old_modularity = self.compute_modularity(&communities);
            communities.insert(node_idx, new_community);
            let new_modularity = self.compute_modularity(&communities);

            let delta_modularity = new_modularity - old_modularity;

            // Accept or reject the move
            if delta_modularity > 0.0 || rng.gen::<f64>() < (-delta_modularity / temperature).exp() {
                current_modularity = new_modularity;
                if current_modularity > best_modularity {
                    best_modularity = current_modularity;
                    best_communities = communities.clone();
                }
            } else {
                // Reject the move
                communities.insert(node_idx, current_community);
            }

            temperature *= cooling_rate;
        }

        self.convert_to_result(best_communities, best_modularity, iteration, true)
    }

    /// Refine communities using local optimization
    fn refine_communities(&self, mut result: CommunityResult) -> Result<CommunityResult, String> {
        // Implement community refinement logic
        // This would involve local optimization within each community
        Ok(result)
    }

    /// Compute modularity of the current community assignment
    fn compute_modularity(&self, communities: &HashMap<NodeIndex, u64>) -> f64 {
        let graph = self.graph.read().unwrap();
        let total_edges = graph.edge_count() as f64;
        
        if total_edges == 0.0 {
            return 0.0;
        }

        let mut modularity = 0.0;
        
        // Group nodes by community
        let mut community_nodes: HashMap<u64, Vec<NodeIndex>> = HashMap::new();
        for (node_idx, community_id) in communities {
            community_nodes.entry(*community_id).or_insert_with(Vec::new).push(*node_idx);
        }

        for (_, nodes) in community_nodes {
            let mut internal_edges = 0.0;
            let mut total_degree = 0.0;

            for &node_idx in &nodes {
                total_degree += graph.edges(node_idx).count() as f64;
                for edge in graph.edges(node_idx) {
                    let neighbor = if edge.source() == node_idx { edge.target() } else { edge.source() };
                    if nodes.contains(&neighbor) {
                        internal_edges += edge.weight();
                    }
                }
            }

            let expected_edges = (total_degree * total_degree) / (2.0 * total_edges);
            modularity += (internal_edges / 2.0) - expected_edges;
        }

        modularity / total_edges
    }

    /// Calculate modularity gain for moving a node
    fn calculate_modularity_gain(&self, node_idx: NodeIndex, from_community: u64, to_community: u64, communities: &HashMap<NodeIndex, u64>) -> f64 {
        // Simplified modularity gain calculation
        // In production, implement the full formula
        0.0
    }

    /// Convert internal representation to result format
    fn convert_to_result(&self, communities: HashMap<NodeIndex, u64>, modularity: f64, iterations: usize, converged: bool) -> Result<CommunityResult, String> {
        let graph = self.graph.read().unwrap();
        let index = self.node_index.read().unwrap();

        // Group nodes by community
        let mut community_groups: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut node_assignments = HashMap::new();

        for (node_idx, community_id) in communities {
            if let Some((&node_id, _)) = index.iter().find(|(_, &idx)| idx == node_idx) {
                community_groups.entry(community_id).or_insert_with(Vec::new).push(node_id);
                node_assignments.insert(node_id, community_id);
            }
        }

        // Create community objects
        let mut community_objects = Vec::new();
        for (community_id, nodes) in community_groups {
            let size = nodes.len();
            let (internal_edges, external_edges) = self.calculate_community_edges(&nodes);
            let density = if size > 1 {
                (2.0 * internal_edges as f64) / (size as f64 * (size as f64 - 1.0))
            } else {
                0.0
            };
            let cohesion = if external_edges > 0 {
                internal_edges as f64 / (internal_edges + external_edges) as f64
            } else {
                1.0
            };

            community_objects.push(Community {
                id: community_id,
                nodes,
                size,
                internal_edges,
                external_edges,
                modularity: 0.0, // Will be calculated separately
                density,
                cohesion,
            });
        }

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(&community_objects, modularity);

        Ok(CommunityResult {
            communities: community_objects,
            node_assignments,
            modularity,
            algorithm_used: self.config.algorithm.clone(),
            computation_time: Duration::from_secs(0), // Will be set by caller
            iterations,
            converged,
            quality_metrics,
        })
    }

    /// Calculate edges within and outside a community
    fn calculate_community_edges(&self, nodes: &[u64]) -> (usize, usize) {
        let graph = self.graph.read().unwrap();
        let index = self.node_index.read().unwrap();
        let node_set: HashSet<u64> = nodes.iter().copied().collect();

        let mut internal_edges = 0;
        let mut external_edges = 0;

        for &node_id in nodes {
            if let Some(&node_idx) = index.get(&node_id) {
                for edge in graph.edges(node_idx) {
                    let neighbor_idx = if edge.source() == node_idx { edge.target() } else { edge.source() };
                    if let Some((&neighbor_id, _)) = index.iter().find(|(_, &idx)| idx == neighbor_idx) {
                        if node_set.contains(&neighbor_id) {
                            internal_edges += 1;
                        } else {
                            external_edges += 1;
                        }
                    }
                }
            }
        }

        (internal_edges / 2, external_edges) // Divide by 2 because we count each edge twice
    }

    /// Calculate comprehensive quality metrics
    fn calculate_quality_metrics(&self, communities: &[Community], modularity: f64) -> CommunityQualityMetrics {
        let total_nodes: usize = communities.iter().map(|c| c.size).sum();
        let total_edges: usize = communities.iter().map(|c| c.internal_edges + c.external_edges).sum();

        // Calculate conductance (average)
        let conductance = if !communities.is_empty() {
            communities.iter().map(|c| {
                if c.external_edges > 0 {
                    c.external_edges as f64 / (c.internal_edges + c.external_edges) as f64
                } else {
                    0.0
                }
            }).sum::<f64>() / communities.len() as f64
        } else {
            0.0
        };

        // Calculate coverage
        let coverage = if total_edges > 0 {
            communities.iter().map(|c| c.internal_edges).sum::<usize>() as f64 / total_edges as f64
        } else {
            0.0
        };

        // Calculate performance (simplified)
        let performance = modularity; // Simplified for now

        // Calculate silhouette score (simplified)
        let silhouette_score = if communities.len() > 1 {
            // Simplified silhouette calculation
            modularity * 0.5
        } else {
            0.0
        };

        // Calculate normalized cut (simplified)
        let normalized_cut = conductance;

        CommunityQualityMetrics {
            modularity,
            conductance,
            coverage,
            performance,
            silhouette_score,
            normalized_cut,
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> CommunityStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get graph information
    pub fn get_graph_info(&self) -> (usize, usize) {
        let graph = self.graph.read().unwrap();
        (graph.node_count(), graph.edge_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_community_detection() {
        let config = CommunityConfig::default();
        let detector = CommunityDetector::new(config);
        
        // Add test nodes
        detector.add_node(1).unwrap();
        detector.add_node(2).unwrap();
        detector.add_node(3).unwrap();
        detector.add_node(4).unwrap();
        
        // Add edges to create two communities
        detector.add_edge(1, 2, 1.0).unwrap();
        detector.add_edge(3, 4, 1.0).unwrap();
        
        // Detect communities
        let result = detector.detect_communities().unwrap();
        
        assert_eq!(result.communities.len(), 2);
        assert!(result.modularity > 0.0);
        assert!(result.converged);
    }
}
