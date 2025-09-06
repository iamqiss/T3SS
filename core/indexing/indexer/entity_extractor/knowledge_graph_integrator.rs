// T3SS Project
// File: core/indexing/indexer/entity_extractor/knowledge_graph_integrator.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use uuid::Uuid;
use petgraph::{Graph, Directed, NodeIndex, EdgeIndex};
use petgraph::algo::dijkstra;
use petgraph::visit::{Dfs, EdgeRef, NodeRef};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Norm;

/// Knowledge graph entity types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Event,
    Concept,
    Product,
    Technology,
    Date,
    Money,
    Percent,
    Unknown,
}

/// Entity confidence levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfidenceLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    VeryHigh = 4,
}

/// Knowledge graph entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: EntityType,
    pub confidence: f64,
    pub confidence_level: ConfidenceLevel,
    pub aliases: Vec<String>,
    pub description: Option<String>,
    pub properties: HashMap<String, String>,
    pub embeddings: Option<Vec<f64>>,
    pub created_at: u64,
    pub updated_at: u64,
    pub source_documents: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Knowledge graph relation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub id: String,
    pub subject_id: String,
    pub predicate: String,
    pub object_id: String,
    pub confidence: f64,
    pub weight: f64,
    pub evidence: Vec<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Knowledge graph triple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    pub subject: Entity,
    pub predicate: String,
    pub object: Entity,
    pub confidence: f64,
    pub weight: f64,
}

/// Knowledge graph configuration
#[derive(Debug, Clone)]
pub struct KnowledgeGraphConfig {
    pub max_entities: usize,
    pub max_relations: usize,
    pub min_confidence: f64,
    pub enable_embeddings: bool,
    pub embedding_dimension: usize,
    pub enable_inference: bool,
    pub enable_consistency_checking: bool,
    pub batch_size: usize,
    pub max_concurrency: usize,
    pub cache_size: usize,
    pub enable_persistence: bool,
    pub persistence_interval: Duration,
}

impl Default for KnowledgeGraphConfig {
    fn default() -> Self {
        Self {
            max_entities: 1_000_000,
            max_relations: 10_000_000,
            min_confidence: 0.5,
            enable_embeddings: true,
            embedding_dimension: 384,
            enable_inference: true,
            enable_consistency_checking: true,
            batch_size: 1000,
            max_concurrency: 8,
            cache_size: 100_000,
            enable_persistence: true,
            persistence_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Knowledge graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphStats {
    pub total_entities: usize,
    pub total_relations: usize,
    pub entities_by_type: HashMap<EntityType, usize>,
    pub average_confidence: f64,
    pub graph_density: f64,
    pub connected_components: usize,
    pub last_updated: u64,
    pub processing_time: Duration,
    pub cache_hit_rate: f64,
    pub inference_rules_applied: usize,
}

/// Knowledge graph integrator
pub struct KnowledgeGraphIntegrator {
    config: KnowledgeGraphConfig,
    entities: Arc<RwLock<HashMap<String, Entity>>>,
    relations: Arc<RwLock<HashMap<String, Relation>>>,
    entity_index: Arc<RwLock<HashMap<String, Vec<String>>>>, // name -> entity_ids
    type_index: Arc<RwLock<HashMap<EntityType, Vec<String>>>>, // type -> entity_ids
    graph: Arc<RwLock<Graph<String, f64, Directed>>>,
    node_mapping: Arc<RwLock<HashMap<String, NodeIndex>>>,
    stats: Arc<Mutex<KnowledgeGraphStats>>,
    cache: Arc<RwLock<HashMap<String, Vec<String>>>>, // query -> results
    inference_rules: Arc<RwLock<Vec<InferenceRule>>>,
}

/// Inference rule for knowledge graph reasoning
#[derive(Debug, Clone)]
pub struct InferenceRule {
    pub id: String,
    pub name: String,
    pub pattern: String,
    pub conclusion: String,
    pub confidence: f64,
    pub enabled: bool,
}

/// Entity extraction result
#[derive(Debug, Clone)]
pub struct EntityExtractionResult {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub confidence: f64,
    pub processing_time: Duration,
    pub source_document: String,
}

/// Entity linking result
#[derive(Debug, Clone)]
pub struct EntityLinkingResult {
    pub entity_id: String,
    pub confidence: f64,
    pub match_type: MatchType,
    pub evidence: Vec<String>,
}

/// Match types for entity linking
#[derive(Debug, Clone, PartialEq)]
pub enum MatchType {
    Exact,
    Partial,
    Fuzzy,
    Semantic,
    Inferred,
}

impl KnowledgeGraphIntegrator {
    /// Create a new knowledge graph integrator
    pub fn new(config: KnowledgeGraphConfig) -> Self {
        let mut stats = KnowledgeGraphStats {
            total_entities: 0,
            total_relations: 0,
            entities_by_type: HashMap::new(),
            average_confidence: 0.0,
            graph_density: 0.0,
            connected_components: 0,
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            processing_time: Duration::from_secs(0),
            cache_hit_rate: 0.0,
            inference_rules_applied: 0,
        };

        // Initialize entity type counts
        for entity_type in [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Event,
            EntityType::Concept,
            EntityType::Product,
            EntityType::Technology,
            EntityType::Date,
            EntityType::Money,
            EntityType::Percent,
            EntityType::Unknown,
        ] {
            stats.entities_by_type.insert(entity_type, 0);
        }

        Self {
            config,
            entities: Arc::new(RwLock::new(HashMap::new())),
            relations: Arc::new(RwLock::new(HashMap::new())),
            entity_index: Arc::new(RwLock::new(HashMap::new())),
            type_index: Arc::new(RwLock::new(HashMap::new())),
            graph: Arc::new(RwLock::new(Graph::new())),
            node_mapping: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(stats)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            inference_rules: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add an entity to the knowledge graph
    pub fn add_entity(&self, entity: Entity) -> Result<(), String> {
        if entity.confidence < self.config.min_confidence {
            return Err("Entity confidence below threshold".to_string());
        }

        let entity_id = entity.id.clone();
        let entity_name = entity.name.clone();
        let entity_type = entity.entity_type.clone();

        // Add to entities map
        {
            let mut entities = self.entities.write().unwrap();
            entities.insert(entity_id.clone(), entity);
        }

        // Update indexes
        self.update_entity_index(&entity_id, &entity_name)?;
        self.update_type_index(&entity_id, &entity_type)?;

        // Add to graph
        self.add_entity_to_graph(&entity_id)?;

        // Update statistics
        self.update_stats()?;

        Ok(())
    }

    /// Add a relation to the knowledge graph
    pub fn add_relation(&self, relation: Relation) -> Result<(), String> {
        if relation.confidence < self.config.min_confidence {
            return Err("Relation confidence below threshold".to_string());
        }

        let relation_id = relation.id.clone();

        // Add to relations map
        {
            let mut relations = self.relations.write().unwrap();
            relations.insert(relation_id.clone(), relation);
        }

        // Add to graph
        self.add_relation_to_graph(&relation_id)?;

        // Apply inference rules if enabled
        if self.config.enable_inference {
            self.apply_inference_rules(&relation_id)?;
        }

        // Update statistics
        self.update_stats()?;

        Ok(())
    }

    /// Extract entities from text
    pub fn extract_entities(&self, text: &str, document_id: &str) -> Result<EntityExtractionResult, String> {
        let start_time = Instant::now();

        // Use NER model to extract entities (placeholder implementation)
        let entities = self.perform_ner_extraction(text, document_id)?;
        
        // Extract relations between entities
        let relations = self.extract_relations(&entities, text, document_id)?;

        // Calculate overall confidence
        let confidence = self.calculate_extraction_confidence(&entities, &relations);

        let processing_time = start_time.elapsed();

        Ok(EntityExtractionResult {
            entities,
            relations,
            confidence,
            processing_time,
            source_document: document_id.to_string(),
        })
    }

    /// Link entities to existing knowledge graph
    pub fn link_entities(&self, entities: &[Entity]) -> Result<Vec<EntityLinkingResult>, String> {
        let mut results = Vec::new();

        for entity in entities {
            let linking_result = self.link_single_entity(entity)?;
            results.push(linking_result);
        }

        Ok(results)
    }

    /// Search for entities by name
    pub fn search_entities(&self, query: &str, limit: usize) -> Result<Vec<Entity>, String> {
        // Check cache first
        if let Some(cached_results) = self.get_cached_results(query) {
            return Ok(cached_results);
        }

        let mut results = Vec::new();
        let query_lower = query.to_lowercase();

        // Search in entity index
        {
            let entity_index = self.entity_index.read().unwrap();
            let entities = self.entities.read().unwrap();

            for (name, entity_ids) in entity_index.iter() {
                if name.to_lowercase().contains(&query_lower) {
                    for entity_id in entity_ids {
                        if let Some(entity) = entities.get(entity_id) {
                            results.push(entity.clone());
                        }
                    }
                }
            }
        }

        // Sort by confidence and relevance
        results.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        if results.len() > limit {
            results.truncate(limit);
        }

        // Cache results
        self.cache_results(query, &results);

        Ok(results)
    }

    /// Get entity by ID
    pub fn get_entity(&self, entity_id: &str) -> Result<Option<Entity>, String> {
        let entities = self.entities.read().unwrap();
        Ok(entities.get(entity_id).cloned())
    }

    /// Get relations for an entity
    pub fn get_entity_relations(&self, entity_id: &str) -> Result<Vec<Relation>, String> {
        let mut relations = Vec::new();
        let relations_map = self.relations.read().unwrap();

        for relation in relations_map.values() {
            if relation.subject_id == entity_id || relation.object_id == entity_id {
                relations.push(relation.clone());
            }
        }

        Ok(relations)
    }

    /// Find shortest path between entities
    pub fn find_shortest_path(&self, from_entity_id: &str, to_entity_id: &str) -> Result<Vec<String>, String> {
        let graph = self.graph.read().unwrap();
        let node_mapping = self.node_mapping.read().unwrap();

        let from_node = node_mapping.get(from_entity_id)
            .ok_or_else(|| "Source entity not found in graph".to_string())?;
        let to_node = node_mapping.get(to_entity_id)
            .ok_or_else(|| "Target entity not found in graph".to_string())?;

        // Use Dijkstra's algorithm to find shortest path
        let path = dijkstra(&*graph, *from_node, Some(*to_node), |_| 1.0);

        if let Some(distance) = path.get(to_node) {
            // Reconstruct path (simplified implementation)
            Ok(vec![from_entity_id.to_string(), to_entity_id.to_string()])
        } else {
            Err("No path found between entities".to_string())
        }
    }

    /// Get knowledge graph statistics
    pub fn get_stats(&self) -> KnowledgeGraphStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear the knowledge graph
    pub fn clear(&self) -> Result<(), String> {
        {
            let mut entities = self.entities.write().unwrap();
            entities.clear();
        }

        {
            let mut relations = self.relations.write().unwrap();
            relations.clear();
        }

        {
            let mut entity_index = self.entity_index.write().unwrap();
            entity_index.clear();
        }

        {
            let mut type_index = self.type_index.write().unwrap();
            type_index.clear();
        }

        {
            let mut graph = self.graph.write().unwrap();
            *graph = Graph::new();
        }

        {
            let mut node_mapping = self.node_mapping.write().unwrap();
            node_mapping.clear();
        }

        {
            let mut cache = self.cache.write().unwrap();
            cache.clear();
        }

        self.update_stats()?;

        Ok(())
    }

    // Private helper methods

    fn perform_ner_extraction(&self, text: &str, document_id: &str) -> Result<Vec<Entity>, String> {
        // Placeholder NER implementation
        // In production, this would use a trained NER model
        let mut entities = Vec::new();

        // Simple pattern-based extraction for demonstration
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            if word.len() > 3 && word.chars().next().unwrap().is_uppercase() {
                let entity = Entity {
                    id: Uuid::new_v4().to_string(),
                    name: word.to_string(),
                    entity_type: EntityType::Unknown,
                    confidence: 0.7,
                    confidence_level: ConfidenceLevel::Medium,
                    aliases: vec![],
                    description: None,
                    properties: HashMap::new(),
                    embeddings: None,
                    created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                    updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                    source_documents: vec![document_id.to_string()],
                    metadata: HashMap::new(),
                };
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    fn extract_relations(&self, entities: &[Entity], text: &str, document_id: &str) -> Result<Vec<Relation>, String> {
        let mut relations = Vec::new();

        // Simple relation extraction based on proximity
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                if self.are_entities_related(&entities[i], &entities[j], text) {
                    let relation = Relation {
                        id: Uuid::new_v4().to_string(),
                        subject_id: entities[i].id.clone(),
                        predicate: "related_to".to_string(),
                        object_id: entities[j].id.clone(),
                        confidence: 0.6,
                        weight: 1.0,
                        evidence: vec![document_id.to_string()],
                        created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        metadata: HashMap::new(),
                    };
                    relations.push(relation);
                }
            }
        }

        Ok(relations)
    }

    fn are_entities_related(&self, entity1: &Entity, entity2: &Entity, text: &str) -> bool {
        // Simple proximity-based relation detection
        let pos1 = text.find(&entity1.name);
        let pos2 = text.find(&entity2.name);

        if let (Some(p1), Some(p2)) = (pos1, pos2) {
            let distance = (p1 as i32 - p2 as i32).abs();
            distance < 100 // Entities within 100 characters are considered related
        } else {
            false
        }
    }

    fn calculate_extraction_confidence(&self, entities: &[Entity], relations: &[Relation]) -> f64 {
        if entities.is_empty() {
            return 0.0;
        }

        let entity_confidence: f64 = entities.iter().map(|e| e.confidence).sum();
        let relation_confidence: f64 = relations.iter().map(|r| r.confidence).sum();
        
        (entity_confidence + relation_confidence) / (entities.len() + relations.len()) as f64
    }

    fn link_single_entity(&self, entity: &Entity) -> Result<EntityLinkingResult, String> {
        // Try exact match first
        if let Some(existing_entity) = self.find_exact_match(entity) {
            return Ok(EntityLinkingResult {
                entity_id: existing_entity.id,
                confidence: 1.0,
                match_type: MatchType::Exact,
                evidence: vec!["exact_name_match".to_string()],
            });
        }

        // Try fuzzy match
        if let Some((existing_entity, confidence)) = self.find_fuzzy_match(entity) {
            return Ok(EntityLinkingResult {
                entity_id: existing_entity.id,
                confidence,
                match_type: MatchType::Fuzzy,
                evidence: vec!["fuzzy_name_match".to_string()],
            });
        }

        // No match found
        Ok(EntityLinkingResult {
            entity_id: entity.id.clone(),
            confidence: 0.0,
            match_type: MatchType::Exact,
            evidence: vec![],
        })
    }

    fn find_exact_match(&self, entity: &Entity) -> Option<Entity> {
        let entity_index = self.entity_index.read().unwrap();
        let entities = self.entities.read().unwrap();

        if let Some(entity_ids) = entity_index.get(&entity.name.to_lowercase()) {
            for entity_id in entity_ids {
                if let Some(existing_entity) = entities.get(entity_id) {
                    if existing_entity.entity_type == entity.entity_type {
                        return Some(existing_entity.clone());
                    }
                }
            }
        }

        None
    }

    fn find_fuzzy_match(&self, entity: &Entity) -> Option<(Entity, f64)> {
        let entities = self.entities.read().unwrap();
        let mut best_match = None;
        let mut best_confidence = 0.0;

        for existing_entity in entities.values() {
            if existing_entity.entity_type == entity.entity_type {
                let similarity = self.calculate_string_similarity(&entity.name, &existing_entity.name);
                if similarity > 0.8 && similarity > best_confidence {
                    best_confidence = similarity;
                    best_match = Some(existing_entity.clone());
                }
            }
        }

        best_match.map(|entity| (entity, best_confidence))
    }

    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        // Simple Jaccard similarity
        let set1: HashSet<char> = s1.to_lowercase().chars().collect();
        let set2: HashSet<char> = s2.to_lowercase().chars().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn update_entity_index(&self, entity_id: &str, entity_name: &str) -> Result<(), String> {
        let mut entity_index = self.entity_index.write().unwrap();
        let key = entity_name.to_lowercase();
        
        entity_index.entry(key).or_insert_with(Vec::new).push(entity_id.to_string());
        
        Ok(())
    }

    fn update_type_index(&self, entity_id: &str, entity_type: &EntityType) -> Result<(), String> {
        let mut type_index = self.type_index.write().unwrap();
        
        type_index.entry(entity_type.clone()).or_insert_with(Vec::new).push(entity_id.to_string());
        
        Ok(())
    }

    fn add_entity_to_graph(&self, entity_id: &str) -> Result<(), String> {
        let mut graph = self.graph.write().unwrap();
        let mut node_mapping = self.node_mapping.write().unwrap();

        let node_index = graph.add_node(entity_id.to_string());
        node_mapping.insert(entity_id.to_string(), node_index);

        Ok(())
    }

    fn add_relation_to_graph(&self, relation_id: &str) -> Result<(), String> {
        let relations = self.relations.read().unwrap();
        let graph = self.graph.read().unwrap();
        let node_mapping = self.node_mapping.read().unwrap();

        if let Some(relation) = relations.get(relation_id) {
            if let (Some(&subject_node), Some(&object_node)) = (
                node_mapping.get(&relation.subject_id),
                node_mapping.get(&relation.object_id)
            ) {
                let mut graph = self.graph.write().unwrap();
                graph.add_edge(subject_node, object_node, relation.weight);
            }
        }

        Ok(())
    }

    fn apply_inference_rules(&self, relation_id: &str) -> Result<(), String> {
        let rules = self.inference_rules.read().unwrap();
        let mut stats = self.stats.lock().unwrap();

        for rule in rules.iter() {
            if rule.enabled {
                // Apply inference rule (placeholder implementation)
                stats.inference_rules_applied += 1;
            }
        }

        Ok(())
    }

    fn update_stats(&self) -> Result<(), String> {
        let mut stats = self.stats.lock().unwrap();
        let entities = self.entities.read().unwrap();
        let relations = self.relations.read().unwrap();

        stats.total_entities = entities.len();
        stats.total_relations = relations.len();
        stats.last_updated = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        // Update entity type counts
        for entity_type in stats.entities_by_type.keys().cloned().collect::<Vec<_>>() {
            stats.entities_by_type.insert(entity_type.clone(), 0);
        }

        for entity in entities.values() {
            let count = stats.entities_by_type.entry(entity.entity_type.clone()).or_insert(0);
            *count += 1;
        }

        // Calculate average confidence
        if !entities.is_empty() {
            let total_confidence: f64 = entities.values().map(|e| e.confidence).sum();
            stats.average_confidence = total_confidence / entities.len() as f64;
        }

        Ok(())
    }

    fn get_cached_results(&self, query: &str) -> Option<Vec<Entity>> {
        let cache = self.cache.read().unwrap();
        cache.get(query).cloned()
    }

    fn cache_results(&self, query: &str, results: &[Entity]) {
        let mut cache = self.cache.write().unwrap();
        cache.insert(query.to_string(), results.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph_integration() {
        let config = KnowledgeGraphConfig::default();
        let integrator = KnowledgeGraphIntegrator::new(config);

        // Create test entity
        let entity = Entity {
            id: "test_entity_1".to_string(),
            name: "Test Entity".to_string(),
            entity_type: EntityType::Person,
            confidence: 0.9,
            confidence_level: ConfidenceLevel::High,
            aliases: vec!["Test".to_string()],
            description: Some("A test entity".to_string()),
            properties: HashMap::new(),
            embeddings: None,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            source_documents: vec!["test_doc".to_string()],
            metadata: HashMap::new(),
        };

        // Add entity
        assert!(integrator.add_entity(entity).is_ok());

        // Search for entity
        let results = integrator.search_entities("Test", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Test Entity");

        // Get stats
        let stats = integrator.get_stats();
        assert_eq!(stats.total_entities, 1);
    }

    #[test]
    fn test_entity_extraction() {
        let config = KnowledgeGraphConfig::default();
        let integrator = KnowledgeGraphIntegrator::new(config);

        let text = "John Smith works at Microsoft Corporation in Seattle.";
        let result = integrator.extract_entities(text, "test_doc").unwrap();

        assert!(!result.entities.is_empty());
        assert!(result.confidence > 0.0);
    }
}
