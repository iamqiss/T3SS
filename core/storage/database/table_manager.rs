// T3SS Project
// File: core/storage/database/table_manager.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use serde::{Deserialize, Serialize};

/// Table manager for database operations
pub struct TableManager {
    tables: Arc<RwLock<HashMap<String, Table>>>,
}

/// Database table representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    pub columns: Vec<Column>,
    pub indexes: Vec<Index>,
    pub constraints: Vec<Constraint>,
}

/// Table column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default_value: Option<String>,
}

/// Data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Integer,
    String,
    Boolean,
    Float,
    DateTime,
}

/// Table index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub name: String,
    pub columns: Vec<String>,
    pub unique: bool,
}

/// Table constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub columns: Vec<String>,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    PrimaryKey,
    ForeignKey,
    Unique,
    Check,
}

impl TableManager {
    pub fn new() -> Self {
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn create_table(&self, table: Table) -> Result<(), String> {
        let mut tables = self.tables.write().unwrap();
        tables.insert(table.name.clone(), table);
        Ok(())
    }

    pub fn get_table(&self, name: &str) -> Option<Table> {
        let tables = self.tables.read().unwrap();
        tables.get(name).cloned()
    }
}