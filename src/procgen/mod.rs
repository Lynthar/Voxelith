//! Procedural generation algorithms.
//!
//! This module provides:
//! - Noise-based terrain generation
//! - Wave Function Collapse
//! - L-System vegetation
//! - Shape grammars for buildings
//!
//! All generators implement the `VoxelGenerator` trait for uniform access.

// TODO: Implement procedural generation algorithms

// Core types will be used when implementing generators
#[allow(unused_imports)]
use crate::core::Voxel;
use std::time::Duration;

/// Result type for generation operations
pub type GenResult<T> = Result<T, GenError>;

/// Generation error types
#[derive(Debug, thiserror::Error)]
pub enum GenError {
    #[error("Generation failed: {0}")]
    Failed(String),
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    #[error("Generation timeout")]
    Timeout,
}

/// Generator category for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorCategory {
    Terrain,
    Building,
    Character,
    Prop,
    Vegetation,
    General,
}

/// Generator backend type (for future AI integration)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorBackend {
    /// Traditional algorithmic generation (WFC, Noise, etc.)
    Algorithmic,
    /// Local AI model inference
    LocalModel,
    /// Remote API call
    RemoteAPI,
    /// Hybrid approach
    Hybrid,
}

/// Metadata for a generator
pub struct GeneratorMeta {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: GeneratorCategory,
    pub backend: GeneratorBackend,
}

/// Trait for all voxel generators (algorithmic and AI)
/// This unified interface allows mixing traditional and AI generators
pub trait VoxelGenerator: Send + Sync {
    /// Get generator metadata
    fn metadata(&self) -> GeneratorMeta;

    /// Estimate generation time (for UI progress)
    fn estimate_duration(&self, params: &GeneratorParams) -> Duration;

    /// Check if generator supports incremental/partial generation
    fn supports_incremental(&self) -> bool {
        false
    }
}

/// Parameters for generation (placeholder)
pub struct GeneratorParams {
    pub seed: u64,
    pub dimensions: [u32; 3],
}
