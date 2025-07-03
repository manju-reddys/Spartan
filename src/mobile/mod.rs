//! Internal mobile-optimized memory management for Spartan
//! 
//! This module provides automatic memory optimization that is transparent to the public API.
//! All optimizations are applied automatically based on the target platform.

mod memory_manager;
mod adaptive_vectors;
pub(crate) mod internal;

// Internal re-exports for use within the library
pub(crate) use internal::{
    InternalVector, create_vector_with_value, create_ones_vector, create_zeros_vector, collect_to_vector
};