//! Adaptive vector types for mobile memory management
//! 
//! This module provides memory-efficient vector alternatives that adapt their storage
//! strategy based on size and platform constraints.

#[cfg(feature = "mobile")]
use smallvec::SmallVec;
#[cfg(feature = "mobile")]
use tinyvec::ArrayVec;
#[cfg(feature = "mobile")]
use dashmap::DashMap;
#[cfg(feature = "mobile")]
use futures::Stream;
#[cfg(feature = "mobile")]
use async_stream::stream;

use crate::scalar::Scalar;
use super::memory_manager::{MobileMemoryManager, MemoryError};

/// Adaptive vector that chooses storage strategy based on size and platform
#[cfg(feature = "mobile")]
pub enum AdaptiveScalarVec {
    /// Stack allocation for very small vectors (≤16 elements)
    Tiny(ArrayVec<[Scalar; 16]>),
    /// Stack allocation for small vectors (≤64 elements)  
    Small(SmallVec<[Scalar; 64]>),
    /// Heap allocation for medium vectors with chunked processing
    Chunked(ChunkedVec<Scalar>),
    /// Streaming evaluation for large vectors
    Streamed(StreamingVec<Scalar>),
    /// Lazy evaluation for very large vectors
    Lazy(LazyVec<Scalar>),
}

#[cfg(feature = "mobile")]
impl AdaptiveScalarVec {
    /// Create a new adaptive vector optimized for the given size
    pub fn new_for_size(estimated_size: usize, memory_manager: &MobileMemoryManager) -> Self {
        match estimated_size {
            0..=16 => Self::Tiny(ArrayVec::new()),
            17..=64 => Self::Small(SmallVec::new()),
            65..=4096 => {
                let chunk_size = memory_manager.get_chunk_size();
                Self::Chunked(ChunkedVec::new(estimated_size, chunk_size))
            },
            4097..=65536 => Self::Streamed(StreamingVec::new(estimated_size)),
            _ => Self::Lazy(LazyVec::new(estimated_size)),
        }
    }

    /// Get an element by index
    pub fn get(&self, index: usize) -> Option<Scalar> {
        match self {
            Self::Tiny(v) => v.get(index).copied(),
            Self::Small(v) => v.get(index).copied(),
            Self::Chunked(v) => v.get(index),
            Self::Streamed(v) => v.get(index),
            Self::Lazy(v) => v.get(index),
        }
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        match self {
            Self::Tiny(v) => v.len(),
            Self::Small(v) => v.len(),
            Self::Chunked(v) => v.len(),
            Self::Streamed(v) => v.len(),
            Self::Lazy(v) => v.len(),
        }
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a new element (if supported by the storage type)
    pub fn push(&mut self, value: Scalar) -> Result<(), MemoryError> {
        match self {
            Self::Tiny(v) => {
                v.push(value);
                Ok(())
            },
            Self::Small(v) => {
                v.push(value);
                Ok(())
            },
            Self::Chunked(v) => v.push(value),
            Self::Streamed(_) => Err(MemoryError::AllocationTooLarge),
            Self::Lazy(_) => Err(MemoryError::AllocationTooLarge),
        }
    }
}

/// Chunked vector for medium-sized data
#[cfg(feature = "mobile")]
pub struct ChunkedVec<T> {
    chunks: Vec<SmallVec<[T; 64]>>,
    chunk_size: usize,
    total_len: usize,
}

#[cfg(feature = "mobile")]
impl<T: Clone> ChunkedVec<T> {
    pub fn new(estimated_size: usize, chunk_size: usize) -> Self {
        let num_chunks = (estimated_size + chunk_size - 1) / chunk_size;
        Self {
            chunks: Vec::with_capacity(num_chunks),
            chunk_size,
            total_len: 0,
        }
    }

    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.total_len {
            return None;
        }
        
        let chunk_idx = index / self.chunk_size;
        let local_idx = index % self.chunk_size;
        
        self.chunks.get(chunk_idx)?.get(local_idx).cloned()
    }

    pub fn len(&self) -> usize {
        self.total_len
    }

    pub fn push(&mut self, value: T) -> Result<(), MemoryError> {
        let chunk_idx = self.total_len / self.chunk_size;

        // Ensure we have enough chunks
        while self.chunks.len() <= chunk_idx {
            self.chunks.push(SmallVec::new());
        }

        self.chunks[chunk_idx].push(value);
        self.total_len += 1;
        Ok(())
    }
}

/// Streaming vector for large data that doesn't fit in memory
#[cfg(feature = "mobile")]
pub struct StreamingVec<T> {
    cache: DashMap<usize, T>,
    evaluator: Option<Box<dyn Fn(usize) -> T + Send + Sync>>,
    len: usize,
}

#[cfg(feature = "mobile")]
impl<T: Clone> StreamingVec<T> {
    pub fn new(len: usize) -> Self {
        Self {
            cache: DashMap::new(),
            evaluator: None,
            len,
        }
    }

    pub fn new_with_evaluator(len: usize, evaluator: Box<dyn Fn(usize) -> T + Send + Sync>) -> Self {
        Self {
            cache: DashMap::new(),
            evaluator: Some(evaluator),
            len,
        }
    }

    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }

        if let Some(cached) = self.cache.get(&index) {
            Some(cached.clone())
        } else if let Some(ref evaluator) = self.evaluator {
            let value = evaluator(index);
            // Only cache if we have reasonable cache size
            if self.cache.len() < 1000 {
                self.cache.insert(index, value.clone());
            }
            Some(value)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub async fn stream(&self) -> impl Stream<Item = T> + '_ {
        stream! {
            for i in 0..self.len {
                if let Some(value) = self.get(i) {
                    yield value;
                }
            }
        }
    }
}

/// Lazy vector for very large data with on-demand evaluation
#[cfg(feature = "mobile")]
pub struct LazyVec<T> {
    cache: DashMap<usize, T>,
    evaluator: Box<dyn Fn(usize) -> T + Send + Sync>,
    len: usize,
}

#[cfg(feature = "mobile")]
impl LazyVec<Scalar> {
    pub fn new(len: usize) -> Self {
        Self {
            cache: DashMap::new(),
            evaluator: Box::new(|_| Scalar::zero()),
            len,
        }
    }

    pub fn new_with_evaluator(len: usize, evaluator: Box<dyn Fn(usize) -> Scalar + Send + Sync>) -> Self {
        Self {
            cache: DashMap::new(),
            evaluator,
            len,
        }
    }

    pub fn get(&self, index: usize) -> Option<Scalar> {
        if index >= self.len {
            return None;
        }

        if let Some(cached) = self.cache.get(&index) {
            Some(*cached)
        } else {
            let value = (self.evaluator)(index);
            // Only cache if we have memory budget
            if self.cache.len() < 1000 {
                self.cache.insert(index, value);
            }
            Some(value)
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

// Stub implementations when mobile feature is not enabled
#[cfg(not(feature = "mobile"))]
/// Stub adaptive vector when mobile feature is not enabled
pub struct AdaptiveScalarVec {
    data: Vec<Scalar>,
}

#[cfg(not(feature = "mobile"))]
impl AdaptiveScalarVec {
    /// Create a new adaptive vector (stub implementation)
    pub fn new_for_size(estimated_size: usize, _memory_manager: &MobileMemoryManager) -> Self {
        Self {
            data: Vec::with_capacity(estimated_size),
        }
    }

    /// Get element at index (stub implementation)
    pub fn get(&self, index: usize) -> Option<Scalar> {
        self.data.get(index).copied()
    }

    /// Get length (stub implementation)
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty (stub implementation)
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Push element (stub implementation)
    pub fn push(&mut self, value: Scalar) -> Result<(), MemoryError> {
        self.data.push(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "mobile")]
    use super::*;

    #[cfg(feature = "mobile")]
    #[test]
    fn test_adaptive_vector_creation() {
        let memory_manager = MobileMemoryManager::new_for_platform();
        
        // Test tiny vector
        let tiny = AdaptiveScalarVec::new_for_size(10, &memory_manager);
        assert!(matches!(tiny, AdaptiveScalarVec::Tiny(_)));
        
        // Test small vector
        let small = AdaptiveScalarVec::new_for_size(50, &memory_manager);
        assert!(matches!(small, AdaptiveScalarVec::Small(_)));
        
        // Test chunked vector
        let chunked = AdaptiveScalarVec::new_for_size(1000, &memory_manager);
        assert!(matches!(chunked, AdaptiveScalarVec::Chunked(_)));
    }

    #[cfg(feature = "mobile")]
    #[test]
    fn test_chunked_vector() {
        let mut chunked = ChunkedVec::new(100, 10);
        
        // Add some elements
        for i in 0..50 {
            chunked.push(Scalar::from(i as u64)).unwrap();
        }
        
        assert_eq!(chunked.len(), 50);
        assert_eq!(chunked.get(0), Some(Scalar::zero()));
        assert_eq!(chunked.get(49), Some(Scalar::from(49u64)));
        assert_eq!(chunked.get(50), None);
    }

    #[cfg(feature = "mobile")]
    #[test]
    fn test_lazy_vector() {
        let lazy = LazyVec::new_with_evaluator(
            100,
            Box::new(|index| Scalar::from(index as u64))
        );
        
        assert_eq!(lazy.len(), 100);
        assert_eq!(lazy.get(0), Some(Scalar::zero()));
        assert_eq!(lazy.get(42), Some(Scalar::from(42u64)));
        assert_eq!(lazy.get(100), None);
    }
}