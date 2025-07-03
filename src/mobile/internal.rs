//! Internal memory management that automatically optimizes for the target platform
//! This module is not exposed in the public API - optimizations are automatic and transparent

use crate::scalar::Scalar;
use super::memory_manager::MobileMemoryManager;
use super::adaptive_vectors::AdaptiveScalarVec;

/// Global memory manager singleton - automatically initialized based on platform
static MEMORY_MANAGER: std::sync::OnceLock<MobileMemoryManager> = std::sync::OnceLock::new();

/// Get the global memory manager, initializing it on first access
fn get_memory_manager() -> &'static MobileMemoryManager {
    MEMORY_MANAGER.get_or_init(|| MobileMemoryManager::new_for_platform())
}

/// Internal vector type that automatically optimizes for platform and size
/// This replaces Vec<Scalar> throughout the codebase transparently
pub(crate) struct InternalVector {
    inner: AdaptiveScalarVec,
}

impl InternalVector {
    /// Create a new vector with automatic optimization
    pub(crate) fn new() -> Self {
        Self {
            inner: AdaptiveScalarVec::new_for_size(0, get_memory_manager()),
        }
    }

    /// Create a new vector with a size hint for optimization
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: AdaptiveScalarVec::new_for_size(capacity, get_memory_manager()),
        }
    }

    /// Create from existing data (replaces Vec::from)
    pub(crate) fn from_vec(data: Vec<Scalar>) -> Self {
        let mut vec = Self::with_capacity(data.len());
        for item in data {
            vec.push(item);
        }
        vec
    }

    /// Push an element
    pub(crate) fn push(&mut self, value: Scalar) {
        let _ = self.inner.push(value); // Ignore memory errors in internal use
    }

    /// Get element by index
    pub(crate) fn get(&self, index: usize) -> Option<Scalar> {
        self.inner.get(index)
    }

    /// Get length
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub(crate) fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterator over elements (for compatibility with Vec<Scalar>)
    pub(crate) fn iter(&self) -> InternalVectorIter {
        InternalVectorIter {
            vector: self,
            index: 0,
        }
    }

    /// Convert to Vec<Scalar> when needed for existing APIs
    pub(crate) fn to_vec(&self) -> Vec<Scalar> {
        (0..self.len())
            .filter_map(|i| self.get(i))
            .collect()
    }

    /// Get element by index with panic (for compatibility with Vec indexing)
    pub(crate) fn index(&self, index: usize) -> Scalar {
        self.get(index).expect("Index out of bounds")
    }

    /// Set element by index (grows vector if needed)
    pub(crate) fn set(&mut self, index: usize, value: Scalar) {
        // Ensure vector is large enough
        while self.len() <= index {
            self.push(Scalar::zero());
        }
        // For now, we'll need to rebuild - this is inefficient but maintains compatibility
        let mut new_vec = self.to_vec();
        if index < new_vec.len() {
            new_vec[index] = value;
        }
        *self = Self::from_vec(new_vec);
    }
}

/// Iterator for InternalVector
pub(crate) struct InternalVectorIter<'a> {
    vector: &'a InternalVector,
    index: usize,
}

impl<'a> Iterator for InternalVectorIter<'a> {
    type Item = Scalar;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.vector.len() {
            let result = self.vector.get(self.index);
            self.index += 1;
            result
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for InternalVectorIter<'a> {
    fn len(&self) -> usize {
        self.vector.len().saturating_sub(self.index)
    }
}

/// Factory function to create optimized vectors - replaces vec![] macros
pub(crate) fn create_vector_with_value(value: Scalar, size: usize) -> InternalVector {
    let mut vec = InternalVector::with_capacity(size);
    for _ in 0..size {
        vec.push(value);
    }
    vec
}

/// Create a vector filled with ones - optimized replacement for vec![Scalar::one(); size]
pub(crate) fn create_ones_vector(size: usize) -> InternalVector {
    create_vector_with_value(Scalar::one(), size)
}

/// Create a vector filled with zeros - optimized replacement for vec![Scalar::zero(); size]  
pub(crate) fn create_zeros_vector(size: usize) -> InternalVector {
    create_vector_with_value(Scalar::zero(), size)
}

/// Factory function to create vectors from iterators
pub(crate) fn collect_to_vector<I>(iter: I) -> InternalVector 
where 
    I: Iterator<Item = Scalar>,
    I: ExactSizeIterator,
{
    let mut vec = InternalVector::with_capacity(iter.len());
    for item in iter {
        vec.push(item);
    }
    vec
}

/// Check if platform optimizations are available
pub(crate) fn has_mobile_optimizations() -> bool {
    cfg!(feature = "mobile")
}

/// Get platform info for debugging (internal use only)
#[allow(dead_code)]
pub(crate) fn platform_info() -> String {
    #[cfg(feature = "mobile")]
    {
        format!("Mobile optimizations enabled: {:?}", get_memory_manager().get_platform())
    }
    #[cfg(not(feature = "mobile"))]
    {
        "Standard mode".to_string()
    }
}