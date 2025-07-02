# Spartan Mobile Memory Management Strategy

## Executive Summary

This document outlines a comprehensive strategy to replace Spartan's memory-intensive `Vec<Scalar>` allocations with mobile-optimized alternatives. The current implementation suffers from exponential memory growth (2^num_vars) making it unsuitable for mobile deployment. Our strategy introduces adaptive memory management with multiple fallback levels based on device constraints.

## Current Memory Issues Analysis

### Critical Problems Identified

1. **Exponential Memory Growth**
   - `dense_mlpoly.rs:71`: `vec![Scalar::one(); ell.pow2()]` → 2^ell allocations
   - `DensePolynomial::Z` → 2^num_vars evaluations 
   - For 20 variables: **1M Scalars = 32MB per polynomial**
   - For 24 variables: **16M Scalars = 512MB per polynomial**

2. **Linear Memory Growth**
   - R1CS variable assignments: O(num_vars)
   - Sparse matrix operations: O(num_entries) 
   - Commitment generators: O(commitment_size)

3. **Memory Hotspots by File**
   - `dense_mlpoly.rs`: Polynomial evaluation storage
   - `sparse_mlpoly.rs`: Matrix operations
   - `r1cs.rs` & `r1csproof.rs`: Constraint system variables
   - `commitments.rs`: Generator vectors

## Memory Management Strategy

### Phase 1: Dependency Updates

#### Required New Dependencies

```toml
[dependencies]
# Memory-efficient vector alternatives
tinyvec = "1.6.0"           # Lightweight vectors for mobile
smallvec = "1.11.0"         # Stack-allocated small vectors  
sized-chunks = "0.6.5"      # Fixed-size chunked processing
thin_vec = "0.2.12"         # Optimized Vec alternative

# Streaming and async processing
futures = "0.3.28"          # Async streaming primitives
tokio-stream = "0.1.14"     # Stream utilities
async-stream = "0.3.5"      # Async stream macros

# Memory mapping and lazy evaluation
memmap2 = "0.7.1"          # Memory-mapped files
once_cell = "1.18.0"       # Lazy static initialization
dashmap = "5.5.3"          # Concurrent HashMap

# Mobile-specific optimizations
cap = "0.1.2"              # Memory capacity limits
bytemuck = "1.14.0"        # Safe transmutation

[target.'cfg(any(target_os = "ios", target_os = "android"))'.dependencies]
# Mobile-specific memory management
jemalloc-sys = "0.5.4"     # Better memory allocator for mobile
```

### Phase 2: Core Memory Management Infrastructure

#### 2.1 Mobile Memory Manager

```rust
// src/mobile/memory_manager.rs
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use smallvec::SmallVec;
use tinyvec::TinyVec;

pub struct MobileMemoryManager {
    max_allocation_mb: usize,
    current_usage_mb: AtomicUsize,
    chunk_size: usize,
    platform: Platform,
}

#[derive(Debug, Clone)]
pub enum Platform {
    iOS { memory_gb: u8 },
    Android { memory_gb: u8, api_level: u32 },
    Desktop,
}

impl MobileMemoryManager {
    pub fn new_for_platform() -> Self {
        let platform = detect_platform();
        let (max_allocation_mb, chunk_size) = match platform {
            Platform::iOS { memory_gb } => {
                let max_mb = match memory_gb {
                    0..=2 => 256,   // Older devices
                    3..=4 => 512,   // Mid-range devices  
                    _ => 1024,      // High-end devices
                };
                (max_mb, 1024)
            },
            Platform::Android { memory_gb, .. } => {
                let max_mb = match memory_gb {
                    0..=2 => 512,   // Android can vary more
                    3..=6 => 1024,  
                    _ => 2048,
                };
                (max_mb, 1024)
            },
            Platform::Desktop => (usize::MAX, 8192),
        };

        Self {
            max_allocation_mb,
            current_usage_mb: AtomicUsize::new(0),
            chunk_size,
            platform,
        }
    }

    pub fn can_allocate(&self, size_bytes: usize) -> bool {
        let size_mb = size_bytes / (1024 * 1024);
        let current = self.current_usage_mb.load(Ordering::Relaxed);
        current + size_mb <= self.max_allocation_mb
    }

    pub fn allocate_tracked(&self, size_bytes: usize) -> Result<(), MemoryError> {
        if !self.can_allocate(size_bytes) {
            return Err(MemoryError::InsufficientMemory {
                requested: size_bytes,
                available: (self.max_allocation_mb - 
                           self.current_usage_mb.load(Ordering::Relaxed)) * 1024 * 1024,
            });
        }
        
        let size_mb = size_bytes / (1024 * 1024);
        self.current_usage_mb.fetch_add(size_mb, Ordering::Relaxed);
        Ok(())
    }

    pub fn deallocate_tracked(&self, size_bytes: usize) {
        let size_mb = size_bytes / (1024 * 1024);
        self.current_usage_mb.fetch_sub(size_mb, Ordering::Relaxed);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Insufficient memory: requested {requested} bytes, available {available} bytes")]
    InsufficientMemory { requested: usize, available: usize },
    #[error("Allocation too large for mobile platform")]
    AllocationTooLarge,
    #[error("Memory pressure detected")]
    MemoryPressure,
}
```

#### 2.2 Adaptive Vector Types

```rust
// src/mobile/adaptive_vectors.rs
use smallvec::SmallVec;
use tinyvec::TinyVec;
use crate::scalar::Scalar;

/// Adaptive vector that chooses storage strategy based on size and platform
pub enum AdaptiveScalarVec {
    /// Stack allocation for very small vectors (≤16 elements)
    Tiny(TinyVec<[Scalar; 16]>),
    /// Stack allocation for small vectors (≤64 elements)  
    Small(SmallVec<[Scalar; 64]>),
    /// Heap allocation for medium vectors with chunked processing
    Chunked(ChunkedVec<Scalar>),
    /// Streaming evaluation for large vectors
    Streamed(StreamingVec<Scalar>),
    /// Lazy evaluation for very large vectors
    Lazy(LazyVec<Scalar>),
}

impl AdaptiveScalarVec {
    pub fn new_for_size(estimated_size: usize, memory_manager: &MobileMemoryManager) -> Self {
        match estimated_size {
            0..=16 => Self::Tiny(TinyVec::new()),
            17..=64 => Self::Small(SmallVec::new()),
            65..=4096 => {
                let chunk_size = memory_manager.chunk_size;
                Self::Chunked(ChunkedVec::new(estimated_size, chunk_size))
            },
            4097..=65536 => Self::Streamed(StreamingVec::new(estimated_size)),
            _ => Self::Lazy(LazyVec::new(estimated_size)),
        }
    }

    pub fn get(&self, index: usize) -> Option<Scalar> {
        match self {
            Self::Tiny(v) => v.get(index).copied(),
            Self::Small(v) => v.get(index).copied(),
            Self::Chunked(v) => v.get(index),
            Self::Streamed(v) => v.get(index),
            Self::Lazy(v) => v.get(index),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Tiny(v) => v.len(),
            Self::Small(v) => v.len(),
            Self::Chunked(v) => v.len(),
            Self::Streamed(v) => v.len(),
            Self::Lazy(v) => v.len(),
        }
    }
}
```

### Phase 3: File-by-File Migration Strategy

#### 3.1 Dense Multilinear Polynomials (`dense_mlpoly.rs`)

**Current Issue**: `vec![Scalar::one(); ell.pow2()]` creates exponentially large vectors

**Strategy**: Replace with lazy evaluation and chunked processing

```rust
// Before (memory-intensive):
pub struct DensePolynomial {
    Z: Vec<Scalar>,  // 2^num_vars elements
    num_vars: usize,
}

impl DensePolynomial {
    pub fn new(Z: Vec<Scalar>) -> Self {
        // Allocates 2^num_vars * 32 bytes
        DensePolynomial { Z, num_vars: Z.len().trailing_zeros() as usize }
    }
}

// After (memory-efficient):
pub struct MobileDensePolynomial {
    evaluations: AdaptiveScalarVec,
    num_vars: usize,
    evaluator: Option<Box<dyn PolynomialEvaluator>>,
    memory_manager: Arc<MobileMemoryManager>,
}

impl MobileDensePolynomial {
    pub fn new_mobile(num_vars: usize, memory_manager: Arc<MobileMemoryManager>) -> Self {
        let estimated_size = 1 << num_vars; // 2^num_vars
        let evaluations = if estimated_size <= 4096 {
            // Small enough for direct storage
            AdaptiveScalarVec::new_for_size(estimated_size, &memory_manager)
        } else {
            // Use lazy evaluation for large polynomials
            AdaptiveScalarVec::Lazy(LazyVec::new_with_evaluator(
                estimated_size,
                Box::new(DensePolynomialEvaluator::new(num_vars))
            ))
        };

        Self {
            evaluations,
            num_vars,
            evaluator: None,
            memory_manager,
        }
    }

    pub fn evaluate(&self, index: usize) -> Scalar {
        self.evaluations.get(index).unwrap_or_else(|| {
            // Lazy evaluation fallback
            self.compute_evaluation_at_index(index)
        })
    }
}

// Lazy evaluation strategy
pub struct LazyVec<T> {
    cache: DashMap<usize, T>,
    evaluator: Box<dyn Fn(usize) -> T + Send + Sync>,
    len: usize,
}

impl LazyVec<Scalar> {
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
            self.cache.insert(index, value);
            Some(value)
        }
    }
}
```

#### 3.2 Sparse Multilinear Polynomials (`sparse_mlpoly.rs`)

**Current Issue**: Large vector allocations for matrix operations

**Strategy**: Use sparse representations and chunked processing

```rust
// Before:
impl SparsePolynomial {
    pub fn multiply_vec(&self, z: &[Scalar]) -> Vec<Scalar> {
        let mut result = vec![Scalar::zero(); self.num_vars];  // Large allocation
        // ... matrix multiplication
        result
    }
}

// After:
impl MobileSparsePolynomial {
    pub fn multiply_vec_chunked(&self, z: &[Scalar], chunk_size: usize) -> ChunkedResult<Scalar> {
        ChunkedResult::new_with_processor(
            self.num_vars,
            chunk_size,
            Box::new(move |start, end| {
                let mut chunk_result = SmallVec::<[Scalar; 64]>::new();
                for i in start..end {
                    chunk_result.push(self.multiply_row(i, z));
                }
                chunk_result.into_vec()
            })
        )
    }

    pub async fn multiply_vec_streaming(&self, z: &[Scalar]) -> impl Stream<Item = Scalar> {
        use async_stream::stream;
        
        stream! {
            for i in 0..self.num_vars {
                yield self.multiply_row(i, z);
            }
        }
    }
}

pub struct ChunkedResult<T> {
    chunks: Vec<SmallVec<[T; 64]>>,
    chunk_size: usize,
    total_len: usize,
}

impl<T: Clone> ChunkedResult<T> {
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.total_len {
            return None;
        }
        
        let chunk_idx = index / self.chunk_size;
        let local_idx = index % self.chunk_size;
        
        self.chunks.get(chunk_idx)?.get(local_idx).cloned()
    }
}
```

#### 3.3 R1CS Operations (`r1cs.rs` & `r1csproof.rs`)

**Current Issue**: Large variable assignment vectors

**Strategy**: Segmented storage with overflow handling

```rust
// Before:
pub struct VarsAssignment {
    assignment: Vec<Scalar>,  // Can be very large
}

// After:
pub struct MobileVarsAssignment {
    segments: Vec<SmallVec<[Scalar; 256]>>,  // 256 scalars per segment = 8KB
    segment_size: usize,
    total_len: usize,
    memory_manager: Arc<MobileMemoryManager>,
}

impl MobileVarsAssignment {
    pub fn new_mobile(assignment: &[Scalar], memory_manager: Arc<MobileMemoryManager>) -> Result<Self, MemoryError> {
        const SEGMENT_SIZE: usize = 256;
        let total_len = assignment.len();
        
        // Check if we can allocate all segments
        let total_memory = total_len * std::mem::size_of::<Scalar>();
        memory_manager.allocate_tracked(total_memory)?;
        
        let mut segments = Vec::new();
        for chunk in assignment.chunks(SEGMENT_SIZE) {
            let mut segment = SmallVec::new();
            segment.extend_from_slice(chunk);
            segments.push(segment);
        }
        
        Ok(Self {
            segments,
            segment_size: SEGMENT_SIZE,
            total_len,
            memory_manager,
        })
    }

    pub fn get(&self, index: usize) -> Option<Scalar> {
        if index >= self.total_len {
            return None;
        }
        
        let segment_idx = index / self.segment_size;
        let local_idx = index % self.segment_size;
        
        self.segments.get(segment_idx)?.get(local_idx).copied()
    }

    pub fn iter_chunked(&self, chunk_size: usize) -> ChunkedIterator<Scalar> {
        ChunkedIterator::new(self, chunk_size)
    }
}

impl Drop for MobileVarsAssignment {
    fn drop(&mut self) {
        let total_memory = self.total_len * std::mem::size_of::<Scalar>();
        self.memory_manager.deallocate_tracked(total_memory);
    }
}
```

#### 3.4 Commitments (`commitments.rs`)

**Current Issue**: Large generator vectors

**Strategy**: Lazy generator computation with caching

```rust
// Before:
pub struct MultiCommitGens {
    G: Vec<GroupElement>,  // Large generator vector
    h: GroupElement,
}

// After:
pub struct MobileMultiCommitGens {
    generator_cache: Arc<DashMap<usize, GroupElement>>,
    base_generator: GroupElement,
    max_generators: usize,
    memory_manager: Arc<MobileMemoryManager>,
}

impl MobileMultiCommitGens {
    pub fn new_mobile(label: &[u8], n: usize, memory_manager: Arc<MobileMemoryManager>) -> Self {
        Self {
            generator_cache: Arc::new(DashMap::new()),
            base_generator: GroupElement::from_uniform_bytes(label),
            max_generators: n,
            memory_manager,
        }
    }

    pub fn get_generator(&self, index: usize) -> Option<GroupElement> {
        if index >= self.max_generators {
            return None;
        }

        if let Some(cached) = self.generator_cache.get(&index) {
            Some(*cached)
        } else {
            // Compute generator on-demand
            let generator = self.compute_generator(index);
            
            // Only cache if we have memory budget
            if self.generator_cache.len() < 1000 {  // Limit cache size
                self.generator_cache.insert(index, generator);
            }
            
            Some(generator)
        }
    }

    fn compute_generator(&self, index: usize) -> GroupElement {
        // Deterministic generator computation
        let mut hasher = Sha3_512::new();
        hasher.update(b"mobile_generator");
        hasher.update(&index.to_le_bytes());
        GroupElement::from_uniform_bytes(&hasher.finalize()[..])
    }
}
```

### Phase 4: Algorithm Adaptation Layer

#### 4.1 Mobile Algorithm Selector

```rust
// src/mobile/algorithm_selector.rs
pub struct MobileAlgorithmSelector {
    memory_manager: Arc<MobileMemoryManager>,
    performance_profiler: PerformanceProfiler,
}

impl MobileAlgorithmSelector {
    pub fn select_snark_strategy(&self, num_vars: usize, num_cons: usize) -> SNARKStrategy {
        let estimated_memory = estimate_snark_memory_usage(num_vars, num_cons);
        let available_memory = self.memory_manager.get_available_memory();
        
        match estimated_memory {
            mem if mem <= available_memory / 4 => {
                SNARKStrategy::FullInMemory
            },
            mem if mem <= available_memory / 2 => {
                SNARKStrategy::ChunkedProcessing { chunk_size: 1024 }
            },
            mem if mem <= available_memory => {
                SNARKStrategy::StreamingWithCache { cache_size: 512 }
            },
            _ => {
                SNARKStrategy::MinimalMemory { 
                    use_disk_cache: true,
                    max_memory_mb: available_memory / (1024 * 1024) / 2
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum SNARKStrategy {
    FullInMemory,
    ChunkedProcessing { chunk_size: usize },
    StreamingWithCache { cache_size: usize },
    MinimalMemory { use_disk_cache: bool, max_memory_mb: usize },
}
```

#### 4.2 Fallback Mechanisms

```rust
// src/mobile/fallback.rs
pub struct MobileFallbackHandler {
    strategies: Vec<SNARKStrategy>,
    current_strategy_index: usize,
}

impl MobileFallbackHandler {
    pub fn new() -> Self {
        Self {
            strategies: vec![
                SNARKStrategy::FullInMemory,
                SNARKStrategy::ChunkedProcessing { chunk_size: 1024 },
                SNARKStrategy::StreamingWithCache { cache_size: 256 },
                SNARKStrategy::MinimalMemory { use_disk_cache: true, max_memory_mb: 128 },
            ],
            current_strategy_index: 0,
        }
    }

    pub async fn execute_with_fallback<T, F, Fut>(&mut self, operation: F) -> Result<T, ProofError>
    where
        F: Fn(SNARKStrategy) -> Fut + Clone,
        Fut: Future<Output = Result<T, ProofError>>,
    {
        for (index, strategy) in self.strategies.iter().enumerate() {
            self.current_strategy_index = index;
            
            match operation(strategy.clone()).await {
                Ok(result) => return Ok(result),
                Err(ProofError::MemoryError(_)) => {
                    // Try next strategy
                    continue;
                },
                Err(other) => return Err(other),
            }
        }
        
        Err(ProofError::AllStrategiesFailed)
    }
}
```

### Phase 5: Testing and Validation

#### 5.1 Memory Usage Testing

```rust
// tests/mobile_memory_tests.rs
#[cfg(test)]
mod mobile_memory_tests {
    use super::*;
    
    #[test]
    fn test_small_polynomial_memory_usage() {
        let memory_manager = Arc::new(MobileMemoryManager::new_for_platform());
        let poly = MobileDensePolynomial::new_mobile(10, memory_manager.clone()); // 2^10 = 1024 elements
        
        // Should use SmallVec for small polynomials
        assert!(memory_manager.current_usage_mb.load(Ordering::Relaxed) < 10);
    }
    
    #[test]
    fn test_large_polynomial_lazy_evaluation() {
        let memory_manager = Arc::new(MobileMemoryManager::new_for_platform());
        let poly = MobileDensePolynomial::new_mobile(20, memory_manager.clone()); // 2^20 = 1M elements
        
        // Should use lazy evaluation, minimal memory upfront
        assert!(memory_manager.current_usage_mb.load(Ordering::Relaxed) < 50);
        
        // Access a few elements
        let _val1 = poly.evaluate(0);
        let _val2 = poly.evaluate(100);
        
        // Memory usage should still be reasonable
        assert!(memory_manager.current_usage_mb.load(Ordering::Relaxed) < 100);
    }
    
    #[test]
    fn test_memory_pressure_handling() {
        let mut memory_manager = MobileMemoryManager::new_for_platform();
        memory_manager.max_allocation_mb = 10; // Artificially low limit
        
        let result = MobileVarsAssignment::new_mobile(
            &vec![Scalar::zero(); 100000], // Large assignment
            Arc::new(memory_manager)
        );
        
        assert!(matches!(result, Err(MemoryError::InsufficientMemory { .. })));
    }
}
```

### Phase 6: Performance Benchmarks

#### 6.1 Mobile Performance Tests

```rust
// benches/mobile_benchmarks.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_polynomial_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_evaluation");
    
    for num_vars in [10, 12, 14, 16, 18].iter() {
        group.bench_with_input(
            BenchmarkId::new("original", num_vars),
            num_vars,
            |b, &num_vars| {
                let poly = DensePolynomial::new(vec![Scalar::random(); 1 << num_vars]);
                b.iter(|| poly.evaluate(&vec![Scalar::random(); num_vars]))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("mobile_optimized", num_vars),
            num_vars,
            |b, &num_vars| {
                let memory_manager = Arc::new(MobileMemoryManager::new_for_platform());
                let poly = MobileDensePolynomial::new_mobile(num_vars, memory_manager);
                b.iter(|| poly.evaluate_mobile(&vec![Scalar::random(); num_vars]))
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_polynomial_evaluation);
criterion_main!(benches);
```

## Implementation Timeline

### Week 1-2: Infrastructure Setup
- [ ] Add new dependencies to Cargo.toml
- [ ] Create mobile memory manager
- [ ] Implement adaptive vector types
- [ ] Set up testing framework

### Week 3-4: Core Algorithm Migration
- [ ] Migrate dense_mlpoly.rs to lazy evaluation
- [ ] Update sparse_mlpoly.rs for chunked processing
- [ ] Implement segmented variable assignments

### Week 5-6: Integration and Testing
- [ ] Update SNARK/NIZK APIs to use mobile types
- [ ] Implement fallback mechanisms
- [ ] Comprehensive memory testing

### Week 7-8: Optimization and Validation
- [ ] Performance benchmarking
- [ ] Mobile device testing
- [ ] Documentation and examples

## Migration Checklist

### Phase 1: Preparation
- [ ] Backup current implementation
- [ ] Create feature flag for mobile optimizations
- [ ] Set up continuous integration for mobile targets

### Phase 2: Core Migration
- [ ] Replace `Vec<Scalar>` with `AdaptiveScalarVec` in dense_mlpoly.rs
- [ ] Update sparse_mlpoly.rs matrix operations
- [ ] Migrate commitment generators to lazy evaluation
- [ ] Update R1CS variable assignments

### Phase 3: API Updates
- [ ] Update public APIs to accept memory managers
- [ ] Add mobile-specific configuration options
- [ ] Implement backward compatibility layer

### Phase 4: Testing
- [ ] Unit tests for all new components
- [ ] Integration tests with real mobile constraints
- [ ] Performance regression testing
- [ ] Memory leak detection

### Phase 5: Documentation
- [ ] Update API documentation
- [ ] Create mobile deployment guide
- [ ] Performance tuning recommendations
- [ ] Troubleshooting guide

## Success Metrics

### Memory Usage Targets
- **Small proofs (≤2^12 vars)**: ≤64MB peak memory
- **Medium proofs (≤2^16 vars)**: ≤256MB peak memory  
- **Large proofs (≤2^20 vars)**: ≤512MB peak memory (with streaming)

### Performance Targets
- **Memory allocation overhead**: <10% vs original
- **Proof generation**: 2-5x slower acceptable for mobile
- **Proof verification**: <50% slower vs original

### Compatibility Targets
- **iOS**: Support iOS 12+ on devices with ≥2GB RAM
- **Android**: Support API level 21+ on devices with ≥3GB RAM
- **Battery impact**: <5% drain for typical proof operations

## Risk Mitigation

### Technical Risks
1. **Performance degradation**: Mitigated by multiple algorithm strategies
2. **Memory fragmentation**: Addressed by consistent chunk sizes
3. **Cache thrashing**: Prevented by intelligent cache eviction

### Platform Risks
1. **iOS memory limits**: Handled by conservative memory budgets
2. **Android fragmentation**: Addressed by device capability detection
3. **Background processing**: Solved by pauseable operations

## Conclusion

This strategy transforms Spartan from a memory-intensive desktop library to a mobile-friendly cryptographic toolkit. By implementing adaptive memory management with multiple fallback strategies, we can support proof generation and verification on resource-constrained mobile devices while maintaining the security guarantees of the original implementation.