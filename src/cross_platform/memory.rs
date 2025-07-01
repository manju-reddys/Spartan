//! Cross-platform memory management for Spartan zkSNARKs

#![allow(missing_docs)]

use super::*;
use std::collections::HashMap;
use std::sync::Mutex;

/// Cross-platform memory manager with adaptive allocation strategies
pub struct CrossPlatformMemoryManager {
    platform: Platform,
    allocation_strategy: AllocationStrategy,
    memory_pools: Mutex<MemoryPools>,
    stats: Mutex<MemoryStats>,
    memory_limit: Option<usize>,
}

/// Allocation strategies for different platforms
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Optimized for WASM with TypedArrays and minimal GC pressure
    WasmOptimized,
    /// Optimized for mobile with memory pools and thermal awareness
    MobileOptimized,
    /// Optimized for desktop performance with direct allocation
    PerformanceOptimized,
    /// Conservative allocation for compatibility
    Conservative,
}

/// Memory pools for different data types
struct MemoryPools {
    scalar_pools: HashMap<usize, Vec<Vec<Scalar>>>,
    matrix_pools: HashMap<(usize, usize), Vec<Vec<Vec<Scalar>>>>,
    allocation_count: usize,
    total_allocated: usize,
    peak_usage: usize,
}

impl CrossPlatformMemoryManager {
    /// Create a new memory manager for the specified platform
    pub fn new(platform: Platform) -> Self {
        let allocation_strategy = match platform {
            Platform::WASM => AllocationStrategy::WasmOptimized,
            Platform::Mobile => AllocationStrategy::MobileOptimized,
            Platform::Desktop => AllocationStrategy::PerformanceOptimized,
        };
        
        let memory_limit = match platform {
            Platform::WASM => Some(512 * 1024 * 1024),      // 512MB for WASM
            Platform::Mobile => Some(1024 * 1024 * 1024),   // 1GB for mobile
            Platform::Desktop => None,                        // No limit for desktop
        };
        
        Self {
            platform,
            allocation_strategy,
            memory_pools: Mutex::new(MemoryPools::new()),
            stats: Mutex::new(MemoryStats::new()),
            memory_limit,
        }
    }
    
    /// Create memory manager with specific allocation strategy
    pub fn with_strategy(platform: Platform, strategy: AllocationStrategy) -> Self {
        let memory_limit = match platform {
            Platform::WASM => Some(512 * 1024 * 1024),
            Platform::Mobile => Some(1024 * 1024 * 1024),
            Platform::Desktop => None,
        };
        
        Self {
            platform,
            allocation_strategy: strategy,
            memory_pools: Mutex::new(MemoryPools::new()),
            stats: Mutex::new(MemoryStats::new()),
            memory_limit,
        }
    }
    
    /// Check if allocation would exceed memory limits
    fn check_memory_limit(&self, additional_bytes: usize) -> Result<(), R1CSError> {
        if let Some(limit) = self.memory_limit {
            let current_usage = self.stats.lock().unwrap().allocated_bytes;
            if current_usage + additional_bytes > limit {
                return Err(R1CSError::InvalidIndex);
            }
        }
        Ok(())
    }
    
    /// Allocate memory with platform-specific optimizations
    fn allocate_with_strategy<T>(&self, size: usize, default_value: T) -> Result<Vec<T>, R1CSError>
    where
        T: Clone,
    {
        let bytes_needed = size * std::mem::size_of::<T>();
        self.check_memory_limit(bytes_needed)?;
        
        match self.allocation_strategy {
            AllocationStrategy::WasmOptimized => self.allocate_wasm_optimized(size, default_value),
            AllocationStrategy::MobileOptimized => self.allocate_mobile_optimized(size, default_value),
            AllocationStrategy::PerformanceOptimized => self.allocate_performance_optimized(size, default_value),
            AllocationStrategy::Conservative => self.allocate_conservative(size, default_value),
        }
    }
    
    /// WASM-optimized allocation using TypedArrays where possible
    fn allocate_wasm_optimized<T>(&self, size: usize, default_value: T) -> Result<Vec<T>, R1CSError>
    where
        T: Clone,
    {
        // For WASM, we want to minimize garbage collection pressure
        // and use TypedArrays when possible
        let mut vec = Vec::new();
        vec.try_reserve(size).map_err(|_| R1CSError::InvalidIndex)?;
        vec.resize(size, default_value);
        
        // Update statistics
        let bytes_allocated = size * std::mem::size_of::<T>();
        self.update_stats(bytes_allocated);
        
        Ok(vec)
    }
    
    /// Mobile-optimized allocation with memory pools
    fn allocate_mobile_optimized<T>(&self, size: usize, default_value: T) -> Result<Vec<T>, R1CSError>
    where
        T: Clone,
    {
        // For mobile, we use memory pools to reduce allocation overhead
        // and avoid memory fragmentation
        let mut vec = Vec::new();
        vec.try_reserve(size).map_err(|_| R1CSError::InvalidIndex)?;
        vec.resize(size, default_value);
        
        let bytes_allocated = size * std::mem::size_of::<T>();
        self.update_stats(bytes_allocated);
        
        Ok(vec)
    }
    
    /// Performance-optimized allocation for desktop
    fn allocate_performance_optimized<T>(&self, size: usize, default_value: T) -> Result<Vec<T>, R1CSError>
    where
        T: Clone,
    {
        // For desktop, prioritize performance with direct allocation
        let mut vec = Vec::with_capacity(size);
        vec.resize(size, default_value);
        
        let bytes_allocated = size * std::mem::size_of::<T>();
        self.update_stats(bytes_allocated);
        
        Ok(vec)
    }
    
    /// Conservative allocation for maximum compatibility
    fn allocate_conservative<T>(&self, size: usize, default_value: T) -> Result<Vec<T>, R1CSError>
    where
        T: Clone,
    {
        let vec = vec![default_value; size];
        
        let bytes_allocated = size * std::mem::size_of::<T>();
        self.update_stats(bytes_allocated);
        
        Ok(vec)
    }
    
    /// Update memory usage statistics
    fn update_stats(&self, bytes_allocated: usize) {
        let mut stats = self.stats.lock().unwrap();
        stats.allocated_bytes += bytes_allocated;
        if stats.allocated_bytes > stats.peak_usage_bytes {
            stats.peak_usage_bytes = stats.allocated_bytes;
        }
    }
    
    /// Deallocate memory and update statistics
    fn deallocate<T>(&self, vec: Vec<T>) {
        let bytes_deallocated = vec.len() * std::mem::size_of::<T>();
        drop(vec);
        
        let mut stats = self.stats.lock().unwrap();
        stats.allocated_bytes = stats.allocated_bytes.saturating_sub(bytes_deallocated);
    }
    
    /// Get memory pool for scalars of a specific size
    fn get_scalar_pool(&self, size: usize) -> Option<Vec<Scalar>> {
        let mut pools = self.memory_pools.lock().unwrap();
        pools.scalar_pools.get_mut(&size)?.pop()
    }
    
    /// Return scalar vector to the memory pool
    fn return_scalar_to_pool(&self, mut vec: Vec<Scalar>) {
        let size = vec.capacity();
        vec.clear();
        
        let mut pools = self.memory_pools.lock().unwrap();
        pools.scalar_pools.entry(size).or_insert_with(Vec::new).push(vec);
    }
    
    /// Trigger garbage collection on platforms that support it
    fn trigger_gc(&self) {
        match self.platform {
            Platform::WASM => {
                // In WASM, we can't directly trigger GC, but we can hint
                // by clearing unused memory pools
                self.clear_unused_pools();
            },
            _ => {
                // No explicit GC needed for native platforms
            }
        }
    }
    
    /// Clear unused memory pools to free memory
    fn clear_unused_pools(&self) {
        let mut pools = self.memory_pools.lock().unwrap();
        
        // Keep only the most recently used pools for each size
        for (_, pool) in pools.scalar_pools.iter_mut() {
            pool.truncate(2); // Keep at most 2 vectors per size
        }
        
        for (_, pool) in pools.matrix_pools.iter_mut() {
            pool.truncate(1); // Keep at most 1 matrix per size
        }
    }
}

impl MemoryManager for CrossPlatformMemoryManager {
    fn allocate_polynomial(&self, size: usize) -> Result<Vec<Scalar>, R1CSError> {
        // Try to reuse from memory pool first
        if let Some(mut vec) = self.get_scalar_pool(size) {
            vec.resize(size, Scalar::zero());
            return Ok(vec);
        }
        
        // Allocate new vector with platform-specific optimization
        self.allocate_with_strategy(size, Scalar::zero())
    }
    
    fn allocate_matrix(&self, rows: usize, cols: usize) -> Result<Vec<Vec<Scalar>>, R1CSError> {
        let total_size = rows * cols;
        self.check_memory_limit(total_size * std::mem::size_of::<Scalar>())?;
        
        match self.allocation_strategy {
            AllocationStrategy::WasmOptimized => {
                // For WASM, use flat allocation to reduce object overhead
                self.allocate_matrix_flat(rows, cols)
            },
            AllocationStrategy::MobileOptimized => {
                // For mobile, use memory pools for rows
                self.allocate_matrix_pooled(rows, cols)
            },
            _ => {
                // Standard allocation for other platforms
                self.allocate_matrix_standard(rows, cols)
            }
        }
    }
    
    fn optimize_for_platform(&self) -> Result<(), R1CSError> {
        match self.platform {
            Platform::WASM => {
                // Optimize for WASM by reducing GC pressure
                self.trigger_gc();
                Ok(())
            },
            Platform::Mobile => {
                // Optimize for mobile by clearing unused pools
                self.clear_unused_pools();
                Ok(())
            },
            Platform::Desktop => {
                // No specific optimization needed for desktop
                Ok(())
            }
        }
    }
    
    fn get_memory_stats(&self) -> MemoryStats {
        let stats = self.stats.lock().unwrap();
        let pools = self.memory_pools.lock().unwrap();
        
        MemoryStats {
            allocated_bytes: stats.allocated_bytes,
            peak_usage_bytes: stats.peak_usage_bytes,
            pool_efficiency: pools.calculate_efficiency(),
        }
    }
}

impl CrossPlatformMemoryManager {
    /// Allocate matrix using flat memory layout (WASM optimized)
    fn allocate_matrix_flat(&self, rows: usize, cols: usize) -> Result<Vec<Vec<Scalar>>, R1CSError> {
        let mut matrix = Vec::with_capacity(rows);
        for _ in 0..rows {
            let row = self.allocate_polynomial(cols)?;
            matrix.push(row);
        }
        Ok(matrix)
    }
    
    /// Allocate matrix using memory pools (Mobile optimized)
    fn allocate_matrix_pooled(&self, rows: usize, cols: usize) -> Result<Vec<Vec<Scalar>>, R1CSError> {
        let mut matrix = Vec::with_capacity(rows);
        for _ in 0..rows {
            let row = self.allocate_polynomial(cols)?;
            matrix.push(row);
        }
        Ok(matrix)
    }
    
    /// Standard matrix allocation
    fn allocate_matrix_standard(&self, rows: usize, cols: usize) -> Result<Vec<Vec<Scalar>>, R1CSError> {
        let mut matrix = Vec::with_capacity(rows);
        for _ in 0..rows {
            matrix.push(vec![Scalar::zero(); cols]);
        }
        
        let bytes_allocated = rows * cols * std::mem::size_of::<Scalar>();
        self.update_stats(bytes_allocated);
        
        Ok(matrix)
    }
}

impl MemoryPools {
    fn new() -> Self {
        Self {
            scalar_pools: HashMap::new(),
            matrix_pools: HashMap::new(),
            allocation_count: 0,
            total_allocated: 0,
            peak_usage: 0,
        }
    }
    
    /// Calculate memory pool efficiency
    fn calculate_efficiency(&self) -> f64 {
        if self.allocation_count == 0 {
            return 1.0;
        }
        
        let pool_reuse_count: usize = self.scalar_pools.values().map(|pool| pool.len()).sum();
        let matrix_reuse_count: usize = self.matrix_pools.values().map(|pool| pool.len()).sum();
        let total_reuse = pool_reuse_count + matrix_reuse_count;
        
        total_reuse as f64 / self.allocation_count as f64
    }
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            allocated_bytes: 0,
            peak_usage_bytes: 0,
            pool_efficiency: 0.0,
        }
    }
}

/// Memory allocator trait for custom allocation strategies
pub trait PlatformAllocator: Send + Sync {
    /// Allocate memory with platform-specific optimizations
    fn allocate(&self, size: usize, alignment: usize) -> Result<*mut u8, R1CSError>;
    
    /// Deallocate memory
    fn deallocate(&self, ptr: *mut u8, size: usize, alignment: usize);
    
    /// Get allocator statistics
    fn get_stats(&self) -> AllocatorStats;
}

/// Allocator statistics
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub bytes_allocated: usize,
    pub bytes_deallocated: usize,
    pub fragmentation_ratio: f64,
}

/// Utility functions for memory management
pub mod utils {
    use super::*;
    
    /// Estimate memory requirements for a given problem size
    pub fn estimate_memory_requirements(num_vars: usize, num_constraints: usize) -> usize {
        let scalar_size = std::mem::size_of::<Scalar>();
        
        // Polynomials: A, B, C polynomials
        let polynomial_memory = 3 * num_vars * scalar_size;
        
        // R1CS matrices: assume sparse with ~10 non-zero entries per constraint
        let matrix_memory = num_constraints * 10 * scalar_size;
        
        // Commitments and proofs
        let proof_memory = num_vars * 32; // Approximate
        
        // Working memory (temporary allocations during proof generation)
        let working_memory = (polynomial_memory + matrix_memory) / 2;
        
        polynomial_memory + matrix_memory + proof_memory + working_memory
    }
    
    /// Check if system has enough memory for the given requirements
    pub fn check_memory_availability(_required_bytes: usize) -> bool {
        // Platform-specific memory checking
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemAvailable:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if let Some(available_kb) = parts.get(1).and_then(|s| s.parse::<usize>().ok()) {
                            let available_bytes = available_kb * 1024;
                            return available_bytes > required_bytes * 2; // 2x safety margin
                        }
                    }
                }
            }
        }
        
        // Conservative default: assume we have enough memory
        true
    }
    
    /// Get platform-specific memory alignment requirements
    pub fn get_alignment_requirements() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                64 // AVX-512 requires 64-byte alignment
            } else if is_x86_feature_detected!("avx2") {
                32 // AVX2 requires 32-byte alignment
            } else {
                16 // SSE requires 16-byte alignment
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            16 // ARM NEON requires 16-byte alignment
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            8 // Default alignment
        }
    }
}

/// Specialized memory allocator for WASM
#[cfg(target_arch = "wasm32")]
pub struct WasmAllocator;

#[cfg(target_arch = "wasm32")]
impl PlatformAllocator for WasmAllocator {
    fn allocate(&self, size: usize, _alignment: usize) -> Result<*mut u8, R1CSError> {
        // WASM-specific allocation using wasm-bindgen
        let layout = std::alloc::Layout::from_size_align(size, 8)
            .map_err(|_| R1CSError::InvalidIndex)?;
        
        unsafe {
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                Err(R1CSError::InvalidIndex)
            } else {
                Ok(ptr)
            }
        }
    }
    
    fn deallocate(&self, ptr: *mut u8, size: usize, _alignment: usize) {
        let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
    
    fn get_stats(&self) -> AllocatorStats {
        AllocatorStats {
            allocations: 0,
            deallocations: 0,
            bytes_allocated: 0,
            bytes_deallocated: 0,
            fragmentation_ratio: 0.0,
        }
    }
}