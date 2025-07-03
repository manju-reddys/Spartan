//! Mobile memory management for Spartan zkSNARK library
//! 
//! This module provides adaptive memory management specifically designed for mobile platforms
//! (iOS and Android) where memory is constrained and requires careful management.

#[cfg(feature = "mobile")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "mobile")]
use thiserror::Error;

/// Mobile memory manager that tracks and limits memory usage based on platform constraints
#[cfg(feature = "mobile")]
pub struct MobileMemoryManager {
    max_allocation_mb: usize,
    current_usage_mb: AtomicUsize,
    chunk_size: usize,
    platform: Platform,
}

/// Platform detection for mobile-specific optimizations
#[cfg(feature = "mobile")]
#[derive(Debug, Clone)]
pub enum Platform {
    iOS { memory_gb: u8 },
    Android { memory_gb: u8, api_level: u32 },
    Desktop,
}

#[cfg(feature = "mobile")]
impl MobileMemoryManager {
    /// Create a new memory manager optimized for the current platform
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

    /// Check if the requested allocation would exceed platform limits
    pub fn can_allocate(&self, size_bytes: usize) -> bool {
        let size_mb = size_bytes / (1024 * 1024);
        let current = self.current_usage_mb.load(Ordering::Relaxed);
        current + size_mb <= self.max_allocation_mb
    }

    /// Track a memory allocation
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

    /// Track a memory deallocation
    pub fn deallocate_tracked(&self, size_bytes: usize) {
        let size_mb = size_bytes / (1024 * 1024);
        self.current_usage_mb.fetch_sub(size_mb, Ordering::Relaxed);
    }

    /// Get the recommended chunk size for this platform
    pub fn get_chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get current memory usage in bytes
    pub fn get_current_usage_bytes(&self) -> usize {
        self.current_usage_mb.load(Ordering::Relaxed) * 1024 * 1024
    }

    /// Get available memory in bytes
    pub fn get_available_memory(&self) -> usize {
        let current_mb = self.current_usage_mb.load(Ordering::Relaxed);
        if current_mb >= self.max_allocation_mb {
            0
        } else {
            (self.max_allocation_mb - current_mb) * 1024 * 1024
        }
    }

    /// Get platform information
    pub fn get_platform(&self) -> &Platform {
        &self.platform
    }
}

/// Error types for mobile memory management
#[cfg(feature = "mobile")]
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Insufficient memory: requested {requested} bytes, available {available} bytes")]
    InsufficientMemory { requested: usize, available: usize },
    #[error("Allocation too large for mobile platform")]
    AllocationTooLarge,
    #[error("Memory pressure detected")]
    MemoryPressure,
}

/// Detect the current platform and available memory
#[cfg(feature = "mobile")]
fn detect_platform() -> Platform {
    // In a real implementation, this would use platform-specific APIs
    // For now, we'll provide a conservative default
    #[cfg(target_os = "ios")]
    {
        // On iOS, we'd use sysctl to get physical memory
        Platform::iOS { memory_gb: 4 } // Conservative estimate
    }
    #[cfg(target_os = "android")]
    {
        // On Android, we'd use /proc/meminfo
        Platform::Android { memory_gb: 6, api_level: 30 }
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        Platform::Desktop
    }
}

// Stub implementations when mobile feature is not enabled
#[cfg(not(feature = "mobile"))]
/// Stub memory manager when mobile feature is not enabled
pub struct MobileMemoryManager;

#[cfg(not(feature = "mobile"))]
impl MobileMemoryManager {
    /// Create a new memory manager (stub implementation)
    pub fn new_for_platform() -> Self {
        Self
    }
    
    /// Check if allocation is possible (always true in stub)
    pub fn can_allocate(&self, _size_bytes: usize) -> bool {
        true
    }
    
    /// Get chunk size (default value in stub)
    pub fn get_chunk_size(&self) -> usize {
        8192
    }
}

#[cfg(not(feature = "mobile"))]
#[derive(Debug)]
/// Error types for memory management (stub implementation)
pub enum MemoryError {
    /// Insufficient memory error
    InsufficientMemory { 
        /// Requested memory size
        requested: usize, 
        /// Available memory size
        available: usize 
    },
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "mobile")]
    use super::*;

    #[cfg(feature = "mobile")]
    #[test]
    fn test_memory_manager_creation() {
        let manager = MobileMemoryManager::new_for_platform();
        assert!(manager.can_allocate(1024)); // Should be able to allocate 1KB
    }

    #[cfg(feature = "mobile")]
    #[test]
    fn test_memory_tracking() {
        let manager = MobileMemoryManager::new_for_platform();
        let initial_usage = manager.get_current_usage_bytes();
        
        // Track an allocation
        let size = 1024 * 1024; // 1MB
        manager.allocate_tracked(size).unwrap();
        assert!(manager.get_current_usage_bytes() >= initial_usage);
        
        // Track deallocation
        manager.deallocate_tracked(size);
        assert_eq!(manager.get_current_usage_bytes(), initial_usage);
    }

    #[cfg(feature = "mobile")]
    #[test]
    fn test_allocation_limits() {
        let mut manager = MobileMemoryManager::new_for_platform();
        manager.max_allocation_mb = 1; // Artificially low limit for testing
        
        let large_size = 2 * 1024 * 1024; // 2MB
        let result = manager.allocate_tracked(large_size);
        assert!(matches!(result, Err(MemoryError::InsufficientMemory { .. })));
    }
}