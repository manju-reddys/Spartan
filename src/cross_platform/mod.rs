//! Cross-platform optimization framework for Spartan zkSNARKs
//! 
//! This module provides a unified interface for running Spartan across different platforms
//! (WASM, Android, iOS, desktop) with adaptive optimizations for each target.

#![allow(missing_docs)]

use std::sync::Arc;
use crate::scalar::Scalar;
use crate::errors::{ProofVerifyError, R1CSError};
use crate::r1cs::R1CSShape;
use serde::{Deserialize, Serialize};

pub mod backend;
pub mod memory;
pub mod capabilities;
pub mod univariate_skip;
pub mod native_opt;

#[cfg(test)]
pub mod tests;

pub mod benchmarks;
pub mod spartan_integration;
pub mod integrated_backends;

// Re-export key types for easier access
pub use integrated_backends::{IntegratedSpartanCrossPlatform, IntegratedNativeBackend};
pub use spartan_integration::{ProvingMode, ProblemParameters};

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(any(feature = "wasm", target_arch = "wasm32"))]
pub mod wasm;

#[cfg(any(target_os = "android", target_os = "ios"))]
pub mod mobile;

/// Cross-platform Spartan implementation with adaptive optimizations
pub struct SpartanCrossPlatform {
    backend: Arc<dyn SpartanBackend>,
    memory_manager: Arc<dyn MemoryManager>,
    #[cfg(feature = "gpu")]
    gpu_accelerator: Option<Arc<dyn GpuAccelerator>>,
    platform_caps: PlatformCapabilities,
}

/// Platform-specific backend trait
pub trait SpartanBackend: Send + Sync {
    /// Generate a proof for the given R1CS instance and witness
    fn prove(&self, r1cs: &R1CSShape, witness: &[Scalar]) -> Result<SpartanProof, ProofVerifyError>;
    
    /// Verify a proof against the given public inputs
    fn verify(&self, proof: &SpartanProof, public_inputs: &[Scalar]) -> Result<bool, ProofVerifyError>;
    
    /// Get performance metrics for this backend
    fn get_performance_metrics(&self) -> PerformanceMetrics;
    
    /// Get the backend type identifier
    fn backend_type(&self) -> BackendType;
}

/// Memory management abstraction for cross-platform optimization
pub trait MemoryManager: Send + Sync {
    /// Allocate memory for a polynomial of the given size
    fn allocate_polynomial(&self, size: usize) -> Result<Vec<Scalar>, R1CSError>;
    
    /// Allocate memory for a matrix with given dimensions
    fn allocate_matrix(&self, rows: usize, cols: usize) -> Result<Vec<Vec<Scalar>>, R1CSError>;
    
    /// Optimize memory layout for the current platform
    fn optimize_for_platform(&self) -> Result<(), R1CSError>;
    
    /// Get memory usage statistics
    fn get_memory_stats(&self) -> MemoryStats;
}

/// GPU acceleration interface
#[cfg(feature = "gpu")]
pub trait GpuAccelerator: Send + Sync {
    /// Compute multi-scalar multiplication on GPU
    fn compute_msm(&self, scalars: &[Scalar], points: &[crate::group::GroupElement]) -> Result<crate::group::GroupElement, ProofVerifyError>;
    
    /// Compute polynomial evaluations on GPU
    fn evaluate_polynomials(&self, polys: &[Vec<Scalar>], points: &[Scalar]) -> Result<Vec<Scalar>, ProofVerifyError>;
    
    /// Check if GPU is available and initialized
    fn is_available(&self) -> bool;
}

/// Supported backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    Native,
    WASM,
    Mobile,
    #[cfg(feature = "gpu")]
    GPU,
}

/// Platform capabilities detection
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    pub platform: Platform,
    pub has_gpu: bool,
    pub simd_level: SIMDLevel,
    pub core_count: usize,
    pub memory_limit: Option<usize>,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub is_mobile: bool,
    pub thermal_management: bool,
}

/// Supported platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Platform {
    Desktop,
    WASM,
    Mobile,
}

/// SIMD capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SIMDLevel {
    None,
    Basic,
    SSE4,
    AVX2,
    AVX512,
    WASM128,
    NEON,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub proof_time_ms: u64,
    pub verify_time_ms: u64,
    pub memory_usage_bytes: usize,
    pub cpu_usage_percent: f64,
    pub gpu_usage_percent: Option<f64>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_usage_bytes: usize,
    pub pool_efficiency: f64,
}

/// Enhanced Spartan proof structure for cross-platform use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpartanProof {
    pub commitments: Vec<crate::group::GroupElement>,
    pub sumcheck_proof: Vec<u8>, // Serialized proof data
    pub timing_info: PerformanceMetrics,
    pub instance_digest: Vec<u8>, // Store instance digest for verification
    pub proving_mode: crate::cross_platform::spartan_integration::ProvingMode, // Store which mode was used
    pub problem_params: crate::cross_platform::spartan_integration::ProblemParameters, // Store problem parameters
    pub computation_commitment: Option<Vec<u8>>, // Serialized ComputationCommitment for SNARK verification
}

impl SpartanCrossPlatform {
    /// Create a new cross-platform Spartan instance with automatic backend selection
    pub fn new() -> Self {
        let platform_caps = PlatformCapabilities::detect();
        let backend = Self::select_optimal_backend(&platform_caps);
        let memory_manager = Self::create_memory_manager(&platform_caps);
        
        #[cfg(feature = "gpu")]
        let gpu_accelerator = Self::initialize_gpu_if_available(&platform_caps);
        
        Self {
            backend,
            memory_manager,
            #[cfg(feature = "gpu")]
            gpu_accelerator,
            platform_caps,
        }
    }
    
    /// Create instance with specific backend type
    pub fn with_backend(backend_type: BackendType) -> Result<Self, ProofVerifyError> {
        let platform_caps = PlatformCapabilities::detect();
        let backend = Self::create_backend(backend_type, &platform_caps)?;
        let memory_manager = Self::create_memory_manager(&platform_caps);
        
        #[cfg(feature = "gpu")]
        let gpu_accelerator = if backend_type == BackendType::GPU {
            Self::initialize_gpu_if_available(&platform_caps)
        } else {
            None
        };
        
        Ok(Self {
            backend,
            memory_manager,
            #[cfg(feature = "gpu")]
            gpu_accelerator,
            platform_caps,
        })
    }
    
    /// Generate a proof using the optimal backend for this platform
    pub fn prove(&self, r1cs: &R1CSShape, witness: &[Scalar]) -> Result<SpartanProof, ProofVerifyError> {
        // Optimize memory allocation for the proof generation
        self.memory_manager.optimize_for_platform()
            .map_err(|e| ProofVerifyError::InternalError)?;
        
        // Generate proof using the selected backend
        self.backend.prove(r1cs, witness)
    }
    
    /// Verify a proof using the optimal backend for this platform
    pub fn verify(&self, proof: &SpartanProof, public_inputs: &[Scalar]) -> Result<bool, ProofVerifyError> {
        self.backend.verify(proof, public_inputs)
    }
    
    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.backend.get_performance_metrics()
    }
    
    /// Get platform capabilities
    pub fn get_platform_capabilities(&self) -> &PlatformCapabilities {
        &self.platform_caps
    }
    
    fn select_optimal_backend(caps: &PlatformCapabilities) -> Arc<dyn SpartanBackend> {
        match (caps.has_gpu, caps.platform) {
            #[cfg(feature = "gpu")]
            (true, Platform::Desktop) => Arc::new(gpu::GpuBackend::new()),
            #[cfg(any(feature = "wasm", target_arch = "wasm32"))]
            (_, Platform::WASM) => Arc::new(wasm::WasmBackend::new()),
            #[cfg(any(target_os = "android", target_os = "ios"))]
            (_, Platform::Mobile) => Arc::new(mobile::MobileBackend::new()),
            _ => Arc::new(backend::NativeBackend::new()),
        }
    }
    
    fn create_backend(backend_type: BackendType, _caps: &PlatformCapabilities) -> Result<Arc<dyn SpartanBackend>, ProofVerifyError> {
        match backend_type {
            BackendType::Native => Ok(Arc::new(backend::NativeBackend::new())),
            #[cfg(any(feature = "wasm", target_arch = "wasm32"))]
            BackendType::WASM => Ok(Arc::new(wasm::WasmBackend::new())),
            #[cfg(any(target_os = "android", target_os = "ios"))]
            BackendType::Mobile => Ok(Arc::new(mobile::MobileBackend::new())),
            #[cfg(feature = "gpu")]
            BackendType::GPU => {
                if _caps.has_gpu {
                    Ok(Arc::new(gpu::GpuBackend::new()))
                } else {
                    Err(ProofVerifyError::InternalError)
                }
            }
            #[cfg(not(feature = "gpu"))]
            _ => Err(ProofVerifyError::InternalError),
        }
    }
    
    fn create_memory_manager(caps: &PlatformCapabilities) -> Arc<dyn MemoryManager> {
        Arc::new(memory::CrossPlatformMemoryManager::new(caps.platform))
    }
    
    #[cfg(feature = "gpu")]
    fn initialize_gpu_if_available(caps: &PlatformCapabilities) -> Option<Arc<dyn GpuAccelerator>> {
        if caps.has_gpu {
            gpu::GpuAccelerator::new().ok().map(|acc| Arc::new(acc) as Arc<dyn GpuAccelerator>)
        } else {
            None
        }
    }
}

impl Default for SpartanCrossPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformCapabilities {
    /// Detect the current platform's capabilities
    pub fn detect() -> Self {
        Self {
            platform: Self::detect_platform(),
            has_gpu: Self::detect_gpu(),
            simd_level: Self::detect_simd(),
            core_count: Self::detect_core_count(),
            memory_limit: Self::detect_memory_limit(),
            has_avx2: Self::detect_avx2(),
            has_avx512: Self::detect_avx512(),
            is_mobile: Self::is_mobile_platform(),
            thermal_management: Self::has_thermal_management(),
        }
    }
    
    fn detect_platform() -> Platform {
        #[cfg(target_arch = "wasm32")]
        return Platform::WASM;
        
        #[cfg(any(target_os = "android", target_os = "ios"))]
        return Platform::Mobile;
        
        #[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
        Platform::Desktop
    }
    
    fn detect_gpu() -> bool {
        #[cfg(feature = "gpu")]
        {
            // Try to initialize a GPU context to check availability
            false // Placeholder - will be implemented with actual GPU detection
        }
        #[cfg(not(feature = "gpu"))]
        false
    }
    
    fn detect_simd() -> SIMDLevel {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                SIMDLevel::AVX512
            } else if is_x86_feature_detected!("avx2") {
                SIMDLevel::AVX2
            } else if is_x86_feature_detected!("sse4.1") {
                SIMDLevel::SSE4
            } else {
                SIMDLevel::Basic
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            // Check for WASM SIMD support
            SIMDLevel::WASM128 // Placeholder - will be implemented with actual WASM SIMD detection
        }
        
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                SIMDLevel::NEON
            } else {
                SIMDLevel::Basic
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "wasm32", target_arch = "aarch64", target_arch = "arm")))]
        SIMDLevel::Basic
    }
    
    fn detect_core_count() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    }
    
    fn detect_memory_limit() -> Option<usize> {
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            Some(1024 * 1024 * 1024) // 1GB limit for mobile
        }
        #[cfg(target_arch = "wasm32")]
        {
            Some(512 * 1024 * 1024) // 512MB limit for WASM
        }
        #[cfg(not(any(target_os = "android", target_os = "ios", target_arch = "wasm32")))]
        None
    }
    
    fn detect_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
    
    fn detect_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
    
    fn is_mobile_platform() -> bool {
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            true
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            false
        }
    }
    
    fn has_thermal_management() -> bool {
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            true
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            false
        }
    }
}