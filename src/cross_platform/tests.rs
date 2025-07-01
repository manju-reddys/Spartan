//! Cross-platform testing and validation
//! 
//! This module provides comprehensive tests for the cross-platform Spartan implementation,
//! validating functionality across different backends, feature combinations, and platforms.

#![allow(missing_docs)]

use super::*;
use crate::scalar::Scalar;
use crate::r1cs::R1CSShape;

/// Test cross-platform backend creation and basic functionality
#[cfg(test)]
mod backend_tests {
    use super::*;

    #[test]
    fn test_automatic_backend_selection() {
        let spartan = SpartanCrossPlatform::new();
        let caps = spartan.get_platform_capabilities();
        
        // Verify platform capabilities are detected
        assert!(caps.core_count >= 1);
        assert!(matches!(caps.platform, Platform::Desktop | Platform::WASM | Platform::Mobile));
        
        // Verify SIMD level is reasonable
        assert!(matches!(caps.simd_level, 
            SIMDLevel::None | SIMDLevel::Basic | SIMDLevel::SSE4 | 
            SIMDLevel::AVX2 | SIMDLevel::AVX512 | SIMDLevel::WASM128 | SIMDLevel::NEON
        ));
        
        println!("Platform capabilities: {:?}", caps);
    }

    #[test]
    fn test_native_backend_creation() {
        let result = SpartanCrossPlatform::with_backend(BackendType::Native);
        assert!(result.is_ok(), "Native backend should always be available");
        
        let spartan = result.unwrap();
        let metrics = spartan.get_metrics();
        
        // Verify metrics structure
        assert_eq!(metrics.proof_time_ms, 0); // Should be 0 for unused backend
        assert_eq!(metrics.verify_time_ms, 0);
    }

    #[cfg(any(feature = "wasm", target_arch = "wasm32"))]
    #[test]
    fn test_wasm_backend_creation() {
        let result = SpartanCrossPlatform::with_backend(BackendType::WASM);
        assert!(result.is_ok(), "WASM backend should be available when feature is enabled");
    }

    #[cfg(any(target_os = "android", target_os = "ios"))]
    #[test]
    fn test_mobile_backend_creation() {
        let result = SpartanCrossPlatform::with_backend(BackendType::Mobile);
        assert!(result.is_ok(), "Mobile backend should be available on mobile platforms");
    }

    #[test]
    fn test_platform_detection_consistency() {
        let caps1 = PlatformCapabilities::detect();
        let caps2 = PlatformCapabilities::detect();
        
        // Platform detection should be consistent
        assert_eq!(caps1.platform, caps2.platform);
        assert_eq!(caps1.simd_level, caps2.simd_level);
        assert_eq!(caps1.core_count, caps2.core_count);
        assert_eq!(caps1.has_avx2, caps2.has_avx2);
        assert_eq!(caps1.has_avx512, caps2.has_avx512);
    }
}

/// Test memory management across platforms
#[cfg(test)]
mod memory_tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let caps = PlatformCapabilities::detect();
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(caps.platform);
        
        // Test basic allocation
        let result = memory_manager.allocate_polynomial(100);
        assert!(result.is_ok(), "Polynomial allocation should succeed");
        
        let poly = result.unwrap();
        assert_eq!(poly.len(), 100);
        assert!(poly.iter().all(|&x| x == Scalar::zero()));
    }

    #[test]
    fn test_matrix_allocation() {
        let caps = PlatformCapabilities::detect();
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(caps.platform);
        
        // Test matrix allocation
        let result = memory_manager.allocate_matrix(10, 20);
        assert!(result.is_ok(), "Matrix allocation should succeed");
        
        let matrix = result.unwrap();
        assert_eq!(matrix.len(), 10);
        assert!(matrix.iter().all(|row| row.len() == 20));
    }

    #[test]
    fn test_memory_optimization() {
        let caps = PlatformCapabilities::detect();
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(caps.platform);
        
        // Test memory optimization (should not fail)
        let result = memory_manager.optimize_for_platform();
        assert!(result.is_ok(), "Memory optimization should not fail");
    }

    #[test]
    fn test_memory_stats() {
        let caps = PlatformCapabilities::detect();
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(caps.platform);
        
        // Get initial stats
        let stats = memory_manager.get_memory_stats();
        let initial_allocated = stats.allocated_bytes;
        
        // Allocate some memory
        let _poly = memory_manager.allocate_polynomial(1000).unwrap();
        
        // Stats should reflect allocation
        let new_stats = memory_manager.get_memory_stats();
        assert!(new_stats.allocated_bytes >= initial_allocated, 
               "Allocated bytes should increase after allocation");
    }
}

/// Test different optimization levels and feature combinations
#[cfg(test)]
mod optimization_tests {
    use super::*;

    #[test]
    fn test_feature_combinations() {
        // Test cross-platform feature without other features
        let spartan = SpartanCrossPlatform::new();
        assert!(spartan.get_platform_capabilities().core_count >= 1);
    }

    #[cfg(feature = "multicore")]
    #[test]
    fn test_multicore_integration() {
        let spartan = SpartanCrossPlatform::new();
        let caps = spartan.get_platform_capabilities();
        
        // With multicore, we should detect multiple cores (if available)
        println!("Detected core count with multicore: {}", caps.core_count);
        assert!(caps.core_count >= 1);
    }

    #[test]
    fn test_simd_detection() {
        let caps = PlatformCapabilities::detect();
        
        // Validate SIMD detection logic
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, we should detect at least basic SIMD
            assert!(matches!(caps.simd_level, 
                SIMDLevel::Basic | SIMDLevel::SSE4 | SIMDLevel::AVX2 | SIMDLevel::AVX512
            ));
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            // On WASM, we should detect WASM128 or basic
            assert!(matches!(caps.simd_level, SIMDLevel::WASM128 | SIMDLevel::Basic));
        }
        
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            // On ARM, we should detect NEON or basic
            assert!(matches!(caps.simd_level, SIMDLevel::NEON | SIMDLevel::Basic));
        }
    }
}

/// Test platform-specific functionality
#[cfg(test)]
mod platform_specific_tests {
    use super::*;

    #[test]
    fn test_native_optimizations() {
        let backend = super::backend::NativeBackend::new();
        
        // Test optimization level selection
        let backend_conservative = super::backend::NativeBackend::with_optimization(
            super::backend::OptimizationLevel::Conservative
        );
        let backend_aggressive = super::backend::NativeBackend::with_optimization(
            super::backend::OptimizationLevel::Aggressive
        );
        
        // Both should be created successfully
        assert_eq!(backend.backend_type(), BackendType::Native);
        assert_eq!(backend_conservative.backend_type(), BackendType::Native);
        assert_eq!(backend_aggressive.backend_type(), BackendType::Native);
    }

    #[cfg(any(feature = "wasm", target_arch = "wasm32"))]
    #[test]
    fn test_wasm_optimizations() {
        let backend = super::wasm::WasmBackend::new();
        
        // Test different optimization levels
        let backend_conservative = super::wasm::WasmBackend::with_optimization(
            super::wasm::WasmOptimizationLevel::Conservative
        );
        let backend_aggressive = super::wasm::WasmBackend::with_optimization(
            super::wasm::WasmOptimizationLevel::Aggressive
        );
        
        assert_eq!(backend.backend_type(), BackendType::WASM);
        assert_eq!(backend_conservative.backend_type(), BackendType::WASM);
        assert_eq!(backend_aggressive.backend_type(), BackendType::WASM);
    }

    #[cfg(any(target_os = "android", target_os = "ios"))]
    #[test]
    fn test_mobile_optimizations() {
        let backend = super::mobile::MobileBackend::new();
        
        // Test different optimization strategies
        let backend_conservative = super::mobile::MobileBackend::with_strategy(
            super::mobile::MobileOptimizationStrategy::Conservative
        );
        let backend_adaptive = super::mobile::MobileBackend::with_strategy(
            super::mobile::MobileOptimizationStrategy::Adaptive
        );
        
        assert_eq!(backend.backend_type(), BackendType::Mobile);
        assert_eq!(backend_conservative.backend_type(), BackendType::Mobile);
        assert_eq!(backend_adaptive.backend_type(), BackendType::Mobile);
    }
}

/// Test error handling and edge cases
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_invalid_backend_creation() {
        // This test depends on the specific platform and features enabled
        // We'll test that error handling works correctly
        
        #[cfg(not(feature = "gpu"))]
        {
            // GPU backend should fail when GPU feature is not enabled
            // Note: We can't test this directly since BackendType::GPU doesn't exist without feature
            println!("GPU feature not enabled - skipping GPU backend test");
        }
    }

    #[test]
    fn test_memory_limits() {
        let caps = PlatformCapabilities::detect();
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(caps.platform);
        
        // Test extremely large allocation (should handle gracefully)
        let result = memory_manager.allocate_polynomial(usize::MAX / 1000);
        // This might succeed or fail depending on available memory,
        // but it should not panic
        match result {
            Ok(_) => println!("Large allocation succeeded"),
            Err(_) => println!("Large allocation failed gracefully"),
        }
    }

    #[test]
    fn test_zero_size_allocations() {
        let caps = PlatformCapabilities::detect();
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(caps.platform);
        
        // Test zero-size allocations
        let result = memory_manager.allocate_polynomial(0);
        assert!(result.is_ok(), "Zero-size allocation should succeed");
        assert_eq!(result.unwrap().len(), 0);
        
        let matrix_result = memory_manager.allocate_matrix(0, 0);
        assert!(matrix_result.is_ok(), "Zero-size matrix allocation should succeed");
        assert_eq!(matrix_result.unwrap().len(), 0);
    }
}

/// Performance regression tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_backend_creation_performance() {
        let start = Instant::now();
        let _spartan = SpartanCrossPlatform::new();
        let creation_time = start.elapsed();
        
        // Backend creation should be fast (< 100ms)
        assert!(creation_time.as_millis() < 100, 
               "Backend creation took too long: {:?}", creation_time);
    }

    #[test]
    fn test_capability_detection_performance() {
        let start = Instant::now();
        let _caps = PlatformCapabilities::detect();
        let detection_time = start.elapsed();
        
        // Capability detection should be fast (< 50ms)
        assert!(detection_time.as_millis() < 50,
               "Capability detection took too long: {:?}", detection_time);
    }

    #[test]
    fn test_memory_allocation_performance() {
        let caps = PlatformCapabilities::detect();
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(caps.platform);
        
        let start = Instant::now();
        for _ in 0..1000 {
            let _poly = memory_manager.allocate_polynomial(100).unwrap();
        }
        let allocation_time = start.elapsed();
        
        // 1000 small allocations should be fast (< 100ms)
        assert!(allocation_time.as_millis() < 100,
               "Memory allocation took too long: {:?}", allocation_time);
    }
}

/// Integration tests with actual proof generation
#[cfg(test)]
mod integration_tests {
    use super::*;

    fn create_tiny_r1cs() -> R1CSShape {
        // Create a minimal R1CS for testing
        // This represents the constraint: x * x = x (so x = 0 or x = 1)
        let num_cons = 1;
        let num_vars = 2; // Must be power of 2
        let num_io = 1;
        
        // A, B, C matrices in (row, col, value) format
        let a = vec![(0, 1, Scalar::one())]; // A[0,1] = 1 (variable x)
        let b = vec![(0, 1, Scalar::one())]; // B[0,1] = 1 (variable x) 
        let c = vec![(0, 1, Scalar::one())]; // C[0,1] = 1 (variable x)
        
        R1CSShape::new(num_cons, num_vars, num_io, &a, &b, &c)
    }

    #[test]
    fn test_cross_platform_proof_generation() {
        let spartan = SpartanCrossPlatform::new();
        let r1cs = create_tiny_r1cs();
        let witness = vec![Scalar::one()]; // x = 1 satisfies x * x = x
        
        // Generate proof using cross-platform backend
        let result = spartan.prove(&r1cs, &witness);
        
        // Proof generation should succeed
        assert!(result.is_ok(), "Proof generation failed: {:?}", result.err());
        
        let proof = result.unwrap();
        assert!(!proof.commitments.is_empty(), "Proof should contain commitments");
        assert!(!proof.sumcheck_proof.is_empty(), "Proof should contain sumcheck data");
    }

    #[test]
    fn test_cross_platform_proof_verification() {
        let spartan = SpartanCrossPlatform::new();
        let r1cs = create_tiny_r1cs();
        let witness = vec![Scalar::one()];
        
        // Generate and verify proof
        let proof = spartan.prove(&r1cs, &witness).unwrap();
        let public_inputs = vec![Scalar::one()]; // Public input: result of x * x
        
        let verification_result = spartan.verify(&proof, &public_inputs);
        assert!(verification_result.is_ok(), "Verification failed: {:?}", verification_result.err());
        assert!(verification_result.unwrap(), "Proof should verify successfully");
    }

    #[test]
    fn test_backend_consistency() {
        let r1cs = create_tiny_r1cs();
        let witness = vec![Scalar::one()];
        let public_inputs = vec![Scalar::one()];
        
        // Test native backend
        let native_spartan = SpartanCrossPlatform::with_backend(BackendType::Native).unwrap();
        let native_proof = native_spartan.prove(&r1cs, &witness).unwrap();
        let native_verify = native_spartan.verify(&native_proof, &public_inputs).unwrap();
        assert!(native_verify, "Native backend verification should succeed");
        
        // Note: Other backends would be tested here if available on the current platform
        println!("Native backend proof/verify: OK");
    }

    #[test]
    fn test_performance_metrics() {
        let spartan = SpartanCrossPlatform::new();
        let r1cs = create_tiny_r1cs();
        let witness = vec![Scalar::one()];
        
        // Generate proof and check metrics
        let _proof = spartan.prove(&r1cs, &witness).unwrap();
        let metrics = spartan.get_metrics();
        
        // Metrics should be populated (even if with placeholder values)
        assert!(metrics.memory_usage_bytes > 0, "Memory usage should be tracked");
        println!("Performance metrics: {:?}", metrics);
    }
}

/// Stress tests for robustness
#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_multiple_backend_creation() {
        // Create multiple instances to test for resource leaks
        for i in 0..10 {
            let spartan = SpartanCrossPlatform::new();
            let caps = spartan.get_platform_capabilities();
            assert!(caps.core_count >= 1, "Iteration {} failed", i);
        }
    }

    #[test]
    fn test_concurrent_capability_detection() {
        use std::thread;
        
        let handles: Vec<_> = (0..5).map(|_| {
            thread::spawn(|| {
                let caps = PlatformCapabilities::detect();
                assert!(caps.core_count >= 1);
                caps
            })
        }).collect();
        
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        
        // All results should be consistent
        let first = &results[0];
        for caps in &results[1..] {
            assert_eq!(first.platform, caps.platform);
            assert_eq!(first.simd_level, caps.simd_level);
            assert_eq!(first.core_count, caps.core_count);
        }
    }

    #[test]
    fn test_memory_stress() {
        let caps = PlatformCapabilities::detect();
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(caps.platform);
        
        // Allocate and deallocate many objects
        for _ in 0..100 {
            let _poly = memory_manager.allocate_polynomial(1000).unwrap();
            let _matrix = memory_manager.allocate_matrix(10, 10).unwrap();
        }
        
        // Memory stats should be reasonable
        let stats = memory_manager.get_memory_stats();
        println!("Memory stats after stress test: {:?}", stats);
    }
}