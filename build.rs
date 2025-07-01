//! Build script for cross-platform Spartan zkSNARK library

use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    
    println!("cargo:rerun-if-changed=build.rs");
    
    // Platform-specific compilation flags
    configure_platform_flags(&target, &target_arch, &target_os);
    
    // SIMD feature detection and configuration
    configure_simd_features(&target_arch);
    
    // GPU acceleration configuration
    configure_gpu_features(&target_os);
    
    // Memory management optimization flags
    configure_memory_optimizations(&target_os);
    
    // Cross-platform specific configurations
    configure_cross_platform_features();
    
    // Generate version and build information
    generate_version_info();
}

fn configure_platform_flags(target: &str, target_arch: &str, target_os: &str) {
    match target_os {
        "android" => {
            println!("cargo:rustc-cfg=mobile");
            println!("cargo:rustc-cfg=android");
            // Android-specific flags
            println!("cargo:rustc-link-lib=m");
        },
        "ios" => {
            println!("cargo:rustc-cfg=mobile");
            println!("cargo:rustc-cfg=ios");
            // iOS-specific flags
            println!("cargo:rustc-link-lib=framework=Foundation");
        },
        _ => {}
    }
    
    match target_arch {
        "wasm32" => {
            println!("cargo:rustc-cfg=wasm");
            println!("cargo:rustc-cfg=no_std_atomics");
        },
        "x86_64" => {
            println!("cargo:rustc-cfg=x86_64");
            // Enable target-cpu=native for better performance when available
            if env::var("CARGO_CFG_TARGET_FEATURE").is_ok() {
                println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
            }
        },
        "aarch64" => {
            println!("cargo:rustc-cfg=aarch64");
        },
        _ => {}
    }
    
    // Special handling for specific targets
    match target {
        "wasm32-unknown-unknown" => {
            println!("cargo:rustc-cfg=wasm_unknown");
        },
        "wasm32-wasi" => {
            println!("cargo:rustc-cfg=wasm_wasi");
        },
        _ => {}
    }
}

fn configure_simd_features(target_arch: &str) {
    match target_arch {
        "x86_64" => {
            // Check for x86_64 SIMD features
            if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |features| features.contains("avx512f")) {
                println!("cargo:rustc-cfg=has_avx512");
                println!("cargo:rustc-cfg=simd_avx512");
            } else if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |features| features.contains("avx2")) {
                println!("cargo:rustc-cfg=has_avx2");
                println!("cargo:rustc-cfg=simd_avx2");
            } else if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |features| features.contains("sse4.1")) {
                println!("cargo:rustc-cfg=has_sse4");
                println!("cargo:rustc-cfg=simd_sse4");
            }
            
            // Enable specific optimizations based on detected features
            configure_x86_optimizations();
        },
        "aarch64" | "arm" => {
            // Check for ARM SIMD features
            if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |features| features.contains("neon")) {
                println!("cargo:rustc-cfg=has_neon");
                println!("cargo:rustc-cfg=simd_neon");
            }
            
            if target_arch == "aarch64" && env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |features| features.contains("sve")) {
                println!("cargo:rustc-cfg=has_sve");
                println!("cargo:rustc-cfg=simd_sve");
            }
        },
        "wasm32" => {
            // Check for WASM SIMD support
            if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |features| features.contains("simd128")) {
                println!("cargo:rustc-cfg=has_wasm_simd");
                println!("cargo:rustc-cfg=simd_wasm128");
            }
        },
        _ => {}
    }
}

fn configure_x86_optimizations() {
    // x86_64 specific optimization flags
    println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+crt-static");
    
    // Enable curve25519-dalek backend optimization
    if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |features| features.contains("avx2")) {
        println!("cargo:rustc-cfg=curve25519_dalek_backend=\"avx2\"");
    } else {
        println!("cargo:rustc-cfg=curve25519_dalek_backend=\"serial\"");
    }
}

fn configure_gpu_features(target_os: &str) {
    if cfg!(feature = "gpu") {
        match target_os {
            "linux" | "windows" | "macos" => {
                println!("cargo:rustc-cfg=gpu_desktop");
                // Desktop GPU support via wgpu
            },
            "android" => {
                println!("cargo:rustc-cfg=gpu_mobile_android");
                // Android GPU support via wgpu + Vulkan
            },
            "ios" => {
                println!("cargo:rustc-cfg=gpu_mobile_ios");
                // iOS GPU support via wgpu + Metal
            },
            _ => {
                println!("cargo:warning=GPU support not available for target OS: {}", target_os);
            }
        }
    }
}

fn configure_memory_optimizations(target_os: &str) {
    match target_os {
        "android" | "ios" => {
            // Mobile memory optimizations
            println!("cargo:rustc-cfg=memory_constrained");
            println!("cargo:rustc-cfg=use_memory_pools");
        },
        _ => {
            // Desktop memory optimizations
            println!("cargo:rustc-cfg=memory_unconstrained");
        }
    }
}

fn configure_cross_platform_features() {
    if cfg!(feature = "cross-platform") {
        println!("cargo:rustc-cfg=cross_platform_enabled");
        
        // Enable LRU cache for cross-platform memory management
        if cfg!(feature = "lru") {
            println!("cargo:rustc-cfg=lru_cache_enabled");
        }
    }
    
    // Feature-specific configurations
    if cfg!(feature = "aggressive-opts") {
        println!("cargo:rustc-cfg=aggressive_optimizations");
        configure_aggressive_optimizations();
    } else if cfg!(feature = "balanced-opts") {
        println!("cargo:rustc-cfg=balanced_optimizations");
    } else {
        println!("cargo:rustc-cfg=conservative_optimizations");
    }
}

fn configure_aggressive_optimizations() {
    // Aggressive optimization flags
    println!("cargo:rustc-env=RUSTFLAGS=-C opt-level=3 -C target-cpu=native -C codegen-units=1");
    
    // Enable all available SIMD features
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    match target_arch.as_str() {
        "x86_64" => {
            println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+avx,+avx2,+fma");
        },
        "aarch64" => {
            println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+neon");
        },
        _ => {}
    }
}

/// Check if we're in a cross-compilation environment
fn is_cross_compiling() -> bool {
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    host != target
}

/// Generate version information for cross-platform builds
fn generate_version_info() {
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let target = env::var("TARGET").unwrap();
    let profile = env::var("PROFILE").unwrap();
    
    println!("cargo:rustc-env=SPARTAN_VERSION={}", version);
    println!("cargo:rustc-env=SPARTAN_TARGET={}", target);
    println!("cargo:rustc-env=SPARTAN_PROFILE={}", profile);
    
    if is_cross_compiling() {
        println!("cargo:rustc-cfg=cross_compiling");
    }
}