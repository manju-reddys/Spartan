//! Cross-platform benchmarking suite for Spartan
//! 
//! This module provides comprehensive benchmarks to measure and compare performance
//! across different platforms, backends, and optimization levels.

#![allow(missing_docs)]

use super::*;
use crate::scalar::Scalar;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub problem_sizes: Vec<usize>,
    pub num_iterations: usize,
    pub warmup_iterations: usize,
    pub backends_to_test: Vec<BackendType>,
    pub measure_memory: bool,
    pub measure_cpu: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            problem_sizes: vec![64, 256, 1024, 4096],
            num_iterations: 10,
            warmup_iterations: 3,
            backends_to_test: vec![BackendType::Native],
            measure_memory: true,
            measure_cpu: false, // CPU measurement is platform-dependent
        }
    }
}

/// Benchmark results for a specific test
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub backend_type: BackendType,
    pub problem_size: usize,
    pub avg_duration_ms: f64,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub std_deviation_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: Option<f64>,
    pub platform_info: PlatformInfo,
}

/// Platform information for benchmark context
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    pub platform: Platform,
    pub simd_level: SIMDLevel,
    pub core_count: usize,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub memory_limit_mb: Option<usize>,
}

impl From<&PlatformCapabilities> for PlatformInfo {
    fn from(caps: &PlatformCapabilities) -> Self {
        Self {
            platform: caps.platform,
            simd_level: caps.simd_level,
            core_count: caps.core_count,
            has_avx2: caps.has_avx2,
            has_avx512: caps.has_avx512,
            memory_limit_mb: caps.memory_limit.map(|bytes| bytes / (1024 * 1024)),
        }
    }
}

/// Cross-platform benchmark runner
pub struct CrossPlatformBenchmark {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
    platform_info: PlatformInfo,
}

impl CrossPlatformBenchmark {
    /// Create a new benchmark runner with default configuration
    pub fn new() -> Self {
        let caps = PlatformCapabilities::detect();
        let platform_info = PlatformInfo::from(&caps);
        
        Self {
            config: BenchmarkConfig::default(),
            results: Vec::new(),
            platform_info,
        }
    }
    
    /// Create benchmark runner with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        let caps = PlatformCapabilities::detect();
        let platform_info = PlatformInfo::from(&caps);
        
        Self {
            config,
            results: Vec::new(),
            platform_info,
        }
    }
    
    /// Run all benchmarks
    pub fn run_all_benchmarks(&mut self) {
        println!("=== Cross-Platform Spartan Benchmarks ===");
        println!("Platform: {:?}", self.platform_info.platform);
        println!("SIMD Level: {:?}", self.platform_info.simd_level);
        println!("Core Count: {}", self.platform_info.core_count);
        println!("AVX2: {}, AVX512: {}", self.platform_info.has_avx2, self.platform_info.has_avx512);
        println!();
        
        // Backend creation benchmarks
        self.benchmark_backend_creation();
        
        // Platform capability detection benchmarks
        self.benchmark_capability_detection();
        
        // Memory management benchmarks
        self.benchmark_memory_operations();
        
        // SIMD detection benchmarks  
        self.benchmark_simd_detection();
        
        // Cross-platform feature benchmarks
        self.benchmark_cross_platform_features();
        
        // Optimization level comparison
        self.benchmark_optimization_levels();
        
        // Polynomial operations (where implemented)
        self.benchmark_polynomial_operations();
        
        // Print summary
        self.print_benchmark_summary();
    }
    
    /// Benchmark backend creation performance
    fn benchmark_backend_creation(&mut self) {
        println!("📊 Benchmarking Backend Creation...");
        
        for &backend_type in &self.config.backends_to_test {
            let timings = self.measure_operation(
                &format!("Backend Creation - {:?}", backend_type),
                || {
                    let start = Instant::now();
                    let _spartan = SpartanCrossPlatform::with_backend(backend_type);
                    start.elapsed()
                }
            );
            
            let result = BenchmarkResult {
                test_name: format!("backend_creation_{:?}", backend_type),
                backend_type,
                problem_size: 1,
                avg_duration_ms: timings.0,
                min_duration_ms: timings.1,
                max_duration_ms: timings.2,
                std_deviation_ms: timings.3,
                throughput_ops_per_sec: 1000.0 / timings.0,
                memory_usage_mb: 0.0, // Minimal for creation
                cpu_usage_percent: None,
                platform_info: self.platform_info.clone(),
            };
            
            self.results.push(result);
            println!("  {:?}: {:.2}ms avg", backend_type, timings.0);
        }
    }
    
    /// Benchmark platform capability detection
    fn benchmark_capability_detection(&mut self) {
        println!("📊 Benchmarking Capability Detection...");
        
        let timings = self.measure_operation(
            "Platform Capability Detection",
            || {
                let start = Instant::now();
                let _caps = PlatformCapabilities::detect();
                start.elapsed()
            }
        );
        
        let result = BenchmarkResult {
            test_name: "capability_detection".to_string(),
            backend_type: BackendType::Native, // Not backend-specific
            problem_size: 1,
            avg_duration_ms: timings.0,
            min_duration_ms: timings.1,
            max_duration_ms: timings.2,
            std_deviation_ms: timings.3,
            throughput_ops_per_sec: 1000.0 / timings.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: None,
            platform_info: self.platform_info.clone(),
        };
        
        self.results.push(result);
        println!("  Detection: {:.2}ms avg", timings.0);
    }
    
    /// Benchmark memory management operations
    fn benchmark_memory_operations(&mut self) {
        println!("📊 Benchmarking Memory Operations...");
        
        let memory_manager = super::memory::CrossPlatformMemoryManager::new(self.platform_info.platform);
        
        for &size in &self.config.problem_sizes {
            // Benchmark polynomial allocation
            let timings = self.measure_operation(
                &format!("Polynomial Allocation (size {})", size),
                || {
                    let start = Instant::now();
                    let _poly = memory_manager.allocate_polynomial(size).unwrap();
                    start.elapsed()
                }
            );
            
            let result = BenchmarkResult {
                test_name: format!("polynomial_allocation_{}", size),
                backend_type: BackendType::Native,
                problem_size: size,
                avg_duration_ms: timings.0,
                min_duration_ms: timings.1,
                max_duration_ms: timings.2,
                std_deviation_ms: timings.3,
                throughput_ops_per_sec: (size as f64) / (timings.0 / 1000.0),
                memory_usage_mb: (size * std::mem::size_of::<Scalar>()) as f64 / (1024.0 * 1024.0),
                cpu_usage_percent: None,
                platform_info: self.platform_info.clone(),
            };
            
            self.results.push(result);
            println!("  Polynomial alloc ({}): {:.3}ms", size, timings.0);
            
            // Benchmark matrix allocation
            let matrix_size = (size as f64).sqrt() as usize;
            if matrix_size > 0 {
                let timings = self.measure_operation(
                    &format!("Matrix Allocation ({}x{})", matrix_size, matrix_size),
                    || {
                        let start = Instant::now();
                        let _matrix = memory_manager.allocate_matrix(matrix_size, matrix_size).unwrap();
                        start.elapsed()
                    }
                );
                
                let result = BenchmarkResult {
                    test_name: format!("matrix_allocation_{}x{}", matrix_size, matrix_size),
                    backend_type: BackendType::Native,
                    problem_size: matrix_size * matrix_size,
                    avg_duration_ms: timings.0,
                    min_duration_ms: timings.1,
                    max_duration_ms: timings.2,
                    std_deviation_ms: timings.3,
                    throughput_ops_per_sec: (matrix_size * matrix_size) as f64 / (timings.0 / 1000.0),
                    memory_usage_mb: (matrix_size * matrix_size * std::mem::size_of::<Scalar>()) as f64 / (1024.0 * 1024.0),
                    cpu_usage_percent: None,
                    platform_info: self.platform_info.clone(),
                };
                
                self.results.push(result);
                println!("  Matrix alloc ({}x{}): {:.3}ms", matrix_size, matrix_size, timings.0);
            }
        }
    }
    
    /// Benchmark SIMD detection performance
    fn benchmark_simd_detection(&mut self) {
        println!("📊 Benchmarking SIMD Detection...");
        
        #[cfg(target_arch = "x86_64")]
        {
            let avx2_timings = self.measure_operation(
                "AVX2 Detection",
                || {
                    let start = Instant::now();
                    let _has_avx2 = is_x86_feature_detected!("avx2");
                    start.elapsed()
                }
            );
            
            let avx512_timings = self.measure_operation(
                "AVX512 Detection", 
                || {
                    let start = Instant::now();
                    let _has_avx512 = is_x86_feature_detected!("avx512f");
                    start.elapsed()
                }
            );
            
            println!("  AVX2 detection: {:.3}ms", avx2_timings.0);
            println!("  AVX512 detection: {:.3}ms", avx512_timings.0);
        }
        
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            let neon_timings = self.measure_operation(
                "NEON Detection",
                || {
                    let start = Instant::now();
                    let _has_neon = std::arch::is_aarch64_feature_detected!("neon");
                    start.elapsed()
                }
            );
            
            println!("  NEON detection: {:.3}ms", neon_timings.0);
        }
    }
    
    /// Benchmark cross-platform features
    fn benchmark_cross_platform_features(&mut self) {
        println!("📊 Benchmarking Cross-Platform Features...");
        
        // Benchmark univariate skip optimization
        let config = super::univariate_skip::UnivariateSkipConfig {
            k_skip: Some(3),
            enable_lde: true,
            optimize_for_platform: self.platform_info.platform,
            memory_limit: Some(1024 * 1024 * 1024), // 1GB
        };
        
        let timings = self.measure_operation(
            "Univariate Skip Config Creation",
            || {
                let start = Instant::now();
                let _optimized_sumcheck = super::univariate_skip::OptimizedSumcheck::new(config.clone());
                start.elapsed()
            }
        );
        
        println!("  Univariate skip config: {:.3}ms", timings.0);
        
        // Benchmark capability-based optimization selection
        let timings = self.measure_operation(
            "Platform Capability Detection",
            || {
                let start = Instant::now();
                let _caps = PlatformCapabilities::detect();
                start.elapsed()
            }
        );
        
        println!("  Capability detection: {:.3}ms", timings.0);
    }
    
    /// Benchmark different optimization levels
    fn benchmark_optimization_levels(&mut self) {
        println!("📊 Benchmarking Optimization Levels...");
        
        // Native backend optimization levels
        let conservative_timings = self.measure_operation(
            "Conservative Native Backend Creation",
            || {
                let start = Instant::now();
                let _backend = super::backend::NativeBackend::with_optimization(
                    super::backend::OptimizationLevel::Conservative
                );
                start.elapsed()
            }
        );
        
        let aggressive_timings = self.measure_operation(
            "Aggressive Native Backend Creation",
            || {
                let start = Instant::now();
                let _backend = super::backend::NativeBackend::with_optimization(
                    super::backend::OptimizationLevel::Aggressive
                );
                start.elapsed()
            }
        );
        
        println!("  Conservative backend: {:.3}ms", conservative_timings.0);
        println!("  Aggressive backend: {:.3}ms", aggressive_timings.0);
        
        // Advanced native backend
        #[cfg(feature = "multicore")]
        {
            let advanced_timings = self.measure_operation(
                "Advanced Native Backend Creation",
                || {
                    let start = Instant::now();
                    let _backend = super::native_opt::AdvancedNativeBackend::new();
                    start.elapsed()
                }
            );
            
            println!("  Advanced backend: {:.3}ms", advanced_timings.0);
        }
    }
    
    /// Benchmark polynomial operations (where available)
    fn benchmark_polynomial_operations(&mut self) {
        println!("📊 Benchmarking Polynomial Operations...");
        
        #[cfg(feature = "multicore")]
        {
            let advanced_backend = super::native_opt::AdvancedNativeBackend::new();
            
            for &size in &[64, 256, 1024] {
                let polys: Vec<_> = (0..3).map(|_| {
                    crate::dense_mlpoly::DensePolynomial::new(vec![Scalar::zero(); size])
                }).collect();
                let poly_refs: Vec<_> = polys.iter().collect();
                
                let timings = self.measure_operation(
                    &format!("Polynomial Evaluation (size {})", size),
                    || {
                        let start = Instant::now();
                        let _result = advanced_backend.evaluate_polynomials_parallel(&poly_refs, &[]);
                        start.elapsed()
                    }
                );
                
                let result = BenchmarkResult {
                    test_name: format!("polynomial_evaluation_{}", size),
                    backend_type: BackendType::Native,
                    problem_size: size,
                    avg_duration_ms: timings.0,
                    min_duration_ms: timings.1,
                    max_duration_ms: timings.2,
                    std_deviation_ms: timings.3,
                    throughput_ops_per_sec: (size as f64) / (timings.0 / 1000.0),
                    memory_usage_mb: (size * std::mem::size_of::<Scalar>()) as f64 / (1024.0 * 1024.0),
                    cpu_usage_percent: None,
                    platform_info: self.platform_info.clone(),
                };
                
                self.results.push(result);
                println!("  Polynomial eval ({}): {:.3}ms", size, timings.0);
            }
        }
    }
    
    /// Measure operation performance with multiple iterations
    fn measure_operation<F>(&self, _name: &str, mut operation: F) -> (f64, f64, f64, f64)
    where
        F: FnMut() -> Duration,
    {
        let mut durations = Vec::new();
        
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = operation();
        }
        
        // Actual measurements
        for _ in 0..self.config.num_iterations {
            let duration = operation();
            durations.push(duration.as_secs_f64() * 1000.0); // Convert to milliseconds
        }
        
        // Calculate statistics
        let avg = durations.iter().sum::<f64>() / durations.len() as f64;
        let min = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = durations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let variance = durations.iter()
            .map(|x| (x - avg).powi(2))
            .sum::<f64>() / durations.len() as f64;
        let std_dev = variance.sqrt();
        
        (avg, min, max, std_dev)
    }
    
    /// Print comprehensive benchmark summary
    fn print_benchmark_summary(&self) {
        println!("\n=== Benchmark Summary ===");
        
        // Group results by test type
        let mut by_test_type: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.results {
            let test_type = result.test_name.split('_').next().unwrap_or("unknown").to_string();
            by_test_type.entry(test_type).or_insert_with(Vec::new).push(result);
        }
        
        for (test_type, results) in by_test_type {
            println!("\n📈 {} Performance:", test_type.to_uppercase());
            
            for result in results {
                println!("  {} (size {}): {:.3}ms ± {:.3}ms", 
                    result.test_name, 
                    result.problem_size,
                    result.avg_duration_ms, 
                    result.std_deviation_ms
                );
                
                if result.memory_usage_mb > 0.0 {
                    println!("    Memory: {:.2}MB", result.memory_usage_mb);
                }
                
                if result.throughput_ops_per_sec > 0.0 {
                    println!("    Throughput: {:.0} ops/sec", result.throughput_ops_per_sec);
                }
            }
        }
        
        // Performance insights
        println!("\n🔍 Performance Insights:");
        
        // Find fastest operations
        if let Some(fastest) = self.results.iter().min_by(|a, b| a.avg_duration_ms.partial_cmp(&b.avg_duration_ms).unwrap()) {
            println!("  Fastest operation: {} ({:.3}ms)", fastest.test_name, fastest.avg_duration_ms);
        }
        
        // Find highest throughput
        if let Some(highest_throughput) = self.results.iter().max_by(|a, b| a.throughput_ops_per_sec.partial_cmp(&b.throughput_ops_per_sec).unwrap()) {
            println!("  Highest throughput: {} ({:.0} ops/sec)", highest_throughput.test_name, highest_throughput.throughput_ops_per_sec);
        }
        
        // Platform-specific insights
        match self.platform_info.simd_level {
            SIMDLevel::AVX512 => println!("  💡 AVX512 detected - Advanced optimizations available"),
            SIMDLevel::AVX2 => println!("  💡 AVX2 detected - Good SIMD optimization potential"),
            SIMDLevel::NEON => println!("  💡 NEON detected - ARM SIMD optimizations available"),
            SIMDLevel::WASM128 => println!("  💡 WASM SIMD detected - Web-optimized performance"),
            _ => println!("  💡 Basic SIMD - Consider upgrading hardware for better performance"),
        }
        
        if self.platform_info.core_count > 4 {
            println!("  💡 Multi-core system detected - Parallel optimizations beneficial");
        }
        
        println!("\n✅ Benchmarking completed successfully!");
    }
    
    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }
    
    /// Export results to CSV format
    pub fn export_to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("test_name,backend_type,problem_size,avg_duration_ms,min_duration_ms,max_duration_ms,std_deviation_ms,throughput_ops_per_sec,memory_usage_mb,platform,simd_level,core_count\n");
        
        for result in &self.results {
            csv.push_str(&format!(
                "{},{:?},{},{:.3},{:.3},{:.3},{:.3},{:.0},{:.2},{:?},{:?},{}\n",
                result.test_name,
                result.backend_type,
                result.problem_size,
                result.avg_duration_ms,
                result.min_duration_ms,
                result.max_duration_ms,
                result.std_deviation_ms,
                result.throughput_ops_per_sec,
                result.memory_usage_mb,
                result.platform_info.platform,
                result.platform_info.simd_level,
                result.platform_info.core_count
            ));
        }
        
        csv
    }
}

impl Default for CrossPlatformBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick benchmark runner for basic performance validation
pub fn run_quick_benchmark() {
    println!("🚀 Running Quick Cross-Platform Benchmark...");
    
    let mut benchmark = CrossPlatformBenchmark::new();
    benchmark.config.num_iterations = 5;
    benchmark.config.warmup_iterations = 1;
    benchmark.config.problem_sizes = vec![64, 256];
    
    benchmark.run_all_benchmarks();
}

/// Comprehensive benchmark runner for detailed analysis
pub fn run_comprehensive_benchmark() {
    println!("🔬 Running Comprehensive Cross-Platform Benchmark...");
    
    let config = BenchmarkConfig {
        problem_sizes: vec![64, 256, 1024, 4096, 16384],
        num_iterations: 20,
        warmup_iterations: 5,
        backends_to_test: vec![BackendType::Native],
        measure_memory: true,
        measure_cpu: true,
    };
    
    let mut benchmark = CrossPlatformBenchmark::with_config(config);
    benchmark.run_all_benchmarks();
    
    // Export results
    let csv_data = benchmark.export_to_csv();
    println!("\n📄 CSV Export Available:");
    println!("{}", csv_data);
}