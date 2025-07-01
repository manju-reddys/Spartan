//! Cross-platform benchmarking example
//! 
//! This example demonstrates how to use the cross-platform benchmarking suite
//! to measure performance across different platforms and optimization levels.

use libspartan::cross_platform::benchmarks::{CrossPlatformBenchmark, BenchmarkConfig, run_quick_benchmark};
use libspartan::cross_platform::{BackendType, SpartanCrossPlatform};

fn main() {
    println!("🔬 Spartan Cross-Platform Benchmark Suite");
    println!("==========================================");
    
    // Run quick benchmark for basic validation
    run_quick_benchmark();
    
    println!("\n{}", "=".repeat(50));
    
    // Run custom benchmark configuration
    let config = BenchmarkConfig {
        problem_sizes: vec![64, 256, 1024],
        num_iterations: 10,
        warmup_iterations: 2,
        backends_to_test: vec![BackendType::Native],
        measure_memory: true,
        measure_cpu: false,
    };
    
    let mut benchmark = CrossPlatformBenchmark::with_config(config);
    benchmark.run_all_benchmarks();
    
    // Print results summary
    let results = benchmark.get_results();
    println!("\n📊 Total benchmarks completed: {}", results.len());
    
    // Export to CSV
    let csv_data = benchmark.export_to_csv();
    println!("\n💾 CSV Export Preview:");
    let lines: Vec<&str> = csv_data.lines().take(5).collect();
    for line in lines {
        println!("  {}", line);
    }
    if csv_data.lines().count() > 5 {
        println!("  ... ({} more lines)", csv_data.lines().count() - 5);
    }
    
    // Demonstrate platform capabilities
    println!("\n🔍 Platform Analysis:");
    let spartan = SpartanCrossPlatform::new();
    let caps = spartan.get_platform_capabilities();
    
    println!("  Platform: {:?}", caps.platform);
    println!("  SIMD Level: {:?}", caps.simd_level);
    println!("  Core Count: {}", caps.core_count);
    println!("  Memory Limit: {:?}MB", caps.memory_limit.map(|b| b / (1024 * 1024)));
    println!("  Advanced Features: AVX2={}, AVX512={}", caps.has_avx2, caps.has_avx512);
    
    println!("\n✅ Benchmark suite completed successfully!");
}