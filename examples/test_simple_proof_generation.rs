//! Test simple proof generation with cross-platform integration

#[cfg(feature = "cross-platform")]
use libspartan::Instance;

#[cfg(feature = "cross-platform")]
use libspartan::cross_platform::{IntegratedSpartanCrossPlatform, ProblemParameters};

#[cfg(feature = "cross-platform")]
fn main() {
    println!("Testing Simple Proof Generation with Cross-Platform Integration...");
    
    // Create a simple R1CS instance for testing
    let (inst, vars, inputs) = Instance::produce_synthetic_r1cs(4, 4, 1);
    
    println!("✅ Created synthetic R1CS instance");
    
    // Create problem parameters (using reasonable defaults since we can't access private fields)
    let params = ProblemParameters {
        num_cons: 4,
        num_vars: 4, 
        num_inputs: 1,
        num_nz_entries: 12, // Reasonable estimate for small circuit
    };
    
    // Create integrated cross-platform backend
    let spartan = IntegratedSpartanCrossPlatform::new(params);
    println!("✅ Created IntegratedSpartanCrossPlatform");
    
    // Check what backend was selected
    let caps = spartan.get_platform_capabilities();
    println!("📊 Selected backend for platform: {:?}", caps.platform);
    println!("🔧 SIMD capabilities: {:?}", caps.simd_level);
    println!("💾 Available cores: {}", caps.core_count);
    
    // Create a simple R1CS shape for testing (we need to work around private fields)
    // For now, let's test with the problem parameters we can control
    
    println!("📐 Problem configuration:");
    let problem_params = spartan.get_problem_parameters();
    println!("   Constraints: {}", problem_params.num_cons);
    println!("   Variables: {}", problem_params.num_vars);
    println!("   Inputs: {}", problem_params.num_inputs);
    println!("   Non-zero entries: {}", problem_params.num_nz_entries);
    
    // For this basic test, we'll demonstrate the cross-platform infrastructure is working
    // Real proof generation would require accessing the Instance's internal R1CSShape
    println!("⚠️  Note: Full proof generation requires enhanced Instance API");
    println!("   Current implementation demonstrates cross-platform infrastructure");
    
    // Test performance metrics
    let metrics = spartan.get_metrics();
    println!("📊 Current performance metrics:");
    println!("   Proof time: {}ms", metrics.proof_time_ms);
    println!("   Verify time: {}ms", metrics.verify_time_ms);
    println!("   Memory usage: {}bytes", metrics.memory_usage_bytes);
    println!("   CPU usage: {:.1}%", metrics.cpu_usage_percent);
    
    println!("✅ Cross-platform integration infrastructure test completed successfully!");
    println!("🔧 Next steps: Enhance Instance API to expose R1CSShape for full integration");
}

#[cfg(not(feature = "cross-platform"))]
fn main() {
    println!("Cross-platform feature not enabled. Use --features cross-platform");
}