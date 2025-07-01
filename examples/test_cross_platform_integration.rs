//! Simple test to verify cross-platform integration works

#[cfg(feature = "cross-platform")]
use libspartan::cross_platform::{IntegratedSpartanCrossPlatform, ProblemParameters};

#[cfg(feature = "cross-platform")]
fn main() {
    println!("Testing Cross-Platform Spartan Integration...");
    
    // Create a small problem for testing
    let params = ProblemParameters {
        num_cons: 4,
        num_vars: 4,
        num_inputs: 1,
        num_nz_entries: 8,
    };
    
    // Create integrated cross-platform instance
    match IntegratedSpartanCrossPlatform::for_small_circuit(4, 4, 1) {
        spartan => {
            println!("✅ Successfully created IntegratedSpartanCrossPlatform");
            
            // Check platform capabilities
            let caps = spartan.get_platform_capabilities();
            println!("📊 Platform: {:?}", caps.platform);
            println!("🔧 SIMD Level: {:?}", caps.simd_level);
            println!("💾 Core Count: {}", caps.core_count);
            
            // Check problem parameters
            let problem_params = spartan.get_problem_parameters();
            println!("📐 Problem Parameters: {:?}", problem_params);
            
            println!("✅ Cross-platform integration test completed successfully!");
        }
    }
}

#[cfg(not(feature = "cross-platform"))]
fn main() {
    println!("Cross-platform feature not enabled. Use --features cross-platform");
}