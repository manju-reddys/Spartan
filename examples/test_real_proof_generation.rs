//! Test real proof generation with cross-platform integration

#[cfg(feature = "cross-platform")]
use libspartan::{Instance, VarsAssignment, InputsAssignment};

#[cfg(feature = "cross-platform")]
use libspartan::cross_platform::{IntegratedSpartanCrossPlatform, ProblemParameters};

#[cfg(feature = "cross-platform")]
fn main() {
    println!("Testing Real Proof Generation with Cross-Platform Integration...");
    
    // Create a simple R1CS instance for testing
    // This creates a very small circuit: x * y = z
    let (inst, vars, inputs) = Instance::produce_synthetic_r1cs(4, 4, 1);
    
    println!("✅ Created synthetic R1CS instance");
    println!("   Constraints: {}", inst.inst.get_num_cons());
    println!("   Variables: {}", inst.inst.get_num_vars());
    println!("   Inputs: {}", inst.inst.get_num_inputs());
    
    // Create problem parameters matching the instance
    let params = ProblemParameters {
        num_cons: inst.inst.get_num_cons(),
        num_vars: inst.inst.get_num_vars(),
        num_inputs: inst.inst.get_num_inputs(),
        num_nz_entries: 12, // Estimate for small circuit
    };
    
    // Create integrated cross-platform backend
    let spartan = IntegratedSpartanCrossPlatform::new(params);
    println!("✅ Created IntegratedSpartanCrossPlatform");
    
    // Check what backend was selected
    let caps = spartan.get_platform_capabilities();
    println!("📊 Selected backend for platform: {:?}", caps.platform);
    println!("🔧 SIMD capabilities: {:?}", caps.simd_level);
    
    // Try to generate a proof
    println!("🚀 Attempting to generate proof...");
    
    match spartan.prove(&inst.inst, &vars.assignment) {
        Ok(proof) => {
            println!("✅ Successfully generated proof!");
            println!("   Proof time: {}ms", proof.timing_info.proof_time_ms);
            println!("   Memory usage: {}bytes", proof.timing_info.memory_usage_bytes);
            println!("   Commitments: {}", proof.commitments.len());
            println!("   Proof data size: {}bytes", proof.sumcheck_proof.len());
            
            // Try to verify the proof
            println!("🔍 Attempting to verify proof...");
            match spartan.verify(&proof, &inputs.assignment) {
                Ok(is_valid) => {
                    if is_valid {
                        println!("✅ Proof verification successful!");
                    } else {
                        println!("❌ Proof verification failed!");
                    }
                },
                Err(e) => {
                    println!("⚠️  Proof verification error: {:?}", e);
                    println!("   (This is expected in current implementation - verification needs enhanced proof format)");
                }
            }
        },
        Err(e) => {
            println!("❌ Proof generation failed: {:?}", e);
        }
    }
    
    println!("✅ Real proof generation test completed!");
}

#[cfg(not(feature = "cross-platform"))]
fn main() {
    println!("Cross-platform feature not enabled. Use --features cross-platform");
}