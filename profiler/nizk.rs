#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]

//! # NIZK Profiler
//! 
//! This profiler measures the performance characteristics of Spartan's NIZK (Non-Interactive Zero-Knowledge)
//! proof system across different circuit sizes. Unlike SNARKs, NIZKs don't require polynomial commitments
//! but still provide zero-knowledge proofs of R1CS satisfiability.
//! 
//! ## What is a NIZK?
//! A NIZK allows a prover to convince a verifier that they know a witness to an R1CS instance
//! without revealing the witness itself. Spartan's NIZK provides:
//! - Zero-knowledge proofs (hides the witness)
//! - No trusted setup required
//! - Simpler structure than SNARKs (no commitments needed)
//! - Direct proof of R1CS satisfiability
//! 
//! ## Key Differences from SNARK:
//! - No polynomial commitment phase (simpler workflow)
//! - May have different proof sizes and verification times
//! - Still provides zero-knowledge guarantees
//! 
//! ## Measurement Phases:
//! 1. **Setup**: Generate R1CS instance and public parameters
//! 2. **Proving**: Generate the NIZK proof directly
//! 3. **Verification**: Verify the proof
//! 4. **Compression**: Measure proof compression efficiency

extern crate flate2;
extern crate libspartan;
extern crate merlin;
extern crate rand;

use flate2::{write::ZlibEncoder, Compression};
use libspartan::{Instance, NIZKGens, NIZK};
use merlin::Transcript;
use std::io::Write;
use std::time::Instant;

/// Metrics collected for each circuit size
#[derive(Debug)]
struct NIZKMetrics {
    /// Circuit parameters
    circuit_size: usize,          // 2^s variables/constraints
    num_vars: usize,             // Number of variables
    num_cons: usize,             // Number of constraints  
    num_inputs: usize,           // Number of public inputs
    
    /// Timing measurements (in milliseconds)
    setup_time: u128,            // R1CS generation time
    keygen_time: u128,           // Public parameter generation time
    proving_time: u128,          // Proof generation time
    verification_time: u128,     // Proof verification time
    total_time: u128,           // Total end-to-end time
    
    /// Detailed proof generation breakdown
    proof_serialization_time: u128,  // Time to serialize proof
    proof_compression_time: u128,    // Time to compress proof
    
    /// Memory and size metrics
    proof_size_uncompressed: usize,  // Raw proof size in bytes
    proof_size_compressed: usize,    // Compressed proof size in bytes
    compression_ratio: f64,          // Compression efficiency
    
    /// Performance metrics
    constraints_per_second: f64,     // Proving throughput
    verification_efficiency: f64,    // Verification speed (constraints/ms)
    space_efficiency: f64,           // Bytes per constraint
    proof_generation_rate: f64,      // MB/sec during proof generation
    verification_rate: f64,          // MB/sec during verification
    amortized_proving_cost: f64,     // ms per constraint during proving
    amortized_verification_cost: f64, // ms per constraint during verification
}

impl NIZKMetrics {
    /// Create a new metrics instance
    fn new(circuit_size: usize, num_vars: usize, num_cons: usize, num_inputs: usize) -> Self {
        Self {
            circuit_size,
            num_vars,
            num_cons, 
            num_inputs,
            setup_time: 0,
            keygen_time: 0,
            proving_time: 0,
            verification_time: 0,
            total_time: 0,
            proof_serialization_time: 0,
            proof_compression_time: 0,
            proof_size_uncompressed: 0,
            proof_size_compressed: 0,
            compression_ratio: 0.0,
            constraints_per_second: 0.0,
            verification_efficiency: 0.0,
            space_efficiency: 0.0,
            proof_generation_rate: 0.0,
            verification_rate: 0.0,
            amortized_proving_cost: 0.0,
            amortized_verification_cost: 0.0,
        }
    }
    
    /// Calculate derived metrics
    fn finalize(&mut self) {
        self.total_time = self.setup_time + self.keygen_time + self.proving_time + self.verification_time;
        
        self.compression_ratio = if self.proof_size_uncompressed > 0 {
            self.proof_size_compressed as f64 / self.proof_size_uncompressed as f64
        } else {
            0.0
        };
        
        self.constraints_per_second = if self.proving_time > 0 {
            (self.num_cons as f64 * 1000.0) / self.proving_time as f64
        } else {
            0.0
        };
        
        self.verification_efficiency = if self.verification_time > 0 {
            (self.num_cons as f64) / self.verification_time as f64
        } else {
            0.0
        };
        
        self.space_efficiency = if self.num_cons > 0 {
            self.proof_size_compressed as f64 / self.num_cons as f64
        } else {
            0.0
        };
        
        self.proof_generation_rate = if self.proving_time > 0 {
            (self.proof_size_uncompressed as f64) / (self.proving_time as f64 * 1024.0 * 1024.0)
        } else {
            0.0
        };
        
        self.verification_rate = if self.verification_time > 0 {
            (self.proof_size_uncompressed as f64) / (self.verification_time as f64 * 1024.0 * 1024.0)
        } else {
            0.0
        };
        
        self.amortized_proving_cost = if self.proving_time > 0 {
            self.proving_time as f64 / self.num_cons as f64
        } else {
            0.0
        };
        
        self.amortized_verification_cost = if self.verification_time > 0 {
            self.verification_time as f64 / self.num_cons as f64
        } else {
            0.0
        };
    }
}

/// Print a formatted header for the metrics table
fn print_header() {
    println!("\n{}", "=".repeat(135));
    println!("{:^135}", "SPARTAN NIZK PERFORMANCE PROFILER");
    println!("{}", "=".repeat(135));
    println!("{:>6} | {:>8} | {:>8} | {:>10} | {:>10} | {:>8} | {:>10} | {:>10} | {:>8} | {:>8} | {:>10} | {:>10}", 
             "Size", "Setup", "KeyGen", "ProveTime", "VerifyTime", "Total", "Proof(KB)", "Comp(KB)", "Ratio", "B/Cons", "Prove(K/s)", "Verify(K/s)");
    println!("{}", "-".repeat(135));
}

/// Print metrics for a single circuit size
fn print_metrics(metrics: &NIZKMetrics) {
    println!("{:>6} | {:>8} | {:>8} | {:>10} | {:>10} | {:>8} | {:>10.1} | {:>10.1} | {:>8.2} | {:>8.1} | {:>10.0} | {:>10.0}",
             metrics.circuit_size,
             metrics.setup_time,
             metrics.keygen_time, 
             metrics.proving_time,
             metrics.verification_time,
             metrics.total_time,
             metrics.proof_size_uncompressed as f64 / 1024.0,
             metrics.proof_size_compressed as f64 / 1024.0,
             metrics.compression_ratio,
             metrics.space_efficiency,
             metrics.constraints_per_second / 1000.0,
             metrics.verification_efficiency
    );
}

/// Print detailed analysis for a single circuit
fn print_detailed_analysis(metrics: &NIZKMetrics) {
    println!("\n    📊 Circuit Analysis (Size: 2^{}):", metrics.circuit_size);
    println!("       • Variables: {:>10}  • Constraints: {:>10}  • Inputs: {:>5}", 
             metrics.num_vars, metrics.num_cons, metrics.num_inputs);
    
    println!("\n    ⚡ Proof Generation Performance:");
    println!("       • Proving Time: {:>8} ms ({:.3} ms/constraint)", 
             metrics.proving_time, metrics.amortized_proving_cost);
    println!("       • Proving Throughput: {:>8.0} constraints/sec", metrics.constraints_per_second);
    println!("       • Proof Generation Rate: {:>8.2} MB/sec", metrics.proof_generation_rate * 1000.0);
    
    println!("\n    🔍 Verification Performance:");
    println!("       • Verification Time: {:>8} ms ({:.6} ms/constraint)", 
             metrics.verification_time, metrics.amortized_verification_cost);
    println!("       • Verification Throughput: {:>8.2} constraints/ms", metrics.verification_efficiency);
    println!("       • Verification Rate: {:>8.2} MB/sec", metrics.verification_rate * 1000.0);
    
    println!("\n    💾 Space Efficiency:");
    println!("       • Compression: {:>8.1}% space saved, {:.1} bytes/constraint", 
             (1.0 - metrics.compression_ratio) * 100.0, metrics.space_efficiency);
}

/// Print summary statistics across all circuit sizes
fn print_summary(all_metrics: &[NIZKMetrics]) {
    println!("\n{}", "=".repeat(135));
    println!("{:^135}", "NIZK PROOF GENERATION & VERIFICATION ANALYSIS");
    println!("{}", "=".repeat(135));
    
    if all_metrics.is_empty() {
        return;
    }
    
    // Calculate proof generation and verification statistics
    let total_proving_time: u128 = all_metrics.iter().map(|m| m.proving_time).sum();
    let total_constraints: usize = all_metrics.iter().map(|m| m.num_cons).sum();
    let overall_throughput = if total_proving_time > 0 {
        (total_constraints as f64 * 1000.0) / total_proving_time as f64
    } else {
        0.0
    };
    
    let min_proving_time = all_metrics.iter().map(|m| m.proving_time).min().unwrap();
    let max_proving_time = all_metrics.iter().map(|m| m.proving_time).max().unwrap();
    let min_verification_time = all_metrics.iter().map(|m| m.verification_time).min().unwrap();
    let max_verification_time = all_metrics.iter().map(|m| m.verification_time).max().unwrap();
    let avg_compression = all_metrics.iter().map(|m| m.compression_ratio).sum::<f64>() / all_metrics.len() as f64;
    
    // Calculate average performance metrics
    let avg_proving_cost = all_metrics.iter().map(|m| m.amortized_proving_cost).sum::<f64>() / all_metrics.len() as f64;
    let avg_verification_cost = all_metrics.iter().map(|m| m.amortized_verification_cost).sum::<f64>() / all_metrics.len() as f64;
    let max_proving_throughput = all_metrics.iter().map(|m| m.constraints_per_second).fold(0.0, f64::max);
    let max_verification_throughput = all_metrics.iter().map(|m| m.verification_efficiency).fold(0.0, f64::max);
    
    println!("🚀 NIZK Proof Generation Performance:");
    println!("   • Peak Proving Throughput: {:.0} constraints/sec", max_proving_throughput);
    println!("   • Overall Proving Throughput: {:.0} constraints/sec", overall_throughput);
    println!("   • Proving Time Range: {} - {} ms", min_proving_time, max_proving_time);
    println!("   • Average Cost per Constraint: {:.3} ms/constraint", avg_proving_cost);
    
    println!("\n⚡ NIZK Verification Performance:");
    println!("   • Peak Verification Throughput: {:.2} constraints/ms", max_verification_throughput);
    println!("   • Verification Time Range: {} - {} ms", min_verification_time, max_verification_time);
    println!("   • Average Verification Cost: {:.6} ms/constraint", avg_verification_cost);
    
    println!("\n📦 NIZK Proof Characteristics:");
    println!("   • Total Constraints Processed: {}", total_constraints);
    println!("   • Average Compression Ratio: {:.1}%", (1.0 - avg_compression) * 100.0);
    
    println!("{}", "=".repeat(135));
}

/// Compare NIZK vs SNARK characteristics
fn print_nizk_analysis(all_metrics: &[NIZKMetrics]) {
    println!("\n{}", "=".repeat(135));
    println!("{:^135}", "NIZK CHARACTERISTICS & SCALABILITY ANALYSIS");
    println!("{}", "=".repeat(135));
    
    if all_metrics.is_empty() {
        return;
    }
    
    // Calculate NIZK-specific insights
    let avg_compression = all_metrics.iter().map(|m| m.compression_ratio).sum::<f64>() / all_metrics.len() as f64;
    let avg_space_efficiency = all_metrics.iter().map(|m| m.space_efficiency).sum::<f64>() / all_metrics.len() as f64;
    let max_proving_throughput = all_metrics.iter().map(|m| m.constraints_per_second).fold(0.0, f64::max);
    let min_verification_time = all_metrics.iter().map(|m| m.verification_time).min().unwrap();
    
    println!("🔍 NIZK Performance Insights:");
    println!("   • Best Proving Throughput: {:.0} constraints/sec", max_proving_throughput);
    println!("   • Fastest Verification: {} ms", min_verification_time);
    println!("   • Average Compression: {:.1}%", (1.0 - avg_compression) * 100.0);
    println!("   • Space Overhead: {:.1} bytes per constraint", avg_space_efficiency);
    
    // Workflow comparison
    println!("\n🔄 NIZK vs SNARK Workflow Comparison:");
    println!("   • NIZK: Setup → Prove → Verify (3 phases)");
    println!("   • SNARK: Setup → KeyGen → Commit → Prove → Verify (5 phases)");
    println!("   • NIZK eliminates polynomial commitment overhead");
    println!("   • NIZK focuses on direct R1CS satisfiability proof");
    
    // Scalability analysis focused on proving and verification
    if all_metrics.len() >= 2 {
        let first = &all_metrics[0];
        let last = &all_metrics[all_metrics.len() - 1];
        let size_ratio = last.num_cons as f64 / first.num_cons as f64;
        let proving_time_ratio = last.proving_time as f64 / first.proving_time as f64;
        let verification_time_ratio = last.verification_time as f64 / first.verification_time as f64;
        let space_ratio = last.proof_size_compressed as f64 / first.proof_size_compressed as f64;
        
        println!("\n📈 NIZK Scalability Analysis:");
        println!("   • Circuit Size Increase: {:.0}x (from 2^{} to 2^{})", 
                 size_ratio, first.circuit_size, last.circuit_size);
        println!("   • Proving Time Scaling: {:.1}x increase", proving_time_ratio);
        println!("   • Verification Time Scaling: {:.1}x increase", verification_time_ratio);
        println!("   • Proof Size Scaling: {:.1}x increase", space_ratio);
        
        // Calculate complexity estimates
        let proving_complexity = proving_time_ratio.log2() / size_ratio.log2();
        let verification_complexity = verification_time_ratio.log2() / size_ratio.log2();
        
        println!("\n🧮 Estimated Complexity:");
        println!("   • Proving Time Complexity: O(n^{:.2})", proving_complexity);
        println!("   • Verification Time Complexity: O(n^{:.2})", verification_complexity);
        
        // Performance efficiency assessment
        let proving_efficiency = if proving_time_ratio < size_ratio { "✅ Sub-linear" } else { "⚠️ Linear or worse" };
        let verification_efficiency = if verification_time_ratio < size_ratio.sqrt() { "✅ Sub-linear" } else { "⚠️ Linear or worse" };
        
        println!("\n🎯 NIZK Efficiency Assessment:");
        println!("   • Proving Efficiency: {}", proving_efficiency);
        println!("   • Verification Efficiency: {}", verification_efficiency);
    }
    
    println!("{}", "=".repeat(135));
}

pub fn main() {
    // Circuit sizes to test: 2^10 to 2^20 (1K to 1M constraints)
    let circuit_sizes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    let mut all_metrics = Vec::new();
    
    print_header();
    
    for &s in circuit_sizes.iter() {
        // Calculate circuit parameters
        // In R1CS, we typically have equal numbers of variables and constraints
        let num_vars = (2_usize).pow(s as u32);  // 2^s variables
        let num_cons = num_vars;                  // Same number of constraints
        let num_inputs = 10;                      // Fixed number of public inputs
        
        let mut metrics = NIZKMetrics::new(s, num_vars, num_cons, num_inputs);
        let total_start = Instant::now();
        
        // Phase 1: Setup - Generate synthetic R1CS instance
        // This creates a random but satisfiable constraint system for benchmarking
        let setup_start = Instant::now();
        let (inst, vars, inputs) = Instance::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);
        metrics.setup_time = setup_start.elapsed().as_millis();
        
        // Phase 2: Key Generation - Create public parameters for NIZK
        // NIZK generators are simpler than SNARK generators (no commitment needed)
        let keygen_start = Instant::now();
        let gens = NIZKGens::new(num_cons, num_vars, num_inputs);
        metrics.keygen_time = keygen_start.elapsed().as_millis();
        
        // Phase 3: Proving - Generate the NIZK proof directly
        // NIZKs skip the commitment phase and prove R1CS satisfiability directly
        // This is the core operation that generates a zero-knowledge proof
        let proving_start = Instant::now();
        let mut prover_transcript = Transcript::new(b"nizk_example");
        let proof = NIZK::prove(&inst, vars, &inputs, &gens, &mut prover_transcript);
        metrics.proving_time = proving_start.elapsed().as_millis();
        
        // Phase 4: Proof Serialization and Compression
        // Measure both raw and compressed proof sizes to understand space efficiency
        let serialization_start = Instant::now();
        let proof_uncompressed = bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        metrics.proof_serialization_time = serialization_start.elapsed().as_millis();
        metrics.proof_size_uncompressed = proof_uncompressed.len();
        
        // Apply compression to measure space efficiency
        let compression_start = Instant::now();
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&proof_uncompressed).unwrap();
        let proof_compressed = encoder.finish().unwrap();
        metrics.proof_compression_time = compression_start.elapsed().as_millis();
        metrics.proof_size_compressed = proof_compressed.len();
        
        // Phase 5: Verification - Verify the NIZK proof
        // Verification should be efficient and independent of prover computation
        let verification_start = Instant::now();
        let mut verifier_transcript = Transcript::new(b"nizk_example");
        let verification_result = proof.verify(&inst, &inputs, &mut verifier_transcript, &gens);
        metrics.verification_time = verification_start.elapsed().as_millis();
        
        metrics.total_time = total_start.elapsed().as_millis();
        
        // Verify the proof is correct
        assert!(verification_result.is_ok(), "NIZK verification failed for size 2^{}", s);
        
        // Calculate derived metrics
        metrics.finalize();
        
        // Display results
        print_metrics(&metrics);
        print_detailed_analysis(&metrics);
        
        all_metrics.push(metrics);
    }
    
    // Print comprehensive analysis
    print_summary(&all_metrics);
    print_nizk_analysis(&all_metrics);
    
    println!("\n✅ All NIZK proofs verified successfully!");
    println!("💡 Note: NIZKs provide zero-knowledge without polynomial commitments");
    println!("🔬 Use this data to compare NIZK vs SNARK performance characteristics");
    println!("📈 Focus on proof generation and verification times for practical applications");
}
