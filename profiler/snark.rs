#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]

//! # SNARK Profiler
//! 
//! This profiler measures the performance characteristics of Spartan's SNARK (Succinct Non-interactive 
//! ARgument of Knowledge) system across different circuit sizes. It provides comprehensive metrics 
//! including timing, memory usage, proof sizes, and circuit complexity analysis.
//! 
//! ## What is a SNARK?
//! A SNARK allows a prover to convince a verifier that they know a witness to an R1CS instance
//! without revealing the witness itself. Spartan's SNARK provides:
//! - Succinct proofs (logarithmic in circuit size)
//! - No trusted setup required
//! - Fast verification time
//! 
//! ## Measurement Phases:
//! 1. **Setup**: Generate R1CS instance and public parameters
//! 2. **Commitment**: Create polynomial commitment to the R1CS matrices
//! 3. **Proving**: Generate the SNARK proof
//! 4. **Verification**: Verify the proof
//! 5. **Compression**: Measure proof compression efficiency

extern crate flate2;
extern crate libspartan;
extern crate merlin;

use flate2::{write::ZlibEncoder, Compression};
use libspartan::{Instance, SNARKGens, SNARK};
use merlin::Transcript;
use std::io::Write;
use std::time::Instant;

/// Metrics collected for each circuit size
#[derive(Debug)]
struct SNARKMetrics {
    /// Circuit parameters
    circuit_size: usize,          // 2^s variables/constraints
    num_vars: usize,             // Number of variables
    num_cons: usize,             // Number of constraints  
    num_inputs: usize,           // Number of public inputs
    
    /// Timing measurements (in milliseconds)
    setup_time: u128,            // R1CS generation time
    keygen_time: u128,           // Public parameter generation time
    commitment_time: u128,       // Polynomial commitment time
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
    proof_generation_rate: f64,      // MB/sec during proof generation
    verification_rate: f64,          // MB/sec during verification
    amortized_proving_cost: f64,     // ms per constraint during proving
    amortized_verification_cost: f64, // ms per constraint during verification
}

impl SNARKMetrics {
    /// Create a new metrics instance
    fn new(circuit_size: usize, num_vars: usize, num_cons: usize, num_inputs: usize) -> Self {
        Self {
            circuit_size,
            num_vars,
            num_cons, 
            num_inputs,
            setup_time: 0,
            keygen_time: 0,
            commitment_time: 0,
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
            proof_generation_rate: 0.0,
            verification_rate: 0.0,
            amortized_proving_cost: 0.0,
            amortized_verification_cost: 0.0,
        }
    }
    
    /// Calculate derived metrics
    fn finalize(&mut self) {
        self.total_time = self.setup_time + self.keygen_time + self.commitment_time 
                         + self.proving_time + self.verification_time;
        
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
    println!("\n{}", "=".repeat(140));
    println!("{:^140}", "SPARTAN SNARK PERFORMANCE PROFILER");
    println!("{}", "=".repeat(140));
    println!("{:>6} | {:>8} | {:>8} | {:>8} | {:>10} | {:>10} | {:>8} | {:>10} | {:>10} | {:>8} | {:>10} | {:>10}", 
             "Size", "Setup", "KeyGen", "Commit", "ProveTime", "VerifyTime", "Total", "Proof(KB)", "Comp(KB)", "Ratio", "Prove(K/s)", "Verify(K/s)");
    println!("{}", "-".repeat(140));
}

/// Print metrics for a single circuit size
fn print_metrics(metrics: &SNARKMetrics) {
    println!("{:>6} | {:>8} | {:>8} | {:>8} | {:>10} | {:>10} | {:>8} | {:>10.1} | {:>10.1} | {:>8.2} | {:>10.0} | {:>10.0}",
             metrics.circuit_size,
             metrics.setup_time,
             metrics.keygen_time, 
             metrics.commitment_time,
             metrics.proving_time,
             metrics.verification_time,
             metrics.total_time,
             metrics.proof_size_uncompressed as f64 / 1024.0,
             metrics.proof_size_compressed as f64 / 1024.0,
             metrics.compression_ratio,
             metrics.constraints_per_second / 1000.0,
             metrics.verification_efficiency
    );
}

/// Print detailed analysis for a single circuit
fn print_detailed_analysis(metrics: &SNARKMetrics) {
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
    println!("       • Compression: {:>8.1}% space saved", (1.0 - metrics.compression_ratio) * 100.0);
    println!("       • Proof Overhead: {:>8.1} bytes/constraint", 
             metrics.proof_size_compressed as f64 / metrics.num_cons as f64);
}

/// Print summary statistics across all circuit sizes
fn print_summary(all_metrics: &[SNARKMetrics]) {
    println!("\n{}", "=".repeat(140));
    println!("{:^140}", "PROOF GENERATION & VERIFICATION ANALYSIS");
    println!("{}", "=".repeat(140));
    
    if all_metrics.is_empty() {
        return;
    }
    
    // Calculate proof generation and verification statistics
    let avg_compression = all_metrics.iter().map(|m| m.compression_ratio).sum::<f64>() / all_metrics.len() as f64;
    let max_proving_throughput = all_metrics.iter().map(|m| m.constraints_per_second).fold(0.0, f64::max);
    let min_proving_time = all_metrics.iter().map(|m| m.proving_time).min().unwrap();
    let max_proving_time = all_metrics.iter().map(|m| m.proving_time).max().unwrap();
    let min_verification_time = all_metrics.iter().map(|m| m.verification_time).min().unwrap();
    let max_verification_time = all_metrics.iter().map(|m| m.verification_time).max().unwrap();
    
    // Calculate average performance metrics
    let avg_proving_cost = all_metrics.iter().map(|m| m.amortized_proving_cost).sum::<f64>() / all_metrics.len() as f64;
    let avg_verification_cost = all_metrics.iter().map(|m| m.amortized_verification_cost).sum::<f64>() / all_metrics.len() as f64;
    let max_verification_throughput = all_metrics.iter().map(|m| m.verification_efficiency).fold(0.0, f64::max);
    
    println!("🚀 Proof Generation Performance:");
    println!("   • Peak Proving Throughput: {:.0} constraints/sec", max_proving_throughput);
    println!("   • Proving Time Range: {} - {} ms", min_proving_time, max_proving_time);
    println!("   • Average Cost per Constraint: {:.3} ms/constraint", avg_proving_cost);
    
    println!("\n⚡ Verification Performance:");
    println!("   • Peak Verification Throughput: {:.2} constraints/ms", max_verification_throughput);
    println!("   • Verification Time Range: {} - {} ms", min_verification_time, max_verification_time);
    println!("   • Average Verification Cost: {:.6} ms/constraint", avg_verification_cost);
    
    println!("\n📦 Proof Size & Compression:");
    println!("   • Average Compression Ratio: {:.1}%", (1.0 - avg_compression) * 100.0);
    
    // Scalability analysis focused on proving and verification
    if all_metrics.len() >= 2 {
        let first = &all_metrics[0];
        let last = &all_metrics[all_metrics.len() - 1];
        let size_ratio = last.num_cons as f64 / first.num_cons as f64;
        let proving_time_ratio = last.proving_time as f64 / first.proving_time as f64;
        let verification_time_ratio = last.verification_time as f64 / first.verification_time as f64;
        
        println!("\n📈 Scalability Analysis:");
        println!("   • Circuit Size Increase: {:.0}x (from 2^{} to 2^{})", 
                 size_ratio, first.circuit_size, last.circuit_size);
        println!("   • Proving Time Scaling: {:.1}x increase", proving_time_ratio);
        println!("   • Verification Time Scaling: {:.1}x increase", verification_time_ratio);
        
        // Calculate complexity estimates
        let proving_complexity = proving_time_ratio.log2() / size_ratio.log2();
        let verification_complexity = verification_time_ratio.log2() / size_ratio.log2();
        
        println!("   • Estimated Proving Complexity: O(n^{:.2})", proving_complexity);
        println!("   • Estimated Verification Complexity: O(n^{:.2})", verification_complexity);
        
        // Performance efficiency metrics
        let proving_efficiency = if proving_time_ratio < size_ratio { "✅ Sub-linear" } else { "⚠️ Linear or worse" };
        let verification_efficiency = if verification_time_ratio < (size_ratio.log2()) { "✅ Logarithmic" } else { "⚠️ Worse than logarithmic" };
        
        println!("\n🎯 Efficiency Assessment:");
        println!("   • Proving Efficiency: {}", proving_efficiency);
        println!("   • Verification Efficiency: {}", verification_efficiency);
    }
    
    println!("{}", "=".repeat(140));
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
        
        let mut metrics = SNARKMetrics::new(s, num_vars, num_cons, num_inputs);
        let total_start = Instant::now();
        
        // Phase 1: Setup - Generate synthetic R1CS instance
        // This creates a random but satisfiable constraint system for benchmarking
        let setup_start = Instant::now();
        let (inst, vars, inputs) = Instance::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);
        metrics.setup_time = setup_start.elapsed().as_millis();
        
        // Phase 2: Key Generation - Create public parameters
        // These parameters are reusable for any R1CS of the same size
        let keygen_start = Instant::now();
        let gens = SNARKGens::new(num_cons, num_vars, num_inputs, num_cons);
        metrics.keygen_time = keygen_start.elapsed().as_millis();
        
        // Phase 3: Commitment - Create polynomial commitments to R1CS matrices
        // This step hides the constraint matrices while allowing later verification
        let commit_start = Instant::now();
        let (comm, decomm) = SNARK::encode(&inst, &gens);
        metrics.commitment_time = commit_start.elapsed().as_millis();
        
        // Phase 4: Proving - Generate the SNARK proof
        // This is the core cryptographic operation that proves satisfiability
        let proving_start = Instant::now();
        let mut prover_transcript = Transcript::new(b"snark_example");
        let proof = SNARK::prove(
            &inst,
            &comm,
            &decomm,
            vars,
            &inputs,
            &gens,
            &mut prover_transcript,
        );
        metrics.proving_time = proving_start.elapsed().as_millis();
        
        // Phase 5: Proof Serialization and Compression
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
        
        // Phase 6: Verification - Verify the proof
        // This should be fast regardless of circuit size (succinct verification)
        let verification_start = Instant::now();
        let mut verifier_transcript = Transcript::new(b"snark_example");
        let verification_result = proof.verify(&comm, &inputs, &mut verifier_transcript, &gens);
        metrics.verification_time = verification_start.elapsed().as_millis();
        
        metrics.total_time = total_start.elapsed().as_millis();
        
        // Verify the proof is correct
        assert!(verification_result.is_ok(), "SNARK verification failed for size 2^{}", s);
        
        // Calculate derived metrics
        metrics.finalize();
        
        // Display results
        print_metrics(&metrics);
        print_detailed_analysis(&metrics);
        
        all_metrics.push(metrics);
    }
    
    // Print comprehensive summary
    print_summary(&all_metrics);
    
    println!("\n✅ All SNARK proofs verified successfully!");
    println!("💡 Note: Proving time scales quasi-linearly, verification time is succinct (logarithmic)");
    println!("📈 Proof generation and verification times are the key metrics for practical usage");
}
