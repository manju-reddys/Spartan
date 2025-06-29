//! # Cubic Equation SNARK Example
//! 
//! This example demonstrates how to create a SNARK proof for the cubic equation: `x³ + x + 5 = y`.
//! It walks through the complete process of:
//! 1. Converting the equation into an R1CS (Rank-1 Constraint System)
//! 2. Creating a satisfying assignment
//! 3. Generating a SNARK proof
//! 4. Verifying the proof
//! 
//! ## The Mathematical Problem
//! We want to prove knowledge of a secret value `x` such that:
//! ```
//! x³ + x + 5 = y
//! ```
//! where `y` is public and `x` is the secret witness.
//! 
//! ## R1CS Construction
//! R1CS constrains computations to the form: A⃗z ∘ B⃗z = C⃗z
//! where z = (variables, 1, public_inputs) and ∘ is element-wise multiplication.
//! 
//! To represent `x³ + x + 5 = y`, we introduce intermediate variables:
//! - Z0 = x (our secret input)
//! - Z1 = x² (intermediate: x squared)  
//! - Z2 = x³ (intermediate: x cubed)
//! - Z3 = x³ + x (intermediate: sum)
//! - I0 = y (public output)
//! 
//! This gives us the constraint system:
//! 1. `Z0 * Z0 - Z1 = 0`        (compute x²)
//! 2. `Z1 * Z0 - Z2 = 0`        (compute x³)  
//! 3. `(Z2 + Z0) * 1 - Z3 = 0`  (compute x³ + x)
//! 4. `(Z3 + 5) * 1 - I0 = 0`   (add constant and check against public output)
//! 
//! For more details, see: https://medium.com/@VitalikButerin/quadratic-arithmetic-programs-from-zero-to-hero-f6d558cea649

#![allow(clippy::assertions_on_result_states)]
use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, SNARKGens, VarsAssignment, SNARK};
use merlin::Transcript;
use rand::rngs::OsRng;

/// Constructs the R1CS instance for the cubic equation x³ + x + 5 = y
/// 
/// Returns:
/// - Circuit parameters (constraints, variables, inputs, non-zero entries)
/// - R1CS instance
/// - Satisfying variable assignment  
/// - Public input assignment
#[allow(non_snake_case)]
fn produce_r1cs() -> (
  usize,
  usize,
  usize,
  usize,
  Instance,
  VarsAssignment,
  InputsAssignment,
) {
  // Circuit parameters for our cubic equation
  let num_cons = 4;                  // 4 constraints (one per intermediate computation)
  let num_vars = 4;                  // 4 variables: Z0=x, Z1=x², Z2=x³, Z3=x³+x
  let num_inputs = 1;                // 1 public input: y (the result)
  let num_non_zero_entries = 8;      // Total non-zero entries across all constraint matrices

  // R1CS constraint matrices A, B, C
  // Each constraint is of the form: (A⃗z) ∘ (B⃗z) = (C⃗z)
  // where z = [Z0, Z1, Z2, Z3, 1, I0] = [x, x², x³, x³+x, 1, y]
  let mut A: Vec<(usize, usize, [u8; 32])> = Vec::new();  // Left operand matrix
  let mut B: Vec<(usize, usize, [u8; 32])> = Vec::new();  // Right operand matrix  
  let mut C: Vec<(usize, usize, [u8; 32])> = Vec::new();  // Result matrix

  let one = Scalar::ONE.to_bytes();  // Constant 1 in field representation

  println!("🔧 Constructing R1CS for cubic equation: x³ + x + 5 = y");
  println!("   Variables: Z0=x, Z1=x², Z2=x³, Z3=x³+x");
  println!("   Public inputs: I0=y");
  println!("   Constraint vector: z = [Z0, Z1, Z2, Z3, 1, I0]");

  // Constraint 0: Z0 * Z0 = Z1  (compute x²)
  // This constraint computes the square of our input
  println!("\n📝 Constraint 0: Z0 * Z0 = Z1  (x² computation)");
  A.push((0, 0, one));               // A[0,0] = 1 (coefficient of Z0)
  B.push((0, 0, one));               // B[0,0] = 1 (coefficient of Z0)  
  C.push((0, 1, one));               // C[0,1] = 1 (coefficient of Z1)
  println!("   A⃗z = Z0, B⃗z = Z0, C⃗z = Z1");

  // Constraint 1: Z1 * Z0 = Z2  (compute x³)
  // This constraint multiplies x² by x to get x³
  println!("\n📝 Constraint 1: Z1 * Z0 = Z2  (x³ computation)");
  A.push((1, 1, one));               // A[1,1] = 1 (coefficient of Z1 = x²)
  B.push((1, 0, one));               // B[1,0] = 1 (coefficient of Z0 = x)
  C.push((1, 2, one));               // C[1,2] = 1 (coefficient of Z2 = x³)
  println!("   A⃗z = Z1, B⃗z = Z0, C⃗z = Z2");

  // Constraint 2: (Z2 + Z0) * 1 = Z3  (compute x³ + x)
  // This constraint adds x³ and x together
  println!("\n📝 Constraint 2: (Z2 + Z0) * 1 = Z3  (x³ + x computation)");
  A.push((2, 2, one));               // A[2,2] = 1 (coefficient of Z2 = x³)
  A.push((2, 0, one));               // A[2,0] = 1 (coefficient of Z0 = x)
  B.push((2, num_vars, one));        // B[2,4] = 1 (coefficient of constant 1)
  C.push((2, 3, one));               // C[2,3] = 1 (coefficient of Z3 = x³+x)
  println!("   A⃗z = Z2 + Z0, B⃗z = 1, C⃗z = Z3");

  // Constraint 3: (Z3 + 5) * 1 = I0  (add constant and check public output)
  // This constraint adds the constant 5 and checks against the public output y
  println!("\n📝 Constraint 3: (Z3 + 5) * 1 = I0  (final computation with constant)");
  A.push((3, 3, one));               // A[3,3] = 1 (coefficient of Z3 = x³+x)
  A.push((3, num_vars, Scalar::from(5u32).to_bytes())); // A[3,4] = 5 (constant term)
  B.push((3, num_vars, one));        // B[3,4] = 1 (coefficient of constant 1)
  C.push((3, num_vars + 1, one));    // C[3,5] = 1 (coefficient of I0 = y)
  println!("   A⃗z = Z3 + 5, B⃗z = 1, C⃗z = I0");

  // Create the R1CS instance
  println!("\n🏗️  Creating R1CS instance with {} constraints, {} variables, {} inputs", 
           num_cons, num_vars, num_inputs);
  let inst = Instance::new(num_cons, num_vars, num_inputs, &A, &B, &C).unwrap();

  // Generate a satisfying assignment
  println!("\n🎲 Generating satisfying assignment...");
  let mut csprng: OsRng = OsRng;
  let z0 = Scalar::random(&mut csprng);    // Random secret input x
  let z1 = z0 * z0;                        // x²  
  let z2 = z1 * z0;                        // x³
  let z3 = z2 + z0;                        // x³ + x
  let i0 = z3 + Scalar::from(5u32);       // x³ + x + 5 = y (public output)

  println!("   Secret input x (Z0): [random 32-byte scalar]");
  println!("   x² (Z1): [computed from Z0]");  
  println!("   x³ (Z2): [computed from Z1 * Z0]");
  println!("   x³ + x (Z3): [computed from Z2 + Z0]");
  println!("   Public output y (I0): [computed from Z3 + 5]");

  // Create variable assignment (private witness)
  let mut vars = vec![Scalar::ZERO.to_bytes(); num_vars];
  vars[0] = z0.to_bytes();  // Z0 = x
  vars[1] = z1.to_bytes();  // Z1 = x²  
  vars[2] = z2.to_bytes();  // Z2 = x³
  vars[3] = z3.to_bytes();  // Z3 = x³ + x
  let assignment_vars = VarsAssignment::new(&vars).unwrap();

  // Create input assignment (public values)
  let mut inputs = vec![Scalar::ZERO.to_bytes(); num_inputs];
  inputs[0] = i0.to_bytes();  // I0 = y = x³ + x + 5
  let assignment_inputs = InputsAssignment::new(&inputs).unwrap();

  // Verify our assignment satisfies the R1CS
  println!("\n✅ Verifying R1CS satisfiability...");
  let res = inst.is_sat(&assignment_vars, &assignment_inputs);
  assert!(res.unwrap(), "R1CS should be satisfied by our assignment");
  println!("   ✓ All constraints satisfied!");

  (
    num_cons,
    num_vars,
    num_inputs,
    num_non_zero_entries,
    inst,
    assignment_vars,
    assignment_inputs,
  )
}

fn main() {
  println!("{}", "=".repeat(80));
  println!("{:^80}", "SPARTAN SNARK: CUBIC EQUATION EXAMPLE");
  println!("{:^80}", "Proving knowledge of x such that x³ + x + 5 = y");
  println!("{}", "=".repeat(80));

  // Phase 1: Construct the R1CS instance
  println!("\n🚀 Phase 1: Constructing R1CS for cubic equation");
  let (
    num_cons,
    num_vars,
    num_inputs,
    num_non_zero_entries,
    inst,
    assignment_vars,
    assignment_inputs,
  ) = produce_r1cs();

  // Phase 2: Generate public parameters
  println!("\n🔑 Phase 2: Generating SNARK public parameters");
  println!("   This creates reusable parameters for any cubic equation proof");
  let gens = SNARKGens::new(num_cons, num_vars, num_inputs, num_non_zero_entries);
  println!("   ✓ Public parameters generated");

  // Phase 3: Create polynomial commitment to R1CS
  println!("\n🔒 Phase 3: Creating polynomial commitments");
  println!("   This hides the constraint matrices while enabling verification");
  let (comm, decomm) = SNARK::encode(&inst, &gens);
  println!("   ✓ Polynomial commitments created");

  // Phase 4: Generate the SNARK proof
  println!("\n🧮 Phase 4: Generating SNARK proof");
  println!("   This proves knowledge of x without revealing it");
  let mut prover_transcript = Transcript::new(b"snark_example");
  let proof = SNARK::prove(
    &inst,
    &comm,
    &decomm,
    assignment_vars,
    &assignment_inputs,
    &gens,
    &mut prover_transcript,
  );
  println!("   ✓ SNARK proof generated successfully!");

  // Phase 5: Verify the proof
  println!("\n🔍 Phase 5: Verifying SNARK proof");
  println!("   The verifier only sees the public output y, not the secret x");
  let mut verifier_transcript = Transcript::new(b"snark_example");
  let verification_result = proof.verify(
    &comm, 
    &assignment_inputs, 
    &mut verifier_transcript, 
    &gens
  );
  
  assert!(verification_result.is_ok(), "Proof verification failed!");
  
  println!("   ✓ Proof verification successful!");
  println!("\n{}", "=".repeat(80));
  println!("🎉 SUCCESS: Proved knowledge of x such that x³ + x + 5 = y");
  println!("   • The prover knows a secret x");
  println!("   • The verifier only sees the result y"); 
  println!("   • The proof is succinct and zero-knowledge");
  println!("   • No trusted setup was required");
  println!("{}", "=".repeat(80));

  println!("\n💡 Key Takeaways:");
  println!("   1. Complex computations can be proven efficiently using R1CS");
  println!("   2. SNARKs provide succinct proofs regardless of computation size");
  println!("   3. Zero-knowledge property hides all intermediate values");
  println!("   4. Verification is fast and independent of proof generation time");
}
