extern crate libspartan;
extern crate merlin;
use libspartan::{Instance, NIZKGens, NIZK};
use merlin::Transcript;
fn main() {
    // specify the size of an R1CS instance
    let num_vars = 1024;
    let num_cons = 1024;
    let num_inputs = 10;

    // produce public parameters
    let gens = NIZKGens::new(num_cons, num_vars, num_inputs);

    // ask the library to produce a synthentic R1CS instance
    let (inst, vars, inputs) = Instance::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);

    // produce a proof of satisfiability
    let mut prover_transcript = Transcript::new(b"nizk_example");
    let proof = NIZK::prove(&inst, vars, &inputs, &gens, &mut prover_transcript);

    // verify the proof of satisfiability
    let mut verifier_transcript = Transcript::new(b"nizk_example");
    assert!(proof
      .verify(&inst, &inputs, &mut verifier_transcript, &gens)
      .is_ok());
    println!("proof verification successful!");
}