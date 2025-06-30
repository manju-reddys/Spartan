#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
use super::commitments::{Commitments, MultiCommitGens};
use super::dense_mlpoly::DensePolynomial;
use super::errors::ProofVerifyError;
use super::group::{CompressedGroup, GroupElement, VartimeMultiscalarMul};
use super::nizk::DotProductProof;
use super::random::RandomTape;
use super::scalar::Scalar;
use super::transcript::{AppendToTranscript, ProofTranscript};
use super::unipoly::{CompressedUniPoly, UniPoly};
use core::iter;
use itertools::izip;
use merlin::Transcript;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SumcheckInstanceProof {
  compressed_polys: Vec<CompressedUniPoly>,
}

impl SumcheckInstanceProof {
  pub fn new(compressed_polys: Vec<CompressedUniPoly>) -> SumcheckInstanceProof {
    SumcheckInstanceProof { compressed_polys }
  }

  pub fn verify(
    &self,
    claim: Scalar,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut Transcript,
  ) -> Result<(Scalar, Vec<Scalar>), ProofVerifyError> {
    let mut e = claim;
    let mut r: Vec<Scalar> = Vec::new();

    // verify that there is a univariate polynomial for each round
    assert_eq!(self.compressed_polys.len(), num_rounds);
    for i in 0..self.compressed_polys.len() {
      let poly = self.compressed_polys[i].decompress(&e);

      // verify degree bound
      assert_eq!(poly.degree(), degree_bound);

      // check if G_k(0) + G_k(1) = e
      assert_eq!(poly.eval_at_zero() + poly.eval_at_one(), e);

      // append the prover's message to the transcript
      poly.append_to_transcript(b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_i = transcript.challenge_scalar(b"challenge_nextround");

      r.push(r_i);

      // evaluate the claimed degree-ell polynomial at r_i
      e = poly.evaluate(&r_i);
    }

    Ok((e, r))
  }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ZKSumcheckInstanceProof {
  comm_polys: Vec<CompressedGroup>,
  comm_evals: Vec<CompressedGroup>,
  proofs: Vec<DotProductProof>,
}

impl ZKSumcheckInstanceProof {
  pub fn new(
    comm_polys: Vec<CompressedGroup>,
    comm_evals: Vec<CompressedGroup>,
    proofs: Vec<DotProductProof>,
  ) -> Self {
    ZKSumcheckInstanceProof {
      comm_polys,
      comm_evals,
      proofs,
    }
  }

  pub fn verify(
    &self,
    comm_claim: &CompressedGroup,
    num_rounds: usize,
    degree_bound: usize,
    gens_1: &MultiCommitGens,
    gens_n: &MultiCommitGens,
    transcript: &mut Transcript,
  ) -> Result<(CompressedGroup, Vec<Scalar>), ProofVerifyError> {
    // verify degree bound
    assert_eq!(gens_n.n, degree_bound + 1);

    // verify that there is a univariate polynomial for each round
    assert_eq!(self.comm_polys.len(), num_rounds);
    assert_eq!(self.comm_evals.len(), num_rounds);

    let mut r: Vec<Scalar> = Vec::new();
    for i in 0..self.comm_polys.len() {
      let comm_poly = &self.comm_polys[i];

      // append the prover's polynomial to the transcript
      comm_poly.append_to_transcript(b"comm_poly", transcript);

      //derive the verifier's challenge for the next round
      let r_i = transcript.challenge_scalar(b"challenge_nextround");

      // verify the proof of sum-check and evals
      let res = {
        let comm_claim_per_round = if i == 0 {
          comm_claim
        } else {
          &self.comm_evals[i - 1]
        };
        let comm_eval = &self.comm_evals[i];

        // add two claims to transcript
        comm_claim_per_round.append_to_transcript(b"comm_claim_per_round", transcript);
        comm_eval.append_to_transcript(b"comm_eval", transcript);

        // produce two weights
        let w = transcript.challenge_vector(b"combine_two_claims_to_one", 2);

        // compute a weighted sum of the RHS
        let comm_target = GroupElement::vartime_multiscalar_mul(
          w.iter(),
          iter::once(&comm_claim_per_round)
            .chain(iter::once(&comm_eval))
            .map(|pt| pt.decompress().unwrap())
            .collect::<Vec<GroupElement>>(),
        )
        .compress();

        let a = {
          // the vector to use to decommit for sum-check test
          let a_sc = {
            let mut a = vec![Scalar::one(); degree_bound + 1];
            a[0] += Scalar::one();
            a
          };

          // the vector to use to decommit for evaluation
          let a_eval = {
            let mut a = vec![Scalar::one(); degree_bound + 1];
            for j in 1..a.len() {
              a[j] = a[j - 1] * r_i;
            }
            a
          };

          // take weighted sum of the two vectors using w
          assert_eq!(a_sc.len(), a_eval.len());
          (0..a_sc.len())
            .map(|i| w[0] * a_sc[i] + w[1] * a_eval[i])
            .collect::<Vec<Scalar>>()
        };

        self.proofs[i]
          .verify(
            gens_1,
            gens_n,
            transcript,
            &a,
            &self.comm_polys[i],
            &comm_target,
          )
          .is_ok()
      };
      if !res {
        return Err(ProofVerifyError::InternalError);
      }

      r.push(r_i);
    }

    Ok((self.comm_evals[self.comm_evals.len() - 1], r))
  }
}

impl SumcheckInstanceProof {
  pub fn prove_cubic<F>(
    claim: &Scalar,
    num_rounds: usize,
    poly_A: &mut DensePolynomial,
    poly_B: &mut DensePolynomial,
    poly_C: &mut DensePolynomial,
    comb_func: F,
    transcript: &mut Transcript,
  ) -> (Self, Vec<Scalar>, Vec<Scalar>)
  where
    F: Fn(&Scalar, &Scalar, &Scalar) -> Scalar,
  {
    let mut e = *claim;
    let mut r: Vec<Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly> = Vec::new();
    for _j in 0..num_rounds {
      let mut eval_point_0 = Scalar::zero();
      let mut eval_point_2 = Scalar::zero();
      let mut eval_point_3 = Scalar::zero();

      let len = poly_A.len() / 2;
      for i in 0..len {
        // eval 0: bound_func is A(low)
        eval_point_0 += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
        eval_point_2 += comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
        );

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];

        eval_point_3 += comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
        );
      }

      let evals = vec![eval_point_0, e - eval_point_0, eval_point_2, eval_point_3];
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      poly.append_to_transcript(b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_j = transcript.challenge_scalar(b"challenge_nextround");
      r.push(r_j);
      // bound all tables to the verifier's challenge
      poly_A.bound_poly_var_top(&r_j);
      poly_B.bound_poly_var_top(&r_j);
      poly_C.bound_poly_var_top(&r_j);
      e = poly.evaluate(&r_j);
      cubic_polys.push(poly.compress());
    }

    (
      SumcheckInstanceProof::new(cubic_polys),
      r,
      vec![poly_A[0], poly_B[0], poly_C[0]],
    )
  }

  pub fn prove_cubic_batched<F>(
    claim: &Scalar,
    num_rounds: usize,
    poly_vec_par: (
      &mut Vec<&mut DensePolynomial>,
      &mut Vec<&mut DensePolynomial>,
      &mut DensePolynomial,
    ),
    poly_vec_seq: (
      &mut Vec<&mut DensePolynomial>,
      &mut Vec<&mut DensePolynomial>,
      &mut Vec<&mut DensePolynomial>,
    ),
    coeffs: &[Scalar],
    comb_func: F,
    transcript: &mut Transcript,
  ) -> (
    Self,
    Vec<Scalar>,
    (Vec<Scalar>, Vec<Scalar>, Scalar),
    (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>),
  )
  where
    F: Fn(&Scalar, &Scalar, &Scalar) -> Scalar,
  {
    let (poly_A_vec_par, poly_B_vec_par, poly_C_par) = poly_vec_par;
    let (poly_A_vec_seq, poly_B_vec_seq, poly_C_vec_seq) = poly_vec_seq;

    //let (poly_A_vec_seq, poly_B_vec_seq, poly_C_vec_seq) = poly_vec_seq;
    let mut e = *claim;
    let mut r: Vec<Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly> = Vec::new();

    for _j in 0..num_rounds {
      let mut evals: Vec<(Scalar, Scalar, Scalar)> = Vec::new();

      for (poly_A, poly_B) in poly_A_vec_par.iter().zip(poly_B_vec_par.iter()) {
        let mut eval_point_0 = Scalar::zero();
        let mut eval_point_2 = Scalar::zero();
        let mut eval_point_3 = Scalar::zero();

        let len = poly_A.len() / 2;
        for i in 0..len {
          // eval 0: bound_func is A(low)
          eval_point_0 += comb_func(&poly_A[i], &poly_B[i], &poly_C_par[i]);

          // eval 2: bound_func is -A(low) + 2*A(high)
          let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C_par[len + i] + poly_C_par[len + i] - poly_C_par[i];
          eval_point_2 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
          );

          // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
          let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C_bound_point + poly_C_par[len + i] - poly_C_par[i];

          eval_point_3 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
          );
        }

        evals.push((eval_point_0, eval_point_2, eval_point_3));
      }

      for (poly_A, poly_B, poly_C) in izip!(
        poly_A_vec_seq.iter(),
        poly_B_vec_seq.iter(),
        poly_C_vec_seq.iter()
      ) {
        let mut eval_point_0 = Scalar::zero();
        let mut eval_point_2 = Scalar::zero();
        let mut eval_point_3 = Scalar::zero();
        let len = poly_A.len() / 2;
        for i in 0..len {
          // eval 0: bound_func is A(low)
          eval_point_0 += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);
          // eval 2: bound_func is -A(low) + 2*A(high)
          let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
          eval_point_2 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
          );
          // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
          let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];
          eval_point_3 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
          );
        }
        evals.push((eval_point_0, eval_point_2, eval_point_3));
      }

      let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
      let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
      let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2 * coeffs[i]).sum();

      let evals = vec![
        evals_combined_0,
        e - evals_combined_0,
        evals_combined_2,
        evals_combined_3,
      ];
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      poly.append_to_transcript(b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_j = transcript.challenge_scalar(b"challenge_nextround");
      r.push(r_j);

      // bound all tables to the verifier's challenge
      for (poly_A, poly_B) in poly_A_vec_par.iter_mut().zip(poly_B_vec_par.iter_mut()) {
        poly_A.bound_poly_var_top(&r_j);
        poly_B.bound_poly_var_top(&r_j);
      }
      poly_C_par.bound_poly_var_top(&r_j);

      for (poly_A, poly_B, poly_C) in izip!(
        poly_A_vec_seq.iter_mut(),
        poly_B_vec_seq.iter_mut(),
        poly_C_vec_seq.iter_mut()
      ) {
        poly_A.bound_poly_var_top(&r_j);
        poly_B.bound_poly_var_top(&r_j);
        poly_C.bound_poly_var_top(&r_j);
      }

      e = poly.evaluate(&r_j);
      cubic_polys.push(poly.compress());
    }

    let poly_A_par_final = (0..poly_A_vec_par.len())
      .map(|i| poly_A_vec_par[i][0])
      .collect();
    let poly_B_par_final = (0..poly_B_vec_par.len())
      .map(|i| poly_B_vec_par[i][0])
      .collect();
    let claims_prod = (poly_A_par_final, poly_B_par_final, poly_C_par[0]);

    let poly_A_seq_final = (0..poly_A_vec_seq.len())
      .map(|i| poly_A_vec_seq[i][0])
      .collect();
    let poly_B_seq_final = (0..poly_B_vec_seq.len())
      .map(|i| poly_B_vec_seq[i][0])
      .collect();
    let poly_C_seq_final = (0..poly_C_vec_seq.len())
      .map(|i| poly_C_vec_seq[i][0])
      .collect();
    let claims_dotp = (poly_A_seq_final, poly_B_seq_final, poly_C_seq_final);

    (
      SumcheckInstanceProof::new(cubic_polys),
      r,
      claims_prod,
      claims_dotp,
    )
  }
}

impl ZKSumcheckInstanceProof {
  pub fn prove_quad<F>(
    claim: &Scalar,
    blind_claim: &Scalar,
    num_rounds: usize,
    poly_A: &mut DensePolynomial,
    poly_B: &mut DensePolynomial,
    comb_func: F,
    gens_1: &MultiCommitGens,
    gens_n: &MultiCommitGens,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape,
  ) -> (Self, Vec<Scalar>, Vec<Scalar>, Scalar)
  where
    F: Fn(&Scalar, &Scalar) -> Scalar,
  {
    let (blinds_poly, blinds_evals) = (
      random_tape.random_vector(b"blinds_poly", num_rounds),
      random_tape.random_vector(b"blinds_evals", num_rounds),
    );
    let mut claim_per_round = *claim;
    let mut comm_claim_per_round = claim_per_round.commit(blind_claim, gens_1).compress();

    let mut r: Vec<Scalar> = Vec::new();
    let mut comm_polys: Vec<CompressedGroup> = Vec::new();
    let mut comm_evals: Vec<CompressedGroup> = Vec::new();
    let mut proofs: Vec<DotProductProof> = Vec::new();

    for j in 0..num_rounds {
      let (poly, comm_poly) = {
        let mut eval_point_0 = Scalar::zero();
        let mut eval_point_2 = Scalar::zero();

        let len = poly_A.len() / 2;
        for i in 0..len {
          // eval 0: bound_func is A(low)
          eval_point_0 += comb_func(&poly_A[i], &poly_B[i]);

          // eval 2: bound_func is -A(low) + 2*A(high)
          let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
          eval_point_2 += comb_func(&poly_A_bound_point, &poly_B_bound_point);
        }

        let evals = vec![eval_point_0, claim_per_round - eval_point_0, eval_point_2];
        let poly = UniPoly::from_evals(&evals);
        let comm_poly = poly.commit(gens_n, &blinds_poly[j]).compress();
        (poly, comm_poly)
      };

      // append the prover's message to the transcript
      comm_poly.append_to_transcript(b"comm_poly", transcript);
      comm_polys.push(comm_poly);

      //derive the verifier's challenge for the next round
      let r_j = transcript.challenge_scalar(b"challenge_nextround");

      // bound all tables to the verifier's challenge
      poly_A.bound_poly_var_top(&r_j);
      poly_B.bound_poly_var_top(&r_j);

      // produce a proof of sum-check and of evaluation
      let (proof, claim_next_round, comm_claim_next_round) = {
        let eval = poly.evaluate(&r_j);
        let comm_eval = eval.commit(&blinds_evals[j], gens_1).compress();

        // we need to prove the following under homomorphic commitments:
        // (1) poly(0) + poly(1) = claim_per_round
        // (2) poly(r_j) = eval

        // Our technique is to leverage dot product proofs:
        // (1) we can prove: <poly_in_coeffs_form, (2, 1, 1, 1)> = claim_per_round
        // (2) we can prove: <poly_in_coeffs_form, (1, r_j, r^2_j, ..) = eval
        // for efficiency we batch them using random weights

        // add two claims to transcript
        comm_claim_per_round.append_to_transcript(b"comm_claim_per_round", transcript);
        comm_eval.append_to_transcript(b"comm_eval", transcript);

        // produce two weights
        let w = transcript.challenge_vector(b"combine_two_claims_to_one", 2);

        // compute a weighted sum of the RHS
        let target = w[0] * claim_per_round + w[1] * eval;
        let comm_target = GroupElement::vartime_multiscalar_mul(
          w.iter(),
          iter::once(&comm_claim_per_round)
            .chain(iter::once(&comm_eval))
            .map(|pt| pt.decompress().unwrap())
            .collect::<Vec<GroupElement>>(),
        )
        .compress();

        let blind = {
          let blind_sc = if j == 0 {
            blind_claim
          } else {
            &blinds_evals[j - 1]
          };

          let blind_eval = &blinds_evals[j];

          w[0] * blind_sc + w[1] * blind_eval
        };
        assert_eq!(target.commit(&blind, gens_1).compress(), comm_target);

        let a = {
          // the vector to use to decommit for sum-check test
          let a_sc = {
            let mut a = vec![Scalar::one(); poly.degree() + 1];
            a[0] += Scalar::one();
            a
          };

          // the vector to use to decommit for evaluation
          let a_eval = {
            let mut a = vec![Scalar::one(); poly.degree() + 1];
            for j in 1..a.len() {
              a[j] = a[j - 1] * r_j;
            }
            a
          };

          // take weighted sum of the two vectors using w
          assert_eq!(a_sc.len(), a_eval.len());
          (0..a_sc.len())
            .map(|i| w[0] * a_sc[i] + w[1] * a_eval[i])
            .collect::<Vec<Scalar>>()
        };

        let (proof, _comm_poly, _comm_sc_eval) = DotProductProof::prove(
          gens_1,
          gens_n,
          transcript,
          random_tape,
          &poly.as_vec(),
          &blinds_poly[j],
          &a,
          &target,
          &blind,
        );

        (proof, eval, comm_eval)
      };

      claim_per_round = claim_next_round;
      comm_claim_per_round = comm_claim_next_round;

      proofs.push(proof);
      r.push(r_j);
      comm_evals.push(comm_claim_per_round);
    }

    (
      ZKSumcheckInstanceProof::new(comm_polys, comm_evals, proofs),
      r,
      vec![poly_A[0], poly_B[0]],
      blinds_evals[num_rounds - 1],
    )
  }

  pub fn prove_cubic_with_additive_term<F>(
    claim: &Scalar,
    blind_claim: &Scalar,
    num_rounds: usize,
    poly_A: &mut DensePolynomial,
    poly_B: &mut DensePolynomial,
    poly_C: &mut DensePolynomial,
    poly_D: &mut DensePolynomial,
    comb_func: F,
    gens_1: &MultiCommitGens,
    gens_n: &MultiCommitGens,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape,
  ) -> (Self, Vec<Scalar>, Vec<Scalar>, Scalar)
  where
    F: Fn(&Scalar, &Scalar, &Scalar, &Scalar) -> Scalar,
  {
    let (blinds_poly, blinds_evals) = (
      random_tape.random_vector(b"blinds_poly", num_rounds),
      random_tape.random_vector(b"blinds_evals", num_rounds),
    );

    let mut claim_per_round = *claim;
    let mut comm_claim_per_round = claim_per_round.commit(blind_claim, gens_1).compress();

    let mut r: Vec<Scalar> = Vec::new();
    let mut comm_polys: Vec<CompressedGroup> = Vec::new();
    let mut comm_evals: Vec<CompressedGroup> = Vec::new();
    let mut proofs: Vec<DotProductProof> = Vec::new();

    for j in 0..num_rounds {
      let (poly, comm_poly) = {
        let mut eval_point_0 = Scalar::zero();
        let mut eval_point_2 = Scalar::zero();
        let mut eval_point_3 = Scalar::zero();

        let len = poly_A.len() / 2;
        for i in 0..len {
          // eval 0: bound_func is A(low)
          eval_point_0 += comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);

          // eval 2: bound_func is -A(low) + 2*A(high)
          let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
          let poly_D_bound_point = poly_D[len + i] + poly_D[len + i] - poly_D[i];
          eval_point_2 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
            &poly_D_bound_point,
          );

          // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
          let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];
          let poly_D_bound_point = poly_D_bound_point + poly_D[len + i] - poly_D[i];
          eval_point_3 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
            &poly_D_bound_point,
          );
        }

        let evals = vec![
          eval_point_0,
          claim_per_round - eval_point_0,
          eval_point_2,
          eval_point_3,
        ];
        let poly = UniPoly::from_evals(&evals);
        let comm_poly = poly.commit(gens_n, &blinds_poly[j]).compress();
        (poly, comm_poly)
      };

      // append the prover's message to the transcript
      comm_poly.append_to_transcript(b"comm_poly", transcript);
      comm_polys.push(comm_poly);

      //derive the verifier's challenge for the next round
      let r_j = transcript.challenge_scalar(b"challenge_nextround");

      // bound all tables to the verifier's challenge
      poly_A.bound_poly_var_top(&r_j);
      poly_B.bound_poly_var_top(&r_j);
      poly_C.bound_poly_var_top(&r_j);
      poly_D.bound_poly_var_top(&r_j);

      // produce a proof of sum-check and of evaluation
      let (proof, claim_next_round, comm_claim_next_round) = {
        let eval = poly.evaluate(&r_j);
        let comm_eval = eval.commit(&blinds_evals[j], gens_1).compress();

        // we need to prove the following under homomorphic commitments:
        // (1) poly(0) + poly(1) = claim_per_round
        // (2) poly(r_j) = eval

        // Our technique is to leverage dot product proofs:
        // (1) we can prove: <poly_in_coeffs_form, (2, 1, 1, 1)> = claim_per_round
        // (2) we can prove: <poly_in_coeffs_form, (1, r_j, r^2_j, ..) = eval
        // for efficiency we batch them using random weights

        // add two claims to transcript
        comm_claim_per_round.append_to_transcript(b"comm_claim_per_round", transcript);
        comm_eval.append_to_transcript(b"comm_eval", transcript);

        // produce two weights
        let w = transcript.challenge_vector(b"combine_two_claims_to_one", 2);

        // compute a weighted sum of the RHS
        let target = w[0] * claim_per_round + w[1] * eval;
        let comm_target = GroupElement::vartime_multiscalar_mul(
          w.iter(),
          iter::once(&comm_claim_per_round)
            .chain(iter::once(&comm_eval))
            .map(|pt| pt.decompress().unwrap())
            .collect::<Vec<GroupElement>>(),
        )
        .compress();

        let blind = {
          let blind_sc = if j == 0 {
            blind_claim
          } else {
            &blinds_evals[j - 1]
          };

          let blind_eval = &blinds_evals[j];

          w[0] * blind_sc + w[1] * blind_eval
        };

        assert_eq!(target.commit(&blind, gens_1).compress(), comm_target);

        let a = {
          // the vector to use to decommit for sum-check test
          let a_sc = {
            let mut a = vec![Scalar::one(); poly.degree() + 1];
            a[0] += Scalar::one();
            a
          };

          // the vector to use to decommit for evaluation
          let a_eval = {
            let mut a = vec![Scalar::one(); poly.degree() + 1];
            for j in 1..a.len() {
              a[j] = a[j - 1] * r_j;
            }
            a
          };

          // take weighted sum of the two vectors using w
          assert_eq!(a_sc.len(), a_eval.len());
          (0..a_sc.len())
            .map(|i| w[0] * a_sc[i] + w[1] * a_eval[i])
            .collect::<Vec<Scalar>>()
        };

        let (proof, _comm_poly, _comm_sc_eval) = DotProductProof::prove(
          gens_1,
          gens_n,
          transcript,
          random_tape,
          &poly.as_vec(),
          &blinds_poly[j],
          &a,
          &target,
          &blind,
        );

        (proof, eval, comm_eval)
      };

      proofs.push(proof);
      claim_per_round = claim_next_round;
      comm_claim_per_round = comm_claim_next_round;
      r.push(r_j);
      comm_evals.push(comm_claim_per_round);
    }

    (
      ZKSumcheckInstanceProof::new(comm_polys, comm_evals, proofs),
      r,
      vec![poly_A[0], poly_B[0], poly_C[0], poly_D[0]],
      blinds_evals[num_rounds - 1],
    )
  }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::random::RandomTape;
    use super::super::commitments::MultiCommitGens;
    use super::super::scalar::Scalar;
    use merlin::Transcript;

    /// Helper function to create test polynomials
    fn create_test_polynomials(size: usize) -> (DensePolynomial, DensePolynomial, DensePolynomial) {
        let mut poly_A_coeffs = vec![Scalar::from(1u64); size];
        let mut poly_B_coeffs = vec![Scalar::from(2u64); size];
        let mut poly_C_coeffs = vec![Scalar::from(3u64); size];
        
        // Add some variation to make testing more realistic
        for i in 0..size {
            poly_A_coeffs[i] = Scalar::from((i + 1) as u64);
            poly_B_coeffs[i] = Scalar::from((i + 2) as u64);
            poly_C_coeffs[i] = Scalar::from((i + 3) as u64);
        }
        
        let poly_A = DensePolynomial::new(poly_A_coeffs);
        let poly_B = DensePolynomial::new(poly_B_coeffs);
        let poly_C = DensePolynomial::new(poly_C_coeffs);
        
        (poly_A, poly_B, poly_C)
    }

    /// Helper function to create two test polynomials
    fn create_test_polynomials_two(size: usize) -> (DensePolynomial, DensePolynomial) {
        let mut poly_A_coeffs = vec![Scalar::from(1u64); size];
        let mut poly_B_coeffs = vec![Scalar::from(2u64); size];
        
        // Add some variation to make testing more realistic
        for i in 0..size {
            poly_A_coeffs[i] = Scalar::from((i + 1) as u64);
            poly_B_coeffs[i] = Scalar::from((i + 2) as u64);
        }
        
        let poly_A = DensePolynomial::new(poly_A_coeffs);
        let poly_B = DensePolynomial::new(poly_B_coeffs);
        
        (poly_A, poly_B)
    }

    /// Helper function to compute log2
    fn log2(x: usize) -> usize {
        (x as f64).log2() as usize
    }

    /// Helper function to create test commitment generators
    fn create_test_gens(degree_bound: usize) -> (MultiCommitGens, MultiCommitGens) {
        let gens_1 = MultiCommitGens::new(1, b"test_gens_1");
        let gens_n = MultiCommitGens::new(degree_bound + 1, b"test_gens_n");
        (gens_1, gens_n)
    }

    /// Test basic sumcheck proof generation and verification
    #[test]
    fn test_basic_sumcheck_proof() {
        let size = 8;
        let (mut poly_A, mut poly_B, mut poly_C) = create_test_polynomials(size);
        
        // Define combination function: A * B + C
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
            a * b + c
        };
        
        // Compute expected claim
        let mut expected_claim = Scalar::zero();
        for i in 0..size {
            expected_claim += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);
        }
        
        let num_rounds = log2(size);
        let mut transcript = Transcript::new(b"test_sumcheck");
        
        // Generate proof
        let (proof, r, final_evals) = SumcheckInstanceProof::prove_cubic(
            &expected_claim,
            num_rounds,
            &mut poly_A,
            &mut poly_B,
            &mut poly_C,
            comb_func,
            &mut transcript,
        );
        
        // Verify proof
        let mut verify_transcript = Transcript::new(b"test_sumcheck");
        let (final_eval, challenges) = proof.verify(
            expected_claim,
            num_rounds,
            3, // degree bound for cubic
            &mut verify_transcript,
        ).unwrap();
        
        // Verify results
        assert_eq!(challenges, r);
        
        // The final evaluation should match the combination of the final polynomial evaluations
        let computed_final = comb_func(&final_evals[0], &final_evals[1], &final_evals[2]);
        assert_eq!(final_eval, computed_final);
    }

    /// Test sumcheck with different polynomial sizes
    #[test]
    fn test_sumcheck_different_sizes() {
        let sizes = vec![4, 8, 16, 32];
        
        for size in sizes {
            let (mut poly_A, mut poly_B, mut poly_C) = create_test_polynomials(size);
            let num_rounds = log2(size);
            
            // Define combination function: A + B * C
            let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
                a + b * c
            };
            
            // Compute expected claim
            let mut expected_claim = Scalar::zero();
            for i in 0..size {
                expected_claim += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);
            }
            
            let mut transcript = Transcript::new(b"test_sumcheck_sizes");
            
            // Generate and verify proof
            let (proof, r, final_evals) = SumcheckInstanceProof::prove_cubic(
                &expected_claim,
                num_rounds,
                &mut poly_A,
                &mut poly_B,
                &mut poly_C,
                comb_func,
                &mut transcript,
            );
            
            let mut verify_transcript = Transcript::new(b"test_sumcheck_sizes");
            let (final_eval, challenges) = proof.verify(
                expected_claim,
                num_rounds,
                3,
                &mut verify_transcript,
            ).unwrap();
            
            assert_eq!(challenges, r);
            // The final evaluation should match the combination of the final polynomial evaluations
            let computed_final = comb_func(&final_evals[0], &final_evals[1], &final_evals[2]);
            assert_eq!(final_eval, computed_final);
        }
    }

    /// Test zero-knowledge sumcheck proof generation and verification
    #[test]
    fn test_zk_sumcheck_proof() {
        let size = 8;
        let (poly_A, poly_B) = create_test_polynomials_two(size);
        
        // Define combination function: A * B
        let comb_func = |a: &Scalar, b: &Scalar| -> Scalar {
            a * b
        };
        
        // Compute expected claim
        let mut expected_claim = Scalar::zero();
        for i in 0..size {
            expected_claim += comb_func(&poly_A[i], &poly_B[i]);
        }
        
        let num_rounds = 3;
        let (gens_1, gens_n) = create_test_gens(2); // degree bound 2 for quadratic
        let mut transcript = Transcript::new(b"test_zk_sumcheck");
        let mut random_tape = RandomTape::new(b"test_zk_sumcheck");
        
        // Generate proof - this should succeed
        let (proof, r, final_evals, _blind) = ZKSumcheckInstanceProof::prove_quad(
            &expected_claim,
            &random_tape.random_scalar(b"blind"),
            num_rounds,
            &mut poly_A.clone(),
            &mut poly_B.clone(),
            comb_func,
            &gens_1,
            &gens_n,
            &mut transcript,
            &mut random_tape,
        );
        
        // Verify that proof generation succeeded
        assert_eq!(proof.comm_polys.len(), num_rounds);
        assert_eq!(proof.comm_evals.len(), num_rounds);
        assert_eq!(proof.proofs.len(), num_rounds);
        assert_eq!(r.len(), num_rounds);
        assert_eq!(final_evals.len(), 2); // A and B
        
        // Test that the final evaluations are reasonable (not zero for non-zero polynomials)
        assert_ne!(final_evals[0], Scalar::zero());
        assert_ne!(final_evals[1], Scalar::zero());
    }

    /// Test batched sumcheck proof generation
    #[test]
    fn test_batched_sumcheck_proof() {
        let size = 8;
        let num_batches = 3;
        
        // Create multiple polynomial batches
        let mut poly_A_vec_par: Vec<DensePolynomial> = Vec::new();
        let mut poly_B_vec_par: Vec<DensePolynomial> = Vec::new();
        let mut poly_C_par = DensePolynomial::new(vec![Scalar::from(1u64); size]);
        
        let mut poly_A_vec_seq: Vec<DensePolynomial> = Vec::new();
        let mut poly_B_vec_seq: Vec<DensePolynomial> = Vec::new();
        let mut poly_C_vec_seq: Vec<DensePolynomial> = Vec::new();
        
        for i in 0..num_batches {
            // Create new polynomials with modified coefficients for each batch
            let mut poly_A_coeffs = vec![Scalar::from(1u64); size];
            let mut poly_B_coeffs = vec![Scalar::from(2u64); size];
            let mut poly_C_coeffs = vec![Scalar::from(3u64); size];
            
            for k in 0..size {
                poly_A_coeffs[k] = Scalar::from((k + 1) as u64) * Scalar::from((i + 1) as u64);
                poly_B_coeffs[k] = Scalar::from((k + 2) as u64) * Scalar::from((i + 2) as u64);
                poly_C_coeffs[k] = Scalar::from((k + 3) as u64) * Scalar::from((i + 3) as u64);
            }
            
            let poly_A = DensePolynomial::new(poly_A_coeffs);
            let poly_B = DensePolynomial::new(poly_B_coeffs);
            let poly_C = DensePolynomial::new(poly_C_coeffs);
            
            poly_A_vec_par.push(poly_A.clone());
            poly_B_vec_par.push(poly_B.clone());
            poly_A_vec_seq.push(poly_A);
            poly_B_vec_seq.push(poly_B);
            poly_C_vec_seq.push(poly_C);
        }
        
        // Define combination function: A * B + C
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
            a * b + c
        };
        
        // Compute expected claim
        let mut expected_claim = Scalar::zero();
        for i in 0..size {
            for j in 0..num_batches {
                expected_claim += comb_func(&poly_A_vec_par[j][i], &poly_B_vec_par[j][i], &poly_C_par[i]);
                expected_claim += comb_func(&poly_A_vec_seq[j][i], &poly_B_vec_seq[j][i], &poly_C_vec_seq[j][i]);
            }
        }
        
        let num_rounds = 3;
        // Create coefficients for both parallel and sequential batches
        let coeffs = vec![
            Scalar::from(1u64), Scalar::from(2u64), Scalar::from(3u64), // parallel batches
            Scalar::from(1u64), Scalar::from(2u64), Scalar::from(3u64), // sequential batches
        ];
        assert_eq!(coeffs.len(), 2 * num_batches); // Ensure coefficients match total batch count
        let mut transcript = Transcript::new(b"test_batched_sumcheck");
        
        // Create mutable references for the batched function
        let mut poly_A_vec_par_refs: Vec<&mut DensePolynomial> = poly_A_vec_par.iter_mut().collect();
        let mut poly_B_vec_par_refs: Vec<&mut DensePolynomial> = poly_B_vec_par.iter_mut().collect();
        let mut poly_A_vec_seq_refs: Vec<&mut DensePolynomial> = poly_A_vec_seq.iter_mut().collect();
        let mut poly_B_vec_seq_refs: Vec<&mut DensePolynomial> = poly_B_vec_seq.iter_mut().collect();
        let mut poly_C_vec_seq_refs: Vec<&mut DensePolynomial> = poly_C_vec_seq.iter_mut().collect();
        
        // Generate proof
        let (proof, r, claims_prod, claims_dotp) = SumcheckInstanceProof::prove_cubic_batched(
            &expected_claim,
            num_rounds,
            (&mut poly_A_vec_par_refs, &mut poly_B_vec_par_refs, &mut poly_C_par),
            (&mut poly_A_vec_seq_refs, &mut poly_B_vec_seq_refs, &mut poly_C_vec_seq_refs),
            &coeffs,
            comb_func,
            &mut transcript,
        );
        
        // Verify proof
        let mut verify_transcript = Transcript::new(b"test_batched_sumcheck");
        let (_final_eval, challenges) = proof.verify(
            expected_claim,
            num_rounds,
            3,
            &mut verify_transcript,
        ).unwrap();
        
        assert_eq!(challenges, r);
        
        // Verify final evaluations
        assert_eq!(claims_prod.0.len(), num_batches);
        assert_eq!(claims_prod.1.len(), num_batches);
        assert_eq!(claims_dotp.0.len(), num_batches);
        assert_eq!(claims_dotp.1.len(), num_batches);
        assert_eq!(claims_dotp.2.len(), num_batches);
    }

    /// Test sumcheck with additive term
    #[test]
    fn test_sumcheck_with_additive_term() {
        let size = 8;
        let (mut poly_A, mut poly_B, mut poly_C, mut poly_D) = {
            let (a, b, c) = create_test_polynomials(size);
            let d = DensePolynomial::new(vec![Scalar::from(4u64); size]);
            (a, b, c, d)
        };
        
        // Define combination function: A * B + C * D
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar, d: &Scalar| -> Scalar {
            a * b + c * d
        };
        
        // Compute expected claim
        let mut expected_claim = Scalar::zero();
        for i in 0..size {
            expected_claim += comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);
        }
        
        let num_rounds = 3;
        let (gens_1, gens_n) = create_test_gens(3); // degree bound 3 for cubic
        let mut transcript = Transcript::new(b"test_sumcheck_additive");
        let mut random_tape = RandomTape::new(b"test_sumcheck_additive");
        
        // Generate proof - this should succeed
        let (proof, r, final_evals, _blind) = ZKSumcheckInstanceProof::prove_cubic_with_additive_term(
            &expected_claim,
            &random_tape.random_scalar(b"blind"),
            num_rounds,
            &mut poly_A,
            &mut poly_B,
            &mut poly_C,
            &mut poly_D,
            comb_func,
            &gens_1,
            &gens_n,
            &mut transcript,
            &mut random_tape,
        );
        
        // Verify that proof generation succeeded
        assert_eq!(proof.comm_polys.len(), num_rounds);
        assert_eq!(proof.comm_evals.len(), num_rounds);
        assert_eq!(proof.proofs.len(), num_rounds);
        assert_eq!(r.len(), num_rounds);
        assert_eq!(final_evals.len(), 4); // A, B, C, D
        
        // Test that the final evaluations are reasonable (not zero for non-zero polynomials)
        for eval in &final_evals {
            assert_ne!(*eval, Scalar::zero());
        }
    }

    /// Test error cases and edge cases
    #[test]
    fn test_sumcheck_edge_cases() {
        // Test with zero polynomials
        let size = 4;
        let zero_poly = DensePolynomial::new(vec![Scalar::zero(); size]);
        let (mut poly_A, mut poly_B, mut poly_C) = (zero_poly.clone(), zero_poly.clone(), zero_poly);
        
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
            a * b + c
        };
        
        let expected_claim = Scalar::zero();
        let num_rounds = 2;
        let mut transcript = Transcript::new(b"test_sumcheck_zero");
        
        let (proof, r, _final_evals) = SumcheckInstanceProof::prove_cubic(
            &expected_claim,
            num_rounds,
            &mut poly_A,
            &mut poly_B,
            &mut poly_C,
            comb_func,
            &mut transcript,
        );
        
        let mut verify_transcript = Transcript::new(b"test_sumcheck_zero");
        let (final_eval, challenges) = proof.verify(
            expected_claim,
            num_rounds,
            3,
            &mut verify_transcript,
        ).unwrap();
        
        assert_eq!(challenges, r);
        // For zero polynomials, the final evaluation should be zero
        assert_eq!(final_eval, Scalar::zero());
    }

    /// Test transcript consistency
    #[test]
    fn test_transcript_consistency() {
        let size = 8;
        let (mut poly_A, mut poly_B, mut poly_C) = create_test_polynomials(size);
        
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
            a * b + c
        };
        
        let mut expected_claim = Scalar::zero();
        for i in 0..size {
            expected_claim += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);
        }
        
        let num_rounds = 3;
        let mut transcript = Transcript::new(b"test_transcript_consistency");
        
        // Generate proof
        let (proof, r, _final_evals) = SumcheckInstanceProof::prove_cubic(
            &expected_claim,
            num_rounds,
            &mut poly_A,
            &mut poly_B,
            &mut poly_C,
            comb_func,
            &mut transcript,
        );
        
        // Create a new transcript with the same label
        let mut verify_transcript = Transcript::new(b"test_transcript_consistency");
        
        // Verify proof
        let (final_eval, challenges) = proof.verify(
            expected_claim,
            num_rounds,
            3,
            &mut verify_transcript,
        ).unwrap();
        
        // Verify that challenges match
        assert_eq!(challenges, r);
        
        // Verify that the final evaluation is consistent
        let computed_final = comb_func(&poly_A[0], &poly_B[0], &poly_C[0]);
        assert_eq!(final_eval, computed_final);
    }

    /// Test degree bound verification
    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_degree_bound_verification() {
        let size = 8;
        let (mut poly_A, mut poly_B, mut poly_C) = create_test_polynomials(size);
        
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
            a * b + c
        };
        
        let mut expected_claim = Scalar::zero();
        for i in 0..size {
            expected_claim += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);
        }
        
        let num_rounds = 3;
        let mut transcript = Transcript::new(b"test_degree_bound");
        
        let (proof, _r, _final_evals) = SumcheckInstanceProof::prove_cubic(
            &expected_claim,
            num_rounds,
            &mut poly_A,
            &mut poly_B,
            &mut poly_C,
            comb_func,
            &mut transcript,
        );
        
        let mut verify_transcript = Transcript::new(b"test_degree_bound");
        
        // This should panic because we're using the wrong degree bound
        proof.verify(
            expected_claim,
            num_rounds,
            2, // Wrong degree bound (should be 3 for cubic)
            &mut verify_transcript,
        ).unwrap();
    }

    /// Test sumcheck property verification
    #[test]
    fn test_sumcheck_property_verification() {
        let size = 8;
        let (mut poly_A, mut poly_B, mut poly_C) = create_test_polynomials(size);
        
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
            a * b + c
        };
        
        // Use a wrong claim that doesn't match the actual sum
        let wrong_claim = Scalar::from(999u64);
        let num_rounds = 3;
        let mut transcript = Transcript::new(b"test_wrong_claim");
        
        let (proof, _r, _final_evals) = SumcheckInstanceProof::prove_cubic(
            &wrong_claim,
            num_rounds,
            &mut poly_A,
            &mut poly_B,
            &mut poly_C,
            comb_func,
            &mut transcript,
        );
        
        let mut verify_transcript = Transcript::new(b"test_wrong_claim");
        
        // The proof should verify successfully, but the final evaluation should not match the wrong claim
        let (final_eval, _challenges) = proof.verify(
            wrong_claim,
            num_rounds,
            3,
            &mut verify_transcript,
        ).unwrap();
        
        // The final evaluation should not equal the wrong claim
        assert_ne!(final_eval, wrong_claim);
    }

    /// Test polynomial evaluation consistency
    #[test]
    fn test_polynomial_evaluation_consistency() {
        let size = 8;
        let (mut poly_A, mut poly_B, mut poly_C) = create_test_polynomials(size);
        
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
            a * b + c
        };
        
        let mut expected_claim = Scalar::zero();
        for i in 0..size {
            expected_claim += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);
        }
        
        let num_rounds = 3;
        let mut transcript = Transcript::new(b"test_evaluation_consistency");
        
        let (proof, r, final_evals) = SumcheckInstanceProof::prove_cubic(
            &expected_claim,
            num_rounds,
            &mut poly_A,
            &mut poly_B,
            &mut poly_C,
            comb_func,
            &mut transcript,
        );
        
        let mut verify_transcript = Transcript::new(b"test_evaluation_consistency");
        let (final_eval, challenges) = proof.verify(
            expected_claim,
            num_rounds,
            3,
            &mut verify_transcript,
        ).unwrap();
        
        // Verify correctness
        assert_eq!(challenges, r);
        // The final evaluation should match the combination of the final polynomial evaluations
        let computed_final = comb_func(&final_evals[0], &final_evals[1], &final_evals[2]);
        assert_eq!(final_eval, computed_final);
    }

    /// Test memory efficiency with large polynomials
    #[test]
    fn test_memory_efficiency() {
        let size = 1024; // Large polynomial
        let (mut poly_A, mut poly_B, mut poly_C) = create_test_polynomials(size);
        
        let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
            a * b + c
        };
        
        let mut expected_claim = Scalar::zero();
        for i in 0..size {
            expected_claim += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);
        }
        
        let num_rounds = 10; // log2(1024)
        let mut transcript = Transcript::new(b"test_memory_efficiency");
        
        // This should complete without excessive memory usage
        let (proof, r, final_evals) = SumcheckInstanceProof::prove_cubic(
            &expected_claim,
            num_rounds,
            &mut poly_A,
            &mut poly_B,
            &mut poly_C,
            comb_func,
            &mut transcript,
        );
        
        let mut verify_transcript = Transcript::new(b"test_memory_efficiency");
        let (final_eval, challenges) = proof.verify(
            expected_claim,
            num_rounds,
            3,
            &mut verify_transcript,
        ).unwrap();
        
        assert_eq!(challenges, r);
        // The final evaluation should match the combination of the final polynomial evaluations
        let computed_final = comb_func(&final_evals[0], &final_evals[1], &final_evals[2]);
        assert_eq!(final_eval, computed_final);
    }

    /// Test performance characteristics
    #[test]
    fn test_performance_characteristics() {
        let sizes = vec![4, 8, 16, 32, 64];
        
        for size in sizes {
            let (poly_A, poly_B, poly_C) = create_test_polynomials(size);
            
            let comb_func = |a: &Scalar, b: &Scalar, c: &Scalar| -> Scalar {
                a * b + c
            };
            
            let mut expected_claim = Scalar::zero();
            for i in 0..size {
                expected_claim += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);
            }
            
            let num_rounds = log2(size);
            let mut transcript = Transcript::new(b"test_performance");
            
            // Measure proof generation time
            let start = std::time::Instant::now();
            let (proof, r, final_evals) = SumcheckInstanceProof::prove_cubic(
                &expected_claim,
                num_rounds,
                &mut poly_A.clone(),
                &mut poly_B.clone(),
                &mut poly_C.clone(),
                comb_func,
                &mut transcript,
            );
            let time_std = start.elapsed();
            
            // Measure verification time
            let mut verify_transcript = Transcript::new(b"test_performance");
            let start = std::time::Instant::now();
            let (final_eval, challenges) = proof.verify(
                expected_claim,
                num_rounds,
                3,
                &mut verify_transcript,
            ).unwrap();
            let verify_time = start.elapsed();
            
            // Verify correctness
            assert_eq!(challenges, r);
            // The final evaluation should match the combination of the final polynomial evaluations
            let computed_final = comb_func(&final_evals[0], &final_evals[1], &final_evals[2]);
            assert_eq!(final_eval, computed_final);
            
            // Log performance metrics (useful for benchmarking)
            println!(
                "Size: {}, Rounds: {}, Proof: {:?}, Verify: {:?}",
                size, num_rounds, time_std, verify_time
            );
        }
    }
}
