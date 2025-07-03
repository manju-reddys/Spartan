use super::group::{GroupElement, VartimeMultiscalarMul, GROUP_BASEPOINT_COMPRESSED};
use super::mobile::create_zeros_vector;
use super::scalar::Scalar;
use digest::{ExtendableOutput, Update, XofReader};
use serde::{Deserialize, Serialize};
use sha3::Shake256;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct MultiCommitGens {
  pub n: usize,
  pub G: Vec<GroupElement>,
  pub h: GroupElement,
  // Internal lazy computation state (not serialized)
  #[serde(skip)]
  generator_cache: Option<Arc<Mutex<HashMap<usize, GroupElement>>>>,
  #[serde(skip)]
  label_hash: Option<Vec<u8>>,
}

impl MultiCommitGens {
  pub fn new(n: usize, label: &[u8]) -> Self {
    // For mobile optimization: use lazy computation for large generator sets
    #[cfg(feature = "mobile")]
    if n > 1024 { // Threshold for lazy computation on mobile
      return Self::new_lazy(n, label);
    }
    
    let mut shake = Shake256::default();
    shake.update(label);
    shake.update(GROUP_BASEPOINT_COMPRESSED.as_bytes());

    let mut reader = shake.finalize_xof();
    let mut gens: Vec<GroupElement> = Vec::new();
    let mut uniform_bytes = [0u8; 64];
    for _ in 0..n + 1 {
      reader.read(&mut uniform_bytes);
      gens.push(GroupElement::from_uniform_bytes(&uniform_bytes));
    }

    MultiCommitGens {
      n,
      G: gens[..n].to_vec(),
      h: gens[n],
      generator_cache: None,
      label_hash: None,
    }
  }

  // Lazy generator computation for mobile devices
  #[cfg(feature = "mobile")]
  fn new_lazy(n: usize, label: &[u8]) -> Self {
    // Compute only the h generator immediately, defer G generators
    let mut shake = Shake256::default();
    shake.update(label);
    shake.update(GROUP_BASEPOINT_COMPRESSED.as_bytes());
    let mut reader = shake.finalize_xof();
    
    // Skip n generators and compute h
    let mut uniform_bytes = [0u8; 64];
    for _ in 0..n {
      reader.read(&mut uniform_bytes); // Skip G generators
    }
    reader.read(&mut uniform_bytes);
    let h = GroupElement::from_uniform_bytes(&uniform_bytes);
    
    // Store label hash for lazy computation
    let mut label_vec = label.to_vec();
    label_vec.extend_from_slice(GROUP_BASEPOINT_COMPRESSED.as_bytes());
    
    MultiCommitGens {
      n,
      G: Vec::with_capacity(0), // Empty - will be computed lazily
      h,
      generator_cache: Some(Arc::new(Mutex::new(HashMap::new()))),
      label_hash: Some(label_vec),
    }
  }
  
  // Get generator at index i, computing it lazily if needed
  #[cfg(feature = "mobile")]
  fn get_generator(&self, i: usize) -> GroupElement {
    if let (Some(cache), Some(label_hash)) = (&self.generator_cache, &self.label_hash) {
      let mut cache = cache.lock().unwrap();
      if let Some(&gen) = cache.get(&i) {
        return gen;
      }
      
      // Compute generator i
      let mut shake = Shake256::default();
      shake.update(label_hash);
      let mut reader = shake.finalize_xof();
      
      let mut uniform_bytes = [0u8; 64];
      // Skip to the i-th generator
      for _ in 0..=i {
        reader.read(&mut uniform_bytes);
      }
      let gen = GroupElement::from_uniform_bytes(&uniform_bytes);
      cache.insert(i, gen);
      gen
    } else {
      // Fall back to regular array access
      self.G[i]
    }
  }
  
  #[cfg(not(feature = "mobile"))]
  fn get_generator(&self, i: usize) -> GroupElement {
    self.G[i]
  }

  pub fn clone(&self) -> MultiCommitGens {
    MultiCommitGens {
      n: self.n,
      h: self.h,
      G: self.G.clone(),
      generator_cache: self.generator_cache.clone(),
      label_hash: self.label_hash.clone(),
    }
  }

  pub fn scale(&self, s: &Scalar) -> MultiCommitGens {
    MultiCommitGens {
      n: self.n,
      h: self.h,
      G: (0..self.n).map(|i| s * self.get_generator(i)).collect(),
      generator_cache: None, // Clear cache after scaling
      label_hash: None,
    }
  }

  // Helper function for manual construction (used internally)
  pub fn from_parts(n: usize, G: Vec<GroupElement>, h: GroupElement) -> Self {
    MultiCommitGens {
      n,
      G,
      h,
      generator_cache: None,
      label_hash: None,
    }
  }

  pub fn split_at(&self, mid: usize) -> (MultiCommitGens, MultiCommitGens) {
    // For lazy generators, materialize the needed generators
    #[cfg(feature = "mobile")]
    if self.generator_cache.is_some() {
      let G1: Vec<GroupElement> = (0..mid).map(|i| self.get_generator(i)).collect();
      let G2: Vec<GroupElement> = (mid..self.n).map(|i| self.get_generator(i)).collect();
      
      return (
        MultiCommitGens {
          n: G1.len(),
          G: G1,
          h: self.h,
          generator_cache: None,
          label_hash: None,
        },
        MultiCommitGens {
          n: G2.len(), 
          G: G2,
          h: self.h,
          generator_cache: None,
          label_hash: None,
        },
      );
    }
    
    let (G1, G2) = self.G.split_at(mid);

    (
      MultiCommitGens {
        n: G1.len(),
        G: G1.to_vec(),
        h: self.h,
        generator_cache: None,
        label_hash: None,
      },
      MultiCommitGens {
        n: G2.len(),
        G: G2.to_vec(),
        h: self.h,
        generator_cache: None,
        label_hash: None,
      },
    )
  }
}

pub trait Commitments {
  fn commit(&self, blind: &Scalar, gens_n: &MultiCommitGens) -> GroupElement;
}

impl Commitments for Scalar {
  fn commit(&self, blind: &Scalar, gens_n: &MultiCommitGens) -> GroupElement {
    assert_eq!(gens_n.n, 1);
    let g0 = gens_n.get_generator(0);
    GroupElement::vartime_multiscalar_mul(&[*self, *blind], &[g0, gens_n.h])
  }
}

impl Commitments for Vec<Scalar> {
  fn commit(&self, blind: &Scalar, gens_n: &MultiCommitGens) -> GroupElement {
    assert_eq!(gens_n.n, self.len());
    // Use lazy generator access for mobile optimization
    #[cfg(feature = "mobile")]
    if gens_n.generator_cache.is_some() {
      let generators: Vec<GroupElement> = (0..gens_n.n).map(|i| gens_n.get_generator(i)).collect();
      return GroupElement::vartime_multiscalar_mul(self, &generators) + blind * gens_n.h;
    }
    GroupElement::vartime_multiscalar_mul(self, &gens_n.G) + blind * gens_n.h
  }
}

impl Commitments for [Scalar] {
  fn commit(&self, blind: &Scalar, gens_n: &MultiCommitGens) -> GroupElement {
    assert_eq!(gens_n.n, self.len());
    // Use lazy generator access for mobile optimization
    #[cfg(feature = "mobile")]
    if gens_n.generator_cache.is_some() {
      let generators: Vec<GroupElement> = (0..gens_n.n).map(|i| gens_n.get_generator(i)).collect();
      return GroupElement::vartime_multiscalar_mul(self, &generators) + blind * gens_n.h;
    }
    GroupElement::vartime_multiscalar_mul(self, &gens_n.G) + blind * gens_n.h
  }
}
