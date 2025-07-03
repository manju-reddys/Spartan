mod ristretto255;

/// Main scalar type used throughout Spartan
pub type Scalar = ristretto255::Scalar;

/// Byte representation of scalars
pub type ScalarBytes = curve25519_dalek::scalar::Scalar;

/// Trait for converting primitive types to Scalar
pub trait ScalarFromPrimitives {
  /// Convert to Scalar
  fn to_scalar(self) -> Scalar;
}

impl ScalarFromPrimitives for usize {
  #[inline]
  fn to_scalar(self) -> Scalar {
    (0..self).map(|_i| Scalar::one()).sum()
  }
}

impl ScalarFromPrimitives for bool {
  #[inline]
  fn to_scalar(self) -> Scalar {
    if self {
      Scalar::one()
    } else {
      Scalar::zero()
    }
  }
}

/// Trait for converting Scalar to ScalarBytes  
pub trait ScalarBytesFromScalar {
  /// Decompress a Scalar to ScalarBytes
  fn decompress_scalar(s: &Scalar) -> ScalarBytes;
}

impl ScalarBytesFromScalar for Scalar {
  fn decompress_scalar(s: &Scalar) -> ScalarBytes {
    ScalarBytes::from_bytes_mod_order(s.to_bytes())
  }
}
