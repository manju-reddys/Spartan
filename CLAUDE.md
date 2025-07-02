# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spartan is a high-speed zero-knowledge proof system implementing zkSNARKs without trusted setup. This is a Rust library (`libspartan`) implementing transparent zero-knowledge succinct non-interactive arguments of knowledge (zkSNARKs) based on the discrete logarithm problem.

## Build Commands

### Basic Build
```bash
# Standard build
cargo build --release

# With SIMD optimizations (recommended)
RUSTFLAGS='-C target_cpu=native --cfg curve25519_dalek_backend="BACKEND"' cargo build --release
```

### Testing
```bash
# Run tests with optimizations
RUSTFLAGS='-C target_cpu=native --cfg curve25519_dalek_backend="BACKEND"' cargo test

# Run specific test
cargo test test_name
```

### Benchmarks
```bash
# Run end-to-end benchmarks
RUSTFLAGS='-C target_cpu=native --cfg curve25519_dalek_backend="BACKEND"' cargo bench

# Individual benchmarks
cargo bench --bench snark
cargo bench --bench nizk
```

### Documentation
```bash
# Build documentation for public APIs
cargo doc

# Open docs in browser
cargo doc --open
```

### Profiling
```bash
# Build with profiling enabled
cargo build --release --features=profile

# Run profilers
./target/release/snark
./target/release/nizk
```

## Code Architecture

### Core Components

- **R1CS System**: Rank-1 Constraint Satisfiability system for expressing NP statements
  - `src/r1cs.rs` - R1CS instance representation and operations
  - `src/r1csproof.rs` - R1CS proof generation and verification

- **Proof Systems**: Two main variants implemented
  - **SNARK**: Full zkSNARK with preprocessing (`SNARKGens`, `SNARK::prove`, `SNARK::verify`)
  - **NIZK**: Non-interactive zero-knowledge variant (`NIZKGens`, `NIZK::prove`, `NIZK::verify`)

- **Cryptographic Primitives**:
  - `src/commitments.rs` - Polynomial commitment schemes
  - `src/group.rs` - Elliptic curve group operations (ristretto255)
  - `src/scalar/` - Scalar field arithmetic
  - `src/transcript.rs` - Fiat-Shamir transformation using Merlin

- **Mathematical Components**:
  - `src/dense_mlpoly.rs` - Dense multilinear polynomials
  - `src/sparse_mlpoly.rs` - Sparse multilinear polynomials
  - `src/sumcheck.rs` - Sum-check protocol implementation
  - `src/unipoly.rs` - Univariate polynomials

- **Proof Subsystems**:
  - `src/nizk/` - NIZK-specific implementations including Bulletproofs
  - `src/product_tree.rs` - Product tree for batch operations

### Key Features

- Uses `curve25519-dalek` for ristretto255 group operations
- Leverages `merlin` crate for transcript management
- Supports both std and no-std environments
- Optional multicore support via `rayon` feature
- WASM compatibility

### Dependencies

- `curve25519-dalek` - Elliptic curve arithmetic
- `merlin` - Transcript management for Fiat-Shamir
- `rand` - Randomness generation
- `serde` - Serialization support
- `rayon` - Optional parallelization

## Cargo Features

- `std` (default) - Standard library support
- `multicore` - Enable parallel computation with rayon
- `profile` - Enable detailed profiling output

## Common Development Tasks

### Running Examples
```bash
# Run the cubic example
cargo run --example cubic
```

### Lint and Format
```bash
# Check formatting
cargo fmt --check

# Format code
cargo fmt

# Run typo checker (used in CI)
typos
```

### Testing Performance
Use the profilers in `profiler/` directory rather than micro-benchmarks for accurate performance measurement when reporting results.