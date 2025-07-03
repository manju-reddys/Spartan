#!/bin/bash
# Build script for iOS targets with mobile optimizations

set -e

echo "🍎 Building Spartan for iOS with mobile optimizations..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "iOS builds require macOS"
    exit 1
fi

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    print_error "Xcode is required for iOS builds"
    exit 1
fi

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    print_error "Rust/Cargo is required"
    exit 1
fi

# Install iOS targets if not already installed
print_status "Installing iOS Rust targets..."

# iOS targets
TARGETS=(
    "aarch64-apple-ios"          # iOS devices (iPhone/iPad)
    "x86_64-apple-ios"           # iOS Simulator (Intel Macs)
    "aarch64-apple-ios-sim"      # iOS Simulator (Apple Silicon Macs)
)

for target in "${TARGETS[@]}"; do
    print_status "Installing target: $target"
    rustup target add "$target" || {
        print_warning "Target $target already installed or failed to install"
    }
done

# Create output directory
OUTPUT_DIR="target/ios"
mkdir -p "$OUTPUT_DIR"

print_status "Building for iOS targets with mobile optimizations enabled..."

# Build configuration
RUSTFLAGS="-C target-cpu=native"
FEATURES="mobile"

# Function to build for a specific target
build_target() {
    local target=$1
    local target_name=$2
    
    print_status "Building for $target_name ($target)..."
    
    RUSTFLAGS="$RUSTFLAGS" cargo build \
        --target "$target" \
        --release \
        --features "$FEATURES" \
        --lib || {
        print_error "Failed to build for $target"
        return 1
    }
    
    # Copy the built library to output directory
    local lib_path="target/$target/release/libspartan.a"
    if [ -f "$lib_path" ]; then
        cp "$lib_path" "$OUTPUT_DIR/libspartan_${target_name}.a"
        print_success "Built library for $target_name: $OUTPUT_DIR/libspartan_${target_name}.a"
    else
        print_error "Library not found for $target: $lib_path"
        return 1
    fi
}

# Build for each target
build_target "aarch64-apple-ios" "ios_device"
build_target "x86_64-apple-ios" "ios_sim_x64"
build_target "aarch64-apple-ios-sim" "ios_sim_arm64"

# Create universal library for iOS Simulator
print_status "Creating universal iOS Simulator library..."
lipo -create \
    "$OUTPUT_DIR/libspartan_ios_sim_x64.a" \
    "$OUTPUT_DIR/libspartan_ios_sim_arm64.a" \
    -output "$OUTPUT_DIR/libspartan_ios_simulator.a" || {
    print_warning "Failed to create universal simulator library (this is OK if you only need one architecture)"
}

# Generate header file for C interop
print_status "Generating C header file..."
cat > "$OUTPUT_DIR/spartan.h" << 'EOF'
#ifndef SPARTAN_H
#define SPARTAN_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types for Spartan structures
typedef struct SpartanInstance SpartanInstance;
typedef struct SpartanProof SpartanProof;
typedef struct SpartanGens SpartanGens;

// Error codes
typedef enum {
    SPARTAN_SUCCESS = 0,
    SPARTAN_ERROR_INVALID_INPUT = 1,
    SPARTAN_ERROR_PROOF_GENERATION = 2,
    SPARTAN_ERROR_PROOF_VERIFICATION = 3,
    SPARTAN_ERROR_MEMORY = 4,
} SpartanError;

// Memory optimization info
typedef struct {
    uint64_t total_memory_mb;
    uint64_t available_memory_mb;
    uint64_t chunk_size;
    bool mobile_optimizations_enabled;
} SpartanMemoryInfo;

// Function declarations (these would need to be implemented in a C wrapper)
SpartanError spartan_get_memory_info(SpartanMemoryInfo* info);
SpartanError spartan_create_instance(SpartanInstance** instance, 
                                    uint32_t num_cons, 
                                    uint32_t num_vars, 
                                    uint32_t num_inputs);
SpartanError spartan_free_instance(SpartanInstance* instance);

// Note: This header is a template. Actual C bindings would need to be implemented
// using tools like cbindgen or manual FFI wrapper functions.

#ifdef __cplusplus
}
#endif

#endif /* SPARTAN_H */
EOF

# Create Swift Package Manager structure
print_status "Creating Swift Package structure..."
mkdir -p "$OUTPUT_DIR/Sources/CSpartan"
mkdir -p "$OUTPUT_DIR/Sources/Spartan"

# Copy header to Swift package
cp "$OUTPUT_DIR/spartan.h" "$OUTPUT_DIR/Sources/CSpartan/"

# Create module map
cat > "$OUTPUT_DIR/Sources/CSpartan/module.modulemap" << 'EOF'
module CSpartan {
    header "spartan.h"
    link "spartan"
    export *
}
EOF

# Create Package.swift
cat > "$OUTPUT_DIR/Package.swift" << 'EOF'
// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "Spartan",
    platforms: [
        .iOS(.v12),
        .macOS(.v10_15)
    ],
    products: [
        .library(
            name: "Spartan",
            targets: ["Spartan"]
        ),
    ],
    targets: [
        .target(
            name: "CSpartan",
            path: "Sources/CSpartan"
        ),
        .target(
            name: "Spartan",
            dependencies: ["CSpartan"],
            path: "Sources/Spartan"
        ),
        .testTarget(
            name: "SpartanTests",
            dependencies: ["Spartan"]
        ),
    ]
)
EOF

# Create basic Swift wrapper
cat > "$OUTPUT_DIR/Sources/Spartan/Spartan.swift" << 'EOF'
import CSpartan
import Foundation

public class SpartanZKProof {
    public struct MemoryInfo {
        public let totalMemoryMB: UInt64
        public let availableMemoryMB: UInt64
        public let chunkSize: UInt64
        public let mobileOptimizationsEnabled: Bool
    }
    
    public enum SpartanError: Error {
        case invalidInput
        case proofGeneration
        case proofVerification
        case memory
        case unknown(Int32)
        
        init(code: SpartanError) {
            switch code {
            case SPARTAN_SUCCESS:
                self = .unknown(0) // This shouldn't happen
            case SPARTAN_ERROR_INVALID_INPUT:
                self = .invalidInput
            case SPARTAN_ERROR_PROOF_GENERATION:
                self = .proofGeneration
            case SPARTAN_ERROR_PROOF_VERIFICATION:
                self = .proofVerification
            case SPARTAN_ERROR_MEMORY:
                self = .memory
            default:
                self = .unknown(code.rawValue)
            }
        }
    }
    
    public static func getMemoryInfo() throws -> MemoryInfo {
        var info = SpartanMemoryInfo()
        let result = spartan_get_memory_info(&info)
        
        if result != SPARTAN_SUCCESS {
            throw SpartanError(code: result)
        }
        
        return MemoryInfo(
            totalMemoryMB: info.total_memory_mb,
            availableMemoryMB: info.available_memory_mb,
            chunkSize: info.chunk_size,
            mobileOptimizationsEnabled: info.mobile_optimizations_enabled
        )
    }
    
    // Additional Swift wrapper methods would be implemented here
    // This is a basic template showing the structure
}
EOF

# Create build info file
print_status "Creating build information..."
cat > "$OUTPUT_DIR/build_info.txt" << EOF
Spartan iOS Build Information
============================

Built on: $(date)
Host: $(uname -a)
Rust version: $(rustc --version)
Cargo version: $(cargo --version)

Features enabled: $FEATURES
RUSTFLAGS: $RUSTFLAGS

Target libraries:
- libspartan_ios_device.a (aarch64-apple-ios)
- libspartan_ios_sim_x64.a (x86_64-apple-ios)  
- libspartan_ios_sim_arm64.a (aarch64-apple-ios-sim)
- libspartan_ios_simulator.a (universal simulator)

Mobile optimizations: ENABLED
- Platform detection: iOS
- Memory limits: 512MB
- Adaptive vector storage
- Lazy generator computation
- Chunked matrix operations

Usage:
1. Link appropriate .a file to your iOS project
2. Include spartan.h header
3. Implement C FFI wrapper functions as needed
4. Use Swift Package Manager structure for Swift projects

Note: This build includes mobile memory optimizations that automatically
activate on iOS devices to manage memory usage efficiently.
EOF

print_success "iOS build completed successfully!"
print_status "Output directory: $OUTPUT_DIR"
print_status "Built libraries:"
ls -la "$OUTPUT_DIR"/*.a 2>/dev/null || print_warning "No .a files found"

print_status "📱 iOS libraries are ready for integration!"
print_status "See $OUTPUT_DIR/build_info.txt for detailed usage instructions"