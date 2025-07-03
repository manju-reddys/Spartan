#!/bin/bash
# Build script for Android targets with mobile optimizations

set -e

echo "🤖 Building Spartan for Android with mobile optimizations..."

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

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    print_error "Rust/Cargo is required"
    exit 1
fi

# Check for Android NDK
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$NDK_HOME" ]; then
    print_error "Android NDK not found. Please set ANDROID_NDK_HOME or NDK_HOME environment variable"
    print_status "Download from: https://developer.android.com/ndk/downloads"
    exit 1
fi

# Set NDK path
NDK_PATH="${ANDROID_NDK_HOME:-$NDK_HOME}"
print_status "Using Android NDK: $NDK_PATH"

# Verify NDK toolchain
if [ ! -d "$NDK_PATH/toolchains" ]; then
    print_error "Invalid Android NDK path: $NDK_PATH"
    exit 1
fi

# Install Android targets if not already installed
print_status "Installing Android Rust targets..."

# Android targets
TARGETS=(
    "aarch64-linux-android"     # ARM64 (most modern Android devices)
    "armv7-linux-androideabi"   # ARM (older Android devices)
    "x86_64-linux-android"      # x86_64 (Android emulator/Intel devices)
    "i686-linux-android"        # x86 (older Android emulator)
)

for target in "${TARGETS[@]}"; do
    print_status "Installing target: $target"
    rustup target add "$target" || {
        print_warning "Target $target already installed or failed to install"
    }
done

# Set up Android NDK toolchain paths
API_LEVEL="21"  # Android 5.0 (minimum supported)
print_status "Using Android API level: $API_LEVEL"

# Create output directory
OUTPUT_DIR="target/android"
mkdir -p "$OUTPUT_DIR"

print_status "Building for Android targets with mobile optimizations enabled..."

# Build configuration
FEATURES="mobile"

# Set up environment for Android builds
export CC_aarch64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$API_LEVEL-clang"
export CXX_aarch64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$API_LEVEL-clang++"
export AR_aarch64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$API_LEVEL-clang"

export CC_armv7_linux_androideabi="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi$API_LEVEL-clang"
export CXX_armv7_linux_androideabi="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi$API_LEVEL-clang++"
export AR_armv7_linux_androideabi="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi$API_LEVEL-clang"

export CC_x86_64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android$API_LEVEL-clang"
export CXX_x86_64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android$API_LEVEL-clang++"
export AR_x86_64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android$API_LEVEL-clang"

export CC_i686_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android$API_LEVEL-clang"
export CXX_i686_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android$API_LEVEL-clang++"
export AR_i686_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
export CARGO_TARGET_I686_LINUX_ANDROID_LINKER="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android$API_LEVEL-clang"

# Handle macOS NDK paths
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Detected macOS, adjusting NDK paths..."
    export CC_aarch64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android$API_LEVEL-clang"
    export CXX_aarch64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android$API_LEVEL-clang++"
    export AR_aarch64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ar"
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android$API_LEVEL-clang"
    
    export CC_armv7_linux_androideabi="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/armv7a-linux-androideabi$API_LEVEL-clang"
    export CXX_armv7_linux_androideabi="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/armv7a-linux-androideabi$API_LEVEL-clang++"
    export AR_armv7_linux_androideabi="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ar"
    export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/armv7a-linux-androideabi$API_LEVEL-clang"
    
    export CC_x86_64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/x86_64-linux-android$API_LEVEL-clang"
    export CXX_x86_64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/x86_64-linux-android$API_LEVEL-clang++"
    export AR_x86_64_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ar"
    export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/x86_64-linux-android$API_LEVEL-clang"
    
    export CC_i686_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/i686-linux-android$API_LEVEL-clang"
    export CXX_i686_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/i686-linux-android$API_LEVEL-clang++"
    export AR_i686_linux_android="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ar"
    export CARGO_TARGET_I686_LINUX_ANDROID_LINKER="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/i686-linux-android$API_LEVEL-clang"
fi

# Function to build for a specific target
build_target() {
    local target=$1
    local target_name=$2
    
    print_status "Building for $target_name ($target)..."
    
    # Check if the toolchain exists
    local linker_var="CARGO_TARGET_${target^^}_LINKER"
    linker_var=${linker_var//-/_}
    local linker_path="${!linker_var}"
    
    if [ ! -f "$linker_path" ]; then
        print_error "Linker not found for $target: $linker_path"
        print_status "Make sure Android NDK is properly installed and accessible"
        return 1
    fi
    
    cargo build \
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
        
        # Also create .so shared library
        local so_path="target/$target/release/libspartan.so"
        if [ -f "$so_path" ]; then
            cp "$so_path" "$OUTPUT_DIR/libspartan_${target_name}.so"
            print_success "Built shared library for $target_name: $OUTPUT_DIR/libspartan_${target_name}.so"
        fi
    else
        print_error "Library not found for $target: $lib_path"
        return 1
    fi
}

# Build for each target
build_target "aarch64-linux-android" "android_arm64"
build_target "armv7-linux-androideabi" "android_arm"
build_target "x86_64-linux-android" "android_x64"
build_target "i686-linux-android" "android_x86"

# Generate JNI header
print_status "Generating JNI header file..."
cat > "$OUTPUT_DIR/spartan_jni.h" << 'EOF'
#ifndef SPARTAN_JNI_H
#define SPARTAN_JNI_H

#include <jni.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// JNI function declarations for Spartan
JNIEXPORT jlong JNICALL
Java_com_spartan_zkproof_SpartanNative_createInstance(JNIEnv *env, jobject thiz,
                                                     jint num_cons, jint num_vars, jint num_inputs);

JNIEXPORT void JNICALL
Java_com_spartan_zkproof_SpartanNative_destroyInstance(JNIEnv *env, jobject thiz, jlong instance_ptr);

JNIEXPORT jstring JNICALL
Java_com_spartan_zkproof_SpartanNative_getMemoryInfo(JNIEnv *env, jobject thiz);

JNIEXPORT jboolean JNICALL
Java_com_spartan_zkproof_SpartanNative_isMobileOptimized(JNIEnv *env, jobject thiz);

// Error codes
#define SPARTAN_JNI_SUCCESS 0
#define SPARTAN_JNI_ERROR_INVALID_INPUT 1
#define SPARTAN_JNI_ERROR_MEMORY 2
#define SPARTAN_JNI_ERROR_PROOF 3

#ifdef __cplusplus
}
#endif

#endif /* SPARTAN_JNI_H */
EOF

# Generate Android.mk for NDK build
print_status "Generating Android.mk..."
cat > "$OUTPUT_DIR/Android.mk" << 'EOF'
LOCAL_PATH := $(call my-dir)

# Spartan static library
include $(CLEAR_VARS)
LOCAL_MODULE := spartan
LOCAL_SRC_FILES := libspartan_$(TARGET_ARCH_ABI).a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)
include $(PREBUILT_STATIC_LIBRARY)

# JNI wrapper (you need to implement this)
include $(CLEAR_VARS)
LOCAL_MODULE := spartan_jni
LOCAL_SRC_FILES := spartan_jni.c
LOCAL_STATIC_LIBRARIES := spartan
LOCAL_LDLIBS := -llog -lm
include $(BUILD_SHARED_LIBRARY)
EOF

# Generate CMakeLists.txt for modern Android builds
print_status "Generating CMakeLists.txt..."
cat > "$OUTPUT_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.10.2)
project(spartan)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the Spartan library based on Android ABI
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(SPARTAN_LIB_NAME "libspartan_android_arm64.a")
elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set(SPARTAN_LIB_NAME "libspartan_android_arm.a")
elseif(${ANDROID_ABI} STREQUAL "x86_64")
    set(SPARTAN_LIB_NAME "libspartan_android_x64.a")
elseif(${ANDROID_ABI} STREQUAL "x86")
    set(SPARTAN_LIB_NAME "libspartan_android_x86.a")
else()
    message(FATAL_ERROR "Unsupported Android ABI: ${ANDROID_ABI}")
endif()

# Add Spartan static library
add_library(spartan STATIC IMPORTED)
set_target_properties(spartan PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/${SPARTAN_LIB_NAME}
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create JNI shared library (implement spartan_jni.cpp)
add_library(spartan_jni SHARED
    # spartan_jni.cpp  # You need to create this file
)

target_link_libraries(spartan_jni
    spartan
    log
    m
)

target_include_directories(spartan_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
EOF

# Generate Kotlin/Java wrapper class template
print_status "Generating Android Kotlin wrapper..."
mkdir -p "$OUTPUT_DIR/kotlin"
cat > "$OUTPUT_DIR/kotlin/SpartanZKProof.kt" << 'EOF'
package com.spartan.zkproof

class SpartanZKProof {
    companion object {
        init {
            System.loadLibrary("spartan_jni")
        }
    }
    
    data class MemoryInfo(
        val totalMemoryMB: Long,
        val availableMemoryMB: Long,
        val chunkSize: Long,
        val mobileOptimizationsEnabled: Boolean
    )
    
    enum class SpartanError(val code: Int) {
        SUCCESS(0),
        INVALID_INPUT(1),
        MEMORY_ERROR(2),
        PROOF_ERROR(3)
    }
    
    private var nativeInstance: Long = 0
    
    constructor(numCons: Int, numVars: Int, numInputs: Int) {
        nativeInstance = SpartanNative.createInstance(numCons, numVars, numInputs)
        if (nativeInstance == 0L) {
            throw RuntimeException("Failed to create Spartan instance")
        }
    }
    
    fun close() {
        if (nativeInstance != 0L) {
            SpartanNative.destroyInstance(nativeInstance)
            nativeInstance = 0
        }
    }
    
    companion object {
        fun getMemoryInfo(): String {
            return SpartanNative.getMemoryInfo()
        }
        
        fun isMobileOptimized(): Boolean {
            return SpartanNative.isMobileOptimized()
        }
    }
    
    protected fun finalize() {
        close()
    }
}

// Native interface
private object SpartanNative {
    external fun createInstance(numCons: Int, numVars: Int, numInputs: Int): Long
    external fun destroyInstance(instancePtr: Long)
    external fun getMemoryInfo(): String
    external fun isMobileOptimized(): Boolean
}
EOF

# Generate Gradle build file
print_status "Generating build.gradle..."
cat > "$OUTPUT_DIR/build.gradle" << 'EOF'
apply plugin: 'com.android.library'
apply plugin: 'kotlin-android'

android {
    compileSdkVersion 33
    
    defaultConfig {
        minSdkVersion 21  // Android 5.0 (matches our native build)
        targetSdkVersion 33
        
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
        
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a', 'x86_64', 'x86'
        }
        
        externalNativeBuild {
            cmake {
                cppFlags "-std=c++17"
                arguments "-DANDROID_STL=c++_shared"
            }
        }
    }
    
    externalNativeBuild {
        cmake {
            path "CMakeLists.txt"
        }
    }
    
    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    
    kotlinOptions {
        jvmTarget = '1.8'
    }
}

dependencies {
    implementation "org.jetbrains.kotlin:kotlin-stdlib:$kotlin_version"
    implementation 'androidx.core:core-ktx:1.8.0'
    
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.3'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.4.0'
}
EOF

# Create directory structure for Android project
mkdir -p "$OUTPUT_DIR/src/main/cpp"
mkdir -p "$OUTPUT_DIR/src/main/java/com/spartan/zkproof"

# Move Kotlin file to proper location
mv "$OUTPUT_DIR/kotlin/SpartanZKProof.kt" "$OUTPUT_DIR/src/main/java/com/spartan/zkproof/"
rmdir "$OUTPUT_DIR/kotlin"

# Create build info file
print_status "Creating build information..."
cat > "$OUTPUT_DIR/build_info.txt" << EOF
Spartan Android Build Information
================================

Built on: $(date)
Host: $(uname -a)
Rust version: $(rustc --version)
Cargo version: $(cargo --version)
Android NDK: $NDK_PATH
API Level: $API_LEVEL

Features enabled: $FEATURES

Target libraries:
- libspartan_android_arm64.a (aarch64-linux-android) - ARM64 devices
- libspartan_android_arm.a (armv7-linux-androideabi) - ARM devices
- libspartan_android_x64.a (x86_64-linux-android) - x64 emulator
- libspartan_android_x86.a (i686-linux-android) - x86 emulator

Mobile optimizations: ENABLED
- Platform detection: Android
- Memory limits: 1GB (configurable)
- Adaptive vector storage
- Lazy generator computation
- Chunked matrix operations

Android Integration:
1. Copy appropriate .a files to your Android project
2. Use CMakeLists.txt or Android.mk for native builds
3. Implement JNI wrapper functions (spartan_jni.cpp)
4. Use Kotlin/Java wrapper classes

Project structure:
- src/main/java/com/spartan/zkproof/ - Kotlin wrapper
- src/main/cpp/ - Place your JNI implementation here
- CMakeLists.txt - Modern CMake build
- Android.mk - Traditional NDK build
- build.gradle - Android library configuration

Note: You need to implement the actual JNI C++ wrapper functions
to bridge between Rust and Java/Kotlin code.

Mobile memory optimizations are automatically enabled and will
adapt to Android device memory constraints.
EOF

print_success "Android build completed successfully!"
print_status "Output directory: $OUTPUT_DIR"
print_status "Built libraries:"
ls -la "$OUTPUT_DIR"/*.a 2>/dev/null || print_warning "No .a files found"

print_status "🤖 Android libraries are ready for integration!"
print_status "See $OUTPUT_DIR/build_info.txt for detailed usage instructions"
print_status "Next steps:"
print_status "1. Implement JNI wrapper functions (spartan_jni.cpp)"
print_status "2. Copy files to your Android project"
print_status "3. Configure CMake or NDK build"
print_status "4. Test with the provided Kotlin wrapper"