//! Platform capability detection and optimization selection

#![allow(missing_docs)]

use super::*;

/// Extended platform capabilities with detailed hardware information
#[derive(Debug, Clone)]
pub struct ExtendedPlatformCapabilities {
    pub base: PlatformCapabilities,
    pub cpu_info: CpuInfo,
    pub memory_info: MemoryInfo,
    pub thermal_info: Option<ThermalInfo>,
    pub power_info: Option<PowerInfo>,
}

/// CPU-specific information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub brand: String,
    pub core_count: usize,
    pub thread_count: usize,
    pub base_freq_mhz: Option<u32>,
    pub max_freq_mhz: Option<u32>,
    pub cache_sizes: CacheSizes,
    pub instruction_sets: InstructionSets,
}

/// Memory system information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_bytes: usize,
    pub available_bytes: usize,
    pub bandwidth_gbps: Option<f64>,
    pub numa_nodes: usize,
}

/// Thermal management information (primarily for mobile platforms)
#[derive(Debug, Clone)]
pub struct ThermalInfo {
    pub current_temp_celsius: Option<f32>,
    pub thermal_state: ThermalState,
    pub throttling_active: bool,
}

/// Power management information (primarily for mobile platforms)
#[derive(Debug, Clone)]
pub struct PowerInfo {
    pub battery_level_percent: Option<u8>,
    pub is_charging: bool,
    pub power_mode: PowerMode,
}

/// Cache size information
#[derive(Debug, Clone)]
pub struct CacheSizes {
    pub l1_data_kb: Option<usize>,
    pub l1_instruction_kb: Option<usize>,
    pub l2_kb: Option<usize>,
    pub l3_kb: Option<usize>,
}

/// Supported instruction sets
#[derive(Debug, Clone)]
pub struct InstructionSets {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512dq: bool,
    pub fma: bool,
    pub neon: bool,
    pub sve: bool,
}

/// Thermal states for mobile devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

/// Power management modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerMode {
    Performance,
    Balanced,
    PowerSaver,
}

impl ExtendedPlatformCapabilities {
    /// Detect extended platform capabilities
    pub fn detect() -> Self {
        let base = PlatformCapabilities::detect();
        let cpu_info = CpuInfo::detect();
        let memory_info = MemoryInfo::detect();
        let thermal_info = ThermalInfo::detect();
        let power_info = PowerInfo::detect();
        
        Self {
            base,
            cpu_info,
            memory_info,
            thermal_info,
            power_info,
        }
    }
    
    /// Recommend optimal backend based on detailed capabilities
    pub fn recommend_backend(&self) -> BackendType {
        // Consider thermal throttling on mobile
        if self.is_thermally_constrained() {
            return BackendType::Mobile;
        }
        
        // Consider power constraints
        if self.is_power_constrained() {
            return BackendType::Mobile;
        }
        
        // Consider GPU availability and problem size suitability
        #[cfg(feature = "gpu")]
        if self.base.has_gpu && self.is_gpu_beneficial() {
            return BackendType::GPU;
        }
        
        // Consider WASM constraints
        if self.base.platform == Platform::WASM {
            return BackendType::WASM;
        }
        
        // Default to native for desktop
        BackendType::Native
    }
    
    /// Check if the system is thermally constrained
    pub fn is_thermally_constrained(&self) -> bool {
        if let Some(thermal) = &self.thermal_info {
            matches!(thermal.thermal_state, ThermalState::Serious | ThermalState::Critical) ||
            thermal.throttling_active
        } else {
            false
        }
    }
    
    /// Check if the system is power constrained
    pub fn is_power_constrained(&self) -> bool {
        if let Some(power) = &self.power_info {
            matches!(power.power_mode, PowerMode::PowerSaver) ||
            (power.battery_level_percent.unwrap_or(100) < 20 && !power.is_charging)
        } else {
            false
        }
    }
    
    /// Check if GPU acceleration would be beneficial
    #[cfg(feature = "gpu")]
    pub fn is_gpu_beneficial(&self) -> bool {
        // GPU is beneficial for large problems or when CPU is constrained
        self.cpu_info.core_count < 4 || self.memory_info.bandwidth_gbps.unwrap_or(0.0) > 100.0
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn is_gpu_beneficial(&self) -> bool {
        false
    }
    
    /// Get recommended optimization level
    pub fn get_optimization_level(&self) -> crate::cross_platform::backend::OptimizationLevel {
        use crate::cross_platform::backend::OptimizationLevel;
        
        if self.is_thermally_constrained() || self.is_power_constrained() {
            OptimizationLevel::Conservative
        } else if self.cpu_info.instruction_sets.avx512f {
            OptimizationLevel::Aggressive
        } else if self.cpu_info.instruction_sets.avx2 {
            OptimizationLevel::Balanced
        } else {
            OptimizationLevel::Conservative
        }
    }
    
    /// Get recommended parallelism level
    pub fn get_parallelism_level(&self) -> ParallelismLevel {
        if self.is_thermally_constrained() || self.is_power_constrained() {
            ParallelismLevel::Single
        } else if self.cpu_info.core_count >= 8 {
            ParallelismLevel::Maximum
        } else if self.cpu_info.core_count >= 4 {
            ParallelismLevel::Limited
        } else {
            ParallelismLevel::Single
        }
    }
}

/// Parallelism levels for different platform constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismLevel {
    Single,     // No parallelism
    Limited,    // Limited parallelism (2-4 threads)
    Maximum,    // Full parallelism available
}

impl CpuInfo {
    fn detect() -> Self {
        let core_count = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        
        Self {
            brand: Self::detect_cpu_brand(),
            core_count,
            thread_count: core_count, // Simplified assumption
            base_freq_mhz: None,      // Platform-specific detection needed
            max_freq_mhz: None,       // Platform-specific detection needed
            cache_sizes: CacheSizes::detect(),
            instruction_sets: InstructionSets::detect(),
        }
    }
    
    fn detect_cpu_brand() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            // Would use CPUID instruction to get brand string
            "Unknown x86_64".to_string()
        }
        #[cfg(target_arch = "aarch64")]
        {
            "Unknown ARM64".to_string()
        }
        #[cfg(target_arch = "wasm32")]
        {
            "WebAssembly".to_string()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        {
            "Unknown".to_string()
        }
    }
}

impl MemoryInfo {
    fn detect() -> Self {
        Self {
            total_bytes: Self::detect_total_memory(),
            available_bytes: Self::detect_available_memory(),
            bandwidth_gbps: None, // Platform-specific detection needed
            numa_nodes: 1,        // Simplified assumption
        }
    }
    
    #[cfg(target_os = "linux")]
    fn detect_total_memory() -> usize {
        // Read from /proc/meminfo on Linux
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            return parts[1].parse::<usize>().ok().map(|kb| kb * 1024);
                        }
                    }
                }
                None
            })
            .unwrap_or(8 * 1024 * 1024 * 1024) // Default to 8GB
    }
    
    #[cfg(not(target_os = "linux"))]
    fn detect_total_memory() -> usize {
        // Platform-specific implementation needed
        8 * 1024 * 1024 * 1024 // Default to 8GB
    }
    
    fn detect_available_memory() -> usize {
        // Simplified estimation
        Self::detect_total_memory() / 2
    }
}

impl ThermalInfo {
    fn detect() -> Option<Self> {
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            Some(Self {
                current_temp_celsius: None, // Platform-specific detection needed
                thermal_state: ThermalState::Nominal,
                throttling_active: false,
            })
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        None
    }
}

impl PowerInfo {
    fn detect() -> Option<Self> {
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            Some(Self {
                battery_level_percent: None, // Platform-specific detection needed
                is_charging: false,          // Platform-specific detection needed
                power_mode: PowerMode::Balanced,
            })
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        None
    }
}

impl CacheSizes {
    fn detect() -> Self {
        Self {
            l1_data_kb: None,        // Platform-specific detection needed
            l1_instruction_kb: None, // Platform-specific detection needed
            l2_kb: None,             // Platform-specific detection needed
            l3_kb: None,             // Platform-specific detection needed
        }
    }
}

impl InstructionSets {
    fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            sse: is_x86_feature_detected!("sse"),
            #[cfg(not(target_arch = "x86_64"))]
            sse: false,
            
            #[cfg(target_arch = "x86_64")]
            sse2: is_x86_feature_detected!("sse2"),
            #[cfg(not(target_arch = "x86_64"))]
            sse2: false,
            
            #[cfg(target_arch = "x86_64")]
            sse3: is_x86_feature_detected!("sse3"),
            #[cfg(not(target_arch = "x86_64"))]
            sse3: false,
            
            #[cfg(target_arch = "x86_64")]
            sse4_1: is_x86_feature_detected!("sse4.1"),
            #[cfg(not(target_arch = "x86_64"))]
            sse4_1: false,
            
            #[cfg(target_arch = "x86_64")]
            sse4_2: is_x86_feature_detected!("sse4.2"),
            #[cfg(not(target_arch = "x86_64"))]
            sse4_2: false,
            
            #[cfg(target_arch = "x86_64")]
            avx: is_x86_feature_detected!("avx"),
            #[cfg(not(target_arch = "x86_64"))]
            avx: false,
            
            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            avx2: false,
            
            #[cfg(target_arch = "x86_64")]
            avx512f: is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            avx512f: false,
            
            #[cfg(target_arch = "x86_64")]
            avx512dq: is_x86_feature_detected!("avx512dq"),
            #[cfg(not(target_arch = "x86_64"))]
            avx512dq: false,
            
            #[cfg(target_arch = "x86_64")]
            fma: is_x86_feature_detected!("fma"),
            #[cfg(not(target_arch = "x86_64"))]
            fma: false,
            
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
            neon: false,
            
            #[cfg(target_arch = "aarch64")]
            sve: std::arch::is_aarch64_feature_detected!("sve"),
            #[cfg(not(target_arch = "aarch64"))]
            sve: false,
        }
    }
}