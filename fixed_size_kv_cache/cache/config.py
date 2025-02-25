"""
Configuration module for FixedSizeDynamicCache.
"""

import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration settings for the fixed-size KV cache.
    
    This class stores all configuration settings for the cache behavior.
    Values can be provided at initialization or loaded from environment variables.
    """
    
    # Basic configuration
    kv_cache_size: int = 1024
    include_skipped: bool = False
    free_memory: bool = False
    
    # Truncation strategy
    truncation_strategy: str = "attention"  # Options: "attention", "hybrid"
    truncation_threshold: float = 0.0
    hybrid_split_ratio: float = 0.5
    
    # CPU offloading
    offload_to_cpu: bool = False
    offload_size: int = 16384
    use_mmap: bool = False
    
    # Token importance tracking
    auto_restore: bool = True
    token_importance_window: int = 100
    importance_threshold: float = 0.1
    
    # Performance optimizations
    quantize: bool = False
    quantization_bits: int = 8
    parallel_processing: bool = True
    worker_threads: int = 4
    adaptive_precision: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        
    def _validate(self):
        """Validate the configuration values."""
        if not isinstance(self.kv_cache_size, int) or self.kv_cache_size <= 0:
            raise ValueError(f"kv_cache_size must be a positive integer, got {self.kv_cache_size}")
        
        if self.truncation_strategy not in ("attention", "hybrid"):
            raise ValueError(f"Unknown truncation strategy: {self.truncation_strategy}. Must be one of: 'attention', 'hybrid'")
            
        if not 0.0 <= self.truncation_threshold <= 1.0:
            raise ValueError(f"truncation_threshold must be between 0.0 and 1.0, got {self.truncation_threshold}")
            
        if not 0.0 <= self.hybrid_split_ratio <= 1.0:
            raise ValueError(f"hybrid_split_ratio must be between 0.0 and 1.0, got {self.hybrid_split_ratio}")
            
        if not isinstance(self.offload_size, int) or self.offload_size <= 0:
            raise ValueError(f"offload_size must be a positive integer, got {self.offload_size}")
            
        if not isinstance(self.token_importance_window, int) or self.token_importance_window <= 0:
            raise ValueError(f"token_importance_window must be a positive integer, got {self.token_importance_window}")
            
        if not 0.0 <= self.importance_threshold <= 1.0:
            raise ValueError(f"importance_threshold must be between 0.0 and 1.0, got {self.importance_threshold}")
            
        if self.quantize and self.quantization_bits not in (4, 8):
            raise ValueError(f"quantization_bits must be 4 or 8, got {self.quantization_bits}")
            
        if not isinstance(self.worker_threads, int) or self.worker_threads <= 0:
            raise ValueError(f"worker_threads must be a positive integer, got {self.worker_threads}")


def load_config_from_env() -> CacheConfig:
    """Load cache configuration from environment variables.
    
    Returns:
        A CacheConfig object with values loaded from environment variables.
    """
    config = CacheConfig()
    
    # Basic configuration
    kv_cache_size_str = os.getenv("FSDC_KV_CACHE_SIZE")
    if kv_cache_size_str:
        try:
            config.kv_cache_size = int(kv_cache_size_str)
        except ValueError:
            logger.warning(f"Invalid FSDC_KV_CACHE_SIZE value: {kv_cache_size_str}. Using default: {config.kv_cache_size}")
    
    include_skipped_str = os.getenv("FSDC_INCLUDE_SKIPPED")
    if include_skipped_str is not None:
        config.include_skipped = include_skipped_str.lower() in ('true', '1', 'yes')
    
    free_memory_str = os.getenv("FSDC_FREE_MEMORY")
    if free_memory_str is not None:
        config.free_memory = free_memory_str.lower() in ('true', '1', 'yes')
        
    # Truncation strategy
    strategy = os.getenv("FSDC_TRUNCATION_STRATEGY")
    if strategy in ("attention", "hybrid"):
        config.truncation_strategy = strategy
    
    hybrid_split_ratio_str = os.getenv("FSDC_HYBRID_SPLIT_RATIO")
    if hybrid_split_ratio_str:
        try:
            ratio = float(hybrid_split_ratio_str)
            if 0.0 <= ratio <= 1.0:
                config.hybrid_split_ratio = ratio
            else:
                logger.warning(f"Invalid FSDC_HYBRID_SPLIT_RATIO value: {hybrid_split_ratio_str}. Must be between 0.0 and 1.0. Using default: {config.hybrid_split_ratio}")
        except ValueError:
            logger.warning(f"Invalid FSDC_HYBRID_SPLIT_RATIO value: {hybrid_split_ratio_str}. Using default: {config.hybrid_split_ratio}")
    
    # CPU offloading
    offload_to_cpu_str = os.getenv("FSDC_OFFLOAD_TO_CPU")
    if offload_to_cpu_str is not None:
        config.offload_to_cpu = offload_to_cpu_str.lower() in ('true', '1', 'yes')
        
    offload_size_str = os.getenv("FSDC_OFFLOAD_SIZE")
    if offload_size_str:
        try:
            config.offload_size = int(offload_size_str)
        except ValueError:
            logger.warning(f"Invalid FSDC_OFFLOAD_SIZE value: {offload_size_str}. Using default: {config.offload_size}")
            
    use_mmap_str = os.getenv("FSDC_USE_MMAP")
    if use_mmap_str is not None:
        config.use_mmap = use_mmap_str.lower() in ('true', '1', 'yes')
            
    # Token importance tracking
    auto_restore_str = os.getenv("FSDC_AUTO_RESTORE")
    if auto_restore_str is not None:
        config.auto_restore = auto_restore_str.lower() in ('true', '1', 'yes')
        
    token_importance_window_str = os.getenv("FSDC_TOKEN_IMPORTANCE_WINDOW")
    if token_importance_window_str:
        try:
            config.token_importance_window = int(token_importance_window_str)
        except ValueError:
            logger.warning(f"Invalid FSDC_TOKEN_IMPORTANCE_WINDOW value: {token_importance_window_str}. Using default: {config.token_importance_window}")
            
    importance_threshold_str = os.getenv("FSDC_IMPORTANCE_THRESHOLD")
    if importance_threshold_str:
        try:
            threshold = float(importance_threshold_str)
            if 0.0 <= threshold <= 1.0:
                config.importance_threshold = threshold
            else:
                logger.warning(f"Invalid FSDC_IMPORTANCE_THRESHOLD value: {importance_threshold_str}. Must be between 0.0 and 1.0. Using default: {config.importance_threshold}")
        except ValueError:
            logger.warning(f"Invalid FSDC_IMPORTANCE_THRESHOLD value: {importance_threshold_str}. Using default: {config.importance_threshold}")
            
    # Performance optimizations
    quantize_str = os.getenv("FSDC_QUANTIZE")
    if quantize_str is not None:
        config.quantize = quantize_str.lower() in ('true', '1', 'yes')
        
    quantization_bits_str = os.getenv("FSDC_QUANTIZATION_BITS")
    if quantization_bits_str:
        try:
            bits = int(quantization_bits_str)
            if bits in (4, 8):
                config.quantization_bits = bits
            else:
                logger.warning(f"Invalid FSDC_QUANTIZATION_BITS value: {quantization_bits_str}. Must be 4 or 8. Using default: {config.quantization_bits}")
        except ValueError:
            logger.warning(f"Invalid FSDC_QUANTIZATION_BITS value: {quantization_bits_str}. Using default: {config.quantization_bits}")
            
    parallel_processing_str = os.getenv("FSDC_PARALLEL_PROCESSING")
    if parallel_processing_str is not None:
        config.parallel_processing = parallel_processing_str.lower() in ('true', '1', 'yes')
        
    adaptive_precision_str = os.getenv("FSDC_ADAPTIVE_PRECISION")
    if adaptive_precision_str is not None:
        config.adaptive_precision = adaptive_precision_str.lower() in ('true', '1', 'yes')
        
    worker_threads_str = os.getenv("FSDC_WORKER_THREADS")
    if worker_threads_str:
        try:
            config.worker_threads = int(worker_threads_str)
        except ValueError:
            logger.warning(f"Invalid FSDC_WORKER_THREADS value: {worker_threads_str}. Using default: {config.worker_threads}")
    
    return config