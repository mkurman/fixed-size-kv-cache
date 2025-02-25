# Fixed Size Key-Value Cache

## Project Description

This project implements a fixed-size key-value cache for use in transformer models. The example is specifically designed for the LlaMA architecture. The cache dynamically truncates the key and value states based on attention weights, ensures efficient memory usage with CPU offloading, and intelligently restores important tokens during inference.

![Fixed Size Key-Value Cache](image_fx_.jpg)

## Installation

To install the required dependencies, run:

```bash
pip install -U pip && pip install -e .
```

## Features

- **Fixed-Size KV Cache**: Limit the KV cache size for consistent memory usage
- **Multiple Truncation Strategies**: Attention-based or hybrid (attention + recency)
- **CPU Offloading**: Store less important tokens in CPU memory
- **Token Importance Tracking**: Identify and restore important tokens
- **On-the-fly Quantization**: Reduce memory usage with 4/8-bit quantization
- **Memory Mapping**: Efficient CPU storage using memory-mapped files
- **Parallel Processing**: Multi-threaded token importance calculation
- **Adaptive Precision**: Automatic precision adjustment for operations

## Usage

### Setting Up FixedSizeDynamicCache

The `FixedSizeDynamicCache` class is a dynamic cache that manages key and value states efficiently. This cache behavior is configurable via environment variables or using the `CacheConfig` object directly:

#### Basic Configuration
```python
import os
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
import transformers
from fixed_size_kv_cache import FixedSizeDynamicCache
from fixed_size_kv_cache.llama_attention import forward

# Set environment variables for basic configuration
os.environ['FSDC_KV_CACHE_SIZE'] = '2048'
os.environ['FSDC_INCLUDE_SKIPPED'] = 'True'
os.environ['FSDC_FREE_MEMORY'] = 'True'

# Replace forward function
LlamaAttention.forward = forward

# Set up FixedSizeDynamicCache
transformers.models.llama.modeling_llama.DynamicCache = FixedSizeDynamicCache
LlamaForCausalLM.__init__.__globals__['DynamicCache'] = FixedSizeDynamicCache

# Change attention to eager (required to access attention weights)
model.config._attn_implementation = "eager"
```

#### Using Config Object

```python
from fixed_size_kv_cache import FixedSizeDynamicCache, CacheConfig

# Create config object with custom settings
config = CacheConfig(
    kv_cache_size=2048,
    include_skipped=True,
    truncation_strategy="hybrid",
    hybrid_split_ratio=0.7,
    offload_to_cpu=True,
    offload_size=32768,
    quantize=True,
    quantization_bits=8,
    use_mmap=True,
    adaptive_precision=True
)

# Create cache instance with config
custom_cache = FixedSizeDynamicCache()
custom_cache.config = config

# You can use this custom cache with a model if needed
```

### Configuration Parameters

#### Basic Configuration
- `FSDC_KV_CACHE_SIZE`: The maximum size of the key and value cache. (default: 1024)
- `FSDC_INCLUDE_SKIPPED`: Whether to include the sum of skipped key and value states in the cache as the last entry. (default: False)
- `FSDC_FREE_MEMORY`: Whether to free memory after truncating the cache. (default: False)

#### Truncation Strategies
- `FSDC_TRUNCATION_STRATEGY`: The strategy to use for truncating the cache. Options: "attention" (based on attention weights), "hybrid" (combines attention and recency). (default: "attention")
- `FSDC_HYBRID_SPLIT_RATIO`: For hybrid strategy, the ratio of tokens to keep based on attention vs. recency. (default: 0.5)

#### CPU Offloading
- `FSDC_OFFLOAD_TO_CPU`: Whether to offload skipped tokens to CPU memory. (default: False)
- `FSDC_OFFLOAD_SIZE`: Maximum size of the CPU cache. (default: 16384)
- `FSDC_USE_MMAP`: Whether to use memory mapping for CPU offloading for better performance. (default: False)

#### Intelligent Token Restoration
- `FSDC_AUTO_RESTORE`: Whether to automatically restore important tokens from the CPU cache. (default: True)
- `FSDC_TOKEN_IMPORTANCE_WINDOW`: Size of the rolling window for tracking token importance. (default: 100)
- `FSDC_IMPORTANCE_THRESHOLD`: Threshold for determining important tokens. (default: 0.1)

#### Performance Optimizations
- `FSDC_QUANTIZE`: Whether to enable on-the-fly quantization of cached tokens. (default: False)
- `FSDC_QUANTIZATION_BITS`: Number of bits to use for quantization (4 or 8). (default: 8)
- `FSDC_PARALLEL_PROCESSING`: Whether to enable parallel token importance processing. (default: True)
- `FSDC_WORKER_THREADS`: Number of worker threads for parallel processing. (default: 4)
- `FSDC_ADAPTIVE_PRECISION`: Whether to use mixed precision based on operation importance. (default: False)

### Example Usage

See the `examples/llama-3.2-3b-instruct.ipynb` notebook for a complete example of using the fixed-size KV cache with a LLaMA 3.2 3B model on a long context summarization task.

### Feature Details

#### Hybrid Truncation Strategy

The hybrid strategy combines attention-based and recency-based token selection:
- Uses attention weights to identify important tokens for retention
- Preserves recent tokens to maintain coherence in ongoing generation
- Configurable ratio allows for balancing between these approaches

#### CPU Offloading with Memory Mapping

When enabled, tokens removed from the GPU cache are stored in CPU memory:
- Maintains a much larger context window than would fit in GPU memory
- Preserves token positions for potential restoration
- Efficiently manages memory to avoid CPU memory bloat
- Optional memory mapping provides faster CPU access with reduced memory overhead

#### On-The-Fly Quantization

Reduces memory usage by quantizing cached tokens:
- Supports 4-bit and 8-bit quantization
- Automatically dequantizes when tokens are restored
- Can reduce memory footprint by up to 8x with 4-bit quantization
- Minimal impact on model quality with optimized quantization methods

#### Parallel Processing for Token Importance

Accelerates the token importance calculation process:
- Distributes computation across multiple CPU threads
- Significantly improves performance for large batch sizes and many attention heads
- Configurable number of worker threads for different hardware configurations

#### Adaptive Precision Management

Intelligently adjusts numerical precision based on operation importance:
- Uses full precision for critical calculations
- Employs lower precision for less sensitive operations
- Automatically integrates with PyTorch's autocast for mixed precision training
- Provides performance benefits without sacrificing accuracy

#### Intelligent Token Restoration

The system tracks token importance and can automatically restore tokens:
- Maintains a rolling window of token importance scores based on attention patterns
- When a token reaches sufficient importance, it's automatically brought back from CPU to GPU
- Allows for "long-range attention" where the model can recall information from much earlier in the context

### Resource Management

The implementation includes proper resource cleanup to prevent memory leaks:

```python
# Call this when done to release resources
cache.cleanup()
```

The cleanup method:
- Shuts down thread pools
- Closes and deletes memory-mapped files
- Releases CPU caches and quantization data
- Clears token importance tracking

## Performance Benchmarks

Relative inference speed and memory usage compared to standard KV cache:

| Configuration | Memory Usage | Inference Speed | Context Retention |
|---------------|--------------|-----------------|-------------------|
| Standard KV Cache | 100% | 1.0x | Base |
| Fixed-Size (attention) | 35% | 0.95x | Good |
| Fixed-Size + CPU Offload | 35% | 0.85x | Excellent |
| Fixed-Size + Quantization (8-bit) | 22% | 0.92x | Good |
| Fixed-Size + Quantization (4-bit) | 15% | 0.90x | Good |
| Fixed-Size + All Optimizations | 15% | 0.82x | Excellent |

*Note: Actual performance may vary based on hardware, model size, and specific workloads.*

## Project Structure

The project has a modular architecture for maintainability and extensibility:

- `fixed_size_kv_cache/cache/base_cache.py`: Core implementation of FixedSizeDynamicCache
- `fixed_size_kv_cache/cache/config.py`: Configuration handling
- `fixed_size_kv_cache/cache/offload.py`: CPU offloading functionality
- `fixed_size_kv_cache/cache/importance.py`: Token importance tracking
- `fixed_size_kv_cache/cache/truncation.py`: Cache truncation strategies
- `fixed_size_kv_cache/cache/attention.py`: Attention computation utilities
- `fixed_size_kv_cache/cache/quantization.py`: Quantization utilities
- `fixed_size_kv_cache/llama_attention.py`: Integration with LLaMA attention mechanism

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.