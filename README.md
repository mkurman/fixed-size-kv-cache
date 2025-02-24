# Fixed Size Key-Value Cache

## Project Description

This project implements a fixed-size key-value cache for use in transformer models, specifically designed to work with the LLaMA model. The cache dynamically truncates the key and value states based on attention weights, ensuring efficient memory usage and performance.

![Fixed Size Key-Value Cache](image_fx_.jpg)

## Installation

To install the required dependencies, run:

```bash
pip install -U pip && pip install -e .
```

## Usage

### Setting Up FixedSizeDynamicCache

The `FixedSizeDynamicCache` class is a dynamic cache that truncates the key and value states based on the attention weights. This cache behavior is configurable via environment variables:

- `FSDC_KV_CACHE_SIZE`: The maximum size of the key and value cache. (default: 1024)
- `FSDC_INCLUDE_SKIPPED`: Whether to include the sum of skipped key and value states in the cache as the last entry. (default: False)
- `FSDC_FREE_MEMORY`: Whether to free memory after truncating the cache. (default: False)

To set up the cache, you need to initialize it and configure the environment variables as needed. Here is an example:

```python
import os
from transformers import LlamaForCausalLM
from fixed_size_kv_cache.dynamic_cache import FixedSizeDynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention
from fixed_size_kv_cache.llama_attention import forward

# Set environment variables
os.environ['FSDC_KV_CACHE_SIZE'] = '2048'
os.environ['FSDC_INCLUDE_SKIPPED'] = 'True'
os.environ['FSDC_FREE_MEMORY'] = 'True'

# Replace forward function
LlamaAttention.forward = forward

# Set up FixedSizeDynamicCache
transformers.models.llama.modeling_llama.DynamicCache = FixedSizeDynamicCache
LlamaForCausalLM.__init__.__globals__['DynamicCache'] = FixedSizeDynamicCache

# Load model
# ...existing code...

# Change attention to eager (this is a workaround as flash/sdpa attention does not return attention weights)
model.config._attn_implementation = "eager"

```

### Using the Cache in LLaMA Attention

The `FixedSizeDynamicCache` is integrated into the LLaMA model's attention mechanism. The `forward` method in `llama_attention.py` demonstrates how the cache is used:

```python
from fixed_size_kv_cache.dynamic_cache import FixedSizeDynamicCache

# ...existing code...

def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # ...existing code...

    if past_key_value is not None:
        past_key_value.key_cache[self.layer_idx], past_key_value.value_cache[self.layer_idx] = FixedSizeDynamicCache.from_legacy_cache(
            past_key_value, self.layer_idx
        ).cache_truncate(
            self.layer_idx,
            query_states,
            key_states,
            attn_weights,
            attention_mask,
        )
    # ...existing code...

    return attn_output, attn_weights
```

### Workarounds and Considerations

- **Attention Weights Calculation**: If attention weights are not provided, they need to be computed using the query and key states. This is handled within the `cache_truncate` method.
- **Key States Expansion**: If the number of heads in the query states is greater than in the key states, the key states need to be expanded to match the number of heads in the query states.
- **Memory Management**: If the `FSDC_FREE_MEMORY` environment variable is set to `True`, memory will be freed after truncating the cache to avoid memory leaks.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
