import torch
from torch import nn
from typing import Optional, Tuple, Callable
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from fixed_size_kv_cache import FixedSizeDynamicCache
import logging
import os
import types

logger = logging.getLogger(__name__)


def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    # ---------------------------------------------------------
    # Fixed size kv cache

    if past_key_value is not None:
        # Handle case where past_key_value is a DynamicCache but not FixedSizeDynamicCache
        # We need to "convert" the instance to FixedSizeDynamicCache to enable all features
        if isinstance(past_key_value, DynamicCache) and not isinstance(past_key_value, FixedSizeDynamicCache):
            # Store original key and value cache data
            original_key_cache = past_key_value.key_cache
            original_value_cache = past_key_value.value_cache
            
            # Python magic: Change the class of the existing instance
            # This preserves the object's identity while changing its behavior
            past_key_value.__class__ = FixedSizeDynamicCache
            
            # Re-initialize the instance with FixedSizeDynamicCache defaults
            FixedSizeDynamicCache.__init__(past_key_value)
            
            # Restore the original cache data
            past_key_value.key_cache = original_key_cache
            past_key_value.value_cache = original_value_cache
            
            logger.debug(f"Upgraded DynamicCache to FixedSizeDynamicCache")
            
        # Now we can use the FixedSizeDynamicCache methods
        past_key_value.key_cache[self.layer_idx], past_key_value.value_cache[self.layer_idx] = past_key_value.cache_truncate(
            self.layer_idx,
            query_states,
            key_states,
            attn_weights,
            attention_mask,
        )
    # ---------------------------------------------

    return attn_output, attn_weights
