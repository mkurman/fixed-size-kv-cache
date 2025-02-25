"""
Attention computation utilities for the fixed-size KV cache.
"""

import torch
from torch import nn
import math
from typing import Optional, Tuple


def compute_attention_weights(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute attention weights from query and key states.
    
    Args:
        query_states: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key_states: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        attention_mask: Optional attention mask
        
    Returns:
        Attention weights tensor
    """
    # If query_states has more heads than key_states, expand key_states
    if query_states.shape[1] != key_states.shape[1]:
        n_rep = query_states.shape[1] // key_states.shape[1]
        key_states = expand_key_states(key_states, n_rep)

    # Try to use Flash Attention if available and using CUDA
    try:
        from flash_attn import flash_attn_func
        
        device = query_states.device
        if device.type == 'cuda' and query_states.is_cuda and key_states.is_cuda:
            # Reshape to BSNH format required by Flash Attention
            query = query_states.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
            key = key_states.transpose(1, 2)      # [batch_size, seq_len, num_heads, head_dim]
            
            # Prepare causal mask
            use_causal = attention_mask is not None
            
            # Call Flash Attention
            attention_scores = flash_attn_func(
                query, key, key,  # q, k, v (using key as v to get attn weights)
                causal=use_causal,
                return_attn_probs=True
            )[1]
            
            # Reshape back to BNSH format
            return attention_scores.transpose(1, 2)
    except (ImportError, AttributeError):
        # Fall back to regular attention if Flash Attention is not available
        pass

    # Compute attention scores using standard method
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(key_states.shape[-1])

    # Apply attention mask if provided
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Apply softmax to get probabilities
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    
    return attn_weights


def expand_key_states(key_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key states to match query states head dimension.
    
    Args:
        key_states: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each head
        
    Returns:
        Expanded key states
    """
    return key_states[:, :, None, :, :].expand(
        key_states.shape[0],
        key_states.shape[1],
        n_rep,
        key_states.shape[-2],
        key_states.shape[-1],
    ).reshape(
        key_states.shape[0],
        key_states.shape[1] * n_rep,
        key_states.shape[-2],
        key_states.shape[-1],
    )