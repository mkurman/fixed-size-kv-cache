"""
Truncation strategies for the fixed-size KV cache.
"""

import torch
from typing import Tuple, Optional


def select_tokens_to_keep(
    attn_weights: torch.Tensor,
    saved_seq_len: int,
    batch_size: int,
    num_heads: int,
    max_cache_size: int,
    include_skipped: bool = False,
    strategy: str = "attention",
    hybrid_split_ratio: float = 0.5,
    is_incremental: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Select which tokens to keep based on the truncation strategy.
    
    Args:
        attn_weights: Attention weights tensor
        saved_seq_len: Current sequence length in the cache
        batch_size: Batch size
        num_heads: Number of attention heads
        max_cache_size: Maximum size of the cache
        include_skipped: Whether to include skipped tokens
        strategy: The truncation strategy ("attention" or "hybrid")
        hybrid_split_ratio: Ratio of attention-based vs recency-based tokens
        is_incremental: Whether this is an incremental update (token-by-token generation)
        
    Returns:
        Tuple of (indices of tokens to keep, mask of tokens to skip)
    """
    topk = min(saved_seq_len, max_cache_size)
    
    # If we don't need to truncate, return early
    if topk >= saved_seq_len:
        if include_skipped:
            all_indices = torch.arange(saved_seq_len, device=attn_weights.device)
            all_indices = all_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
            # Create mask with all False (no tokens to skip)
            mask = torch.zeros_like(all_indices, dtype=torch.bool)
            return all_indices, mask
        else:
            return None, None
    
    # Different truncation strategies
    if strategy == "attention":
        return _select_attention_based(
            attn_weights, saved_seq_len, batch_size, num_heads, topk, include_skipped, is_incremental
        )
    elif strategy == "hybrid":
        return _select_hybrid(
            attn_weights, saved_seq_len, batch_size, num_heads, topk, 
            include_skipped, hybrid_split_ratio, is_incremental
        )
    else:
        raise ValueError(f"Unknown truncation strategy: {strategy}")


def _select_attention_based(
    attn_weights: torch.Tensor,
    saved_seq_len: int,
    batch_size: int,
    num_heads: int,
    topk: int,
    include_skipped: bool = False,
    is_incremental: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Select tokens to keep based on attention weights.
    
    Args:
        attn_weights: Attention weights tensor
        saved_seq_len: Current sequence length in the cache
        batch_size: Batch size
        num_heads: Number of attention heads
        topk: Number of tokens to keep
        include_skipped: Whether to include skipped tokens
        is_incremental: Whether this is an incremental update
        
    Returns:
        Tuple of (indices of tokens to keep, mask of tokens to skip)
    """
    device = attn_weights.device
    
    if is_incremental:
        # For incremental updates with attention strategy:
        # 1. Always keep the newest token
        # 2. Drop the least important token based on attention scores
        
        # Compute attention for all tokens
        attn_scores = attn_weights.abs().sum(dim=-2)
        
        # Find the most important tokens to keep
        newest_token_idx = saved_seq_len - 1
        
        # Create a mask to exclude the newest token from attention scoring
        exclude_newest = torch.ones((batch_size, num_heads, saved_seq_len), device=device, dtype=torch.bool)
        exclude_newest[:, :, newest_token_idx] = False
        
        # Get attention scores for all tokens except the newest
        masked_attn_scores = attn_scores.clone()
        masked_attn_scores.masked_fill_(~exclude_newest, float('-inf'))
        
        # Select topk-1 most important tokens (excluding newest)
        _, important_indices = masked_attn_scores.topk(
            topk - 1 - (1 if include_skipped else 0),
            dim=-1,
            sorted=False
        )
        
        # Add the newest token
        newest_token = torch.full((batch_size, num_heads, 1), newest_token_idx, device=device)
        topk_indices = torch.cat([important_indices, newest_token], dim=-1)
    else:
        # For non-incremental updates, use regular attention-based selection
        _, topk_indices = (
            attn_weights.abs().sum(dim=-2)
            .topk(topk - (1 if include_skipped else 0), dim=-1, sorted=False)
        )
        
    # Create mask of skipped tokens if needed
    if include_skipped:
        all_indices = torch.arange(saved_seq_len, device=attn_weights.device)
        all_indices = all_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        
        # Sort indices to ensure consistent ordering
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=-1)
        
        # Create mask: True means the token will be skipped (not included in topk)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask.scatter_(-1, topk_indices_sorted, False)
    else:
        mask = None
        
    return topk_indices, mask


def _select_hybrid(
    attn_weights: torch.Tensor,
    saved_seq_len: int,
    batch_size: int,
    num_heads: int,
    topk: int,
    include_skipped: bool = False,
    hybrid_split_ratio: float = 0.5,
    is_incremental: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Select tokens to keep based on hybrid strategy (attention + recency).
    
    Args:
        attn_weights: Attention weights tensor
        saved_seq_len: Current sequence length in the cache
        batch_size: Batch size
        num_heads: Number of attention heads
        topk: Number of tokens to keep
        include_skipped: Whether to include skipped tokens
        hybrid_split_ratio: Ratio of attention-based vs recency-based tokens
        is_incremental: Whether this is an incremental update
        
    Returns:
        Tuple of (indices of tokens to keep, mask of tokens to skip)
    """
    device = attn_weights.device
    
    if is_incremental:
        # For incremental updates, we primarily want to preserve existing tokens
        # and just decide whether to keep the new token based on attention
        
        # Always keep the newest token
        newest_token_idx = saved_seq_len - 1
        
        # Get the rest of the tokens (all previous cache tokens except one to remove)
        num_to_keep = topk - 1 - (1 if include_skipped else 0)
        
        # Compute attention scores for all tokens except the newest
        attn_scores = attn_weights.abs().sum(dim=-2)
        current_scores = attn_scores[:, :, :newest_token_idx]
        
        # Find the least important token to remove
        _, least_important_indices = current_scores.topk(
            num_to_keep, 
            dim=-1, 
            largest=True,  # Keep the most important ones
            sorted=False
        )
        
        # Combine with the newest token
        newest_token = torch.full((batch_size, num_heads, 1), newest_token_idx, device=device)
        topk_indices = torch.cat([least_important_indices, newest_token], dim=-1)
    else:
        # For non-incremental updates (e.g., first pass), use the hybrid strategy normally
        attention_count = max(1, int(topk * hybrid_split_ratio))
        # Ensure we don't exceed topk - 1 (to account for possible skipped token)
        attention_count = min(attention_count, topk - (1 if include_skipped else 0) - 1)
        
        # Get attention-based tokens
        _, attn_indices = (
            attn_weights.abs().sum(dim=-2)
            .topk(attention_count, dim=-1, sorted=False)
        )
        
        # Get recent tokens
        recency_count = topk - attention_count - (1 if include_skipped else 0)
        recent_indices = torch.arange(
            saved_seq_len - recency_count,
            saved_seq_len,
            device=device
        )
        recent_indices = recent_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        
        # Combine indices, ensuring no duplicates
        topk_indices = torch.cat([attn_indices, recent_indices], dim=-1)
        topk_indices = torch.unique(topk_indices, dim=-1)
    
    # If we removed duplicates, we need to add more tokens
    if topk_indices.size(-1) < topk - (1 if include_skipped else 0):
        missing = topk - (1 if include_skipped else 0) - topk_indices.size(-1)
        all_indices = torch.arange(saved_seq_len, device=device)
        all_indices = all_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, False)
        
        # Instead of reshaping masked indices (which can lose position information),
        # we'll create a flat attention score for all tokens and use it to find additional tokens
        
        # Create a full attention score tensor for all tokens
        attn_scores = attn_weights.abs().sum(dim=-2)  # [batch_size, num_heads, seq_len]
        
        # Create a mask for tokens that are already selected
        selected_mask = torch.zeros_like(all_indices, dtype=torch.bool)
        selected_mask.scatter_(-1, topk_indices, True)
        
        # Set attention scores of already selected tokens to a very low value
        masked_attn_scores = attn_scores.clone()
        masked_attn_scores.masked_fill_(selected_mask, float('-inf'))
        
        # Select additional tokens based on attention scores
        _, additional_indices = masked_attn_scores.topk(
            min(missing, saved_seq_len - topk_indices.size(-1)), 
            dim=-1, 
            sorted=False
        )
            
        # Combine original indices with additional indices
        topk_indices = torch.cat([topk_indices, additional_indices], dim=-1)
        
        # Ensure no duplicates
        topk_indices = torch.unique(topk_indices, dim=-1)
        topk_indices = torch.sort(topk_indices, dim=-1)[0]
    
    # Create mask of skipped tokens if needed
    if include_skipped:
        all_indices = torch.arange(saved_seq_len, device=attn_weights.device)
        all_indices = all_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        
        # Sort indices to ensure consistent ordering
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=-1)
        
        # Create mask: True means the token will be skipped (not included in topk)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask.scatter_(-1, topk_indices_sorted, False)
    else:
        mask = None
        
    return topk_indices, mask


def compute_skipped_states(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute aggregated skipped states if include_skipped is True.
    
    Args:
        key_states: Full key states tensor
        value_states: Full value states tensor
        mask: Boolean mask of tokens to skip
        batch_size: Batch size
        num_heads: Number of attention heads
        
    Returns:
        Tuple of (aggregated key states, aggregated value states)
    """
    device = key_states.device
    all_indices = torch.arange(key_states.shape[2], device=device)
    all_indices = all_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
    
    # Check if there are any skipped tokens
    if not mask.any():
        # No tokens to skip, return empty tensors
        empty_key = torch.zeros(
            (batch_size, num_heads, 0, key_states.size(-1)), 
            device=device, 
            dtype=key_states.dtype
        )
        empty_value = torch.zeros(
            (batch_size, num_heads, 0, value_states.size(-1)), 
            device=device, 
            dtype=value_states.dtype
        )
        return empty_key, empty_value
    
    # Create indices for all skipped tokens (where mask is True)
    # We'll compute average embeddings by batch and head, preserving position info
    
    # Initialize accumulators for each batch and head
    num_dim_k = key_states.size(-1)
    num_dim_v = value_states.size(-1)
    sum_key = torch.zeros((batch_size, num_heads, 1, num_dim_k), device=device, dtype=key_states.dtype)
    sum_value = torch.zeros((batch_size, num_heads, 1, num_dim_v), device=device, dtype=value_states.dtype)
    count = torch.zeros((batch_size, num_heads, 1, 1), device=device, dtype=torch.float32)
    
    # Process each batch and head separately
    for b in range(batch_size):
        for h in range(num_heads):
            # Get indices of skipped tokens
            skipped_indices = torch.nonzero(mask[b, h]).squeeze(-1)
            
            if skipped_indices.size(0) > 0:
                # Gather key and value states for skipped tokens
                skipped_keys = torch.index_select(key_states[b, h], 0, skipped_indices)
                skipped_values = torch.index_select(value_states[b, h], 0, skipped_indices)
                
                # Sum them up
                sum_key[b, h, 0] = skipped_keys.sum(dim=0)
                sum_value[b, h, 0] = skipped_values.sum(dim=0)
                count[b, h, 0, 0] = skipped_indices.size(0)
    
    # Average the states (avoid division by zero)
    mask_for_div = (count > 0).to(dtype=key_states.dtype)
    avg_key = sum_key / (count.clamp(min=1) * mask_for_div)
    avg_value = sum_value / (count.clamp(min=1) * mask_for_div)
    
    # If all counts are zero, return empty tensors
    if not mask_for_div.any():
        empty_key = torch.zeros(
            (batch_size, num_heads, 0, key_states.size(-1)), 
            device=device, 
            dtype=key_states.dtype
        )
        empty_value = torch.zeros(
            (batch_size, num_heads, 0, value_states.size(-1)), 
            device=device, 
            dtype=value_states.dtype
        )
        return empty_key, empty_value
        
    # The aggregated (averaged) key and value states are our skipped states
    # They already have the correct shape: [batch_size, num_heads, 1, hidden_dim]
    key_states_skipped = avg_key
    value_states_skipped = avg_value
    
    return key_states_skipped, value_states_skipped