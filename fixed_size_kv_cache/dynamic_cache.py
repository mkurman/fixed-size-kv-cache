from transformers.cache_utils import DynamicCache
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from torch import nn
import torch
import math
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class FixedSizeDynamicCache(DynamicCache):
    """
    A dynamic cache that truncates the key and value states based on the attention weights.
    This cache behavior is configurable via environment variables.
    - FSDC_KV_CACHE_SIZE: The maximum size of the key and value cache. (default: 1024)
    - FSDC_INCLUDE_SKIPPED: Whether to include the sum of skipped key and value states in the cache as the last entry. (default: False)
    - FSDC_FREE_MEMORY: Whether to free memory after truncating the cache. (default: False)
    """

    kv_cache_size: int = 1024
    include_skipped: bool = False
    free_memory: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache_size = int(os.getenv("FSDC_KV_CACHE_SIZE", self.kv_cache_size))
        self.include_skipped = bool(
            os.getenv("FSDC_INCLUDE_SKIPPED", self.include_skipped)
        )
        self.free_memory = bool(os.getenv("FSDC_FREE_MEMORY", self.free_memory))

        logger.debug(
            f"Dynamic cache initialized with kv_cache_size: {self.kv_cache_size}, include_skipped: {self.include_skipped}"
        )

    def cache_truncate(
        self,
        layer_idx: int,
        query_states: Optional[torch.Tensor],
        key_states: Optional[torch.Tensor],
        attn_weights: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ):
        # If we don't have attention weights, we need to compute them
        if attn_weights is None:
            if query_states is None and key_states is None:
                return

            # If query_states has more heads than key_states, we need to expand key_states
            if query_states.shape[1] != key_states.shape[1]:

                n_rep = query_states.shape[1] // key_states.shape[1]

                key_states = key_states[:, :, None, :, :].expand(
                    key_states.shape[0],
                    key_states.shape[1],
                    n_rep,
                    key_states.shape[-2],
                    key_states.shape[-1],
                )
                key_states = key_states.reshape(
                    key_states.shape[0],
                    key_states.shape[1] * n_rep,
                    key_states.shape[-2],
                    key_states.shape[-1],
                )

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(key_states.shape[-1])

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

        # Here is the actual truncation
        key_states_pk, value_states_pk = self[layer_idx]

        saved_seq_len = key_states_pk.shape[2]

        topk = min(saved_seq_len, self.kv_cache_size)

        if topk < self.kv_cache_size:
            return

        # We need to use the absolute value of the attention weights to determine which keys and values to keep
        # We take the topk - 1 if we are including the skipped keys and values
        _, topk_indices = (
            attn_weights.abs()[:, : key_states_pk.shape[1]]
            .sum(dim=-2)
            .topk(topk - (1 if self.include_skipped else 0), dim=-1, sorted=False)
        )

        key_states_topk = torch.gather(
            key_states_pk,
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, key_states_pk.size(-1)),
        )
        value_states_topk = torch.gather(
            value_states_pk,
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, value_states_pk.size(-1)),
        )

        if self.include_skipped and topk == saved_seq_len:
            batch_size, num_heads, seq_len, _ = key_states_pk.shape
            device = key_states_pk.device

            all_indices = (
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, num_heads, -1)
            )

            topk_indices_sorted, _ = torch.sort(topk_indices, dim=-1)

            mask = torch.ones_like(all_indices, dtype=torch.bool)
            mask.scatter_(2, topk_indices_sorted, False)

            skipped_indices = all_indices[mask].view(batch_size, num_heads, -1)

            key_states_skipped = torch.gather(
                key_states_pk,
                2,
                skipped_indices.unsqueeze(-1).expand(
                    -1, -1, -1, key_states_pk.size(-1)
                ),
            ).sum(dim=2, keepdim=True)
            value_states_skipped = torch.gather(
                value_states_pk,
                2,
                skipped_indices.unsqueeze(-1).expand(
                    -1, -1, -1, value_states_pk.size(-1)
                ),
            ).sum(dim=2, keepdim=True)

            key_states_updated = torch.cat([key_states_topk, key_states_skipped], dim=2)
            value_states_updated = torch.cat(
                [value_states_topk, value_states_skipped], dim=2
            )
        else:
            key_states_updated = key_states_topk
            value_states_updated = value_states_topk

        self.key_cache[layer_idx] = key_states_updated
        self.value_cache[layer_idx] = value_states_updated

        # Free up memory if the cache is larger than the kv_cache_size
        if self.free_memory and saved_seq_len > self.kv_cache_size:
            torch.cuda.empty_cache()
