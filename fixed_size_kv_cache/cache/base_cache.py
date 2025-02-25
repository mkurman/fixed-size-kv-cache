"""
Base implementation of the fixed-size dynamic cache.
"""

import torch
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Union
from torch import autocast

from transformers.cache_utils import DynamicCache

from .config import CacheConfig, load_config_from_env
from .offload import OffloadManager
from .importance import ImportanceManager
from .truncation import select_tokens_to_keep, compute_skipped_states
from .attention import compute_attention_weights

logger = logging.getLogger(__name__)


@dataclass
class FixedSizeDynamicCache(DynamicCache):
    """
    A dynamic cache that intelligently manages key and value states based on attention weights,
    with optional offloading of less-used tokens to CPU memory and on-the-fly quantization.
    """
    
    # Statistics for monitoring
    truncation_stats: Dict[str, List[int]] = field(default_factory=lambda: {"truncated_sizes": []})
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load configuration
        config_kwargs = {k: v for k, v in kwargs.items() if hasattr(CacheConfig(), k)}
        env_config = load_config_from_env()
        config = CacheConfig(**config_kwargs)
        
        # Override with env config where not explicitly set
        for key, value in vars(env_config).items():
            if key not in kwargs and hasattr(config, key):
                setattr(config, key, value)
                
        # Store configuration
        self.config = config
        
        # Initialize managers
        self.offload_manager = (
            OffloadManager(
                offload_size=config.offload_size,
                use_mmap=config.use_mmap,
                quantize=config.quantize,
                quantization_bits=config.quantization_bits
            )
            if config.offload_to_cpu
            else None
        )
        
        self.importance_manager = (
            ImportanceManager(
                token_importance_window=config.token_importance_window,
                importance_threshold=config.importance_threshold,
                parallel_processing=config.parallel_processing,
                worker_threads=config.worker_threads,
            )
            if config.offload_to_cpu and config.auto_restore
            else None
        )
        
        # Initialize statistics
        self.truncation_stats = {"truncated_sizes": []}
        
        logger.debug(
            f"FixedSizeDynamicCache initialized with:"
            f" kv_cache_size={config.kv_cache_size},"
            f" strategy={config.truncation_strategy},"
            f" offload_to_cpu={config.offload_to_cpu},"
            f" quantize={config.quantize if config.offload_to_cpu else 'N/A'}"
            f" adaptive_precision={config.adaptive_precision if config.offload_to_cpu else 'N/A'}"
            f" auto_restore={config.auto_restore if config.offload_to_cpu else 'N/A'}"
            f" include_skipped={config.include_skipped if config.offload_to_cpu else 'N/A'}"
            f" free_memory={config.free_memory if config.offload_to_cpu else 'N/A'}"
            f" use_mmap={config.use_mmap if config.offload_to_cpu else 'N/A'}"
            f" hybrid_split_ratio={config.hybrid_split_ratio if config.offload_to_cpu else 'N/A'}"
            f" token_importance_window={config.token_importance_window if config.offload_to_cpu and config.auto_restore else 'N/A'}"
            f" importance_threshold={config.importance_threshold if config.offload_to_cpu and config.auto_restore else 'N/A'}"
            f" parallel_processing={config.parallel_processing if config.offload_to_cpu and config.auto_restore else 'N/A'}"
            f" worker_threads={config.worker_threads if config.offload_to_cpu and config.auto_restore else 'N/A'}"
        )
        
    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about the cache truncation and optimizations."""
        stats = {
            "truncation_count": len(self.truncation_stats["truncated_sizes"]),
            "avg_truncated_size": sum(self.truncation_stats["truncated_sizes"]) / len(self.truncation_stats["truncated_sizes"]) 
            if self.truncation_stats["truncated_sizes"] else 0,
            "strategy": self.config.truncation_strategy,
            "offload_enabled": self.config.offload_to_cpu,
            "quantization": self.config.quantize,
            "adaptive_precision": self.config.adaptive_precision
        }
        
        # Add offload manager stats if available
        if self.offload_manager is not None:
            offload_stats = {
                "cpu_cache_used": bool(self.offload_manager.cpu_key_cache),
                "cpu_cache_layers": len(self.offload_manager.cpu_key_cache),
                "use_mmap": self.config.use_mmap
            }
            stats.update(offload_stats)
            
        # Add importance manager stats if available
        if self.importance_manager is not None:
            importance_stats = {
                "tracked_tokens": len(self.importance_manager.token_importance),
                "parallel_processing": self.config.parallel_processing
            }
            stats.update(importance_stats)
            
        return stats
        
    def cache_truncate(
        self,
        layer_idx: int,
        query_states: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        restore_positions: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Truncate the KV cache based on attention weights or specified strategy.
        
        Args:
            layer_idx: Index of the transformer layer
            query_states: Optional tensor of query states
            key_states: Optional tensor of key states
            attn_weights: Optional tensor of attention weights
            attention_mask: Optional attention mask
            restore_positions: Optional list or tensor of positions to restore from CPU cache
            
        Returns:
            Tuple of (truncated key states, truncated value states) or None
        """
        # Get cached KV states for this layer
        key_states_cached, value_states_cached = self[layer_idx]
        
        # Apply adaptive precision if enabled
        if self.config.adaptive_precision and hasattr(torch, 'autocast'):
            # Check if we're already in a mixed precision context
            in_autocast = torch.is_autocast_enabled()
            
            if not in_autocast:
                # Use FP16 precision for operations that don't need high precision
                with autocast(device_type=key_states_cached.device.type, dtype=torch.float16):
                    return self._cache_truncate_impl(
                        layer_idx, query_states, key_states, attn_weights, 
                        attention_mask, restore_positions
                    )
            else:
                # We're already in a mixed precision context, proceed normally
                return self._cache_truncate_impl(
                    layer_idx, query_states, key_states, attn_weights, 
                    attention_mask, restore_positions
                )
        else:
            # No adaptive precision, use normal implementation
            return self._cache_truncate_impl(
                layer_idx, query_states, key_states, attn_weights, 
                attention_mask, restore_positions
            )
        
    def _cache_truncate_impl(
        self,
        layer_idx: int,
        query_states: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        restore_positions: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Implementation of cache truncation with all optimizations.
        
        Args:
            layer_idx: Index of the transformer layer
            query_states: Optional tensor of query states
            key_states: Optional tensor of key states
            attn_weights: Optional tensor of attention weights
            attention_mask: Optional attention mask
            restore_positions: Optional list or tensor of positions to restore from CPU cache
            
        Returns:
            Tuple of (truncated key states, truncated value states) or None
        """
        # Get cached KV states for this layer
        key_states_cached, value_states_cached = self[layer_idx]
        
        # Handle explicit restoration from CPU cache
        if self.config.offload_to_cpu and restore_positions is not None and self.offload_manager is not None:
            self._restore_from_cpu(layer_idx, restore_positions)
            # Re-get the cached states since they might have been updated
            key_states_cached, value_states_cached = self[layer_idx]
            
        # Auto-restore important tokens if enabled
        if (self.config.offload_to_cpu and self.config.auto_restore and 
            self.offload_manager is not None and self.importance_manager is not None):
            self._auto_restore_important_tokens(layer_idx)
            # Re-get the cached states since they might have been updated
            key_states_cached, value_states_cached = self[layer_idx]
        
        # Check if we need to truncate
        saved_seq_len = key_states_cached.shape[2]
        
        # Detect if this is an incremental update (token-by-token generation)
        is_incremental = saved_seq_len == self.config.kv_cache_size + 1
        
        if saved_seq_len <= self.config.kv_cache_size:
            return self[layer_idx]
            
        # Compute attention weights if not provided
        if attn_weights is None:
            if query_states is None or key_states is None:
                return self[layer_idx]
                
            attn_weights = compute_attention_weights(
                query_states, key_states, attention_mask
            )
        
        batch_size, num_heads = key_states_cached.shape[0], key_states_cached.shape[1]
        
        # Select tokens to keep based on truncation strategy
        topk_indices, mask = select_tokens_to_keep(
            attn_weights[:, :key_states_cached.shape[1]], 
            saved_seq_len,
            batch_size,
            num_heads,
            self.config.kv_cache_size,
            self.config.include_skipped,
            self.config.truncation_strategy,
            self.config.hybrid_split_ratio,
            is_incremental
        )
        
        # If topk_indices is None, no truncation needed
        if topk_indices is None:
            return self[layer_idx]
            
        # Gather the top-k key and value states
        key_states_topk = torch.gather(
            key_states_cached,
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, key_states_cached.size(-1)),
        )
        
        value_states_topk = torch.gather(
            value_states_cached,
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, value_states_cached.size(-1)),
        )
        
        # Update token importance based on attention patterns
        if (attn_weights is not None and self.config.offload_to_cpu and 
            self.importance_manager is not None):
            device = key_states_cached.device
            positions = torch.arange(saved_seq_len, device=device)
            positions = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
            self.importance_manager.update_token_importance(layer_idx, positions, attn_weights)
            
        # Identify and track skipped tokens
        if mask is not None:
            # Get the original position indices for each token in the sequence
            device = key_states_cached.device
            positions = torch.arange(saved_seq_len, device=device)
            positions = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
            
            # If offloading to CPU is enabled, save the skipped tokens
            if self.config.offload_to_cpu and self.offload_manager is not None:
                # Process skipped tokens by batch and head
                skipped_keys_list = []
                skipped_values_list = []
                skipped_positions_list = []
                
                for b in range(batch_size):
                    for h in range(num_heads):
                        # Find indices of skipped tokens for this batch and head
                        skipped_indices = torch.nonzero(mask[b, h]).squeeze(-1)
                        
                        if skipped_indices.numel() > 0:
                            # Extract the skipped key and value states
                            skipped_keys = key_states_cached[b:b+1, h:h+1, skipped_indices]
                            skipped_values = value_states_cached[b:b+1, h:h+1, skipped_indices]
                            
                            # Extract the positions
                            skipped_pos = positions[b:b+1, h:h+1, skipped_indices]
                            
                            # Add to lists
                            skipped_keys_list.append(skipped_keys)
                            skipped_values_list.append(skipped_values)
                            skipped_positions_list.append(skipped_pos)
                
                # Combine all skipped tokens if any were found
                if skipped_keys_list:
                    skipped_keys = torch.cat(skipped_keys_list, dim=2)
                    skipped_values = torch.cat(skipped_values_list, dim=2)
                    skipped_positions = torch.cat(skipped_positions_list, dim=2)
                    
                    # Offload to CPU with quantization if enabled
                    self.offload_manager.offload_to_cpu(
                        layer_idx, skipped_keys, skipped_values, skipped_positions
                    )
                
        # If we need to include skipped tokens, compute their aggregated states
        if self.config.include_skipped and mask is not None:
            key_states_skipped, value_states_skipped = compute_skipped_states(
                key_states_cached,
                value_states_cached,
                mask,
                batch_size,
                num_heads
            )
            
            # Only append skipped states if they're not empty
            if key_states_skipped.size(2) > 0:
                key_states_updated = torch.cat([key_states_topk, key_states_skipped], dim=2)
                value_states_updated = torch.cat([value_states_topk, value_states_skipped], dim=2)
            else:
                key_states_updated = key_states_topk
                value_states_updated = value_states_topk
        else:
            key_states_updated = key_states_topk
            value_states_updated = value_states_topk
        
        # Update cache
        self.key_cache[layer_idx] = key_states_updated
        self.value_cache[layer_idx] = value_states_updated
        
        # Update statistics
        self.truncation_stats["truncated_sizes"].append(saved_seq_len - key_states_updated.shape[2])
        
        # Free memory if requested
        if self.config.free_memory:
            torch.cuda.empty_cache()
        
        logger.debug(
            f"Truncated layer {layer_idx} cache from {saved_seq_len} to {key_states_updated.shape[2]} tokens "
            f"(incremental update: {is_incremental})"
        )
        
        return self[layer_idx]
    
    def _restore_from_cpu(self, layer_idx: int, positions: Union[List[int], torch.Tensor]) -> bool:
        """Restore tokens from CPU cache and add them to the current GPU cache.
        
        Args:
            layer_idx: The layer index
            positions: The positions to restore
            
        Returns:
            True if at least one token was restored, False otherwise
        """
        if not self.config.offload_to_cpu or self.offload_manager is None:
            return False
            
        # Convert positions to list if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.tolist()
            
        # Get tokens from CPU cache
        restored_tokens = self.offload_manager.search_cpu_cache(layer_idx, positions)
        if restored_tokens is None:
            return False
            
        # Unpack the tokens
        key_states, value_states = restored_tokens
        
        # Get the current cache
        current_key, current_value = self[layer_idx]
        
        # Append the restored tokens to the cache
        updated_key = torch.cat([current_key, key_states], dim=2)
        updated_value = torch.cat([current_value, value_states], dim=2)
        
        # Update the cache
        self.key_cache[layer_idx] = updated_key
        self.value_cache[layer_idx] = updated_value
        
        logger.info(f"Restored {key_states.shape[2]} tokens from CPU cache for layer {layer_idx}")
        return True
        
    def _auto_restore_important_tokens(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Automatically restore important tokens based on attention patterns.
        
        Args:
            layer_idx: The layer index
            
        Returns:
            Updated cache or None if no tokens were restored
        """
        if (not self.config.offload_to_cpu or not self.config.auto_restore or 
            self.offload_manager is None or self.importance_manager is None):
            return None
            
        # Get current positions in the cache
        key_states, _ = self[layer_idx]
        batch_size, num_heads, seq_len = key_states.shape[:3]
        device = key_states.device
        
        # Create position tensor
        positions = torch.arange(seq_len, device=device)
        positions = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        
        # Find important tokens to restore
        important_positions = self.importance_manager.find_important_tokens(layer_idx, positions)
        
        # Restore tokens if any important ones found
        if important_positions and len(important_positions) > 0:
            return self._restore_from_cpu(layer_idx, important_positions)
            
        return None
        
    def cleanup(self):
        """Clean up resources used by the cache."""
        # Clean up offload manager
        if self.offload_manager is not None:
            self.offload_manager.cleanup()
            
        # Clean up importance manager
        if self.importance_manager is not None:
            self.importance_manager.cleanup()
            
        logger.debug("Cache resources cleaned up")
        
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()
        
    @classmethod
    def from_legacy_cache(cls, cache, layer_idx=None):
        """Create a FixedSizeDynamicCache from a legacy cache.
        
        Args:
            cache: The original cache to convert
            layer_idx: Optional layer index to convert only a specific layer
            
        Returns:
            A FixedSizeDynamicCache instance with the same content
        """
        # If the cache is already a FixedSizeDynamicCache, return it
        if isinstance(cache, cls):
            return cache
            
        # Create a new instance
        instance = cls()
        
        # Copy the key and value caches
        if hasattr(cache, 'key_cache'):
            instance.key_cache = cache.key_cache.copy()
        
        if hasattr(cache, 'value_cache'):
            instance.value_cache = cache.value_cache.copy()
            
        # Preserve any other attributes if the cache is already a DynamicCache
        if isinstance(cache, DynamicCache):
            # Copy any additional attributes that might be in the parent class
            for attr_name in dir(cache):
                # Skip private attributes, methods, and properties
                if (attr_name.startswith('_') or callable(getattr(cache, attr_name)) or 
                    isinstance(getattr(type(cache), attr_name, None), property)):
                    continue
                
                # Skip key_cache and value_cache as they're already handled
                if attr_name in ('key_cache', 'value_cache'):
                    continue
                    
                # Get the attribute value from the original cache
                attr_value = getattr(cache, attr_name)
                
                # Set the attribute in our new instance
                if hasattr(instance, attr_name):
                    # For collections, we want to update rather than replace if possible
                    if isinstance(attr_value, dict) and isinstance(getattr(instance, attr_name), dict):
                        getattr(instance, attr_name).update(attr_value)
                    elif isinstance(attr_value, list) and isinstance(getattr(instance, attr_name), list):
                        getattr(instance, attr_name).extend(attr_value)
                    else:
                        setattr(instance, attr_name, attr_value)
                else:
                    # Direct copy for attributes we don't already have
                    setattr(instance, attr_name, attr_value)
                    
        return instance