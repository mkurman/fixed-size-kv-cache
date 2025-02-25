"""
Token importance tracking and restoration functionality.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ImportanceManager:
    """Manages token importance tracking and restoration."""
    
    def __init__(
        self,
        token_importance_window: int = 100,
        importance_threshold: float = 0.1,
        parallel_processing: bool = True,
        worker_threads: int = 4
    ):
        """Initialize importance manager.
        
        Args:
            token_importance_window: Size of the rolling window for tracking token importance
            importance_threshold: Threshold for considering a token important
            parallel_processing: Whether to use parallel processing for token importance
            worker_threads: Number of worker threads for parallel processing
        """
        self.token_importance_window = token_importance_window
        self.importance_threshold = importance_threshold
        self.parallel_processing = parallel_processing
        self.worker_threads = worker_threads
        
        # Token importance tracking
        self.token_importance: Dict[int, Dict[int, float]] = {}  # {position: {layer_idx: importance}}
        self.token_access_history: Dict[int, List[float]] = {}  # {position: [recent_importance_scores]}
        
        # Initialize thread pool if parallel processing is enabled
        self.executor = ThreadPoolExecutor(max_workers=worker_threads) if parallel_processing else None
        
    def update_token_importance(self, layer_idx: int, positions: torch.Tensor, attn_weights: torch.Tensor) -> None:
        """Update the importance scores of tokens based on attention weights.
        
        Args:
            layer_idx: The layer index
            positions: Tensor of position indices [batch_size, num_heads, seq_len]
            attn_weights: Attention weights tensor [batch_size, num_heads, seq_len, seq_len]
        """
        # Compute importance in parallel if enabled
        if self.parallel_processing and self.executor is not None:
            batch_size, num_heads, seq_len = positions.shape
            futures = []
            
            for b in range(batch_size):
                for h in range(num_heads):
                    # Extract data for this batch and head
                    pos = positions[b, h].cpu()
                    scores = attn_weights.abs().sum(dim=-2)[b, h].cpu()
                    
                    # Submit the task to the thread pool
                    future = self.executor.submit(
                        self._update_importance_for_batch_head,
                        pos.float().numpy(), scores.float().numpy(), layer_idx
                    )
                    futures.append(future)
                    
            # Wait for all tasks to complete
            for future in futures:
                future.result()
        else:
            # Get importance scores from attention (sum across heads and queries)
            # We use attention to the token as a proxy for its importance
            attn_sum = attn_weights.abs().sum(dim=-2)  # [batch_size, num_heads, seq_len]
            
            # For each position, update its importance score
            batch_size, num_heads, seq_len = positions.shape
            
            # Process by batch and head for efficiency
            for b in range(batch_size):
                for h in range(num_heads):
                    # Get positions and scores for this batch and head
                    pos = positions[b, h].cpu()
                    scores = attn_sum[b, h].cpu()
                    
                    # Update importance for each position
                    self._update_importance_for_batch_head(pos.numpy(), scores.numpy(), layer_idx)
        
        # Clean up old entries if we're tracking too many positions
        if len(self.token_importance) > self.token_importance_window * 20:
            # Sort positions by average recent importance
            positions_to_keep = sorted(
                self.token_importance.keys(),
                key=lambda p: sum(self.token_access_history.get(p, [])) / max(1, len(self.token_access_history.get(p, []))),
                reverse=True
            )[:self.token_importance_window * 10]
            
            # Remove positions that aren't in the top positions_to_keep
            positions_to_remove = [p for p in list(self.token_importance.keys()) if p not in positions_to_keep]
            for p in positions_to_remove:
                if p in self.token_importance:
                    del self.token_importance[p]
                if p in self.token_access_history:
                    del self.token_access_history[p]
                    
    def _update_importance_for_batch_head(self, positions: np.ndarray, scores: np.ndarray, layer_idx: int) -> None:
        """Update token importance scores for a single batch and head (CPU-only operation).
        
        Args:
            positions: NumPy array of positions
            scores: NumPy array of attention scores
            layer_idx: The layer index
        """
        # Update importance for each position
        for i in range(len(positions)):
            position = int(positions[i])
            importance = float(scores[i])
            
            # Initialize if needed
            if position not in self.token_importance:
                self.token_importance[position] = {}
                self.token_access_history[position] = []
            
            # Update layer-specific importance
            self.token_importance[position][layer_idx] = importance
            
            # Update history (append and keep only the last window_size entries)
            self.token_access_history[position].append(importance)
            if len(self.token_access_history[position]) > self.token_importance_window:
                self.token_access_history[position] = self.token_access_history[position][-self.token_importance_window:]
                
    def find_important_tokens(self, layer_idx: int, current_positions: torch.Tensor) -> Optional[List[int]]:
        """Find important tokens to restore based on attention history.
        
        Args:
            layer_idx: The layer index
            current_positions: Tensor of current positions in the cache
            
        Returns:
            List of positions to restore or None if no important tokens found
        """
        # Convert current positions to a set for fast lookup
        current_pos_set = set()
        for b in range(current_positions.shape[0]):
            for h in range(current_positions.shape[1]):
                current_pos_set.update(current_positions[b, h].cpu().tolist())
        
        # Find positions with high importance that aren't in the current cache
        important_positions = []
        
        for pos, layers in self.token_importance.items():
            # Skip if already in the cache
            if pos in current_pos_set:
                continue
                
            # Skip if we don't have importance for this layer
            if layer_idx not in layers:
                continue
                
            # Calculate average importance from history
            avg_importance = sum(self.token_access_history.get(pos, [])) / max(1, len(self.token_access_history.get(pos, [])))
            
            # If above threshold, add to important positions
            if avg_importance > self.importance_threshold:
                important_positions.append((pos, avg_importance))
        
        # Return top positions sorted by importance (up to 10% of cache size)
        if important_positions:
            max_to_restore = max(1, 100)  # Maximum number to restore
            sorted_positions = sorted(important_positions, key=lambda x: x[1], reverse=True)[:max_to_restore]
            return [pos for pos, _ in sorted_positions]
            
        return None
        
    def cleanup(self) -> None:
        """Clean up resources used by the importance manager."""
        # Close thread pool if it exists
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
            
        # Clear importance tracking
        self.token_importance = {}
        self.token_access_history = {}