"""
CPU offloading functionality for the fixed-size KV cache.
"""

import os
import torch
import mmap
import pickle
import tempfile
from typing import Dict, List, Any, Tuple

from .quantization import quantize_tensor, dequantize_tensor


class OffloadManager:
    """Manages offloading KV cache tokens to CPU memory."""
    
    def __init__(
        self, 
        offload_size: int, 
        use_mmap: bool = False, 
        quantize: bool = False, 
        quantization_bits: int = 8
    ):
        """Initialize the offload manager.
        
        Args:
            offload_size: Maximum number of tokens to store in CPU memory
            use_mmap: Whether to use memory mapping for CPU storage
            quantize: Whether to quantize tokens before offloading
            quantization_bits: Number of bits to use for quantization (4 or 8)
        """
        self.offload_size = offload_size
        self.use_mmap = use_mmap
        self.quantize = quantize
        self.quantization_bits = quantization_bits
        
        # CPU memory storage
        self.cpu_key_cache: Dict[int, List[torch.Tensor]] = {}
        self.cpu_value_cache: Dict[int, List[torch.Tensor]] = {}
        self.position_map: Dict[int, List[torch.Tensor]] = {}  # {layer_idx: [position_tensors]}
        
        # Memory-mapped files for CPU offloading
        self.mmap_files: Dict[int, Dict[str, str]] = {}
        self.mmap_handles: Dict[int, Dict[str, Any]] = {}
        
        # Quantization state
        self.quant_scales: Dict[int, Dict[str, List[Tuple[torch.Tensor, Tuple]]]] = {}
        self.quant_zeros: Dict[int, Dict[str, List[Tuple[torch.Tensor, Tuple]]]] = {}
        
    def offload_to_cpu(self, layer_idx: int, skipped_keys: torch.Tensor, 
                      skipped_values: torch.Tensor, positions: torch.Tensor) -> None:
        """Offload skipped tokens to CPU memory with optional quantization and memory mapping.
        
        Args:
            layer_idx: The layer index
            skipped_keys: The key states to offload
            skipped_values: The value states to offload
            positions: The original positions of the tokens
        """
        # Initialize layer caches if not present
        if layer_idx not in self.cpu_key_cache:
            self.cpu_key_cache[layer_idx] = []
            self.cpu_value_cache[layer_idx] = []
            self.position_map[layer_idx] = []
            
        # Apply quantization if enabled
        if self.quantize:
            # Quantize key states
            key_q, key_scale, key_zero = quantize_tensor(
                skipped_keys.detach(), self.quantization_bits
            )
            
            # Quantize value states
            value_q, value_scale, value_zero = quantize_tensor(
                skipped_values.detach(), self.quantization_bits
            )
            
            # Store quantization parameters
            if layer_idx not in self.quant_scales:
                self.quant_scales[layer_idx] = {"key": [], "value": []}
                self.quant_zeros[layer_idx] = {"key": [], "value": []}
                
            self.quant_scales[layer_idx]["key"].append((key_scale.cpu(), skipped_keys.shape))
            self.quant_scales[layer_idx]["value"].append((value_scale.cpu(), skipped_values.shape))
            self.quant_zeros[layer_idx]["key"].append((key_zero.cpu(), skipped_keys.shape))
            self.quant_zeros[layer_idx]["value"].append((value_zero.cpu(), skipped_values.shape))
            
            # Use quantized tensors
            cpu_keys = key_q.cpu()
            cpu_values = value_q.cpu()
        else:
            # Use full precision tensors
            cpu_keys = skipped_keys.detach().cpu()
            cpu_values = skipped_values.detach().cpu()
            
        # If using memory mapping, store in mmap-backed storage
        if self.use_mmap:
            if layer_idx not in self.mmap_files:
                # Create temporary file for this layer
                key_file = tempfile.NamedTemporaryFile(delete=False)
                value_file = tempfile.NamedTemporaryFile(delete=False)
                pos_file = tempfile.NamedTemporaryFile(delete=False)
                
                self.mmap_files[layer_idx] = {
                    "key": key_file.name,
                    "value": value_file.name,
                    "pos": pos_file.name
                }
                
                # Initialize memory-mapped files with placeholder content
                with open(key_file.name, 'wb+') as f:
                    # Write a placeholder byte to allow memory mapping
                    f.write(b'\x00')
                    f.flush()
                    mmap_key = mmap.mmap(f.fileno(), 0)
                    
                with open(value_file.name, 'wb+') as f:
                    # Write a placeholder byte to allow memory mapping
                    f.write(b'\x00')
                    f.flush()
                    mmap_value = mmap.mmap(f.fileno(), 0)
                    
                with open(pos_file.name, 'wb+') as f:
                    # Write a placeholder byte to allow memory mapping
                    f.write(b'\x00')
                    f.flush()
                    mmap_pos = mmap.mmap(f.fileno(), 0)
                    
                self.mmap_handles[layer_idx] = {
                    "key": mmap_key,
                    "value": mmap_value,
                    "pos": mmap_pos
                }
                
            # Serialize and append to mmap file
            key_data = pickle.dumps((cpu_keys, positions.cpu()))
            value_data = pickle.dumps(cpu_values)
            pos_data = pickle.dumps(positions.cpu())
            
            # Append to mmap files
            with open(self.mmap_files[layer_idx]["key"], 'ab') as f:
                f.write(key_data)
                
            with open(self.mmap_files[layer_idx]["value"], 'ab') as f:
                f.write(value_data)
                
            with open(self.mmap_files[layer_idx]["pos"], 'ab') as f:
                f.write(pos_data)
        else:
            # Store directly in memory
            self.cpu_key_cache[layer_idx].append(cpu_keys)
            self.cpu_value_cache[layer_idx].append(cpu_values)
            self.position_map[layer_idx].append(positions.cpu())
        
        # Check if we're exceeding the offload size limit
        if self.use_mmap:
            # For mmap, we estimate based on file sizes
            total_cpu_tokens = sum(
                os.path.getsize(self.mmap_files[layer_idx]["key"]) // (4 if self.quantize else 16)
                for layer_idx in self.mmap_files
            )
        else:
            # For in-memory, we count actual tokens
            total_cpu_tokens = sum(
                sum(kv.shape[2] for kv in layer_kvs) 
                for layer_kvs in self.cpu_key_cache.values()
            )
        
        # If exceeding limit, remove oldest entries
        if total_cpu_tokens > self.offload_size:
            # Start removing from oldest layers
            while total_cpu_tokens > self.offload_size:
                if self.use_mmap:
                    # For mmap, truncate files of the oldest layer
                    oldest_layer = min(self.mmap_files.keys())
                    
                    # Recreate empty files
                    for file_type in ["key", "value", "pos"]:
                        with open(self.mmap_files[oldest_layer][file_type], 'wb') as f:
                            # Write a placeholder byte
                            f.write(b'\x00')
                            
                    # Recalculate token count
                    total_cpu_tokens = sum(
                        os.path.getsize(self.mmap_files[layer_idx]["key"]) // (4 if self.quantize else 16)
                        for layer_idx in self.mmap_files
                    )
                else:
                    # Find the oldest layer with tokens
                    if not self.cpu_key_cache:
                        break
                        
                    oldest_layer = min(self.cpu_key_cache.keys())
                    
                    if not self.cpu_key_cache[oldest_layer]:
                        del self.cpu_key_cache[oldest_layer]
                        del self.cpu_value_cache[oldest_layer]
                        del self.position_map[oldest_layer]
                        continue
                        
                    # Remove the oldest chunk from that layer
                    removed_keys = self.cpu_key_cache[oldest_layer].pop(0)
                    removed_values = self.cpu_value_cache[oldest_layer].pop(0)
                    removed_positions = self.position_map[oldest_layer].pop(0)
                    
                    # Remove quantization parameters if needed
                    if self.quantize and oldest_layer in self.quant_scales:
                        self.quant_scales[oldest_layer]["key"].pop(0)
                        self.quant_scales[oldest_layer]["value"].pop(0)
                        self.quant_zeros[oldest_layer]["key"].pop(0)
                        self.quant_zeros[oldest_layer]["value"].pop(0)
                    
                    # Update token count
                    total_cpu_tokens -= removed_keys.shape[2]
                    
                    # If the layer is now empty, remove it
                    if not self.cpu_key_cache[oldest_layer]:
                        del self.cpu_key_cache[oldest_layer]
                        del self.cpu_value_cache[oldest_layer]
                        del self.position_map[oldest_layer]
                        
                        if self.quantize and oldest_layer in self.quant_scales:
                            del self.quant_scales[oldest_layer]
                            del self.quant_zeros[oldest_layer]
        
    def search_cpu_cache(self, layer_idx: int, query_positions: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search the CPU cache for tokens at specific positions.
        
        Args:
            layer_idx: The layer index
            query_positions: The positions to search for
            
        Returns:
            Tuple of (key_state, value_state) if found, None otherwise
        """
        if layer_idx not in self.position_map and layer_idx not in self.mmap_files:
            return None
            
        # Create lists to store found key and value states
        found_keys = []
        found_values = []
        found_positions = []
        
        # Search mechanism depends on storage type
        if self.use_mmap and layer_idx in self.mmap_files:
            # Search in memory-mapped files
            with open(self.mmap_files[layer_idx]["pos"], 'rb') as f:
                # Read positions data in chunks
                offset = 0
                chunk_idx = 0
                while offset < os.path.getsize(self.mmap_files[layer_idx]["pos"]):
                    f.seek(offset)
                    try:
                        pos_tensor = pickle.load(f)
                        
                        # Check if any query position exists in this tensor
                        for query_position in query_positions:
                            found_indices = torch.nonzero(pos_tensor == query_position)
                            
                            if found_indices.shape[0] > 0:
                                # Extract the batch, head, and position indices
                                batch_idx, head_idx, pos_idx = found_indices[0]
                                
                                # Load the corresponding key and value data
                                with open(self.mmap_files[layer_idx]["key"], 'rb') as key_file:
                                    key_file.seek(offset)
                                    key_data, _ = pickle.load(key_file)
                                    
                                with open(self.mmap_files[layer_idx]["value"], 'rb') as value_file:
                                    value_file.seek(offset)
                                    value_data = pickle.load(value_file)
                                
                                # Extract the specific token
                                key_state = key_data[batch_idx:batch_idx+1, head_idx:head_idx+1, pos_idx:pos_idx+1]
                                value_state = value_data[batch_idx:batch_idx+1, head_idx:head_idx+1, pos_idx:pos_idx+1]
                                
                                # Dequantize if needed
                                if self.quantize:
                                    key_scale, key_shape = self.quant_scales[layer_idx]["key"][chunk_idx]
                                    key_zero, _ = self.quant_zeros[layer_idx]["key"][chunk_idx]
                                    value_scale, value_shape = self.quant_scales[layer_idx]["value"][chunk_idx]
                                    value_zero, _ = self.quant_zeros[layer_idx]["value"][chunk_idx]
                                    
                                    key_state = dequantize_tensor(
                                        key_state, key_scale, key_zero, 
                                        self.quantization_bits, key_shape
                                    )
                                    
                                    value_state = dequantize_tensor(
                                        value_state, value_scale, value_zero,
                                        self.quantization_bits, value_shape
                                    )
                                
                                # Add to found lists
                                found_keys.append(key_state)
                                found_values.append(value_state)
                                found_positions.append(query_position)
                        
                        # Move to next chunk
                        offset = f.tell()
                        chunk_idx += 1
                    except (EOFError, pickle.UnpicklingError):
                        break
        else:
            # Search in in-memory cache
            # For each query position
            for query_position in query_positions:
                position_found = False
                
                # Search through the position maps for this layer
                for i, positions in enumerate(self.position_map[layer_idx]):
                    # Check if the position exists in this chunk
                    found_indices = torch.nonzero(positions == query_position)
                    if found_indices.shape[0] > 0:
                        # Extract the batch, head, and position indices
                        batch_idx, head_idx, pos_idx = found_indices[0]
                        
                        # Get the corresponding key and value states
                        key_state = self.cpu_key_cache[layer_idx][i][batch_idx:batch_idx+1, 
                                                                  head_idx:head_idx+1, 
                                                                  pos_idx:pos_idx+1]
                        value_state = self.cpu_value_cache[layer_idx][i][batch_idx:batch_idx+1, 
                                                                      head_idx:head_idx+1, 
                                                                      pos_idx:pos_idx+1]
                        
                        # Dequantize if needed
                        if self.quantize:
                            key_scale, key_shape = self.quant_scales[layer_idx]["key"][i]
                            key_zero, _ = self.quant_zeros[layer_idx]["key"][i]
                            value_scale, value_shape = self.quant_scales[layer_idx]["value"][i]
                            value_zero, _ = self.quant_zeros[layer_idx]["value"][i]
                            
                            key_state = dequantize_tensor(
                                key_state, key_scale, key_zero, 
                                self.quantization_bits, key_shape
                            )
                            
                            value_state = dequantize_tensor(
                                value_state, value_scale, value_zero,
                                self.quantization_bits, value_shape
                            )
                        
                        # Add to found lists
                        found_keys.append(key_state)
                        found_values.append(value_state)
                        found_positions.append(query_position)
                        position_found = True
                        break
                    
        # If no positions were found, return None
        if not found_keys:
            return None
            
        # Get the device from somewhere - will need to be passed in
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Concatenate all found keys and values
        all_keys = torch.cat(found_keys, dim=2).to(device)
        all_values = torch.cat(found_values, dim=2).to(device)
        
        return all_keys, all_values
    
    def cleanup(self) -> None:
        """Clean up resources used by the offload manager."""
        # Close and delete memory-mapped files
        if hasattr(self, 'mmap_handles') and self.mmap_handles:
            for layer_idx, handles in self.mmap_handles.items():
                for handle_type, handle in handles.items():
                    try:
                        handle.close()
                    except:
                        pass
                
                # Delete the files
                if layer_idx in self.mmap_files:
                    for file_type, file_path in self.mmap_files[layer_idx].items():
                        try:
                            os.remove(file_path)
                        except:
                            pass
            
            self.mmap_handles = {}
            self.mmap_files = {}
        
        # Clear CPU caches
        self.cpu_key_cache = {}
        self.cpu_value_cache = {}
        self.position_map = {}
        
        # Clear quantization data
        self.quant_scales = {}
        self.quant_zeros = {}