"""
Quantization utilities for reducing memory usage of KV cache.
"""

from typing import Tuple, Optional
import torch


def quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a tensor to reduce memory usage.
    
    Args:
        tensor: The tensor to quantize
        bits: Number of bits to use for quantization (4 or 8)
        
    Returns:
        Tuple of (quantized tensor, scale, zero_point)
    """
    if bits not in (4, 8):
        raise ValueError(f"Quantization bits must be 4 or 8, got {bits}")
        
    # Get tensor shape and device
    shape = tensor.shape
    device = tensor.device
    
    # Convert to contiguous format and flatten for easier processing
    tensor = tensor.contiguous().float().view(-1)
    
    # Calculate statistics for quantization
    min_val, max_val = tensor.min(), tensor.max()
    
    # Get quantization range (2^bits - 1 for unsigned)
    quant_range = 2**bits - 1
    
    # Calculate scale and zero point
    scale = (max_val - min_val) / quant_range
    zero_point = -min_val / scale
    
    # Handle edge case where all values are the same
    if scale == 0 or torch.isnan(scale):
        scale = torch.tensor(1.0, device=device)
        zero_point = torch.tensor(0.0, device=device)
    
    # Quantize the tensor
    tensor_q = torch.clamp(torch.round(tensor / scale + zero_point), 0, quant_range)
    
    # For 4-bit quantization, pack two 4-bit values into one 8-bit value
    if bits == 4:
        # Reshape to prepare for packing
        tensor_q = tensor_q.view(-1, 2)
        # Pack two 4-bit values into one 8-bit value
        tensor_q = (tensor_q[:, 0] + (tensor_q[:, 1] << 4)).byte()
    else:
        tensor_q = tensor_q.byte()
    
    # Reshape to original shape, adjusting for packing if needed
    if bits == 4:
        new_shape = list(shape)
        new_shape[-1] = -1  # Adjust the last dimension for packing
        tensor_q = tensor_q.view(*new_shape)
    else:
        tensor_q = tensor_q.view(*shape)
    
    return tensor_q, scale, zero_point


def dequantize_tensor(tensor_q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, 
                      bits: int = 8, original_shape: Optional[Tuple] = None) -> torch.Tensor:
    """Dequantize a quantized tensor.
    
    Args:
        tensor_q: The quantized tensor
        scale: The scale used for quantization
        zero_point: The zero point used for quantization
        bits: Number of bits used for quantization (4 or 8)
        original_shape: The original shape of the tensor (required for 4-bit)
        
    Returns:
        Dequantized tensor with the original values
    """
    if bits not in (4, 8):
        raise ValueError(f"Quantization bits must be 4 or 8, got {bits}")
        
    device = tensor_q.device
    
    # Unpack 4-bit values if needed
    if bits == 4:
        if original_shape is None:
            raise ValueError("Original shape must be provided for 4-bit dequantization")
            
        # Flatten and unpack
        tensor_q_flat = tensor_q.view(-1)
        # Extract low 4 bits and high 4 bits
        unpacked = torch.zeros(tensor_q_flat.size(0) * 2, dtype=torch.uint8, device=device)
        unpacked[0::2] = tensor_q_flat & 0x0F  # Low 4 bits
        unpacked[1::2] = (tensor_q_flat >> 4) & 0x0F  # High 4 bits
        
        # Reshape to original shape
        tensor_q = unpacked.view(*original_shape)
    
    # Dequantize
    tensor_dq = (tensor_q.float() - zero_point) * scale
    
    return tensor_dq