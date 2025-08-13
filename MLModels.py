"""
Custom AI/ML Models using PyTorch

This module contains custom neural network models optimized for various sequence-to-sequence tasks,
including LSTM models, Transformer architectures with Flash Attention, and hybrid models.
"""

# Standard library imports
import sys
import os
import math
import random
from typing import Tuple, Optional, Dict, List, Union, Any

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence

# Flash Attention imports
from flash_attn.flash_attention import FlashMHA, FlashAttention
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from einops import rearrange

# Local imports
# TODO: Update this path to be more flexible/configurable
MODULE_PATH = 'C:\\Users\\hmadmin\\Documents\\DataScienceRepo\\Data Science\\Python Modules\\MLModels\\'
sys.path.append(MODULE_PATH)
from processing import transform_output_to_onehot
      
# %% 
# Function to create mask 
# chunk has dimensions [batch_size, seq_len, embedded_size]
# chunk_lengths has dimensions [batch_size]
# chunk_lengths contains the lengths of each sequence in the batch 
def create_padding_mask(chunk: torch.Tensor, chunk_lengths: torch.Tensor) -> torch.Tensor: 
    batch_size, seq_len, embedded_size = chunk.size()
    # Create a mask where True indicates padding positions
    # Will output True is value should be masked 
    padding_mask = torch.arange(seq_len).unsqueeze(0) >= chunk_lengths.unsqueeze(1).cpu()
    return padding_mask

# %% 
# Positional encoding 
class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for input sequences.

    Args:
        d_model (int): The dimension of the model (embedding size).
        max_len (int): The maximum length of the input sequences.
        dropout (float, optional): Dropout probability applied after adding positional encoding. Default is 0.1.

    Purpose:
        Adds positional information to input embeddings to enable the model to capture sequence order.
    """
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ----------------------------------------- LSTM Models -----------------------------------------------------------
# %% 
# Sequence to Sequence LSTM 
# Select last timestamp if you don't want full sequence as output 
# Must add activation function after model 
class LSTMModel(nn.Module):
    """
    Sequence-to-sequence LSTM model.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in LSTM.
        output_size (int): Number of output features.
        num_layers (int, optional): Number of LSTM layers. Default is 2.
        dropout (float, optional): Dropout probability. Default is 0.

    Output:
        The forward method returns a tensor of shape (batch_size, seq_len, output_size).
        No activation is applied to the output; apply activation (e.g., sigmoid, softmax) externally as needed.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0
    ) -> None:
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for output
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            lengths (Tensor): Sequence lengths for each batch element.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, output_size) without activation.
        """
        device = next(self.parameters()).device
        x = x.to(device)
        lengths = lengths.to(device)

        lengths_cpu = lengths.cpu()
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)  # Apply dropout
        output = self.fc1(output)
        output = self.fc2(output)
        return output

# %%
# Disclaimer: This is a sequence to 1 model, not sequence to sequence
class LSTMWithAttention(nn.Module):
    """
    Sequence-to-one LSTM model with attention mechanism.

    This model applies an LSTM to input sequences and uses an attention mechanism to produce a context vector,
    which is then mapped to the output. Suitable for tasks where a single output is required per sequence.

    Input shape:
        x: Tensor of shape (batch_size, seq_len, input_dim)

    Output shape:
        Tensor of shape (batch_size, output_dim)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ) -> None:
        """
        Initializes the LSTMWithAttention model.

        Args:
            input_dim (int): Number of input features per time step.
            hidden_dim (int): Number of hidden units in the LSTM.
            output_dim (int): Number of output features.
            num_layers (int): Number of LSTM layers.

        Input shape:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Output shape:
            Tensor of shape (batch_size, output_dim)
        """
        super(LSTMWithAttention, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)  # rnn_out shape: (batch_size, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attention(rnn_out), dim=1)  # (batch_size, seq_len, 1)
        attn_weights = attn_weights.squeeze(-1)  # (batch_size, seq_len)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * rnn_out, dim=1)  # Weighted sum
        output = self.fc(context_vector)
        return output
    
# ----------------------------------------------- Attention ----------------------------------------------------
# %% 
# Custom Flash Attention optimized to work with Tesla T4 GPU 
class CustomFlashAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super(CustomFlashAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.softmax_scale = embed_dim ** -0.5  # Scaling factor for softmax

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_s: Optional[int] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Implements the multihead softmax attention with separate query, key, and value tensors.
        Arguments
        ---------
            q: Tensor containing the query. Shape (B, S_q, H, D)
            k: Tensor containing the key. Shape (B, S_k, H, D)
            v: Tensor containing the value. Shape (B, S_v, H, D)
            key_padding_mask: a bool tensor of shape (B, S_k)
            causal: Boolean indicating if causal attention should be used.
            cu_seqlens: Optional for efficient attention on padded sequences.
            max_s: Optional for maximum sequence length.
            need_weights: Boolean indicating if attention weights should be returned.
        """
        #assert not need_weights
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda

        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        seqlen_k = k.shape[1]

        if cu_seqlens is None:
            if key_padding_mask is None:
                # Reshape q, k, v to the required shape for flash attention
                q = rearrange(q, 'b s h d -> (b s) h d')   # Shape: (batch_size * seqlen_q, num_heads, head_dim)
                k = rearrange(k, 'b s h d -> (b s) h d')   # Shape: (batch_size * seqlen_k, num_heads, head_dim)
                v = rearrange(v, 'b s h d -> (b s) h d')   # Shape: (batch_size * seqlen_k, num_heads, head_dim)

                # Concatenate q, k, v into qkv
                # Use stack instead of cat to include the three components
                qkv = torch.stack([q, k, v], dim=1)  # Shape: (batch_size * seq_len_q, 3, num_heads, head_dim)

                # Reshape for flash attention (total, 3, nheads, headdim)
                qkv = qkv.reshape(-1, 3, qkv.size(2), qkv.size(3))  # Shape: (total, 3, num_heads, head_dim)
                
                max_s = seqlen_k
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                        device=q.device)

                # Call the unpadded attention function
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = output.view(batch_size, seqlen_k, -1)  # Reshape to (batch_size, seq_len, hidden_dim)
                if output.dim() != 3:
                    output = rearrange(output, 'b s h d -> b s (h d)'), None
            else:
                # Handle the padded case
                nheads = q.shape[-2]
                # Unpad inputs
                q_unpad, indices_q, cu_seqlens, max_s = unpad_input(q, key_padding_mask)
                k_unpad, _, _, _ = unpad_input(k, key_padding_mask)
                v_unpad, _, _, _ = unpad_input(v, key_padding_mask)

                # Compute sequence lengths based on unpadded data (assuming max_s is computed correctly)
                seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

                # Concatenate unpadded q, k, v for flash attention
                qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)  # Shape: (nnz, 3, h, d)

                # Correct cu_seqlens to have shape (batch_size + 1)
                cu_seqlens = torch.cat([torch.tensor([0], device=q.device), torch.cumsum(seq_lens, dim=0)])

                # Call the unpadded attention function
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    qkv_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )

                # Pad the output back to the original shape
                output = pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                indices_q, batch_size, seqlen_q)

                # Rearrange the output to match the expected shape
                output = rearrange(output, 'b s (h d) -> b s (h d)', h=nheads)

        else:
            # Handle the padded case
            nheads = q.shape[-2]
            q_unpad, indices_q, cu_seqlens, max_s = unpad_input(q, key_padding_mask)
            k_unpad, _, _, _ = unpad_input(k, key_padding_mask)
            v_unpad, _, _, _ = unpad_input(v, key_padding_mask)

            # Ensure max_s is computed (should be the maximum sequence length)
            assert max_s is not None, "max_s must not be None. Ensure unpad_input returns it."
            
            # Ensure cu_seqlens is correctly shaped
            assert cu_seqlens is not None, "cu_seqlens must not be None."

            # Concatenate unpadded q, k, v for flash attention
            qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)  # Shape: (nnz, 3, h, d)

            # Call the unpadded attention function
            output_unpad = flash_attn_unpadded_qkvpacked_func(
                qkv_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

            # Pad the output back to the original shape
            output = pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                            indices_q, batch_size, seqlen_q)

            # Rearrange the output to match the expected shape
            output = rearrange(output, 'b s (h d) -> b s (h d)', h=nheads)
        # Compute attention weights if needed
        if need_weights:
            attn_scores = torch.einsum('bqhd,bkhd->bhqk', q.view(batch_size, seqlen_q, self.num_heads, self.embed_dim//self.num_heads),
                                        k.view(batch_size, seqlen_k, self.num_heads, self.embed_dim//self.num_heads)) * self.softmax_scale
            attn_weights = torch.softmax(attn_scores, dim=-1)  # Shape: (batch, num_heads, seq_len_q, seq_len_k)
        else:
            attn_weights = None
        return output, attn_weights
    
# %% 
class CustomFlashMHA(nn.Module):
    """
    Custom multi-head attention module using Flash Attention.

    Args:
        embed_dim (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads.
        bias (bool, optional): If True, adds a learnable bias to the linear layers. Default is True.
        batch_first (bool, optional): If True, input and output tensors are provided as (batch, seq, feature). Default is True.
        attention_dropout (float, optional): Dropout probability for attention weights. Default is 0.0.
        causal (bool, optional): If True, applies causal masking for autoregressive tasks. Default is False.
        device (torch.device, optional): Device for module parameters.
        dtype (torch.dtype, optional): Data type for module parameters.

    Purpose:
        Implements multi-head attention using Flash Attention for efficient computation on supported hardware.
        Supports cross-attention between query and key-value inputs, and returns attention weights if requested.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        batch_first: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        # Separate projection layers for query and key-value
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)

        self.inner_attn = CustomFlashAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query (Tensor): (batch, seqlen_q, hidden_dim) - the query input (decoder)
            key_value (Tensor): (batch, seqlen_kv, hidden_dim) - the key-value input (encoder)
            key_padding_mask (Tensor, optional): bool tensor of shape (batch, seqlen_kv)
            need_weights (bool, optional): If True, returns attention weights.
            cu_seqlens (Tensor, optional): Cumulative sequence lengths for packed sequences.

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                - output (Tensor): (batch, seqlen_q, hidden_dim) after output projection.
                - attn_weights (Tensor or None): (batch, num_heads, seqlen_q, seqlen_kv) if need_weights=True, else None.
        """
        # Project query and key-value
        q = self.Wq(query)
        kv = self.Wkv(key_value)
        # Reshape for multi-head attention
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = rearrange(kv, 'b s (two h d) -> b s two h d', two=2, h=self.num_heads)
        # Split key_value into key and value
        k, v = kv.unbind(dim=2)  # Separate into key and value, k: (batch, seq_len, n_heads, head_dim), v: same

        # Perform the cross-attention with separate query, key, and value
        context, attn_weights = self.inner_attn(
            q, k, v,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
            cu_seqlens=cu_seqlens
        )
        
        # Apply output projection
        return self.out_proj(context), attn_weights

class FlashAttentionWeights(nn.Module):
    """
    Implements scaled dot-product attention using Flash Attention, with optional output of attention weights.

    Args:
        softmax_scale (float, optional): Scaling factor for softmax normalization. If None, uses default scaling.
        attention_dropout (float, optional): Dropout probability applied to attention weights.

    Input Shapes:
        qkv (Tensor): Shape (batch_size, seq_len, 3, num_heads, head_dim) if key_padding_mask is None.
        key_padding_mask (Tensor, optional): Boolean tensor of shape (batch_size, seq_len) indicating padding positions.
        causal (bool, optional): If True, applies causal masking for autoregressive tasks.
        cu_seqlens (Tensor, optional): Cumulative sequence lengths for packed sequences.
        max_s (int, optional): Maximum sequence length in the batch.
        need_weights (bool, optional): If True, returns attention weights.

    Output:
        output (Tensor): Shape (batch_size, seq_len, num_heads, head_dim), attended values.
        attn_weights (Tensor or None): Shape (batch_size, num_heads, seq_len, seq_len) if need_weights=True, else None.

    Usage:
        Use this module for efficient multi-head attention computation on CUDA devices, optionally retrieving attention weights for interpretability or analysis.
    """
    def __init__(self, softmax_scale: float = None, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(
        self,
        qkv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_s: Optional[int] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Arguments:
            qkv: (B, S, 3, H, D) if key_padding_mask is None
            key_padding_mask: bool tensor of shape (B, S)
            need_weights: If True, return attention weights.
        Returns:
            output: (B, S, H, D)
            attn_weights: (B, H, S, S) if need_weights=True, else None
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        if cu_seqlens is None:
            batch_size, seqlen, _, num_heads, head_dim = qkv.shape
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            max_s = seqlen
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device)
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            if need_weights:
                q, k, _ = qkv.view(batch_size, seqlen, 3, num_heads, head_dim).unbind(dim=2)  # Separate Q, K, V
                attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k)  # (batch, num_heads, seq_len, seq_len)
                attn_weights = torch.softmax(attn_scores * (self.softmax_scale or 1.0 / head_dim ** 0.5), dim=-1)
            else:
                attn_weights = None
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            attn_weights = None  # Not computed for packed sequences
        # Returns:
        #   output: Tensor of shape (batch_size, seq_len, num_heads, head_dim) (attended values)
        #   attn_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len) if need_weights=True, else None
        return output, attn_weights

class FlashMHAWeights(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        batch_first: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttentionWeights(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies multi-head attention using Flash Attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim), where hidden_dim = num_heads * head_dim.
            key_padding_mask (Tensor, optional): Boolean tensor of shape (batch_size, seq_len) indicating padding positions.
            need_weights (bool, optional): If True, returns attention weights for interpretability.

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                - output (Tensor): Tensor of shape (batch_size, seq_len, hidden_dim) after attention and output projection.
                - attn_weights (Tensor or None): Attention weights of shape (batch_size, num_heads, seq_len, seq_len) if need_weights=True, else None.
        """
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
# ------------------------------------------------------ Transformers -------------------------------------------------------
# %% 
# Custom transformer decoder layer that uses flash attention (should be faster)
class CustomTransformerDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer using Flash Attention.

    Args:
        hidden_dim (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Default is 0.1.
        causal (bool, optional): If True, applies causal masking for autoregressive tasks. Default is False.

    Purpose:
        Implements a transformer decoder layer with Flash Attention for both self-attention and cross-attention.
        Includes residual connections, layer normalization, and a feed-forward network.

    Input Shapes:
        tgt (Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len, hidden_dim).
        memory (Tensor): Encoder output tensor of shape (batch_size, src_seq_len, hidden_dim).
        tgt_mask (Tensor, optional): Mask for target sequence (batch_size, tgt_seq_len, tgt_seq_len) or (batch_size, tgt_seq_len).
        memory_mask (Tensor, optional): Mask for encoder memory (batch_size, tgt_seq_len, src_seq_len) or (batch_size, src_seq_len).
        cu_seqlens (Tensor, optional): Cumulative sequence lengths for packed sequences.
        needs_weights (bool, optional): If True, returns attention weights.

    Output:
        Tuple[Tensor, Dict[str, Optional[Tensor]]]:
            - output (Tensor): Output tensor of shape (batch_size, tgt_seq_len, hidden_dim).
            - attn (dict): Dictionary with keys "self" and "cross" containing attention weights (or None).
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = False
    ) -> None:
        super(CustomTransformerDecoderLayer, self).__init__()
        # Flash attention for self-attention and encoder-decoder attention
        self.self_attn = FlashMHAWeights(embed_dim=hidden_dim, num_heads=num_heads, causal=causal)
        self.multihead_attn = CustomFlashMHA(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        # Dropout and normalization layers
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        needs_weights: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Forward pass for the decoder layer.

        Args:
            tgt (Tensor): Target sequence (batch_size, tgt_seq_len, hidden_dim).
            memory (Tensor): Encoder output (batch_size, src_seq_len, hidden_dim).
            tgt_mask (Tensor, optional): Mask for target sequence.
            memory_mask (Tensor, optional): Mask for encoder memory.
            cu_seqlens (Tensor, optional): Cumulative sequence lengths for packed sequences.
            needs_weights (bool, optional): If True, returns attention weights.

        Returns:
            Tuple[Tensor, Dict[str, Optional[Tensor]]]:
                - output (Tensor): Output tensor after decoder layer.
                - attn (dict): Dictionary with "self" and "cross" attention weights.
        """
        # Self-attention with residual connection and layer normalization
        tgt2, self_attention_weights = self.self_attn(
            tgt, key_padding_mask=tgt_mask, need_weights=needs_weights
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with encoder output (memory), residual connection, and layer normalization
        tgt2, cross_attention_weights = self.multihead_attn(
            tgt, memory, key_padding_mask=memory_mask, need_weights=needs_weights, cu_seqlens=cu_seqlens
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network with residual connection and layer normalization
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt, {"self": self_attention_weights, "cross": cross_attention_weights}
# %%    
class CustomTransformerDecoder(nn.Module):
    """
    Custom Transformer Decoder composed of multiple CustomTransformerDecoderLayer modules.

    Args:
        num_layers (int): Number of decoder layers to stack.
        hidden_dim (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads in each layer.
        dropout (float, optional): Dropout probability for each layer. Default is 0.1.

    Purpose:
        Stacks multiple custom transformer decoder layers, applies layer normalization at the end,
        and collects self-attention and cross-attention weights from each layer.

    Input Shapes:
        tgt (torch.Tensor): Target sequence tensor of shape (batch_size, target_seq_len, hidden_dim).
        memory (torch.Tensor): Encoder output tensor of shape (batch_size, source_seq_len, hidden_dim).
        tgt_mask (Optional[torch.Tensor]): Mask for target sequence (batch_size, target_seq_len, target_seq_len) or (batch_size, target_seq_len).
        memory_mask (Optional[torch.Tensor]): Mask for encoder memory (batch_size, target_seq_len, source_seq_len) or (batch_size, source_seq_len).
        cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths for packed sequences.

    Output:
        Tuple[torch.Tensor, Dict[str, List[Optional[torch.Tensor]]]]:
            - output (torch.Tensor): Output tensor of shape (batch_size, target_seq_len, hidden_dim).
            - attn (dict): Dictionary with keys "self" and "cross", each containing a list of attention weights from all layers.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, List[Optional[torch.Tensor]]]]:
        """
        Forward pass for the stacked decoder.

        Args:
            tgt (torch.Tensor): Target sequence (batch_size, target_seq_len, hidden_dim).
            memory (torch.Tensor): Encoder output (batch_size, source_seq_len, hidden_dim).
            tgt_mask (Optional[torch.Tensor]): Mask for target sequence.
            memory_mask (Optional[torch.Tensor]): Mask for encoder memory.
            cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths for packed sequences.

        Returns:
            Tuple[torch.Tensor, Dict[str, List[Optional[torch.Tensor]]]]:
                - output (torch.Tensor): Output tensor after decoder stack.
                - attn (dict): Dictionary with lists of self-attention and cross-attention weights from all layers.
        """
        output = tgt
        self_attentions: List[Optional[torch.Tensor]] = []
        cross_attentions: List[Optional[torch.Tensor]] = []

        for layer in self.layers:
            output, attn = layer(output, memory, tgt_mask, memory_mask, cu_seqlens)
            self_attentions.append(attn["self"])
            cross_attentions.append(attn["cross"])
        output = self.norm(output)
        return output, {"self": self_attentions, "cross": cross_attentions}
# %% 
class TransformerDecoder(nn.Module):
    """
    TransformerDecoder implements a custom transformer-based decoder module for sequence modeling tasks.
    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden layers within the transformer.
        output_dim (int): Dimensionality of the output features.
        num_layers (int): Number of transformer decoder layers.
        num_heads (int, optional): Number of attention heads in each transformer layer. Default is 2.
        max_len (int, optional): Maximum sequence length for positional encoding. Default is 1024.
        dropout (float, optional): Dropout rate applied in positional encoding and transformer layers. Default is 0.1.
        contextual_dim (Optional[int], optional): Dimensionality of contextual information for embedding. If None, contextual embedding is not used.
    Attributes:
        num_heads (int): Number of attention heads.
        input_embedding (nn.Linear): Linear layer for input feature embedding.
        positional_encoding (PositionalEncoding): Positional encoding module.
        decoder_layer (CustomTransformerDecoderLayer): Single custom transformer decoder layer.
        transformer_decoder (CustomTransformerDecoder): Stacked custom transformer decoder layers.
        fc (nn.Linear): Final linear layer mapping hidden states to output dimension.
        contextual_embedding (Optional[nn.Linear]): Linear layer for contextual embedding if provided.
    Methods:
        forward(
            x: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
            contextual_info: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            Forward pass through the decoder.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                lengths (Optional[torch.Tensor]): Sequence lengths for each batch element.
                contextual_info (Optional[torch.Tensor]): Contextual information tensor.
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 2,
        max_len: int = 1024,
        dropout: float = 0.1,
        contextual_dim: Optional[int] = None
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.num_heads = num_heads
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len, dropout)
        
        # Initialize custom transformer decoder layers
        self.decoder_layer = CustomTransformerDecoderLayer(hidden_dim=hidden_dim, num_heads=self.num_heads)
        self.transformer_decoder = CustomTransformerDecoder(num_layers=num_layers, hidden_dim=hidden_dim, num_heads=self.num_heads)
        
        # Fully connected layer to map to output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Contextual embedding if provided
        if contextual_dim:
            self.contextual_embedding = nn.Linear(contextual_dim, hidden_dim)
        else:
            self.contextual_embedding = None

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        contextual_info: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        
        # Apply input embedding
        x = self.input_embedding(x)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Apply contextual embedding if provided
        if self.contextual_embedding is not None and contextual_info is not None:
            contextual_embedded = self.contextual_embedding(contextual_info)
            x = x + contextual_embedded  # Add contextual embedding
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        # Create masks and cumulative lengths only if lengths are provided
        if lengths is not None:
            batch_size, seq_len = x.size(0), x.size(1)

            key_padding_mask = create_padding_mask(x, lengths)
            
            # Create cumulative sequence lengths
            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
            cu_seqlens[1:] = lengths.cumsum(dim=0)

            # Call the transformer decoder with the mask and lengths
            x = self.transformer_decoder(x, x, tgt_mask=key_padding_mask, memory_mask=key_padding_mask, cu_seqlens=cu_seqlens)

        else:
            # If lengths are not provided, just use x directly
            x = self.transformer_decoder(x, x)

        # Apply final fully connected layer
        x = self.fc(x)  # (batch_size, seq_len, output_dim)

        # Apply sigmoid activation to the output
        #x = torch.sigmoid(x) # Add this in the next run 
        
        return x
    
# %% 
#Custom Transformer Encoder Layer using custom flash attention 
class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer with Flash Multi-Head Attention.
    This layer implements a transformer encoder block using a custom FlashMHAWeights attention mechanism,
    followed by a feed-forward network, dropout, and layer normalization. It supports optional masking for attention.
    Args:
        hidden_dim (int): The dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Default is 0.1.
    Attributes:
        self_attn (FlashMHAWeights): Multi-head self-attention module.
        linear1 (nn.Linear): First linear layer of the feed-forward network.
        linear2 (nn.Linear): Second linear layer of the feed-forward network.
        dropout (nn.Dropout): Dropout layer.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feed-forward network.
    Methods:
        forward(src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Performs a forward pass through the encoder layer.
            Args:
                src (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).
                src_mask (Optional[torch.Tensor]): Optional mask tensor for attention.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Output tensor after encoding and attention weights.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = FlashMHAWeights(embed_dim=hidden_dim, num_heads=num_heads, causal=False)

        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

        # Dropout and normalization layers
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection and layer normalization
        src2, attention_weights = self.self_attn(src, key_padding_mask=src_mask, need_weights=True)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feed-forward network with residual connection and layer normalization
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src, attention_weights
# %% 
# Custom Transformer Encoder
class CustomTransformerEncoder(nn.Module):
    """
    Custom Transformer Encoder module.
    This class implements a stack of custom transformer encoder layers followed by a final layer normalization.
    It processes input sequences and returns the encoded output along with attention weights from each layer.
    Args:
        num_layers (int): Number of encoder layers to stack.
        hidden_dim (int): Dimensionality of the hidden representations.
        num_heads (int): Number of attention heads in each encoder layer.
        dropout (float, optional): Dropout probability for attention and feed-forward layers. Default is 0.1.
    Attributes:
        layers (nn.ModuleList): List of CustomTransformerEncoderLayer modules.
        norm (nn.LayerNorm): Final layer normalization module.
    Methods:
        forward(src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            Processes the input sequence through the encoder stack and returns the normalized output and attention weights.
            Args:
                src (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).
                src_mask (Optional[torch.Tensor]): Optional mask tensor for attention mechanism.
            Returns:
                output (torch.Tensor): Encoded and normalized output tensor of shape (batch_size, seq_length, hidden_dim).
                attention_weights_all_layers (List[torch.Tensor]): List of attention weights from each encoder layer.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super(CustomTransformerEncoder, self).__init__()
        # Stack of CustomTransformerEncoderLayers
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # Final normalization layer
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output = src
        attention_weights_all_layers: List[torch.Tensor] = []
        # Pass through all encoder layers
        for layer in self.layers:
            output, attention_weights = layer(output, src_mask)
            attention_weights_all_layers.append(attention_weights)
        
        # Apply final normalization
        output = self.norm(output)
        
        return output, attention_weights_all_layers

# %% 
class TransformerEncoderDecoder(nn.Module):
    """
    A flexible Transformer-based Encoder-Decoder model supporting optional convolutional, LSTM, and contextual embedding layers.
    This model is designed for sequence-to-sequence tasks and allows for various input preprocessing strategies, including Conv1d and LSTM layers.
    It supports custom transformer encoder and decoder layers, positional encoding, and optional contextual embeddings for enhanced representation.
    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers in the transformer.
        output_dim (int): Dimension of output features.
        num_layers (int): Number of transformer encoder/decoder layers.
        num_heads (int, optional): Number of attention heads in transformer layers. Default is 2.
        dropout (float, optional): Dropout rate for regularization. Default is 0.5.
        contextual_dim (int, optional): Dimension of contextual embedding input. If provided, enables contextual embedding.
        lstm_hidden_dim (int, optional): Hidden dimension for LSTM layer. If provided, enables LSTM preprocessing.
        lstm_num_layers (int, optional): Number of LSTM layers. Default is 1.
        conv_out_channels (int, optional): Number of output channels for Conv1d layer. If provided, enables convolutional preprocessing.
        conv_kernel_size (int, optional): Kernel size for Conv1d layer. Default is 3.
        conv_stride (int, optional): Stride for Conv1d layer. Default is 1.
    Attributes:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension size.
        output_dim (int): Output dimension size.
        conv_layer (nn.Conv1d or None): Optional convolutional layer.
        lstm (nn.LSTM or None): Optional LSTM layer.
        input_embedding (nn.Sequential): Input embedding layer.
        positional_encoding (PositionalEncoding): Positional encoding module.
        encoder_layer (CustomTransformerEncoderLayer): Transformer encoder layer.
        transformer_encoder (CustomTransformerEncoder): Transformer encoder stack.
        decoder_layer (CustomTransformerDecoderLayer): Transformer decoder layer.
        transformer_decoder (CustomTransformerDecoder): Transformer decoder stack.
        contextual_embedding (nn.Linear or None): Optional contextual embedding layer.
        fc (nn.Linear): Final fully connected output layer.
    Methods:
        freeze_layers():
            Freezes parameters of input preprocessing and encoder layers to prevent training updates.
        forward(src: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
            Performs a forward pass through the model.
            Args:
                src (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                lengths (Optional[torch.Tensor]): Optional tensor of sequence lengths for masking.
            Returns:
                Tuple[torch.Tensor, Dict[str, Any]]:
                    - Output tensor of shape (batch_size, seq_len, output_dim).
                    - Dictionary containing encoder and decoder attention weights.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 2,
        dropout: float = 0.5,
        contextual_dim: Optional[int] = None,
        lstm_hidden_dim: Optional[int] = None,
        lstm_num_layers: int = 1,
        conv_out_channels: Optional[int] = None,
        conv_kernel_size: int = 3,
        conv_stride: int = 1
    ) -> None:
        super(TransformerEncoderDecoder, self).__init__()
        self.num_heads = num_heads 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Optional Convolutional Layer
        if conv_out_channels is not None:
            self.conv_layer = nn.Conv1d(input_dim, conv_out_channels, kernel_size=conv_kernel_size, 
                                        stride=conv_stride, padding=(conv_kernel_size - 1)//2)
            self.input_embedding = nn.Sequential(nn.Linear(conv_out_channels, hidden_dim), 
                                                 nn.LayerNorm(hidden_dim))  # Adjust input dim based on conv layer output
            self.lstm = None
        elif lstm_hidden_dim is not None:
            self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)
            self.input_embedding = nn.Sequential(nn.Linear(lstm_hidden_dim, hidden_dim), 
                                                 nn.LayerNorm(hidden_dim))  # Embedding after LSTM output
        else:
            self.conv_layer = None       
            self.input_embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                                 nn.LayerNorm(hidden_dim)) 
            self.lstm = None

        self.positional_encoding = PositionalEncoding(hidden_dim, 1024, dropout)

        # Initialize Encoder
        self.encoder_layer = CustomTransformerEncoderLayer(hidden_dim=hidden_dim, num_heads=self.num_heads, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(num_layers=num_layers, hidden_dim=hidden_dim, num_heads=self.num_heads, dropout=dropout)
        
        # Initialize custom transformer decoder layers
        if contextual_dim: 
            self.decoder_layer = CustomTransformerDecoderLayer(hidden_dim=hidden_dim*2, num_heads=self.num_heads, dropout=dropout)
            self.transformer_decoder = CustomTransformerDecoder(num_layers=num_layers-1, hidden_dim=hidden_dim*2, num_heads=self.num_heads, dropout=dropout)
            self.contextual_embedding = nn.Linear(contextual_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else: 
            self.decoder_layer = CustomTransformerDecoderLayer(hidden_dim=hidden_dim, num_heads=self.num_heads, dropout=dropout)
            self.transformer_decoder = CustomTransformerDecoder(num_layers=num_layers-1, hidden_dim=hidden_dim, num_heads=self.num_heads, dropout=dropout)
            self.contextual_embedding = None
            self.fc = nn.Linear(hidden_dim, output_dim)

    def freeze_layers(self) -> None:
        # Freeze LSTM layers (if present)
        if self.lstm is not None:
            for param in self.lstm.parameters():
                param.requires_grad = False

        # Freeze input embedding layer
        for param in self.input_embedding.parameters():
            param.requires_grad = False

        # Freeze positional encoding layer (if it has parameters)
        if hasattr(self.positional_encoding, 'parameters'):
            for param in self.positional_encoding.parameters():
                param.requires_grad = False

        # Freeze encoder layers
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        src: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # src shape: (batch_size, seq_len, input_dim)
        
        # If LSTM is used, pass through LSTM first
        if self.lstm is not None:
            lstm_out, _ = self.lstm(src)  # lstm_out shape: (batch_size, seq_len, lstm_hidden_dim)
            x = self.input_embedding(lstm_out)  # Apply embedding after LSTM
        elif self.conv_layer is not None: 
            # Rearrange dimensions for Conv1d: (batch_size, input_dim, seq_len)
            src_perm = src.permute(0, 2, 1)
            conv_out = self.conv_layer(src_perm)  # Apply convolution
            # Rearrange back: (batch_size, seq_len, conv_out_channels)
            conv_out = conv_out.permute(0, 2, 1)
            x = self.input_embedding(conv_out)  # Apply embedding after Conv1d            
        else:
            x = self.input_embedding(src)  # Apply embedding directly
        
        # Apply positional encoding
        x = self.positional_encoding(x)

        # Pass through the transformer encoder
        memory, encoder_attention = self.transformer_encoder(x)

        # Apply contextual embedding (concatenate memory with src)
        if self.contextual_embedding is not None:
            contextual_embedded = self.contextual_embedding(src)
            memory = torch.cat((memory, contextual_embedded), dim=2)

        # If lengths are provided, create masks and cumulative lengths
        if lengths is not None:
            batch_size, seq_len = x.size(0), x.size(1)
            key_padding_mask = torch.ones(batch_size, seq_len, device=x.device)
            for i in range(batch_size):
                key_padding_mask[i, lengths[i]:] = 0  # Set positions beyond lengths to 1
            
            key_padding_mask = key_padding_mask.bool()  # Convert to boolean
            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
            cu_seqlens[1:] = lengths.cumsum(dim=0)

            # Call the transformer decoder with the mask and lengths
            x, decoder_attention = self.transformer_decoder(memory, memory, tgt_mask=key_padding_mask, memory_mask=key_padding_mask, cu_seqlens=cu_seqlens)
        else:
            # If lengths are not provided, just use memory directly
            x, decoder_attention = self.transformer_decoder(memory, memory)

        # Apply the final fully connected layer
        x = self.fc(x)  # (batch_size, seq_len, output_dim)

        return x, {"enc": encoder_attention, "dec_self": decoder_attention["self"], "dec_cross": decoder_attention["cross"]}
