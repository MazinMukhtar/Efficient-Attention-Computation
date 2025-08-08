import numpy as np 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union, Any
from flash_attn.flash_attention import FlashMHA
from einops import rearrange
from flash_attn.flash_attention import FlashAttention
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """
        Initialize the positional encoding layer.
        
        Args:
            d_model (int): The dimension of the model embeddings
            max_len (int): Maximum sequence length for positional encoding
            dropout (float): Dropout probability for the positional encoding
            
        Returns:
            None
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            torch.Tensor: Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Custom Flash Attention optimized to work with Tesla GPU 
class CustomFlashAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        """
        Initialize the custom flash attention module.
        
        Args:
            embed_dim (int): The embedding dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability for attention weights
            
        Returns:
            None
        """
        super(CustomFlashAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.softmax_scale = embed_dim ** -0.5  # Scaling factor for softmax

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, 
                causal: bool = False, 
                cu_seqlens: Optional[torch.Tensor] = None,
                max_s: Optional[int] = None, 
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Implements the multihead softmax attention with separate query, key, and value tensors.
        
        Args:
            q (torch.Tensor): Query tensor of shape (B, S_q, H, D)
            k (torch.Tensor): Key tensor of shape (B, S_k, H, D)
            v (torch.Tensor): Value tensor of shape (B, S_v, H, D)
            key_padding_mask (Optional[torch.Tensor]): Boolean mask for padding, shape (B, S_k)
            causal (bool): Whether to use causal attention
            cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths for packed sequences
            max_s (Optional[int]): Maximum sequence length
            need_weights (bool): Whether to return attention weights
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and optional attention weights
            
        Raises:
            AssertionError: If q is not on CUDA or not in float16/bfloat16
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
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, batch_first: bool = True, 
                 attention_dropout: float = 0.0, causal: bool = False, device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """
        Initialize the custom flash multi-head attention module.
        
        Args:
            embed_dim (int): The embedding dimension
            num_heads (int): Number of attention heads
            bias (bool): Whether to use bias in linear layers
            batch_first (bool): Whether input is batch first
            attention_dropout (float): Dropout probability for attention
            causal (bool): Whether to use causal attention
            device (Optional[torch.device]): Device to place the module on
            dtype (Optional[torch.dtype]): Data type for the module
            
        Returns:
            None
            
        Raises:
            AssertionError: If batch_first is False or embed_dim is not divisible by num_heads
        """
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

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, 
                need_weights: bool = False, 
                cu_seqlens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform multi-head attention with separate query and key-value projections.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch, seqlen_q, hidden_dim)
            key_value (torch.Tensor): Key-value tensor of shape (batch, seqlen_kv, hidden_dim)
            key_padding_mask (Optional[torch.Tensor]): Boolean mask for padding, shape (batch, seqlen_kv)
            need_weights (bool): Whether to return attention weights
            cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths for packed sequences
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and optional attention weights
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
        context, attn_weights = self.inner_attn(q, k, v, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal, cu_seqlens=cu_seqlens)
        
        # Apply output projection
        return self.out_proj(context), attn_weights

class FlashAttentionWeights(nn.Module):
   """Scaled dot product attention with optional attention weight output."""
   def __init__(self, softmax_scale: Optional[float] = None, attention_dropout: float = 0.0) -> None:
       """
       Initialize the flash attention weights module.
       
       Args:
           softmax_scale (Optional[float]): Scaling factor for softmax
           attention_dropout (float): Dropout probability for attention weights
           
       Returns:
           None
       """
       super().__init__()
       self.softmax_scale = softmax_scale
       self.dropout_p = attention_dropout
       
   def forward(self, qkv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, 
               causal: bool = False, cu_seqlens: Optional[torch.Tensor] = None,
               max_s: Optional[int] = None, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
       """
       Perform flash attention with optional weight computation.
       
       Args:
           qkv (torch.Tensor): Packed query-key-value tensor of shape (B, S, 3, H, D)
           key_padding_mask (Optional[torch.Tensor]): Boolean mask for padding, shape (B, S)
           causal (bool): Whether to use causal attention
           cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths for packed sequences
           max_s (Optional[int]): Maximum sequence length
           need_weights (bool): Whether to return attention weights
           
       Returns:
           Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and optional attention weights
           
       Raises:
           AssertionError: If qkv is not on CUDA or not in float16/bfloat16
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
       return output, attn_weights  

class FlashMHAWeights(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, batch_first: bool = True, 
                 attention_dropout: float = 0.0, causal: bool = False, 
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        """
        Initialize the flash multi-head attention weights module.
        
        Args:
            embed_dim (int): The embedding dimension
            num_heads (int): Number of attention heads
            bias (bool): Whether to use bias in linear layers
            batch_first (bool): Whether input is batch first
            attention_dropout (float): Dropout probability for attention
            causal (bool): Whether to use causal attention
            device (Optional[torch.device]): Device to place the module on
            dtype (Optional[torch.dtype]): Data type for the module
            
        Returns:
            None
            
        Raises:
            AssertionError: If batch_first is False or embed_dim is not divisible by num_heads
        """
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, 
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform multi-head attention with flash attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seqlen, hidden_dim)
            key_padding_mask (Optional[torch.Tensor]): Boolean mask for padding, shape (batch, seqlen)
            need_weights (bool): Whether to return attention weights
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and optional attention weights
        """
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights

# %% 
# Custom transformer decoder layer that uses flash attention (should be faster)
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        """
        Initialize the custom transformer decoder layer.
        
        Args:
            hidden_dim (int): The hidden dimension size
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            
        Returns:
            None
        """
        super(CustomTransformerDecoderLayer, self).__init__()
        # Flash attention for self-attention and encoder-decoder attention
        self.self_attn = FlashMHAWeights(embed_dim=hidden_dim, num_heads=num_heads, causal=True)
        self.multihead_attn = CustomFlashMHA(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        # Dropout and normalization layers
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                memory_mask: Optional[torch.Tensor] = None, 
                cu_seqlens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Forward pass through the transformer decoder layer.
        
        Args:
            tgt (torch.Tensor): Target sequence tensor
            memory (torch.Tensor): Encoder memory tensor
            tgt_mask (Optional[torch.Tensor]): Mask for target sequence
            memory_mask (Optional[torch.Tensor]): Mask for encoder memory
            cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]: Output tensor and attention weights
        """
        # Self-attention with residual connection and layer normalization
        tgt2, self_attention_weights = self.self_attn(tgt, key_padding_mask=tgt_mask, need_weights=True)  # Apply self-attention to tgt
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with encoder output (memory), residual connection, and layer normalization
        tgt2, cross_attention_weights = self.multihead_attn(tgt, memory, key_padding_mask=memory_mask, need_weights=True, cu_seqlens=cu_seqlens)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network with residual connection and layer normalization
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt, {"self": self_attention_weights, "cross": cross_attention_weights}
# %%    
class CustomTransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        """
        Initialize the custom transformer decoder.
        
        Args:
            num_layers (int): Number of decoder layers
            hidden_dim (int): The hidden dimension size
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            
        Returns:
            None
        """
        super(CustomTransformerDecoder, self).__init__()
        # Stack of CustomTransformerDecoderLayers
        self.layers = nn.ModuleList([CustomTransformerDecoderLayer(hidden_dim, num_heads, dropout)
                                     for _ in range(num_layers)])
        # Final normalization layer
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                memory_mask: Optional[torch.Tensor] = None, 
                cu_seqlens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, List[Optional[torch.Tensor]]]]:
        """
        Forward pass through the transformer decoder.
        
        Args:
            tgt (torch.Tensor): Target sequence (batch_size, target_seq_len, hidden_dim)
            memory (torch.Tensor): Encoder output (batch_size, source_seq_len, hidden_dim)
            tgt_mask (Optional[torch.Tensor]): Mask for target sequence (batch_size, target_seq_len, target_seq_len)
            memory_mask (Optional[torch.Tensor]): Mask for encoder memory (batch_size, target_seq_len, source_seq_len)
            cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths
        
        Returns:
            Tuple[torch.Tensor, Dict[str, List[Optional[torch.Tensor]]]]: Output tensor and attention weights from all layers
        """
        output = tgt
        self_attentions: List[Optional[torch.Tensor]] = []
        cross_attentions: List[Optional[torch.Tensor]] = []
        
        # Pass through all decoder layers
        for layer in self.layers:
            output, attn = layer(output, memory, tgt_mask, memory_mask, cu_seqlens)
            self_attentions.append(attn["self"])
            cross_attentions.append(attn["cross"])
        # Apply final normalization
        output = self.norm(output)
        
        return output, {"self": self_attentions, "cross": cross_attentions}

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        """
        Initialize the custom transformer encoder layer.
        
        Args:
            hidden_dim (int): The hidden dimension size
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            
        Returns:
            None
        """
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = FlashMHAWeights(embed_dim=hidden_dim, num_heads=num_heads, causal=True)

        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

        # Dropout and normalization layers
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer encoder layer.
        
        Args:
            src (torch.Tensor): Source sequence tensor
            src_mask (Optional[torch.Tensor]): Mask for source sequence
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and attention weights
        """
        # Self-attention with residual connection and layer normalization
        src2, attention_weights = self.self_attn(src, key_padding_mask=src_mask, need_weights = True)
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
    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        """
        Initialize the custom transformer encoder.
        
        Args:
            num_layers (int): Number of encoder layers
            hidden_dim (int): The hidden dimension size
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            
        Returns:
            None
        """
        super(CustomTransformerEncoder, self).__init__()
        # Stack of CustomTransformerEncoderLayers
        self.layers = nn.ModuleList([CustomTransformerEncoderLayer(hidden_dim, num_heads, dropout)
                                     for _ in range(num_layers)])
        # Final normalization layer
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Forward pass through the transformer encoder.
        
        Args:
            src (torch.Tensor): Source sequence tensor
            src_mask (Optional[torch.Tensor]): Mask for source sequence
        
        Returns:
            Tuple[torch.Tensor, List[Optional[torch.Tensor]]]: Output tensor and attention weights from all layers
        """
        output = src
        attention_weights_all_layers: List[Optional[torch.Tensor]] = []
        # Pass through all encoder layers
        for layer in self.layers:
            output, attention_weights = layer(output, src_mask)
            attention_weights_all_layers.append(attention_weights)
        
        # Apply final normalization
        output = self.norm(output)
        
        return output, attention_weights_all_layers

# %% 
class TransformerEncoderDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, 
                 num_heads: int = 2, dropout: float = 0.5, contextual_dim: Optional[int] = None, 
                 lstm_hidden_dim: Optional[int] = None, lstm_num_layers: int = 1, 
                 conv_out_channels: Optional[int] = None, conv_kernel_size: int = 3, 
                 conv_stride: int = 1) -> None:
        """
        Initialize the transformer encoder-decoder model.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension size
            output_dim (int): Output dimension
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            contextual_dim (Optional[int]): Dimension for contextual embedding
            lstm_hidden_dim (Optional[int]): Hidden dimension for LSTM layer
            lstm_num_layers (int): Number of LSTM layers
            conv_out_channels (Optional[int]): Number of output channels for convolutional layer
            conv_kernel_size (int): Kernel size for convolutional layer
            conv_stride (int): Stride for convolutional layer
            
        Returns:
            None
        """
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
        """
        Freeze specific layers of the model for fine-tuning.
        
        Args:
            None
            
        Returns:
            None
        """
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
            
    def forward(self, src: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the transformer encoder-decoder model.
        
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            lengths (Optional[torch.Tensor]): Sequence lengths for masking
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Output tensor and attention weights from encoder and decoder
        """
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

