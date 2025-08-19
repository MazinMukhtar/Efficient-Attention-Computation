# Library imports
import math 
import torch 
import triton
import triton.language as tl

# Maintain (BLOCK_M + 2 * BLOCK_N) * BLOCK_HEADDIM * TYPESIZE(BLOCK_HEADDIM) <= 64KB
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_wraps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_wraps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_wraps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_wraps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_wraps=8, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_wraps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_wraps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_wraps=8, num_stages=4),
    ],
    key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
)
@triton.heuristics(
    values={
        'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0,
        'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0,
        'EVEN_HEADDIM': lambda args: args['headdim'] % args['BLOCK_HEADDIM'] == 0,
    }
)

def _fwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    Bias: tl.tensor,
    Out: tl.tensor,
    Lse: tl.tensor,
    TMP: tl. tensor, # Scratchpad buffer to workaround compiler bug
    softmax_scale: tl.float32,
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_qm: tl.int32,
    stride_kb: tl.int32,
    stride_kh: tl.int32,
    stride_kn: tl.int32,
    stride_vb: tl.int32,
    stride_vh: tl.int32,
    stride_vn: tl.int32,
    stride_bb: tl.int32,
    stride_bh: tl.int32,
    stride_bm: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_om: tl.int32,
    nheads: tl.int32,
    seqlen_q: tl.int32,
    seqlen_k: tl.int32,
    seqlen_q_rounded: tl.int32,
    headdim: tl.int32,
    CACHE_KEY_SEQLEN_Q: tl.int32,
    CACHE_KEY_SEQLEN_K: tl.int32,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    '''
    Forward kernel for attention computation.

    Args:
        Q (tl.tensor): Query tensor of shape (seqlen_q, nheads, headdim)
        K (tl.tensor): Key tensor of shape (seqlen_k, nheads, headdim)
        V (tl.tensor): Value tensor of shape (seqlen_k, nheads, headdim)
        Bias (tl.tensor): Bias tensor of shape (nheads, seqlen_q, seqlen_k)
        Out (tl.tensor): Output tensor of shape (seqlen_q, nheads, headdim)
        Lse (tl.tensor): Log sum exp tensor of shape (seqlen_q, nheads)
        TMP (tl.tensor): Scratchpad buffer of shape (seqlen_q, nheads, BLOCK_M + 2 * BLOCK_N)
        softmax_scale (tl.float32): Scale factor for softmax
        stride_qb (tl.int32): Stride for query batch
        stride_qh (tl.int32): Stride for query head
        stride_qm (tl.int32): Stride for query sequence length
        stride_kb (tl.int32): Stride for key batch
        stride_kh (tl.int32): Stride for key head
        stride_kn (tl.int32): Stride for key sequence length
        stride_vb (tl.int32): Stride for value batch
        stride_vh (tl.int32): Stride for value head
        stride_vn (tl.int32): Stride for value sequence length
        stride_bb (tl.int32): Stride for bias batch
        stride_bh (tl.int32): Stride for bias head
        stride_bm (tl.int32): Stride for bias sequence length
        stride_ob (tl.int32): Stride for output batch
        stride_oh (tl.int32): Stride for output head
        stride_om (tl.int32): Stride for output sequence length
        nheads (tl.int32): Number of attention heads
        seqlen_q (tl.int32): Length of query sequence
        seqlen_k (tl.int32): Length of key sequence
        seqlen_q_rounded (tl.int32): Rounded query sequence length
        headdim (tl.int32): Dimension of each attention head
        CACHE_KEY_SEQLEN_Q (tl.int32): Cache key for query sequence length
        CACHE_KEY_SEQLEN_K (tl.int32): Cache key for key sequence length
        BIAS_TYPE (tl.constexpr): Type of bias
        IS_CAUSAL (tl.constexpr): Whether to use causal attention
        BLOCK_HEADDIM (tl.constexpr): Block size for head dimension
        EVEN_M (tl.constexpr): Whether query sequence length is divisible by BLOCK_M
        EVEN_N (tl.constexpr): Whether key sequence length is divisible by BLOCK_N
        EVEN_HEADDIM (tl.constexpr): Whether head dimension is divisible by BLOCK_HEADDIM
        BLOCK_M (tl.constexpr): Block size for query sequence length
        BLOCK_N (tl.constexpr): Block size for key sequence length
    '''
    # Get indices for query tokens and batch-head pair
    start_m = tl.program_id(0) # Starting index for query tokens 
    off_hb = tl.program_id(1) # Starting index for batch-head pair 
    off_b = off_hb // nheads # Batch index
    off_h = off_hb % nheads # Head index

    # Initialize offsets for query tokens, key tokens, and head dimension
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # Offset for query tokens
    offs_n = tl.arange(0, BLOCK_N) # Offset for key tokens
    offs_d = tl.arange(0, BLOCK_HEADDIM) # Offset for head dimension

    # Initializing pointers for query, key and value 
    q_ptrs = (
        Q 
        + off_b * stride_qb 
        + off_h * stride_qh 
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
        # Query + (batch offset * batch stride) + (head offset * head stride) + (query token offset * query token stride) + (head dimension offset * head dimension stride)
        # (seqlen_q, BLOCK_M, BLOCK_HEADDIM)
    )
    k_ptrs = (
        K 
        + off_b * stride_kb 
        + off_h * stride_kh 
        + (offs_n[:, None] * stride_kn + offs_d[None, :])
        # Key + (batch offset * batch stride) + (head offset * head stride) + (key token offset * key token stride) + (head dimension offset * head dimension stride)
        # (seqlen_k, BLOCK_N, BLOCK_HEADDIM)
    )
    v_ptrs = (
        V 
        + off_b * stride_vb 
        + off_h * stride_vh 
        + (offs_n[:, None] * stride_vn + offs_d[None, :])
        # Value + (batch offset * batch stride) + (head offset * head stride) + (value token offset * value token stride) + (head dimension offset * head dimension stride)
        # (seqlen_k, BLOCK_N, BLOCK_HEADDIM)
    )

    if BIAS_TYPE == 'vector': # One bias per key position
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n 
        # Bias + (batch offset * batch stride) + (head offset * head stride) + key token offset
        # (seqlen_k)
    elif BIAS_TYPE == 'matrix': # One bias per query-key pair
        b_ptrs = (
            Bias 
            + off_b * stride_bb 
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
        # Bias + (batch offset * batch stride) + (head offset * head stride) + (query token offset * query token stride) + (key token offset)
        # (seqlen_q, seqlen_k)
    # Initializing pointers for scratchpad buffer, log sum exp and m
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m 
    # Scratchpad buffer + (batch-head offset * sequence length) + query token offset
    # (seqlen_q, BLOCK_M + 2 * BLOCK_N)
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    # (BLOCK_M)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # (BLOCK_M)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # (BLOCK_M, BLOCK_HEADDIM)

    # Loading query - staying in SRAM throughout computation
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            # seqlen_q, seqlen_k and headdim all divisible by BLOCK_M, BLOCK_N and BLOCK_HEADDIM
            q = tl.load(
                q_ptrs
                ) # loading query tokens, no padding
        else: 
            # seqlen_q and seqlen_k divisible by BLOCK_M and BLOCK_N, but headdim not divisible by BLOCK_HEADDIM
            q = tl.load(
                q_ptrs, 
                mask=offs_d[:, None] < headdim, 
                other=0.0
                ) # loading query tokens, padding invalid head dimension
    else:
        # seqlen_q or seqlen_k not divisible by BLOCK_M or BLOCK_N, but headdim divisible by BLOCK_HEADDIM
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs, 
                mask=offs_m[:, None] < seqlen_q, 
                other=0.0
                ) # loading query tokens, padding invalid query tokens
        # seqlen_q, seqlen_k and headdim all not divisible by BLOCK_M, BLOCK_N and BLOCK_HEADDIM
        else: 
            q = tl.load(
                q_ptrs, 
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), 
                other=0.0
                ) # loading query tokens, padding invalid query tokens and head dimension

    # Looping over k, v 
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Loading key - staying in SRAM throughout computation
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn
                    ) # loading key tokens, no padding
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn, 
                    mask=offs_d[None, :] < headdim, 
                    other=0.0
                    ) # loading key tokens, padding invalid head dimension
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn, 
                    mask=(start_n + offs_n)[:, None] < seqlen_k, 
                    other=0.0
                    ) # loading key tokens, padding invalid key tokens
            else: 
                k = tl.load(
                    k_ptrs + start_n * stride_kn, 
                    mask=((start_n + offs_n)[:, None] < seqlen_k & (offs_d[None, :] < headdim)), 
                    other=0.0
                    ) # loading key tokens, padding invalid key tokens and head dimension

        # Computing S 
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k, trans_b=True)

        if not EVEN_N: # Masking for softmax calculation
            s += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float('-inf')) # invalid key tokens
        if IS_CAUSAL:
            s += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float('-inf')) # key tokens after query tokens

        # Loading bias  
        if BIAS_TYPE != 'none':
            if BIAS_TYPE == 'vector': # One bias per key position
                if EVEN_N:
                    bias = tl.load(
                        b_ptrs + start_n
                        ).to(tl.float32) # loading bias, no padding
                else:
                    bias = tl.load(
                        b_ptrs + start_n, 
                        mask=(start_n + offs_n) < seqlen_k, 
                        other=0.0
                        ).to(tl.float32) # loading bias, padding invalid key tokens
                bias = bias[None, :] # (1, BLOCK_N)

            elif BIAS_TYPE =='matrix': # One bias per query-key pair
                if EVEN_M & EVEN_N: 
                    bias = tl.load(b_ptrs + start_n).to(tl.float32) # loading bias, no padding
                else: 
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0
                    ).to(tl.float32) # loading bias, padding invalid query tokens and key tokens

            # computing m_ij and p - bias
            s = s * softmax_scale + bias 
            m_ij = tl.maximum(tl.max(s, 1), lse_i)
            p = tl.exp(s - m_ij[:, None])
        else:
            # computing m_ij and p - no bias
            s *= softmax_scale 
            m_ij = tl.maximum(tl.max(s, 1), lse_i)
            p = tl.exp(s - m_ij[:, None]);
        
        # Computing l_ij
        l_ij = tl.sum(p, 1)

        # Scaling o 
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o *= acc_o_scale[:, None]

        # Loading value - staying in SRAM throughout computation
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn
                ) # loading value tokens, no padding
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0
                ) # loading value tokens, padding invalid head dimension
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0
                ) # loading value tokens, padding invalid key tokens
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0
                ) # loading value tokens, padding invalid key tokens and head dimension

        # computing o
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # updating m_1 and lse_i 
        m_i = m_ij 
        l_i_new = tl.exp(lse_i - m_ij) + l_ij 
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o *= o_scale[:, None]

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Writing back l and m 
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m 
    tl.store(lse_ptrs, lse_i)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out 
        + off_b * stride_ob 
        + off_h * stride_oh 
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(
                out_ptrs, acc_o
                ) # storing output, no padding
        else:
            tl.store(
                out_ptrs,
                acc_o, 
                mask=offs_d[None, :] < headdim
            ) # storing output, padding invalid head dimension
    else:
        if EVEN_HEADDIM:
            tl.store(
                out_ptrs, 
                acc_o,
                mask=offs_m[:, None] < seqlen_q
            ) # storing output, padding invalid query tokens 
        else:
            tl.store(
                out_ptrs,
                acc_o, 
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            ) # storing output, padding invalid query tokens and head dimension

@triton.jit 
def _bwd_preprocess_do_o_dot(
    Out: tl.tensor,
    DO: tl.tensor,
    Delta: tl.tensor,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_om: tl.int32,
    stride_dob: tl.int32,
    stride_doh: tl.int32,
    stride_dom: tl.int32,
    nheads: tl.int32,
    seqlen_q: tl.int32,
    seqlen_q_rounded: tl.int32,
    headdim: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr
):
    '''
    Preprocess for backward pass of attention computation.

    Args:
        Out (tl.tensor): Output tensor of shape (seqlen_q, nheads, headdim)
        DO (tl.tensor): Delta of output tensor of shape (seqlen_q, nheads, headdim)
        Delta (tl.tensor): Delta tensor of shape (seqlen_q, nheads, headdim)
        stride_ob (tl.int32): Stride for output batch
        stride_oh (tl.int32): Stride for output head
        stride_om (tl.int32): Stride for output sequence length
        stride_dob (tl.int32): Stride for delta output batch
        stride_doh (tl.int32): Stride for delta output head
        stride_dom (tl.int32): Stride for delta output sequence length
        nheads (tl.int32): Number of attention heads
        seqlen_q (tl.int32): Length of query sequence
        seqlen_q_rounded (tl.int32): Rounded query sequence length
        headdim (tl.int32): Dimension of each attention head
        BLOCK_M (tl.constexpr): Block size for query sequence length
        BLOCK_HEADDIM (tl.constexpr): Block size for head dimension
    '''

    start_m = tl.program_id(0) # Starting index for query tokens 
    off_hb = tl.program_id(1) # Starting index for batch-head pair
    off_b = off_hb // nheads # Batch index
    off_h = off_hb % nheads # Head index

    # Initializing offsets for query and head dimension 
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # Offset for query tokens
    offs_d = tl.arange(0, BLOCK_HEADDIM) # Offset for head dimension

    # Loading output - staying in SRAM throughout computation
    o = tl.load(
        Out 
        + off_b * stride_ob 
        + off_h * stride_oh 
        + (offs_m[:, None] * stride_om + offs_d[None, :]),
        # Out + (batch offset * batch stride) + (head offset * head stride) + (query token offset * query token stride) + head dimension offset
        # (seqlen_q, BLOCK_M, BLOCK_HEADDIm)
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), # masking invalid query tokens and head dimension
        other=0.0 # padding invalid query tokens and head dimension
    ).to(tl.float32)

    # Loading delta of output - staying in SRAM throughout computation
    do = tl.load(
        DO 
        + off_b * stride_dob 
        + off_h * stride_doh 
        + offs_m[:, None] * stride_dom 
        + offs_d[None, :], 
        # Delta of Output + (batch offset * batch stride) + (head offset * head stride) + (query token offset * query token stride) + head dimension offset
        # (seqlen_q, BLOCK_M, BLOCK_HEADDIM)
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), # masking invalid query tokens and head dimension
        other=0.0 # padding invalid query tokens and head dimension
    ).to(tl.float32)

    # Computing delta
    delta = tl.sum(o * do, axis=1) # (BLOCK_M, )

    # Writing back delta
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta) 


# export TRITON_PRINT_AUTOTUNING=1