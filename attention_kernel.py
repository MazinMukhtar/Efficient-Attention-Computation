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

@triton.jit 
def _bwd_store_dk_dv(
    dk_ptrs: tl.tensor,
    dv_ptrs: tl.tensor,
    dk: tl.tensor,
    dv: tl.tensor,
    offs_n: tl.int32,
    offs_d: tl.int32,
    seqlen_k: tl.int32,
    headdim: tl.int32,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr
):
    ''' 
    Store delta of key and value 

    Args:
        dk_ptrs (tl.tensor): Pointers for delta of key
        dv_ptrs (tl.tensor): Pointers for delta of value
        dk (tl.tensor): Delta of key tensor of shape (seqlen_k, nheads, headdim)
        dv (tl.tensor): Delta of value tensor of shape (seqlen_k, nheads, headdim)
        offs_n (tl.int32): Offset for key sequence length
        offs_d (tl.int32): Offset for head dimension
        seqlen_k (tl.int32): Length of key sequence
        headdim (tl.int32): Dimension of each attention head
        EVEN_M (tl.constexpr): Whether query sequence length is divisible by BLOCK_M
        EVEN_N (tl.constexpr): Whether key sequence length is divisible by BLOCK_N
        EVEN_HEADDIM (tl.constexpr): Whether head dimension is divisible by BLOCK_HEADDIM
    '''

    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv) # storing delta of value, no padding
            tl.store(dk_ptrs, dk) # storing delta of key, no padding
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim) # storing delta of value, padding invalid head dimension
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim) # storing delta of key, padding invalid head dimension
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=(offs_n + offs_d)[:, None] < seqlen_k) # storing delta of value, padding invalid key tokens
            tl.store(dk_ptrs, dk, mask=(offs_n + offs_d)[:, None] < seqlen_k) # storing delta of key, padding invalid key tokens
        else:
            tl.store(dv_ptrs, dv, mask=((offs_n + offs_d)[:, None] < seqlen_k) & (offs_d[None, :] < headdim)) # storing delta of value, padding invalid key tokens and head dimension
            tl.store(dk_ptrs, dk, mask=((offs_n + offs_d)[:, None] < seqlen_k) & (offs_d[None, :] < headdim)) # storing delta of key, padding invalid key tokens and head dimension

@triton.jit 
def _bwd_kernel_one_col_block(
    start_n: tl.int32,
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    Bias: tl.tensor,
    DO: tl.tesnor,
    DQ: tl.tensor,
    DK: tl.tensor,
    DV: tl.tensor,
    LSE: tl.tensor,
    D: tl.tensor,
    softmax_scale: tl.float32,
    stride_qm: tl.int32,
    stride_kn: tl.int32,
    stride_vn: tl.int32,
    stride_bm: tl.int32,
    stride_dom: tl.int32,
    stride_dqm: tl.int32,
    stride_dkn: tl.int32,
    stride_dvn: tl.int32,
    seqlen_q: tl.int32,
    seqlen_k: tl.int32,
    headdim: tl.int32,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Ensuring begin_m is a multiple of BLOCK_M - 0 if not causal, otherwise the first block of the current row
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M

    # Initialzing offsets for query, key and head
    offs_qm = begin_m + tl.arange(0, BLOCK_M) # (BLOCK_M, ) 
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N) # (BLOCK_N, )
    offs_m = tl.arange(0, BLOCK_M) # (BLOCK_M, )
    offs_d = tl.arange(0, BLOCK_HEADDIM) # (BLOCK_HEADDIM, )

    # Iniitalizing pointers to query, key, value, delta of output and delta of query
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    # Query + (query token offset * query token stride) + head dimension offset
    # (seqlen_q, BLOCK_M, BLOCK_HEADDIM)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    # Key + (key token offset * key token stride) + head dimension offset
    # (seqlen_k, BLOCK_N, BLOCK_HEADDIM)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    # Value + (key token offset * value token stride) + head dimension offset
    # (seqlen_k, BLOCK_N, BLOCK_HEADDIM)
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    # Delta of Output + (query token offset * delta output stride) + head dimension offset
    # (seqlen_q, BLOCK_M, BLOCK_HEADDIM)
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    # Delta of Query + (query token offset * delta query stride) + head dimension offset
    # (seqlen_q, BLOCK_M, BLOCK_HEADDIM)
    
    # Initializing bias pointers
    if BIAS_TYPE == 'vector':
        b_ptrs = Bias + offs_n
        # Bias + key token offset
        # (seqlen_k, )
    elif BIAS_TYPE == 'matrix':
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
        # Bias + (query token offset * bias stride) + key token offset
        # (seqlen_q, BLOCK_M, BLOCK_N)

    # Initializing delta of key and delta of value 
    dv = tl.zeros((BLOCK_N, BLOCK_HEADDIM), dtype=tl.float32)
    dk = tl.zeros((BLOCK_N, BLOCK_HEADDIM), dtype=tl.float32)

    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        # Delta of Value + (key token offset * delta value stride) + head dimension offset
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        # Delta of Key + (key token offset * delta key stride) + head dimension offset
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M,
            EVEN_N,
            EVEN_HEADDIM
        )
        return 

    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs) # Loading key tokens, no padding
            v = tl.load(v_ptrs) # Loading value tokens, no paddings
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0) # Loading key tokens, padding invalid head dimension
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0) # Loading value tokens, padding invalid head dimension
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0) # Loading key tokens, padding invalid key tokens
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0) # Loading value tokens, padding invalid key tokens
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0
            ) # Loading key tokens, padding invalid key tokens and head dimension
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0
            ) # Loading value tokens, padding invalid key tokens and head dimension

    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)

        offs_m_curr = start_m + offs_m 

        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs) # Loading query tokens, no padding
        else:
            if EVEN_HEADDIM:
                tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0) # Loading query tokens, padding invalid query tokens
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0
                ) # Loading query tokens, padding invalid query tokens and head dimension

        qk = tl.dot(q, k, trans_b=True) # (BLOCK_M, BLOCK_N)

        if not EVEN_M:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float('-inf')) # Masking invalid key tokens

        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float('-inf')) # Masking future key tokens

        if BIAS_TYPE != 'none':
            tl.debug_barrier()
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
            elif BIAS_TYPE == 'matrix':
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0
                    ).to(tl.float32)

            qk = qk * softmax_scale + bias

        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        
        lse_i = tl.load(LSE + offs_m_curr)
        
        if BIAS_TYPE == 'none':
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])

        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0
            )

        dv += tl.dot(p.to(do.dtype), do, trans_a=True)

        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()

        Di = tl.load(D + offs_m_curr)

        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        dk += tl.dot(ds, q, trans_a=True)

        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, eviction_policy='evict_last') # Loading delta of query, no padding
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy='evict_last') # Storing delta of query, no padding
            else:
                if EVEN_HEADDIM:
                    dq=tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy='evict_last'
                    ) # Loading delta of query, padding invalid query tokens
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy='evict_last'
                    ) # Storing delta of query, padding invalid query tokens
                else:
                    dq = tl.load(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy='evict_last'
                    ) # Loading delta of query, padding invalid query tokens and head dimension
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy='evict_last'
                    )
        else: # Parallezing across seqlen_k dimension
            dq = tl.dot(ds, k)
            
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
                    )

        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

        if BIAS_TYPE == 'matrix':
            b_ptrs += BLOCK_M * stride_bm 

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])

    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M,
        EVEN_N, 
        EVEN_HEADDIM
    )

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

@triton.jit
def _bwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    Bias: tl.tensor,
    DO: tl.tensor,
    DQ: tl.tensor,
    DK: tl.tensor,
    DV: tl.tensor,
    LSE: tl.tensor,
    D: tl.tensor,
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
    stride_dob: tl.int32,
    stride_doh: tl.int32,
    stride_dom: tl.int32,
    stride_dqb: tl.int32,
    stride_dqh: tl.int32,
    stride_dqm: tl.int32,
    stride_dkb: tl.int32,
    stride_dkh: tl.int32,
    stride_dkn: tl.int32,
    stride_dvb: tl.int32,
    stride_dvh: tl.int32,
    stride_dvn: tl.int32,
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
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Initialzing offsets for head batch, head and batch
    off_hb = tl.program_id(1) 
    off_b = off_hb // nheads
    off_h = off_hb % nheads 

    # Offset pointers for batch and head
    Q += off_b * stride_qb + off_h * stride_qh
    # Query + (batch offset * query batch stride) + (head offset * query head stride)
    K += off_b * stride_kb + off_h * stride_kh
    # Key + (batch offset * key batch stride) + (head offset * key head stride)
    V += off_b * stride_vb + off_h * stride_vh
    # Value + (batch offset * value batch stride) + (head offset * value head stride)
    DO += off_b * stride_dob + off_h * stride_doh
    # Delta of Output + (batch offset * delta output batch stride) + (head offset * delta output head stride)
    DQ += off_b * stride_dqb + off_h * stride_dqh
    # Delta of Query + (batch offset * delta query batch stride) + (head offset * delta query head stride)
    DK += off_b * stride_dkb + off_h * stride_dkh 
    # Delta of Key + (batch offset * delta key batch stride) + (head offset * delta key head stride)
    DV += off_b * stride_dvb + off_h * stride_dvh 
    # Delta of Value + (batch offset * delta value batch stride) + (head offset * delta value head stride)

    if BIAS_TYPE != 'none':
        Bias += off_b * stride_bb + off_h * stride_bh
        # Bias + (batch offset * bias batch stride) + (head offset * bias head stride)

    D += off_hb * seqlen_q_rounded 
    LSE += off_hb * seqlen_q_rounded 

    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N) 

        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias, 
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N
            )
    else:
        start_n = tl.program_id(0)

        _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias, 
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N
        )

def _flash_attn_forward(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    bias: tl.tensor = None,
    causal: bool = False,
    softmax_scale: tl.float32 = None,
):
    '''
    Flash Attention Forward Pass - Self Attention Only

    Args:
        q: Query tensor of shape (batch_size, num_heads, seqlen_q, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seqlen_k, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seqlen_k, head_dim)
        bias: Bias tensor of shape (batch_size, num_heads, seqlen_q, seqlen_k)
        causal: Whether to apply causal masking
        softmax_scale: Scale factor for softmax
    '''

    # Shapes
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape # self attention ONLY 

    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, 'Current implementation only supports head_dim <= 128, fixing'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same dtype'
    assert q.dtype in [tl.float16, tl.bfloat16], 'Only float16 and bfloat16 are supported'
    assert q.is_cuda and k.is_cuda and v.is_cuda, 'Triton kernels only support CUDA'
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    # Bias
    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float], 'Bias must be same dtype as query or float'
        assert bias.is_cuda, 'Triton kernels only support CUDA'
        assert bias.dim == 4, 'Bias must be 4D'

        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = 'vector'
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = 'matrix'
        else:
            raise RuntimeError(
                'Last two dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)'
            )

        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128

    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads, META['num_wraps'], META['num_stages'])
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_wraps=grid[2],
        num_stages=grid[3]
    )

    return o, lse, softmax_scale

def _flash_attn_backward(
    do,
    q,
    k,
    v,
    o,
    lse,
    dq,
    dk,
    dv,
    bias=None,
    causal=False,
    softmax_scale=None
):
    '''
    Flash Attention Backward Pass - Self Attention Only

    Args:
        do: Delta of Output tensor of shape (batch_size, num_heads, seqlen_q, seqlen_k)
        q: Query tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
        v: Value tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
        o: Output tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
        lse: Log Sum Exponential tensor of shape (batch_size, num_heads, seqlen_q)
        dq: Delta of Query tensor of shape (batch_size, num_heads, seqlen_q, head_dim)
        dk: Delta of Key tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
        dv: Delta of Value tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
        bias: Bias tensor of shape (batch_size, seqlen_q, num_heads, seqlen_k)
        causal: Whether to apply causal masking
        softmax_scale: Scale factor for softmax
    '''

    if do.stride(-1) != 1:
        do = do.contiguous()

    # Shapes
    batch, seqlen_q, nheads, d = q.shape 
    _, seqlen_k, _, _ = k.shape

    assert d <= 128, 'Current implementation only supports head_dim <= 128, fixing'
    
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

    assert lse.shape == (batch, nheads, seqlen_q_rounded), 'LSE must be of shape (batch, num_heads, seqlen_q_rounded)'
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1, 'All tensors must be contiguous'
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1, 'Deltas must be contiguous'

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)

    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float], 'Bias must be same dtype as query or float'
        assert bias.is_cuda, 'Triton kernels only support CUDA'
        assert bias.dim == 4, 'Bias must be 4D'
        assert bias.stride(-1) != 1, 'Bias must be contiguous'

        if bias.shape[2:] == (1, seqlen_k):
            bias_type = 'vector'
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = 'matrix'
        else:
            raise RuntimeError(
                'Last two dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)'
            )

        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    grid = lambda META: (
        triton.cdiv(seqlen_k, META['BLOCK_N']) if META['SEQUENCE_PARALLEL'] else 1,
        batch * nheads,
        META['num_wraps'],
        META['num_stages']
    )

    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        bias_type,
        causal,
        BLOCK_HEADDIM
    )

    dq.copy_(dq_accum)
# export TRITON_PRINT_AUTOTUNING=1