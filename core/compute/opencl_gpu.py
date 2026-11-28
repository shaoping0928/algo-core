"""Zero-dependency OpenCL GPU Compute Backend via native OS ``OpenCL.dll`` / ``libOpenCL.so``.

Automatically selects the highest-memory OpenCL GPU (e.g., AMD Radeon RX 9070 XT
``gfx1201``) and compiles fused float32 compute kernels for:
  - Batched RC / Integrator / Differentiator dynamical circuit node updates
  - Multi-tick on-device dynamical circuit rollouts (zero host PCIe roundtrips)
  - Batched sparse CSR graph spreading activation (SpMV)

Falls back gracefully when no OpenCL GPU is available.
"""
from __future__ import annotations

import ctypes
import sys
from typing import Optional, Tuple

import numpy as np


# OpenCL constants
CL_DEVICE_TYPE_GPU = 1 << 2
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_PLATFORM_NAME = 0x0902
CL_DEVICE_NAME = 0x102B
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_GLOBAL_MEM_SIZE = 0x1010
CL_MEM_READ_WRITE = 1 << 0
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_TRUE = 1
CL_PROGRAM_BUILD_LOG = 0x1183


CIRCUIT_KERNELS_SRC = r"""
__kernel void circuit_step_kernel(
    const int n_total,
    const int n_nodes,
    const float dt,
    const float alpha_int,
    const float w_cap,
    const float w_int,
    const float w_diff,
    __global const float* I_in,
    __global const float* R_arr,
    __global const float* C_arr,
    __global float* V_cap,
    __global float* V_int,
    __global float* V_diff,
    __global float* V_out
) {
    int gid = get_global_id(0);
    if (gid >= n_total) return;
    int node = gid % n_nodes;
    float r = fmax(R_arr[node], 1e-4f);
    float c = fmax(C_arr[node], 1e-4f);
    float vc = V_cap[gid];
    float vi = V_int[gid];
    float i_in = I_in[gid];

    // RC Capacitor update: dV_C/dt = (I_in - V_C / R) / C
    float decay = fmax(1.0f - (dt / (r * c)), 0.0f);
    float vc_new = decay * vc + (dt / c) * i_in;

    // Low-pass Integrator element
    float vi_new = (1.0f - alpha_int) * vi + alpha_int * vc_new;

    // High-pass Differentiator element (onset / surprise / change current)
    float vd_new = vc_new - vi_new;

    // Composite nonlinear node activation
    float raw = w_cap * vc_new + w_int * vi_new + w_diff * vd_new;
    V_cap[gid] = vc_new;
    V_int[gid] = vi_new;
    V_diff[gid] = vd_new;
    V_out[gid] = tanh(raw);
}

__kernel void csr_spmv_batch_kernel(
    const int batch_size,
    const int n_nodes,
    __global const int* row_ptr,
    __global const int* col_idx,
    __global const float* vals,
    __global const float* x_in,
    __global float* y_out
) {
    int gid = get_global_id(0);
    int total = batch_size * n_nodes;
    if (gid >= total) return;
    int b = gid / n_nodes;
    int row = gid - b * n_nodes;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    int base = b * n_nodes;
    float acc = 0.0f;
    for (int e = start; e < end; ++e) {
        acc += vals[e] * x_in[base + col_idx[e]];
    }
    y_out[gid] = acc;
}

__kernel void circuit_beam_branch_kernel(
    const int batch_size,
    const int n_nodes,
    const float dt,
    const float alpha_int,
    const float realize_gain,
    const float w_cap,
    const float w_int,
    const float w_diff,
    __global const int* parent_idx,
    __global const int* action_sym,
    __global const float* action_sal,
    __global const float* I_spread,
    __global const float* R_arr,
    __global const float* C_arr,
    __global const float* V_cap_in,
    __global const float* V_int_in,
    __global float* V_cap_out,
    __global float* V_int_out,
    __global float* V_diff_out,
    __global float* V_out
) {
    int gid = get_global_id(0);
    int total = batch_size * n_nodes;
    if (gid >= total) return;
    int b = gid / n_nodes;
    int node = gid - b * n_nodes;
    int p = parent_idx[b];
    int p_gid = p * n_nodes + node;

    float r = fmax(R_arr[node], 1e-4f);
    float c = fmax(C_arr[node], 1e-4f);
    float decay = fmax(1.0f - (dt / (r * c)), 0.0f);
    float inv_c = dt / c;

    float vc = decay * V_cap_in[p_gid] + inv_c * I_spread[p_gid];
    float vi = (1.0f - alpha_int) * V_int_in[p_gid] + alpha_int * vc;

    int act = action_sym[b];
    float sal = action_sal[b];
    if (node == act && sal > 0.0f) {
        vi += realize_gain * fmax(sal, 0.5f);
        vc *= 0.72f;
    }

    float vd = vc - vi;
    float raw = w_cap * vc + w_int * vi + w_diff * vd;
    V_cap_out[gid] = vc;
    V_int_out[gid] = vi;
    V_diff_out[gid] = vd;
    V_out[gid] = tanh(raw);
}

// 32x32 register-tiled (2x2/thread) LDS SGEMM with bank-conflict-free padding supporting NN/NT/TN
__kernel void sgemm_tiled_kernel(
    const int M,
    const int N,
    const int K,
    const int transA,
    const int transB,
    const float alpha,
    const float beta,
    const int a_offset,
    const int c_offset,
    __global const float* A,
    __global const float* B,
    __global float* C
) {
    A += a_offset;
    C += c_offset;
    __local float As[32][17];
    __local float Bs[16][33];
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int col0 = get_group_id(0) * 32 + lx;
    int col1 = col0 + 16;
    int row0 = get_group_id(1) * 32 + ly;
    int row1 = row0 + 16;

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    int n_tiles = (K + 15) / 16;
    for (int t = 0; t < n_tiles; ++t) {
        int a_k = t * 16 + lx;
        int b_k = t * 16 + ly;
        if (transA) {
            // Swap the load coordinates so adjacent lanes read adjacent rows in A^T.
            int src_row = get_group_id(1) * 32 + lx;
            As[lx][ly] = (src_row < M && b_k < K) ? A[b_k * M + src_row] : 0.0f;
            As[lx + 16][ly] = (src_row + 16 < M && b_k < K) ? A[b_k * M + src_row + 16] : 0.0f;
        } else if (a_k < K) {
            As[ly][lx] = (row0 < M) ? A[row0 * K + a_k] : 0.0f;
            As[ly + 16][lx] = (row1 < M) ? A[row1 * K + a_k] : 0.0f;
        } else {
            As[ly][lx] = 0.0f;
            As[ly + 16][lx] = 0.0f;
        }
        if (transB) {
            int src_col = get_group_id(0) * 32 + ly;
            Bs[lx][ly] = (src_col < N && a_k < K) ? B[src_col * K + a_k] : 0.0f;
            Bs[lx][ly + 16] = (src_col + 16 < N && a_k < K) ? B[(src_col + 16) * K + a_k] : 0.0f;
        } else if (b_k < K) {
            Bs[ly][lx] = (col0 < N) ? B[b_k * N + col0] : 0.0f;
            Bs[ly][lx + 16] = (col1 < N) ? B[b_k * N + col1] : 0.0f;
        } else {
            Bs[ly][lx] = 0.0f;
            Bs[ly][lx + 16] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a0 = As[ly][k];
            float a1 = As[ly + 16][k];
            float b0 = Bs[k][lx];
            float b1 = Bs[k][lx + 16];
            c00 = fma(a0, b0, c00);
            c01 = fma(a0, b1, c01);
            c10 = fma(a1, b0, c10);
            c11 = fma(a1, b1, c11);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row0 < M && col0 < N) {
        int idx = row0 * N + col0;
        C[idx] = (beta == 0.0f) ? (alpha * c00) : fma(alpha, c00, beta * C[idx]);
    }
    if (row0 < M && col1 < N) {
        int idx = row0 * N + col1;
        C[idx] = (beta == 0.0f) ? (alpha * c01) : fma(alpha, c01, beta * C[idx]);
    }
    if (row1 < M && col0 < N) {
        int idx = row1 * N + col0;
        C[idx] = (beta == 0.0f) ? (alpha * c10) : fma(alpha, c10, beta * C[idx]);
    }
    if (row1 < M && col1 < N) {
        int idx = row1 * N + col1;
        C[idx] = (beta == 0.0f) ? (alpha * c11) : fma(alpha, c11, beta * C[idx]);
    }
}

__kernel void gated_bilinear_fwd_kernel(
    const int B,
    const int R,
    __global const float* A_pre,
    __global const float* b_loc,
    __global const float* G_pre,
    __global const float* b_ctx,
    __global float* A_out,
    __global float* G_out,
    __global float* Q_out
) {
    int gid = get_global_id(0);
    int tot = B * R;
    if (gid >= tot) return;
    int c = gid % R;
    float a = tanh(A_pre[gid] + b_loc[c]);
    float g_in = clamp(G_pre[gid] + b_ctx[c], -16.0f, 16.0f);
    float g = 1.0f / (1.0f + exp(-g_in));
    A_out[gid] = a;
    G_out[gid] = g;
    Q_out[gid] = a * g;
}

__kernel void gated_bilinear_bwd_kernel(
    const int B,
    const int R,
    __global const float* dQ,
    __global const float* A,
    __global const float* G,
    __global float* dA_pre,
    __global float* dG_pre
) {
    int gid = get_global_id(0);
    int tot = B * R;
    if (gid >= tot) return;
    float dq = dQ[gid];
    float a = A[gid];
    float g = G[gid];
    dA_pre[gid] = dq * g * (1.0f - a * a);
    dG_pre[gid] = dq * a * g * (1.0f - g);
}

// 1 work-group of 256 threads per row: exact parallel Softmax + Cross-Entropy + in-place dL
__kernel void softmax_ce_grad_kernel(
    const int B,
    const int V,
    const float inv_B,
    __global const float* b_out,
    __global const int* targets,
    __global float* logits_and_dL,
    __global float* row_losses
) {
    int row = get_group_id(0);
    if (row >= B) return;
    int lid = get_local_id(0);
    int base = row * V;
    int tgt = targets[row];
    if (tgt <= 0 || tgt >= V) {
        if (lid == 0) row_losses[row] = 0.0f;
        for (int c = lid; c < V; c += 256) {
            logits_and_dL[base + c] = 0.0f;
        }
        return;
    }

    __local float smem[256];
    // 1. Add bias & mask col 0, find row max
    float local_max = -1e30f;
    for (int c = lid; c < V; c += 256) {
        float v = (c == 0) ? -1e4f : (logits_and_dL[base + c] + b_out[c]);
        logits_and_dL[base + c] = v;
        if (v > local_max) local_max = v;
    }
    smem[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s && smem[lid + s] > smem[lid]) smem[lid] = smem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rmax = smem[0];

    // 2. Exp & row sum
    float local_sum = 0.0f;
    for (int c = lid; c < V; c += 256) {
        float ex = exp(clamp(logits_and_dL[base + c] - rmax, -30.0f, 0.0f));
        logits_and_dL[base + c] = ex;
        local_sum += ex;
    }
    smem[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / fmax(smem[0], 1e-12f);

    // 3. Normalize to probs, record -log(p_true), overwrite with dL = (probs - onehot) / B
    for (int c = lid; c < V; c += 256) {
        float p = logits_and_dL[base + c] * inv_sum;
        if (c == tgt) {
            row_losses[row] = -log(fmax(p, 1e-9f));
            p -= 1.0f;
        }
        logits_and_dL[base + c] = p * inv_B;
    }
}

__kernel void col_sum_kernel(
    const int B,
    const int N,
    __global const float* X,
    __global float* out_sum
) {
    int c = get_global_id(0);
    if (c >= N) return;
    float acc = 0.0f;
    for (int r = 0; r < B; ++r) {
        acc += X[r * N + c];
    }
    out_sum[c] = acc;
}

__kernel void adamw_update_kernel(
    const int n_total,
    const float lr_step,
    const float wd_scale,
    __global float* param,
    __global const float* grad,
    __global float* m_buf,
    __global float* v_buf
) {
    int gid = get_global_id(0);
    if (gid >= n_total) return;
    float g = grad[gid];
    float m = 0.9f * m_buf[gid] + 0.1f * g;
    float v = 0.999f * v_buf[gid] + 0.001f * g * g;
    m_buf[gid] = m;
    v_buf[gid] = v;
    float p = param[gid] * wd_scale - lr_step * (m / (sqrt(v) + 1e-7f));
    param[gid] = p;
}

// Residual 2nd-layer context-gated forward: H2 = tanh(H2_pre + b_h2), G2 = sigmoid(G2_pre + b_ctx2), Q2 = Q1 + H2 * G2
__kernel void res_gated_fwd_kernel(
    const int B,
    const int R,
    __global const float* Q1,
    __global const float* H2_pre,
    __global const float* b_h2,
    __global const float* G2_pre,
    __global const float* b_ctx2,
    __global float* H2_out,
    __global float* G2_out,
    __global float* Q2_out
) {
    int gid = get_global_id(0);
    int tot = B * R;
    if (gid >= tot) return;
    int c = gid % R;
    float h2 = tanh(H2_pre[gid] + b_h2[c]);
    float g_in = clamp(G2_pre[gid] + b_ctx2[c], -16.0f, 16.0f);
    float g2 = 1.0f / (1.0f + exp(-g_in));
    H2_out[gid] = h2;
    G2_out[gid] = g2;
    Q2_out[gid] = Q1[gid] + h2 * g2;
}

// Residual 2nd-layer backward: dH2_pre, dG2_pre from dQ2 (dQ1_res is added via separate SGEMM beta=1)
__kernel void res_gated_bwd_kernel(
    const int B,
    const int R,
    __global const float* dQ2,
    __global const float* H2,
    __global const float* G2,
    __global float* dH2_pre,
    __global float* dG2_pre
) {
    int gid = get_global_id(0);
    int tot = B * R;
    if (gid >= tot) return;
    float dq = dQ2[gid];
    float h = H2[gid];
    float g = G2[gid];
    dH2_pre[gid] = dq * g * (1.0f - h * h);
    dG2_pre[gid] = dq * h * g * (1.0f - g);
}

// Gather dynamic trainable embeddings E_dyn[tok_cur], E_dyn[tok_prev] into Xl[:, 0:2D]
__kernel void embed_gather_2_kernel(
    const int B,
    const int D,
    const int loc_dim,
    __global const int* tok_cur,
    __global const int* tok_prev,
    __global const float* E_dyn,
    __global float* Xl
) {
    int gid = get_global_id(0);
    int tot = B * D;
    if (gid >= tot) return;
    int r = gid / D;
    int d = gid - r * D;
    int c0 = tok_cur[r];
    int c1 = tok_prev[r];
    int base = r * loc_dim;
    Xl[base + d] = E_dyn[c0 * D + d];
    Xl[base + D + d] = E_dyn[c1 * D + d];
}

// Deterministic (no atomic contention!) per-vocab-row gather-reduce gradient for E_dyn[V, D]
__kernel void embed_grad_reduce_2_kernel(
    const int B,
    const int V,
    const int D,
    const int loc_dim,
    __global const int* tok_cur,
    __global const int* tok_prev,
    __global const float* dXl,
    __global float* dE_dyn
) {
    int gid = get_global_id(0);
    int tot = V * D;
    if (gid >= tot) return;
    int v = gid / D;
    int d = gid - v * D;
    if (v == 0) {
        dE_dyn[gid] = 0.0f;
        return;
    }
    float acc = 0.0f;
    for (int r = 0; r < B; ++r) {
        if (tok_cur[r] == v) {
            acc += dXl[r * loc_dim + d];
        }
        if (tok_prev[r] == v) {
            acc += dXl[r * loc_dim + D + d];
        }
    }
    dE_dyn[gid] = acc;
}

// Gather single token embedding E[tok[i]] -> X[i, :] over N = B*T tokens
__kernel void embed_gather_1_kernel(
    const int N,
    const int D,
    __global const int* tok_ids,
    __global const float* E,
    __global float* X_out
) {
    int gid = get_global_id(0);
    int tot = N * D;
    if (gid >= tot) return;
    int r = gid / D;
    int d = gid - r * D;
    int t = tok_ids[r];
    X_out[gid] = (t > 0) ? E[t * D + d] : 0.0f;
}

// Deterministic per-vocab-row gather-reduce gradient for E[V, D] over N = B*T tokens
__kernel void embed_grad_reduce_1_kernel(
    const int N,
    const int V,
    const int D,
    __global const int* tok_ids,
    __global const float* dX,
    __global float* dE
) {
    int gid = get_global_id(0);
    int tot = V * D;
    if (gid >= tot) return;
    int v = gid / D;
    int d = gid - v * D;
    if (v == 0) {
        dE[gid] = 0.0f;
        return;
    }
    float acc = 0.0f;
    for (int r = 0; r < N; ++r) {
        if (tok_ids[r] == v) {
            acc += dX[r * D + d];
        }
    }
    dE[gid] = acc;
}

// Fused Gated RC / CuDNN-GRU-v2 forward step at timestep t for B parallel streams
// Xg_all/Sg_all/cache_zrh: [T, B, 3H], s_cur: [B, H], s_prev_all/s_all: [T, B, H]
__kernel void gru_step_fwd_kernel(
    const int B,
    const int H,
    const int t,
    __global const float* Xg_all,
    __global const float* Sg_all,
    __global const float* b_x,
    __global const float* b_s,
    __global float* s_cur,
    __global float* s_prev_all,
    __global float* s_all,
    __global float* cache_zrh
) {
    int gid = get_global_id(0);
    int tot = B * H;
    if (gid >= tot) return;
    int b = gid / H;
    int h = gid - b * H;
    int H3 = 3 * H;
    int xg_base = (t * B + b) * H3;
    float z_pre = clamp(Xg_all[xg_base + h] + Sg_all[xg_base + h] + b_x[h] + b_s[h], -16.0f, 16.0f);
    float r_pre = clamp(Xg_all[xg_base + H + h] + Sg_all[xg_base + H + h] + b_x[H + h] + b_s[H + h], -16.0f, 16.0f);
    float z = 1.0f / (1.0f + exp(-z_pre));
    float r = 1.0f / (1.0f + exp(-r_pre));

    float sg_h = Sg_all[xg_base + 2 * H + h] + b_s[2 * H + h];
    float h_tilde = tanh(Xg_all[xg_base + 2 * H + h] + b_x[2 * H + h] + r * sg_h);
    float sp = s_cur[gid];
    float s_new = (1.0f - z) * sp + z * h_tilde;

    int state_index = t * tot + gid;
    s_prev_all[state_index] = sp;
    s_all[state_index] = s_new;
    s_cur[gid] = s_new;
    cache_zrh[xg_base + h] = z;
    cache_zrh[xg_base + H + h] = r;
    cache_zrh[xg_base + 2 * H + h] = h_tilde;
}

inline void gru_projection_fused_store(
    const int B, const int H, const int T, const int t, const int b, const int h,
    const float gz, const float gr, const float gh,
    __global const float* Xg, __global const float* bx, __global const float* bs,
    __global float* cur, __global float* prev_all, __global float* state_all,
    __global float* Sg, __global float* cache
) {
    const int H3 = 3 * H;
    const int gidx = (t * B + b) * H3;
    const int sidx = (t * B + b) * H + h;
    const float sp = (t == 0) ? cur[b * H + h] : state_all[sidx - B * H];
    Sg[gidx + h] = gz;
    Sg[gidx + H + h] = gr;
    Sg[gidx + 2 * H + h] = gh;
    const float z_pre = clamp(Xg[gidx + h] + gz + bx[h] + bs[h], -16.0f, 16.0f);
    const float r_pre = clamp(Xg[gidx + H + h] + gr + bx[H + h] + bs[H + h], -16.0f, 16.0f);
    const float z = 1.0f / (1.0f + exp(-z_pre));
    const float r = 1.0f / (1.0f + exp(-r_pre));
    const float h_tilde = tanh(Xg[gidx + 2 * H + h] + bx[2 * H + h] + r * (gh + bs[2 * H + h]));
    const float s_new = (1.0f - z) * sp + z * h_tilde;
    prev_all[sidx] = sp;
    state_all[sidx] = s_new;
    if (t == T - 1) cur[b * H + h] = s_new;
    cache[gidx + h] = z;
    cache[gidx + H + h] = r;
    cache[gidx + 2 * H + h] = h_tilde;
}

// Recurrent projection + GRU update in one launch. The prior state for t>0 is
// read from the completed state slice, avoiding cross-workgroup read/write races.
__kernel void gru_projection_fused_kernel(
    const int B, const int H, const int T, const int t,
    __global const float* Xg, __global const float* U,
    __global const float* bx, __global const float* bs,
    __global float* cur, __global float* prev_all, __global float* state_all,
    __global float* Sg, __global float* cache
) {
    __local float As[32][17];
    __local float Bz[16][33], Br[16][33], Bh[16][33];
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int col0 = get_group_id(0) * 32 + lx, col1 = col0 + 16;
    const int row0 = get_group_id(1) * 32 + ly, row1 = row0 + 16;
    const int H3 = 3 * H;
    float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
    float r00 = 0.0f, r01 = 0.0f, r10 = 0.0f, r11 = 0.0f;
    float h00 = 0.0f, h01 = 0.0f, h10 = 0.0f, h11 = 0.0f;
    const int n_tiles = (H + 15) / 16;
    for (int tile = 0; tile < n_tiles; ++tile) {
        const int ak = tile * 16 + lx, bk = tile * 16 + ly;
        const int prev_base = (t == 0) ? 0 : (t - 1) * B * H;
        As[ly][lx] = (row0 < B && ak < H) ? ((t == 0) ? cur[row0 * H + ak] : state_all[prev_base + row0 * H + ak]) : 0.0f;
        As[ly + 16][lx] = (row1 < B && ak < H) ? ((t == 0) ? cur[row1 * H + ak] : state_all[prev_base + row1 * H + ak]) : 0.0f;
        Bz[ly][lx] = (bk < H && col0 < H) ? U[bk * H3 + col0] : 0.0f;
        Bz[ly][lx + 16] = (bk < H && col1 < H) ? U[bk * H3 + col1] : 0.0f;
        Br[ly][lx] = (bk < H && col0 < H) ? U[bk * H3 + H + col0] : 0.0f;
        Br[ly][lx + 16] = (bk < H && col1 < H) ? U[bk * H3 + H + col1] : 0.0f;
        Bh[ly][lx] = (bk < H && col0 < H) ? U[bk * H3 + 2 * H + col0] : 0.0f;
        Bh[ly][lx + 16] = (bk < H && col1 < H) ? U[bk * H3 + 2 * H + col1] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float a0 = As[ly][k], a1 = As[ly + 16][k];
            const float bz0 = Bz[k][lx], bz1 = Bz[k][lx + 16];
            const float br0 = Br[k][lx], br1 = Br[k][lx + 16];
            const float bh0 = Bh[k][lx], bh1 = Bh[k][lx + 16];
            z00 = fma(a0, bz0, z00); z01 = fma(a0, bz1, z01);
            z10 = fma(a1, bz0, z10); z11 = fma(a1, bz1, z11);
            r00 = fma(a0, br0, r00); r01 = fma(a0, br1, r01);
            r10 = fma(a1, br0, r10); r11 = fma(a1, br1, r11);
            h00 = fma(a0, bh0, h00); h01 = fma(a0, bh1, h01);
            h10 = fma(a1, bh0, h10); h11 = fma(a1, bh1, h11);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row0 < B && col0 < H) gru_projection_fused_store(B, H, T, t, row0, col0, z00, r00, h00, Xg, bx, bs, cur, prev_all, state_all, Sg, cache);
    if (row0 < B && col1 < H) gru_projection_fused_store(B, H, T, t, row0, col1, z01, r01, h01, Xg, bx, bs, cur, prev_all, state_all, Sg, cache);
    if (row1 < B && col0 < H) gru_projection_fused_store(B, H, T, t, row1, col0, z10, r10, h10, Xg, bx, bs, cur, prev_all, state_all, Sg, cache);
    if (row1 < B && col1 < H) gru_projection_fused_store(B, H, T, t, row1, col1, z11, r11, h11, Xg, bx, bs, cur, prev_all, state_all, Sg, cache);
}

// Fused Gated RC / CuDNN-GRU-v2 backward step at timestep t for B parallel streams
__kernel void gru_step_bwd_kernel(
    const int B,
    const int H,
    const int t,
    __global const float* dS_out_all,
    __global const float* dS_carry,
    __global const float* cache_zrh,
    __global const float* s_prev_all,
    __global const float* Sg_all,
    __global const float* b_s,
    __global float* dXg_all,
    __global float* dSg_all,
    __global float* dS_prev_direct
) {
    int gid = get_global_id(0);
    int tot = B * H;
    if (gid >= tot) return;
    int b = gid / H;
    int h = gid - b * H;
    int H3 = 3 * H;
    int tb_h = (t * B + b) * H + h;
    int tb_3h = (t * B + b) * H3;

    float ds = dS_out_all[tb_h] + dS_carry[b * H + h];
    float z = cache_zrh[tb_3h + h];
    float r = cache_zrh[tb_3h + H + h];
    float ht = cache_zrh[tb_3h + 2 * H + h];
    float sp = s_prev_all[tb_h];
    float sg_h = Sg_all[tb_3h + 2 * H + h] + b_s[2 * H + h];

    float dz_pre = ds * (ht - sp) * z * (1.0f - z);
    float dh_pre = ds * z * (1.0f - ht * ht);
    float dr_pre = dh_pre * sg_h * r * (1.0f - r);
    float dsg_h = dh_pre * r;

    dXg_all[tb_3h + h] = dz_pre;
    dXg_all[tb_3h + H + h] = dr_pre;
    dXg_all[tb_3h + 2 * H + h] = dh_pre;

    dSg_all[tb_3h + h] = dz_pre;
    dSg_all[tb_3h + H + h] = dr_pre;
    dSg_all[tb_3h + 2 * H + h] = dsg_h;

    dS_prev_direct[b * H + h] = ds * (1.0f - z);
}

// Elementwise gradient clipping for rock-solid 48-step BPTT
__kernel void clip_grad_kernel(
    const int n_total,
    const float clip_val,
    __global float* grad
) {
    int gid = get_global_id(0);
    if (gid >= n_total) return;
    grad[gid] = clamp(grad[gid], -clip_val, clip_val);
}

// Copy slice of [T, B, C] at timestep t into/from [B, C] (dir=0: src[t]->dst, dir=1: src->dst[t])
__kernel void copy_slice_kernel(
    const int slice_elems,
    const int t,
    const int dir,
    __global float* seq_buf,
    __global float* step_buf
) {
    int gid = get_global_id(0);
    if (gid >= slice_elems) return;
    int off = t * slice_elems + gid;
    if (dir == 0) {
        step_buf[gid] = seq_buf[off];
    } else {
        seq_buf[off] = step_buf[gid];
    }
}

// Zero-copy on-GPU slice of [T, B] time-major tok_in & tok_tgt directly from VRAM-resident [B, stream_len] stream
__kernel void slice_tokens_kernel(
    const int B,
    const int T,
    const int stream_len,
    const int s0,
    __global const int* full_stream,
    __global int* tok_in,
    __global int* tok_tgt
) {
    int gid = get_global_id(0);
    int tot = B * T;
    if (gid >= tot) return;
    int t = gid / B;
    int b = gid - t * B;
    int pos = b * stream_len + s0 + t;
    tok_in[gid] = full_stream[pos];
    tok_tgt[gid] = full_stream[pos + 1];
}

// On-GPU buffer zeroing (eliminates blocking host->device PCIe syncs in BPTT loops)
__kernel void zero_buf_kernel(
    const int n_total,
    __global float* buf
) {
    int gid = get_global_id(0);
    if (gid >= n_total) return;
    buf[gid] = 0.0f;
}

// Elementwise residual addition out[i] = a[i] + b[i]
__kernel void vec_add_kernel(
    const int n_total,
    __global const float* a,
    __global const float* b,
    __global float* out
) {
    int gid = get_global_id(0);
    if (gid >= n_total) return;
    out[gid] = a[gid] + b[gid];
}

// Offset elementwise addition: out[out_off + i] = a[a_off + i] + b[b_off + i]
__kernel void vec_add_offset_kernel(
    const int n_total,
    const int a_off,
    const int b_off,
    const int out_off,
    __global const float* a,
    __global const float* b,
    __global float* out
) {
    int gid = get_global_id(0);
    if (gid >= n_total) return;
    out[out_off + gid] = a[a_off + gid] + b[b_off + gid];
}

// Zero-copy on-GPU slice of [T, B] time-major single int32 stream (e.g. aligned chunk IDs)
__kernel void slice_stream_1_kernel(
    const int B,
    const int T,
    const int stream_len,
    const int s0,
    __global const int* full_stream,
    __global int* out_slice
) {
    int gid = get_global_id(0);
    int tot = B * T;
    if (gid >= tot) return;
    int t = gid / B;
    int b = gid - t * B;
    out_slice[gid] = full_stream[b * stream_len + s0 + t];
}

// Additive dual-stream embedding gather: X[r, d] += E_chunk[chk_ids[r], d] for chk > 0
__kernel void embed_add_1_kernel(
    const int N,
    const int C,
    const int D,
    __global const int* chk_ids,
    __global const float* E_chunk,
    __global float* X_inout
) {
    int gid = get_global_id(0);
    int tot = N * D;
    if (gid >= tot) return;
    int r = gid / D;
    int d = gid - r * D;
    int c = chk_ids[r];
    if (c > 0 && c < C) {
        X_inout[gid] += E_chunk[c * D + d];
    }
}

// Fast-Weight Isotropic Gated Delta-Rule Associative Scan Forward (1 workgroup of 64 threads per stream b, D <= 64).
// Isotropic scalar decay lam_t preserves exact key-query geometry across arbitrary delays and allows topic flushes!
__kernel void fast_weight_scan_fwd_kernel(
    const int B,
    const int T,
    const int D,
    __global const float* K_raw,
    __global const float* QVG_raw,
    __global const float* b_k,
    __global const float* b_qvg,
    __global float* M_cur,
    __global float* M_prev_all,
    __global float* R_hat_all,
    __global float* fw_cache
) {
    int b = get_group_id(0);
    if (b >= B) return;
    int tid = get_local_id(0);
    int QVG_D = 2 * D + 3;
    int CACHE_D = 5 * D + 6;

    __local float sm_M[64 * 65];
    __local float sm_k[64];
    __local float sm_q[64];
    __local float sm_v[64];
    __local float sm_r[64];
    __local float sm_red[64];

    if (tid < D) {
        int mb = b * D * D;
        for (int i = 0; i < D; ++i) {
            sm_M[i * 65 + tid] = M_cur[mb + i * D + tid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < T; ++t) {
        int tb = t * B + b;
        int qvg_base = tb * QVG_D;

        float kr = (tid < D) ? (K_raw[tb * D + tid] + b_k[tid]) : 0.0f;
        float qr = (tid < D) ? (QVG_raw[qvg_base + tid] + b_qvg[tid]) : 0.0f;
        float vr = (tid < D) ? (QVG_raw[qvg_base + D + tid] + b_qvg[D + tid]) : 0.0f;

        sm_red[tid] = kr * kr;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) sm_red[tid] += sm_red[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) sm_red[tid] += sm_red[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) sm_red[tid] += sm_red[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) sm_red[tid] += sm_red[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) sm_red[tid] += sm_red[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) sm_red[tid] += sm_red[tid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        float inv_nk = 1.0f / sqrt(sm_red[0] + 1e-5f);

        sm_red[tid] = qr * qr;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) sm_red[tid] += sm_red[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) sm_red[tid] += sm_red[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) sm_red[tid] += sm_red[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) sm_red[tid] += sm_red[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) sm_red[tid] += sm_red[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) sm_red[tid] += sm_red[tid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        float inv_nq = 1.0f / sqrt(sm_red[0] + 1e-5f);

        float gl = clamp(QVG_raw[qvg_base + 2 * D] + b_qvg[2 * D], -16.0f, 16.0f);
        float ge = clamp(QVG_raw[qvg_base + 2 * D + 1] + b_qvg[2 * D + 1], -16.0f, 16.0f);
        float go = clamp(QVG_raw[qvg_base + 2 * D + 2] + b_qvg[2 * D + 2], -16.0f, 16.0f);
        float lam = 1.0f / (1.0f + exp(-gl));
        float eta = 1.0f / (1.0f + exp(-ge));
        float og = 1.0f / (1.0f + exp(-go));

        if (tid < D) {
            sm_k[tid] = kr * inv_nk;
            sm_q[tid] = qr * inv_nq;
            sm_v[tid] = tanh(vr);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float v_old = 0.0f;
        float r_j = 0.0f;
        if (tid < D) {
            int m_base = tb * D * D;
            for (int i = 0; i < D; ++i) {
                float mij = sm_M[i * 65 + tid];
                M_prev_all[m_base + i * D + tid] = mij;
                v_old += sm_k[i] * mij;
            }
            float delta = sm_v[tid] - lam * v_old;
            for (int i = 0; i < D; ++i) {
                float mij_new = lam * sm_M[i * 65 + tid] + eta * sm_k[i] * delta;
                sm_M[i * 65 + tid] = mij_new;
                r_j += sm_q[i] * mij_new;
            }
            sm_r[tid] = r_j;
        }
        sm_red[tid] = (tid < D) ? (r_j * r_j) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) sm_red[tid] += sm_red[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) sm_red[tid] += sm_red[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) sm_red[tid] += sm_red[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) sm_red[tid] += sm_red[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) sm_red[tid] += sm_red[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) sm_red[tid] += sm_red[tid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        float inv_rms = 1.0f / sqrt(sm_red[0] / (float)D + 1.0f);

        if (tid < D) {
            float r_hat = sm_r[tid] * inv_rms;
            R_hat_all[tb * D + tid] = og * r_hat;
            int cb = tb * CACHE_D;
            fw_cache[cb + tid] = sm_k[tid];
            fw_cache[cb + D + tid] = sm_q[tid];
            fw_cache[cb + 2 * D + tid] = sm_v[tid];
            fw_cache[cb + 3 * D + tid] = v_old;
            fw_cache[cb + 4 * D + tid] = r_hat;
            if (tid == 0) {
                fw_cache[cb + 5 * D + 0] = inv_nk;
                fw_cache[cb + 5 * D + 1] = inv_nq;
                fw_cache[cb + 5 * D + 2] = lam;
                fw_cache[cb + 5 * D + 3] = eta;
                fw_cache[cb + 5 * D + 4] = inv_rms;
                fw_cache[cb + 5 * D + 5] = og;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < D) {
        int mb = b * D * D;
        for (int i = 0; i < D; ++i) {
            M_cur[mb + i * D + tid] = sm_M[i * 65 + tid];
        }
    }
}

// Fast-Weight Isotropic Gated Delta-Rule Associative Scan Backward (1 workgroup of 64 threads/stream, ~19.7 KB LDS).
__kernel void fast_weight_scan_bwd_kernel(
    const int B,
    const int T,
    const int D,
    __global const float* dR_hat_all,
    __global const float* M_prev_all,
    __global const float* fw_cache,
    __global float* dK_raw,
    __global float* dQVG_raw
) {
    int b = get_group_id(0);
    if (b >= B) return;
    int tid = get_local_id(0);
    int QVG_D = 2 * D + 3;
    int CACHE_D = 5 * D + 6;

    __local float sm_dM[64 * 65];
    __local float sm_k[64];
    __local float sm_q[64];
    __local float sm_v[64];
    __local float sm_vold[64];
    __local float sm_rhat[64];
    __local float sm_dr[64];
    __local float sm_delta[64];
    __local float sm_dq[64];
    __local float sm_dk[64];
    __local float sm_dv[64];
    __local float sm_dvold[64];
    __local float sm_red[64];

    if (tid < D) {
        for (int i = 0; i < D; ++i) {
            sm_dM[i * 65 + tid] = 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = T - 1; t >= 0; --t) {
        int tb = t * B + b;
        int cb = tb * CACHE_D;
        int m_base = tb * D * D;

        float drh_j = 0.0f;
        if (tid < D) {
            drh_j = dR_hat_all[tb * D + tid];
            sm_k[tid] = fw_cache[cb + tid];
            sm_q[tid] = fw_cache[cb + D + tid];
            sm_v[tid] = fw_cache[cb + 2 * D + tid];
            sm_vold[tid] = fw_cache[cb + 3 * D + tid];
            sm_rhat[tid] = fw_cache[cb + 4 * D + tid];
        }
        float inv_nk = fw_cache[cb + 5 * D + 0];
        float inv_nq = fw_cache[cb + 5 * D + 1];
        float lam = fw_cache[cb + 5 * D + 2];
        float eta = fw_cache[cb + 5 * D + 3];
        float inv_rms = fw_cache[cb + 5 * D + 4];
        float og = fw_cache[cb + 5 * D + 5];

        sm_red[tid] = (tid < D) ? (drh_j * sm_rhat[tid]) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) sm_red[tid] += sm_red[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) sm_red[tid] += sm_red[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) sm_red[tid] += sm_red[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) sm_red[tid] += sm_red[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) sm_red[tid] += sm_red[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) sm_red[tid] += sm_red[tid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        float do_tot = sm_red[0];
        float r_dot = (do_tot * og) / (float)D;

        if (tid < D) {
            sm_dr[tid] = (og * drh_j - sm_rhat[tid] * r_dot) * inv_rms;
            sm_delta[tid] = sm_v[tid] - lam * sm_vold[tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < D) {
            float dq_i = 0.0f;
            float ki = sm_k[tid];
            int row_off = m_base + tid * D;
            for (int j = 0; j < D; ++j) {
                float m_cur = lam * M_prev_all[row_off + j] + eta * ki * sm_delta[j];
                dq_i += m_cur * sm_dr[j];
            }
            sm_dq[tid] = dq_i;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float deta_part = 0.0f;
        float dlam_part = 0.0f;
        if (tid < D) {
            float dr_j = sm_dr[tid];
            float u_j = 0.0f;
            float dlam2_j = 0.0f;
            for (int i = 0; i < D; ++i) {
                float dm_ij = sm_dM[i * 65 + tid] + sm_q[i] * dr_j;
                sm_dM[i * 65 + tid] = dm_ij;
                u_j += sm_k[i] * dm_ij;
                dlam2_j += M_prev_all[m_base + i * D + tid] * dm_ij;
            }
            float d_delta = eta * u_j;
            sm_dv[tid] = d_delta;
            sm_dvold[tid] = -lam * d_delta;
            deta_part = sm_delta[tid] * u_j;
            dlam_part = dlam2_j - sm_vold[tid] * d_delta;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < D) {
            float dk_i = 0.0f;
            int row_off = m_base + tid * D;
            int dm_row = tid * 65;
            for (int j = 0; j < D; ++j) {
                dk_i += eta * sm_dM[dm_row + j] * sm_delta[j] + M_prev_all[row_off + j] * sm_dvold[j];
            }
            sm_dk[tid] = dk_i;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        sm_red[tid] = deta_part;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) sm_red[tid] += sm_red[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) sm_red[tid] += sm_red[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) sm_red[tid] += sm_red[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) sm_red[tid] += sm_red[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) sm_red[tid] += sm_red[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) sm_red[tid] += sm_red[tid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        float deta_tot = sm_red[0];

        sm_red[tid] = dlam_part;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) sm_red[tid] += sm_red[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) sm_red[tid] += sm_red[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) sm_red[tid] += sm_red[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) sm_red[tid] += sm_red[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) sm_red[tid] += sm_red[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) sm_red[tid] += sm_red[tid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        float dlam_tot = sm_red[0];

        sm_red[tid] = (tid < D) ? (sm_dq[tid] * sm_q[tid]) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) sm_red[tid] += sm_red[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) sm_red[tid] += sm_red[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) sm_red[tid] += sm_red[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) sm_red[tid] += sm_red[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) sm_red[tid] += sm_red[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) sm_red[tid] += sm_red[tid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        float dq_dot = sm_red[0];

        sm_red[tid] = (tid < D) ? (sm_dk[tid] * sm_k[tid]) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 32) sm_red[tid] += sm_red[tid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 16) sm_red[tid] += sm_red[tid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 8) sm_red[tid] += sm_red[tid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 4) sm_red[tid] += sm_red[tid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 2) sm_red[tid] += sm_red[tid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < 1) sm_red[tid] += sm_red[tid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        float dk_dot = sm_red[0];

        if (tid < D) {
            float dvold_j = sm_dvold[tid];
            for (int i = 0; i < D; ++i) {
                sm_dM[i * 65 + tid] = lam * sm_dM[i * 65 + tid] + sm_k[i] * dvold_j;
            }
            dK_raw[tb * D + tid] = (sm_dk[tid] - sm_k[tid] * dk_dot) * inv_nk;
            int qvg_base = tb * QVG_D;
            dQVG_raw[qvg_base + tid] = (sm_dq[tid] - sm_q[tid] * dq_dot) * inv_nq;
            float vj = sm_v[tid];
            dQVG_raw[qvg_base + D + tid] = sm_dv[tid] * (1.0f - vj * vj);
            if (tid == 0) {
                dQVG_raw[qvg_base + 2 * D] = dlam_tot * lam * (1.0f - lam);
                dQVG_raw[qvg_base + 2 * D + 1] = deta_tot * eta * (1.0f - eta);
                dQVG_raw[qvg_base + 2 * D + 2] = do_tot * og * (1.0f - og);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
"""


class OpenCLAccelerator:
    """Thin ctypes OpenCL 1.2+ compute accelerator for AMD / NVIDIA / Intel GPUs."""

    def __init__(self, prefer_vendor: str = "AMD"):
        self.available: bool = False
        self.device_name: str = "unavailable"
        self.platform_name: str = ""
        self.compute_units: int = 0
        self.global_mem_gb: float = 0.0
        self._ocl = None
        self._ctx = None
        self._queue = None
        self._prog = None
        self._k_circuit = None
        self._k_spmv = None
        self._k_beam = None
        self._k_sgemm = None
        self._k_gb_fwd = None
        self._k_gb_bwd = None
        self._k_softmax_ce = None
        self._k_col_sum = None
        self._k_adamw = None
        self._k_res_fwd = None
        self._k_res_bwd = None
        self._k_emb_gather = None
        self._k_emb_grad = None
        self._k_emb_gather_1 = None
        self._k_emb_grad_1 = None
        self._k_gru_fwd = None
        self._k_gru_proj_fused = None
        self._k_gru_bwd = None
        self._k_clip_grad = None
        self._k_copy_slice = None
        self._k_vec_add_offset = None
        self._k_slice_stream_1 = None
        self._k_embed_add_1 = None
        self._k_fw_fwd = None
        self._k_fw_bwd = None
        self._init_opencl(prefer_vendor=prefer_vendor)

    def _init_opencl(self, prefer_vendor: str = "AMD") -> None:
        try:
            lib_name = "OpenCL.dll" if sys.platform == "win32" else "libOpenCL.so.1"
            ocl = ctypes.WinDLL(lib_name) if sys.platform == "win32" else ctypes.CDLL(lib_name)
        except Exception:
            return

        # Declare exact 64-bit pointer signatures
        c_vp = ctypes.c_void_p
        c_u32 = ctypes.c_uint32
        c_i32 = ctypes.c_int32
        c_u64 = ctypes.c_uint64
        c_sz = ctypes.c_size_t
        c_p_i32 = ctypes.POINTER(c_i32)
        c_p_u32 = ctypes.POINTER(c_u32)
        c_p_sz = ctypes.POINTER(c_sz)

        ocl.clGetPlatformIDs.argtypes = [c_u32, c_vp, c_p_u32]
        ocl.clGetPlatformIDs.restype = c_i32
        ocl.clGetPlatformInfo.argtypes = [c_vp, c_u32, c_sz, c_vp, c_p_sz]
        ocl.clGetPlatformInfo.restype = c_i32
        ocl.clGetDeviceIDs.argtypes = [c_vp, c_u64, c_u32, c_vp, c_p_u32]
        ocl.clGetDeviceIDs.restype = c_i32
        ocl.clGetDeviceInfo.argtypes = [c_vp, c_u32, c_sz, c_vp, c_p_sz]
        ocl.clGetDeviceInfo.restype = c_i32
        ocl.clCreateContext.argtypes = [c_vp, c_u32, ctypes.POINTER(c_vp), c_vp, c_vp, c_p_i32]
        ocl.clCreateContext.restype = c_vp
        ocl.clCreateCommandQueue.argtypes = [c_vp, c_vp, c_u64, c_p_i32]
        ocl.clCreateCommandQueue.restype = c_vp
        ocl.clCreateProgramWithSource.argtypes = [c_vp, c_u32, ctypes.POINTER(ctypes.c_char_p), c_p_sz, c_p_i32]
        ocl.clCreateProgramWithSource.restype = c_vp
        ocl.clBuildProgram.argtypes = [c_vp, c_u32, ctypes.POINTER(c_vp), ctypes.c_char_p, c_vp, c_vp]
        ocl.clBuildProgram.restype = c_i32
        ocl.clGetProgramBuildInfo.argtypes = [c_vp, c_vp, c_u32, c_sz, c_vp, c_p_sz]
        ocl.clGetProgramBuildInfo.restype = c_i32
        ocl.clCreateKernel.argtypes = [c_vp, ctypes.c_char_p, c_p_i32]
        ocl.clCreateKernel.restype = c_vp
        ocl.clCreateBuffer.argtypes = [c_vp, c_u64, c_sz, c_vp, c_p_i32]
        ocl.clCreateBuffer.restype = c_vp
        ocl.clEnqueueWriteBuffer.argtypes = [c_vp, c_vp, c_u32, c_sz, c_sz, c_vp, c_u32, c_vp, c_vp]
        ocl.clEnqueueWriteBuffer.restype = c_i32
        ocl.clEnqueueReadBuffer.argtypes = [c_vp, c_vp, c_u32, c_sz, c_sz, c_vp, c_u32, c_vp, c_vp]
        ocl.clEnqueueReadBuffer.restype = c_i32
        ocl.clSetKernelArg.argtypes = [c_vp, c_u32, c_sz, c_vp]
        ocl.clSetKernelArg.restype = c_i32
        ocl.clEnqueueNDRangeKernel.argtypes = [c_vp, c_vp, c_u32, c_p_sz, c_p_sz, c_p_sz, c_u32, c_vp, c_vp]
        ocl.clEnqueueNDRangeKernel.restype = c_i32
        ocl.clFinish.argtypes = [c_vp]
        ocl.clFinish.restype = c_i32
        ocl.clReleaseMemObject.argtypes = [c_vp]
        ocl.clReleaseMemObject.restype = c_i32

        n_plat = c_u32(0)
        if ocl.clGetPlatformIDs(0, None, ctypes.byref(n_plat)) != 0 or n_plat.value == 0:
            return
        plats = (c_vp * n_plat.value)()
        ocl.clGetPlatformIDs(n_plat.value, plats, None)

        best = None
        buf = ctypes.create_string_buffer(256)
        sz = c_sz(0)
        for i in range(n_plat.value):
            ocl.clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, 256, buf, ctypes.byref(sz))
            pname = buf.value.decode("utf-8", "replace")
            n_dev = c_u32(0)
            if ocl.clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(n_dev)) != 0 or n_dev.value == 0:
                continue
            devs = (c_vp * n_dev.value)()
            ocl.clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, n_dev.value, devs, None)
            for j in range(n_dev.value):
                ocl.clGetDeviceInfo(devs[j], CL_DEVICE_NAME, 256, buf, ctypes.byref(sz))
                dname = buf.value.decode("utf-8", "replace").strip()
                cu = c_u32(0)
                ocl.clGetDeviceInfo(devs[j], CL_DEVICE_MAX_COMPUTE_UNITS, 4, ctypes.byref(cu), None)
                mem = c_u64(0)
                ocl.clGetDeviceInfo(devs[j], CL_DEVICE_GLOBAL_MEM_SIZE, 8, ctypes.byref(mem), None)
                vendor_bonus = 1 if (prefer_vendor and prefer_vendor.lower() in pname.lower()) else 0
                score = (vendor_bonus, int(mem.value), int(cu.value))
                if best is None or score > best[0]:
                    best = (score, devs[j], pname, dname, int(cu.value), float(mem.value) / (1024.0 ** 3))

        if best is None:
            return

        _sc, dev_id, pname, dname, cu_val, mem_gb = best
        err = c_i32(0)
        dev_arr = (c_vp * 1)(dev_id)
        ctx = ocl.clCreateContext(None, 1, dev_arr, None, None, ctypes.byref(err))
        if err.value != 0 or not ctx:
            return
        queue = ocl.clCreateCommandQueue(ctx, dev_id, 0, ctypes.byref(err))
        if err.value != 0 or not queue:
            return

        src_bytes = CIRCUIT_KERNELS_SRC.encode("utf-8")
        src_ptr = ctypes.c_char_p(src_bytes)
        src_len = c_sz(len(src_bytes))
        prog = ocl.clCreateProgramWithSource(ctx, 1, ctypes.byref(src_ptr), ctypes.byref(src_len), ctypes.byref(err))
        if err.value != 0 or not prog:
            return
        if ocl.clBuildProgram(prog, 1, dev_arr, b"-cl-fast-relaxed-math", None, None) != 0:
            return
        k_circ = ocl.clCreateKernel(prog, b"circuit_step_kernel", ctypes.byref(err))
        if err.value != 0 or not k_circ:
            return
        k_spmv = ocl.clCreateKernel(prog, b"csr_spmv_batch_kernel", ctypes.byref(err))
        if err.value != 0 or not k_spmv:
            return
        k_beam = ocl.clCreateKernel(prog, b"circuit_beam_branch_kernel", ctypes.byref(err))
        if err.value != 0 or not k_beam:
            return
        k_sgemm = ocl.clCreateKernel(prog, b"sgemm_tiled_kernel", ctypes.byref(err))
        k_gb_fwd = ocl.clCreateKernel(prog, b"gated_bilinear_fwd_kernel", ctypes.byref(err))
        k_gb_bwd = ocl.clCreateKernel(prog, b"gated_bilinear_bwd_kernel", ctypes.byref(err))
        k_sm_ce = ocl.clCreateKernel(prog, b"softmax_ce_grad_kernel", ctypes.byref(err))
        k_col_s = ocl.clCreateKernel(prog, b"col_sum_kernel", ctypes.byref(err))
        k_adamw = ocl.clCreateKernel(prog, b"adamw_update_kernel", ctypes.byref(err))
        k_res_fwd = ocl.clCreateKernel(prog, b"res_gated_fwd_kernel", ctypes.byref(err))
        k_res_bwd = ocl.clCreateKernel(prog, b"res_gated_bwd_kernel", ctypes.byref(err))
        k_emb_g = ocl.clCreateKernel(prog, b"embed_gather_2_kernel", ctypes.byref(err))
        k_emb_gr = ocl.clCreateKernel(prog, b"embed_grad_reduce_2_kernel", ctypes.byref(err))
        k_emb_g1 = ocl.clCreateKernel(prog, b"embed_gather_1_kernel", ctypes.byref(err))
        k_emb_gr1 = ocl.clCreateKernel(prog, b"embed_grad_reduce_1_kernel", ctypes.byref(err))
        k_gru_fwd = ocl.clCreateKernel(prog, b"gru_step_fwd_kernel", ctypes.byref(err))
        k_gru_proj_fused = ocl.clCreateKernel(prog, b"gru_projection_fused_kernel", ctypes.byref(err))
        k_gru_bwd = ocl.clCreateKernel(prog, b"gru_step_bwd_kernel", ctypes.byref(err))
        k_clip_g = ocl.clCreateKernel(prog, b"clip_grad_kernel", ctypes.byref(err))
        k_cp_sl = ocl.clCreateKernel(prog, b"copy_slice_kernel", ctypes.byref(err))
        k_sl_tok = ocl.clCreateKernel(prog, b"slice_tokens_kernel", ctypes.byref(err))
        k_zero = ocl.clCreateKernel(prog, b"zero_buf_kernel", ctypes.byref(err))
        k_vadd = ocl.clCreateKernel(prog, b"vec_add_kernel", ctypes.byref(err))
        k_vadd_off = ocl.clCreateKernel(prog, b"vec_add_offset_kernel", ctypes.byref(err))
        k_sl_s1 = ocl.clCreateKernel(prog, b"slice_stream_1_kernel", ctypes.byref(err))
        k_emb_add1 = ocl.clCreateKernel(prog, b"embed_add_1_kernel", ctypes.byref(err))
        k_fw_fwd = ocl.clCreateKernel(prog, b"fast_weight_scan_fwd_kernel", ctypes.byref(err))
        k_fw_bwd = ocl.clCreateKernel(prog, b"fast_weight_scan_bwd_kernel", ctypes.byref(err))

        self._ocl = ocl
        self._ctx = ctx
        self._queue = queue
        self._prog = prog
        self._k_circuit = k_circ
        self._k_spmv = k_spmv
        self._k_beam = k_beam
        self._k_sgemm = k_sgemm
        self._k_gb_fwd = k_gb_fwd
        self._k_gb_bwd = k_gb_bwd
        self._k_softmax_ce = k_sm_ce
        self._k_col_sum = k_col_s
        self._k_adamw = k_adamw
        self._k_res_fwd = k_res_fwd
        self._k_res_bwd = k_res_bwd
        self._k_emb_gather = k_emb_g
        self._k_emb_grad = k_emb_gr
        self._k_emb_gather_1 = k_emb_g1
        self._k_emb_grad_1 = k_emb_gr1
        self._k_gru_fwd = k_gru_fwd
        self._k_gru_proj_fused = k_gru_proj_fused
        self._k_gru_bwd = k_gru_bwd
        self._k_clip_grad = k_clip_g
        self._k_copy_slice = k_cp_sl
        self._k_slice_tokens = k_sl_tok
        self._k_zero_buf = k_zero
        self._k_vec_add = k_vadd
        self._k_vec_add_offset = k_vadd_off
        self._k_slice_stream_1 = k_sl_s1
        self._k_embed_add_1 = k_emb_add1
        self._k_fw_fwd = k_fw_fwd
        self._k_fw_bwd = k_fw_bwd
        self.platform_name = pname
        self.device_name = dname
        self.compute_units = cu_val
        self.global_mem_gb = round(mem_gb, 2)
        self.available = True

    def create_buffer_from_array(self, arr: np.ndarray, *, read_only: bool = False) -> ctypes.c_void_p:
        cont = np.ascontiguousarray(arr)
        flags = (CL_MEM_READ_ONLY if read_only else CL_MEM_READ_WRITE) | CL_MEM_COPY_HOST_PTR
        err = ctypes.c_int32(0)
        buf = self._ocl.clCreateBuffer(self._ctx, flags, cont.nbytes, cont.ctypes.data, ctypes.byref(err))
        if err.value != 0 or not buf:
            raise RuntimeError(f"clCreateBuffer failed with code {err.value}")
        return buf

    def write_buffer(self, buf: ctypes.c_void_p, arr: np.ndarray) -> None:
        cont = np.ascontiguousarray(arr)
        rc = self._ocl.clEnqueueWriteBuffer(self._queue, buf, CL_TRUE, 0, cont.nbytes, cont.ctypes.data, 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueWriteBuffer failed with code {rc}")

    def read_buffer(self, buf: ctypes.c_void_p, out: np.ndarray) -> np.ndarray:
        cont = np.ascontiguousarray(out)
        rc = self._ocl.clEnqueueReadBuffer(self._queue, buf, CL_TRUE, 0, cont.nbytes, cont.ctypes.data, 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueReadBuffer failed with code {rc}")
        return cont

    def release_buffer(self, buf: Optional[ctypes.c_void_p]) -> None:
        if buf and self._ocl is not None:
            self._ocl.clReleaseMemObject(buf)

    def launch_circuit_step(
        self,
        n_total: int,
        n_nodes: int,
        dt: float,
        alpha_int: float,
        w_cap: float,
        w_int: float,
        w_diff: float,
        b_iin: ctypes.c_void_p,
        b_r: ctypes.c_void_p,
        b_c: ctypes.c_void_p,
        b_vcap: ctypes.c_void_p,
        b_vint: ctypes.c_void_p,
        b_vdiff: ctypes.c_void_p,
        b_vout: ctypes.c_void_p,
        *,
        sync: bool = False,
    ) -> None:
        ocl = self._ocl
        k = self._k_circuit
        args = [
            (ctypes.c_int32(n_total), 4),
            (ctypes.c_int32(n_nodes), 4),
            (ctypes.c_float(dt), 4),
            (ctypes.c_float(alpha_int), 4),
            (ctypes.c_float(w_cap), 4),
            (ctypes.c_float(w_int), 4),
            (ctypes.c_float(w_diff), 4),
            (ctypes.c_void_p(b_iin), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_r), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_c), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_vcap), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_vint), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_vdiff), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_vout), ctypes.sizeof(ctypes.c_void_p)),
        ]
        for idx, (val, sz) in enumerate(args):
            rc = ocl.clSetKernelArg(k, idx, sz, ctypes.byref(val))
            if rc != 0:
                raise RuntimeError(f"clSetKernelArg({idx}) failed: {rc}")
        gs = ctypes.c_size_t(((n_total + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = ocl.clEnqueueNDRangeKernel(self._queue, k, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(circuit) failed: {rc}")
        if sync:
            ocl.clFinish(self._queue)

    def launch_spmv_batch(
        self,
        batch_size: int,
        n_nodes: int,
        b_row_ptr: ctypes.c_void_p,
        b_col_idx: ctypes.c_void_p,
        b_vals: ctypes.c_void_p,
        b_xin: ctypes.c_void_p,
        b_yout: ctypes.c_void_p,
        *,
        sync: bool = False,
    ) -> None:
        ocl = self._ocl
        k = self._k_spmv
        total = int(batch_size) * int(n_nodes)
        args = [
            (ctypes.c_int32(batch_size), 4),
            (ctypes.c_int32(n_nodes), 4),
            (ctypes.c_void_p(b_row_ptr), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_col_idx), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_vals), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_xin), ctypes.sizeof(ctypes.c_void_p)),
            (ctypes.c_void_p(b_yout), ctypes.sizeof(ctypes.c_void_p)),
        ]
        for idx, (val, sz) in enumerate(args):
            rc = ocl.clSetKernelArg(k, idx, sz, ctypes.byref(val))
            if rc != 0:
                raise RuntimeError(f"clSetKernelArg(spmv:{idx}) failed: {rc}")
        gs = ctypes.c_size_t(((total + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = ocl.clEnqueueNDRangeKernel(self._queue, k, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(spmv) failed: {rc}")
        if sync:
            ocl.clFinish(self._queue)

    def launch_beam_branch_step(
        self,
        batch_size: int,
        n_nodes: int,
        dt: float,
        alpha_int: float,
        realize_gain: float,
        w_cap: float,
        w_int: float,
        w_diff: float,
        b_parent_idx: ctypes.c_void_p,
        b_action_sym: ctypes.c_void_p,
        b_action_sal: ctypes.c_void_p,
        b_ispread: ctypes.c_void_p,
        b_r: ctypes.c_void_p,
        b_c: ctypes.c_void_p,
        b_vcap_in: ctypes.c_void_p,
        b_vint_in: ctypes.c_void_p,
        b_vcap_out: ctypes.c_void_p,
        b_vint_out: ctypes.c_void_p,
        b_vdiff_out: ctypes.c_void_p,
        b_vout: ctypes.c_void_p,
        *,
        sync: bool = False,
    ) -> None:
        ocl = self._ocl
        k = self._k_beam
        total = int(batch_size) * int(n_nodes)
        vp_sz = ctypes.sizeof(ctypes.c_void_p)
        args = [
            (ctypes.c_int32(batch_size), 4),
            (ctypes.c_int32(n_nodes), 4),
            (ctypes.c_float(dt), 4),
            (ctypes.c_float(alpha_int), 4),
            (ctypes.c_float(realize_gain), 4),
            (ctypes.c_float(w_cap), 4),
            (ctypes.c_float(w_int), 4),
            (ctypes.c_float(w_diff), 4),
            (ctypes.c_void_p(b_parent_idx), vp_sz),
            (ctypes.c_void_p(b_action_sym), vp_sz),
            (ctypes.c_void_p(b_action_sal), vp_sz),
            (ctypes.c_void_p(b_ispread), vp_sz),
            (ctypes.c_void_p(b_r), vp_sz),
            (ctypes.c_void_p(b_c), vp_sz),
            (ctypes.c_void_p(b_vcap_in), vp_sz),
            (ctypes.c_void_p(b_vint_in), vp_sz),
            (ctypes.c_void_p(b_vcap_out), vp_sz),
            (ctypes.c_void_p(b_vint_out), vp_sz),
            (ctypes.c_void_p(b_vdiff_out), vp_sz),
            (ctypes.c_void_p(b_vout), vp_sz),
        ]
        for idx, (val, sz) in enumerate(args):
            rc = ocl.clSetKernelArg(k, idx, sz, ctypes.byref(val))
            if rc != 0:
                raise RuntimeError(f"clSetKernelArg(beam:{idx}) failed: {rc}")
        gs = ctypes.c_size_t(((total + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = ocl.clEnqueueNDRangeKernel(self._queue, k, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(beam) failed: {rc}")
        if sync:
            ocl.clFinish(self._queue)

    def finish(self) -> None:
        if self._ocl is not None and self._queue is not None:
            self._ocl.clFinish(self._queue)

    def create_empty_buffer(self, nbytes: int) -> ctypes.c_void_p:
        err = ctypes.c_int32(0)
        buf = self._ocl.clCreateBuffer(self._ctx, CL_MEM_READ_WRITE, max(int(nbytes), 4), None, ctypes.byref(err))
        if err.value != 0 or not buf:
            raise RuntimeError(f"clCreateBuffer(empty:{nbytes}) failed: {err.value}")
        return buf

    def _set_args(self, kernel: ctypes.c_void_p, args: list) -> None:
        ocl = self._ocl
        for idx, (val, sz) in enumerate(args):
            rc = ocl.clSetKernelArg(kernel, idx, sz, ctypes.byref(val))
            if rc != 0:
                raise RuntimeError(f"clSetKernelArg({idx}) failed: {rc}")

    def launch_sgemm(
        self,
        M: int,
        N: int,
        K: int,
        b_A: ctypes.c_void_p,
        b_B: ctypes.c_void_p,
        b_C: ctypes.c_void_p,
        *,
        transA: bool = False,
        transB: bool = False,
        alpha: float = 1.0,
        beta: float = 0.0,
        sync: bool = False,
        a_offset: int = 0,
        c_offset: int = 0,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_sgemm, [
            (ctypes.c_int32(M), 4),
            (ctypes.c_int32(N), 4),
            (ctypes.c_int32(K), 4),
            (ctypes.c_int32(1 if transA else 0), 4),
            (ctypes.c_int32(1 if transB else 0), 4),
            (ctypes.c_float(alpha), 4),
            (ctypes.c_float(beta), 4),
            (ctypes.c_int32(a_offset), 4),
            (ctypes.c_int32(c_offset), 4),
            (ctypes.c_void_p(b_A), vp),
            (ctypes.c_void_p(b_B), vp),
            (ctypes.c_void_p(b_C), vp),
        ])
        gs = (ctypes.c_size_t * 2)(((int(N) + 31) // 32) * 16, ((int(M) + 31) // 32) * 16)
        ls = (ctypes.c_size_t * 2)(16, 16)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_sgemm, 2, None, gs, ls, 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(sgemm) failed: {rc}")
        if sync:
            self._ocl.clFinish(self._queue)

    def launch_gated_bilinear_fwd(
        self,
        B: int,
        R: int,
        b_A_pre: ctypes.c_void_p,
        b_bloc: ctypes.c_void_p,
        b_G_pre: ctypes.c_void_p,
        b_bctx: ctypes.c_void_p,
        b_A: ctypes.c_void_p,
        b_G: ctypes.c_void_p,
        b_Q: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(R)
        self._set_args(self._k_gb_fwd, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(R), 4),
            (ctypes.c_void_p(b_A_pre), vp),
            (ctypes.c_void_p(b_bloc), vp),
            (ctypes.c_void_p(b_G_pre), vp),
            (ctypes.c_void_p(b_bctx), vp),
            (ctypes.c_void_p(b_A), vp),
            (ctypes.c_void_p(b_G), vp),
            (ctypes.c_void_p(b_Q), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_gb_fwd, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(gb_fwd) failed: {rc}")

    def launch_gated_bilinear_bwd(
        self,
        B: int,
        R: int,
        b_dQ: ctypes.c_void_p,
        b_A: ctypes.c_void_p,
        b_G: ctypes.c_void_p,
        b_dA_pre: ctypes.c_void_p,
        b_dG_pre: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(R)
        self._set_args(self._k_gb_bwd, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(R), 4),
            (ctypes.c_void_p(b_dQ), vp),
            (ctypes.c_void_p(b_A), vp),
            (ctypes.c_void_p(b_G), vp),
            (ctypes.c_void_p(b_dA_pre), vp),
            (ctypes.c_void_p(b_dG_pre), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_gb_bwd, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(gb_bwd) failed: {rc}")

    def launch_softmax_ce_grad(
        self,
        B: int,
        V: int,
        b_bout: ctypes.c_void_p,
        b_targets: ctypes.c_void_p,
        b_logits_dL: ctypes.c_void_p,
        b_row_losses: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_softmax_ce, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(V), 4),
            (ctypes.c_float(1.0 / max(float(B), 1.0)), 4),
            (ctypes.c_void_p(b_bout), vp),
            (ctypes.c_void_p(b_targets), vp),
            (ctypes.c_void_p(b_logits_dL), vp),
            (ctypes.c_void_p(b_row_losses), vp),
        ])
        gs = ctypes.c_size_t(int(B) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_softmax_ce, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(softmax_ce) failed: {rc}")

    def launch_col_sum(
        self,
        B: int,
        N: int,
        b_X: ctypes.c_void_p,
        b_out: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_col_sum, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(N), 4),
            (ctypes.c_void_p(b_X), vp),
            (ctypes.c_void_p(b_out), vp),
        ])
        gs = ctypes.c_size_t(((int(N) + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_col_sum, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(col_sum) failed: {rc}")

    def launch_adamw_update(
        self,
        n_total: int,
        lr_step: float,
        wd_scale: float,
        b_param: ctypes.c_void_p,
        b_grad: ctypes.c_void_p,
        b_m: ctypes.c_void_p,
        b_v: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_adamw, [
            (ctypes.c_int32(n_total), 4),
            (ctypes.c_float(lr_step), 4),
            (ctypes.c_float(wd_scale), 4),
            (ctypes.c_void_p(b_param), vp),
            (ctypes.c_void_p(b_grad), vp),
            (ctypes.c_void_p(b_m), vp),
            (ctypes.c_void_p(b_v), vp),
        ])
        gs = ctypes.c_size_t(((int(n_total) + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_adamw, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(adamw) failed: {rc}")

    def launch_res_gated_fwd(
        self,
        B: int,
        R: int,
        b_Q1: ctypes.c_void_p,
        b_H2_pre: ctypes.c_void_p,
        b_bh2: ctypes.c_void_p,
        b_G2_pre: ctypes.c_void_p,
        b_bctx2: ctypes.c_void_p,
        b_H2: ctypes.c_void_p,
        b_G2: ctypes.c_void_p,
        b_Q2: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(R)
        self._set_args(self._k_res_fwd, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(R), 4),
            (ctypes.c_void_p(b_Q1), vp),
            (ctypes.c_void_p(b_H2_pre), vp),
            (ctypes.c_void_p(b_bh2), vp),
            (ctypes.c_void_p(b_G2_pre), vp),
            (ctypes.c_void_p(b_bctx2), vp),
            (ctypes.c_void_p(b_H2), vp),
            (ctypes.c_void_p(b_G2), vp),
            (ctypes.c_void_p(b_Q2), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_res_fwd, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(res_fwd) failed: {rc}")

    def launch_res_gated_bwd(
        self,
        B: int,
        R: int,
        b_dQ2: ctypes.c_void_p,
        b_H2: ctypes.c_void_p,
        b_G2: ctypes.c_void_p,
        b_dH2_pre: ctypes.c_void_p,
        b_dG2_pre: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(R)
        self._set_args(self._k_res_bwd, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(R), 4),
            (ctypes.c_void_p(b_dQ2), vp),
            (ctypes.c_void_p(b_H2), vp),
            (ctypes.c_void_p(b_G2), vp),
            (ctypes.c_void_p(b_dH2_pre), vp),
            (ctypes.c_void_p(b_dG2_pre), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_res_bwd, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(res_bwd) failed: {rc}")

    def launch_embed_gather_2(
        self,
        B: int,
        D: int,
        loc_dim: int,
        b_tok_cur: ctypes.c_void_p,
        b_tok_prev: ctypes.c_void_p,
        b_Edyn: ctypes.c_void_p,
        b_Xl: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(D)
        self._set_args(self._k_emb_gather, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(D), 4),
            (ctypes.c_int32(loc_dim), 4),
            (ctypes.c_void_p(b_tok_cur), vp),
            (ctypes.c_void_p(b_tok_prev), vp),
            (ctypes.c_void_p(b_Edyn), vp),
            (ctypes.c_void_p(b_Xl), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_emb_gather, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(emb_gather) failed: {rc}")

    def launch_embed_grad_reduce_2(
        self,
        B: int,
        V: int,
        D: int,
        loc_dim: int,
        b_tok_cur: ctypes.c_void_p,
        b_tok_prev: ctypes.c_void_p,
        b_dXl: ctypes.c_void_p,
        b_dEdyn: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(V) * int(D)
        self._set_args(self._k_emb_grad, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(V), 4),
            (ctypes.c_int32(D), 4),
            (ctypes.c_int32(loc_dim), 4),
            (ctypes.c_void_p(b_tok_cur), vp),
            (ctypes.c_void_p(b_tok_prev), vp),
            (ctypes.c_void_p(b_dXl), vp),
            (ctypes.c_void_p(b_dEdyn), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_emb_grad, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(emb_grad) failed: {rc}")

    def launch_embed_gather_1(
        self,
        N: int,
        D: int,
        b_tok_ids: ctypes.c_void_p,
        b_E: ctypes.c_void_p,
        b_X: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(N) * int(D)
        self._set_args(self._k_emb_gather_1, [
            (ctypes.c_int32(N), 4),
            (ctypes.c_int32(D), 4),
            (ctypes.c_void_p(b_tok_ids), vp),
            (ctypes.c_void_p(b_E), vp),
            (ctypes.c_void_p(b_X), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_emb_gather_1, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(emb_gather_1) failed: {rc}")

    def launch_embed_grad_reduce_1(
        self,
        N: int,
        V: int,
        D: int,
        b_tok_ids: ctypes.c_void_p,
        b_dX: ctypes.c_void_p,
        b_dE: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(V) * int(D)
        self._set_args(self._k_emb_grad_1, [
            (ctypes.c_int32(N), 4),
            (ctypes.c_int32(V), 4),
            (ctypes.c_int32(D), 4),
            (ctypes.c_void_p(b_tok_ids), vp),
            (ctypes.c_void_p(b_dX), vp),
            (ctypes.c_void_p(b_dE), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_emb_grad_1, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(emb_grad_1) failed: {rc}")

    def launch_gru_step_fwd(
        self,
        B: int,
        H: int,
        t: int,
        b_Xg_all: ctypes.c_void_p,
        b_Sg_all: ctypes.c_void_p,
        b_bx: ctypes.c_void_p,
        b_bs: ctypes.c_void_p,
        b_s_cur: ctypes.c_void_p,
        b_s_prev_all: ctypes.c_void_p,
        b_s_all: ctypes.c_void_p,
        b_cache_zrh: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(H)
        self._set_args(self._k_gru_fwd, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(H), 4),
            (ctypes.c_int32(t), 4),
            (ctypes.c_void_p(b_Xg_all), vp),
            (ctypes.c_void_p(b_Sg_all), vp),
            (ctypes.c_void_p(b_bx), vp),
            (ctypes.c_void_p(b_bs), vp),
            (ctypes.c_void_p(b_s_cur), vp),
            (ctypes.c_void_p(b_s_prev_all), vp),
            (ctypes.c_void_p(b_s_all), vp),
            (ctypes.c_void_p(b_cache_zrh), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_gru_fwd, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(gru_fwd) failed: {rc}")

    def launch_gru_projection_fused(
        self,
        B: int,
        H: int,
        T: int,
        t: int,
        b_Xg_all: ctypes.c_void_p,
        b_U: ctypes.c_void_p,
        b_bx: ctypes.c_void_p,
        b_bs: ctypes.c_void_p,
        b_s_cur: ctypes.c_void_p,
        b_s_prev_all: ctypes.c_void_p,
        b_s_all: ctypes.c_void_p,
        b_Sg_all: ctypes.c_void_p,
        b_cache_zrh: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_gru_proj_fused, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(H), 4),
            (ctypes.c_int32(T), 4),
            (ctypes.c_int32(t), 4),
            (ctypes.c_void_p(b_Xg_all), vp),
            (ctypes.c_void_p(b_U), vp),
            (ctypes.c_void_p(b_bx), vp),
            (ctypes.c_void_p(b_bs), vp),
            (ctypes.c_void_p(b_s_cur), vp),
            (ctypes.c_void_p(b_s_prev_all), vp),
            (ctypes.c_void_p(b_s_all), vp),
            (ctypes.c_void_p(b_Sg_all), vp),
            (ctypes.c_void_p(b_cache_zrh), vp),
        ])
        gs = (ctypes.c_size_t * 2)(
            ((int(H) + 31) // 32) * 16,
            ((int(B) + 31) // 32) * 16,
        )
        ls = (ctypes.c_size_t * 2)(16, 16)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_gru_proj_fused, 2, None, gs, ls, 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(gru_projection_fused) failed: {rc}")

    def launch_gru_step_bwd(
        self,
        B: int,
        H: int,
        t: int,
        b_dS_out_all: ctypes.c_void_p,
        b_dS_carry: ctypes.c_void_p,
        b_cache_zrh: ctypes.c_void_p,
        b_s_prev_all: ctypes.c_void_p,
        b_Sg_all: ctypes.c_void_p,
        b_bs: ctypes.c_void_p,
        b_dXg_all: ctypes.c_void_p,
        b_dSg_all: ctypes.c_void_p,
        b_dS_prev_direct: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(H)
        self._set_args(self._k_gru_bwd, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(H), 4),
            (ctypes.c_int32(t), 4),
            (ctypes.c_void_p(b_dS_out_all), vp),
            (ctypes.c_void_p(b_dS_carry), vp),
            (ctypes.c_void_p(b_cache_zrh), vp),
            (ctypes.c_void_p(b_s_prev_all), vp),
            (ctypes.c_void_p(b_Sg_all), vp),
            (ctypes.c_void_p(b_bs), vp),
            (ctypes.c_void_p(b_dXg_all), vp),
            (ctypes.c_void_p(b_dSg_all), vp),
            (ctypes.c_void_p(b_dS_prev_direct), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_gru_bwd, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(gru_bwd) failed: {rc}")

    def launch_clip_grad(self, n_total: int, clip_val: float, b_grad: ctypes.c_void_p) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_clip_grad, [
            (ctypes.c_int32(n_total), 4),
            (ctypes.c_float(clip_val), 4),
            (ctypes.c_void_p(b_grad), vp),
        ])
        gs = ctypes.c_size_t(((int(n_total) + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_clip_grad, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(clip_grad) failed: {rc}")

    def launch_copy_slice(
        self,
        slice_elems: int,
        t: int,
        direction: int,
        b_seq: ctypes.c_void_p,
        b_step: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_copy_slice, [
            (ctypes.c_int32(slice_elems), 4),
            (ctypes.c_int32(t), 4),
            (ctypes.c_int32(direction), 4),
            (ctypes.c_void_p(b_seq), vp),
            (ctypes.c_void_p(b_step), vp),
        ])
        gs = ctypes.c_size_t(((int(slice_elems) + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_copy_slice, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(copy_slice) failed: {rc}")

    def launch_slice_tokens(
        self,
        B: int,
        T: int,
        stream_len: int,
        s0: int,
        b_full_stream: ctypes.c_void_p,
        b_tok_in: ctypes.c_void_p,
        b_tok_tgt: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(T)
        self._set_args(self._k_slice_tokens, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(T), 4),
            (ctypes.c_int32(stream_len), 4),
            (ctypes.c_int32(s0), 4),
            (ctypes.c_void_p(b_full_stream), vp),
            (ctypes.c_void_p(b_tok_in), vp),
            (ctypes.c_void_p(b_tok_tgt), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_slice_tokens, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(slice_tokens) failed: {rc}")

    def launch_zero_buf(self, n_total: int, b_buf: ctypes.c_void_p) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_zero_buf, [
            (ctypes.c_int32(n_total), 4),
            (ctypes.c_void_p(b_buf), vp),
        ])
        gs = ctypes.c_size_t(((int(n_total) + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_zero_buf, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(zero_buf) failed: {rc}")

    def launch_vec_add(
        self,
        n_total: int,
        b_a: ctypes.c_void_p,
        b_b: ctypes.c_void_p,
        b_out: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_vec_add, [
            (ctypes.c_int32(n_total), 4),
            (ctypes.c_void_p(b_a), vp),
            (ctypes.c_void_p(b_b), vp),
            (ctypes.c_void_p(b_out), vp),
        ])
        gs = ctypes.c_size_t(((int(n_total) + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_vec_add, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(vec_add) failed: {rc}")

    def launch_vec_add_offset(
        self,
        n_total: int,
        a_off: int,
        b_off: int,
        out_off: int,
        b_a: ctypes.c_void_p,
        b_b: ctypes.c_void_p,
        b_out: ctypes.c_void_p,
    ) -> None:
        if n_total <= 0:
            return
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_vec_add_offset, [
            (ctypes.c_int32(n_total), 4),
            (ctypes.c_int32(a_off), 4),
            (ctypes.c_int32(b_off), 4),
            (ctypes.c_int32(out_off), 4),
            (ctypes.c_void_p(b_a), vp),
            (ctypes.c_void_p(b_b), vp),
            (ctypes.c_void_p(b_out), vp),
        ])
        gs = ctypes.c_size_t(((int(n_total) + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_vec_add_offset, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(vec_add_offset) failed: {rc}")

    def launch_slice_stream_1(
        self,
        B: int,
        T: int,
        stream_len: int,
        s0: int,
        b_full_stream: ctypes.c_void_p,
        b_out_slice: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(B) * int(T)
        self._set_args(self._k_slice_stream_1, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(T), 4),
            (ctypes.c_int32(stream_len), 4),
            (ctypes.c_int32(s0), 4),
            (ctypes.c_void_p(b_full_stream), vp),
            (ctypes.c_void_p(b_out_slice), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_slice_stream_1, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(slice_stream_1) failed: {rc}")

    def launch_embed_add_1(
        self,
        N: int,
        C: int,
        D: int,
        b_chk_ids: ctypes.c_void_p,
        b_E_chunk: ctypes.c_void_p,
        b_X_inout: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        tot = int(N) * int(D)
        self._set_args(self._k_embed_add_1, [
            (ctypes.c_int32(N), 4),
            (ctypes.c_int32(C), 4),
            (ctypes.c_int32(D), 4),
            (ctypes.c_void_p(b_chk_ids), vp),
            (ctypes.c_void_p(b_E_chunk), vp),
            (ctypes.c_void_p(b_X_inout), vp),
        ])
        gs = ctypes.c_size_t(((tot + 255) // 256) * 256)
        ls = ctypes.c_size_t(256)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_embed_add_1, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(embed_add_1) failed: {rc}")

    def launch_fast_weight_scan_fwd(
        self,
        B: int,
        T: int,
        D: int,
        b_K_raw: ctypes.c_void_p,
        b_QVG_raw: ctypes.c_void_p,
        b_bk: ctypes.c_void_p,
        b_bqvg: ctypes.c_void_p,
        b_M_cur: ctypes.c_void_p,
        b_M_prev_all: ctypes.c_void_p,
        b_R_hat_all: ctypes.c_void_p,
        b_fw_cache: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_fw_fwd, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(T), 4),
            (ctypes.c_int32(D), 4),
            (ctypes.c_void_p(b_K_raw), vp),
            (ctypes.c_void_p(b_QVG_raw), vp),
            (ctypes.c_void_p(b_bk), vp),
            (ctypes.c_void_p(b_bqvg), vp),
            (ctypes.c_void_p(b_M_cur), vp),
            (ctypes.c_void_p(b_M_prev_all), vp),
            (ctypes.c_void_p(b_R_hat_all), vp),
            (ctypes.c_void_p(b_fw_cache), vp),
        ])
        gs = ctypes.c_size_t(int(B) * 64)
        ls = ctypes.c_size_t(64)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_fw_fwd, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(fast_weight_scan_fwd) failed: {rc}")

    def launch_fast_weight_scan_bwd(
        self,
        B: int,
        T: int,
        D: int,
        b_dR_hat_all: ctypes.c_void_p,
        b_M_prev_all: ctypes.c_void_p,
        b_fw_cache: ctypes.c_void_p,
        b_dK_raw: ctypes.c_void_p,
        b_dQVG_raw: ctypes.c_void_p,
    ) -> None:
        vp = ctypes.sizeof(ctypes.c_void_p)
        self._set_args(self._k_fw_bwd, [
            (ctypes.c_int32(B), 4),
            (ctypes.c_int32(T), 4),
            (ctypes.c_int32(D), 4),
            (ctypes.c_void_p(b_dR_hat_all), vp),
            (ctypes.c_void_p(b_M_prev_all), vp),
            (ctypes.c_void_p(b_fw_cache), vp),
            (ctypes.c_void_p(b_dK_raw), vp),
            (ctypes.c_void_p(b_dQVG_raw), vp),
        ])
        gs = ctypes.c_size_t(int(B) * 64)
        ls = ctypes.c_size_t(64)
        rc = self._ocl.clEnqueueNDRangeKernel(self._queue, self._k_fw_bwd, 1, None, ctypes.byref(gs), ctypes.byref(ls), 0, None, None)
        if rc != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel(fast_weight_scan_bwd) failed: {rc}")

    def info(self) -> dict:
        return dict(
            available=self.available,
            platform=self.platform_name,
            device=self.device_name,
            compute_units=self.compute_units,
            global_mem_gb=self.global_mem_gb,
        )
