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

// 16x16 LDS-tiled SGEMM with bank-conflict-free padding (17) supporting NN/NT/TN
__kernel void sgemm_tiled_kernel(
    const int M,
    const int N,
    const int K,
    const int transA,
    const int transB,
    const float alpha,
    const float beta,
    __global const float* A,
    __global const float* B,
    __global float* C
) {
    __local float As[16][17];
    __local float Bs[16][17];
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int col = get_group_id(0) * 16 + lx;
    int row = get_group_id(1) * 16 + ly;

    float acc = 0.0f;
    int n_tiles = (K + 15) / 16;
    for (int t = 0; t < n_tiles; ++t) {
        int a_k = t * 16 + lx;
        int b_k = t * 16 + ly;
        if (row < M && a_k < K) {
            As[ly][lx] = (transA == 0) ? A[row * K + a_k] : A[a_k * M + row];
        } else {
            As[ly][lx] = 0.0f;
        }
        if (b_k < K && col < N) {
            Bs[ly][lx] = (transB == 0) ? B[b_k * N + col] : B[col * K + b_k];
        } else {
            Bs[ly][lx] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc = fma(As[ly][k], Bs[k][lx], acc);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        int idx = row * N + col;
        if (beta == 0.0f) {
            C[idx] = alpha * acc;
        } else {
            C[idx] = fma(alpha, acc, beta * C[idx]);
        }
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
    int tgt = targets[row];

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
            (ctypes.c_void_p(b_A), vp),
            (ctypes.c_void_p(b_B), vp),
            (ctypes.c_void_p(b_C), vp),
        ])
        gs = (ctypes.c_size_t * 2)(((int(N) + 15) // 16) * 16, ((int(M) + 15) // 16) * 16)
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

    def info(self) -> dict:
        return dict(
            available=self.available,
            platform=self.platform_name,
            device=self.device_name,
            compute_units=self.compute_units,
            global_mem_gb=self.global_mem_gb,
        )
