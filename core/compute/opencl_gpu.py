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

        self._ocl = ocl
        self._ctx = ctx
        self._queue = queue
        self._prog = prog
        self._k_circuit = k_circ
        self._k_spmv = k_spmv
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

    def finish(self) -> None:
        if self._ocl is not None and self._queue is not None:
            self._ocl.clFinish(self._queue)

    def info(self) -> dict:
        return dict(
            available=self.available,
            platform=self.platform_name,
            device=self.device_name,
            compute_units=self.compute_units,
            global_mem_gb=self.global_mem_gb,
        )
