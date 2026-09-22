#!/usr/bin/env python3
"""
RGA-accelerated image preprocessor for RKNN inference on RV1126B.

Uses Rockchip RGA2 hardware (im2d API via librga.so) for letterbox resize.
Key optimization: buffers are imported ONCE and reused via handles, avoiding
the ~5 ms per-call kernel overhead of re-registering virtual addresses.

Falls back to OpenCV CPU letterbox if librga is unavailable.

Usage:
    from rga_preprocess import RGAPreprocessor

    preproc = RGAPreprocessor(target_size=(640, 640))

    # Per-frame:
    img_lb, ratio, (dw, dh) = preproc.process(frame)

    # Cleanup:
    preproc.release()
"""

import ctypes
import os
import sys
import time
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# --- librga constants (from im2d_type.h / rga.h) ---
RK_FORMAT_RGB_888 = 0x2 << 8   # 0x200
RK_FORMAT_BGR_888 = 0x7 << 8   # 0x700
RK_FORMAT_YCbCr_420_SP = 0xA << 8  # 0xa00 NV12
IM_STATUS_SUCCESS = 1
IM_SYNC = 1 << 19
_RGA_BUF_SIZE = 96  # rga_buffer_t is 96 bytes on librga v1.10.5


class _rga_buffer_t(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_char * _RGA_BUF_SIZE)]


class _im_rect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]


class _im_handle_param_t(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
    ]


def _cpu_letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """CPU fallback: resize with aspect ratio preserved, pad with gray."""
    shape = img.shape[:2]  # h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)


class RGAPreprocessor:
    """RGA-accelerated letterbox preprocessor.

    Uses import+handle reuse pattern for maximum performance.
    Automatically falls back to CPU letterbox if RGA is unavailable.
    """

    def __init__(self, target_size=(640, 640), src_format="BGR"):
        """
        Args:
            target_size: (width, height) of the output image.
            src_format: "BGR" for cv2.imread/GStreamer BGR input,
                        "RGB" for RGB input,
                        "NV12" for camera NV12 input.
        """
        self.target_w, self.target_h = target_size
        self.src_format = src_format
        self._lib = None
        self._available = False
        self._src_handle = 0
        self._dst_handle = 0
        self._last_src_key = None
        self._dst_buf = None
        self._import_va = None
        self._import_fd = None
        self._release = None
        self._wrap_handle = None
        self._improcess = None

        if src_format == "BGR":
            self._rk_fmt = RK_FORMAT_BGR_888
        elif src_format == "RGB":
            self._rk_fmt = RK_FORMAT_RGB_888
        elif src_format == "NV12":
            self._rk_fmt = RK_FORMAT_YCbCr_420_SP
        else:
            raise ValueError(f"Unsupported src_format: {src_format}")

        self._try_load_rga()

    @property
    def available(self):
        """True if RGA hardware is loaded and functional."""
        return self._available

    def _try_load_rga(self):
        """Attempt to load librga and resolve symbols."""
        lib_paths = [
            "/oem/usr/lib/librga.so",
            "librga.so",
        ]
        for path in lib_paths:
            try:
                lib = ctypes.CDLL(path)
                # Verify key symbols exist
                getattr(lib, "importbuffer_virtualaddr")
                getattr(lib, "releasebuffer_handle")
                getattr(lib, "wrapbuffer_handle_t")
                getattr(lib, "improcess")
                self._lib = lib
                break
            except OSError:
                continue

        if self._lib is None:
            return

        try:
            # importbuffer_virtualaddr(va, param) -> handle
            self._import_va = self._lib.importbuffer_virtualaddr
            self._import_va.restype = ctypes.c_uint32
            self._import_va.argtypes = [ctypes.c_void_p, ctypes.POINTER(_im_handle_param_t)]

            # importbuffer_fd(fd, param) -> handle
            self._import_fd = self._lib.importbuffer_fd
            self._import_fd.restype = ctypes.c_uint32
            self._import_fd.argtypes = [ctypes.c_int, ctypes.POINTER(_im_handle_param_t)]

            # releasebuffer_handle(handle) -> status
            self._release = self._lib.releasebuffer_handle
            self._release.restype = ctypes.c_int
            self._release.argtypes = [ctypes.c_uint32]

            # wrapbuffer_handle_t(handle, w, h, ws, hs, fmt) -> rga_buffer_t
            self._wrap_handle = self._lib.wrapbuffer_handle_t
            self._wrap_handle.restype = _rga_buffer_t
            self._wrap_handle.argtypes = [
                ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]

            # improcess(src, dst, pat, srect, drect, prect, usage, ...) -> status
            self._improcess = self._lib.improcess
            self._improcess.restype = ctypes.c_int
            self._improcess.argtypes = [
                _rga_buffer_t, _rga_buffer_t, _rga_buffer_t,
                _im_rect, _im_rect, _im_rect,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ]

            self._available = True
        except Exception:
            self._available = False
            self._lib = None

    def _ensure_src_imported(self, img_ptr, src_w, src_h, src_fmt):
        """Import source buffer if size/format changed. Returns handle."""
        key = (src_w, src_h, src_fmt)
        if self._last_src_key == key and self._src_handle > 0:
            return self._src_handle

        # Release old source handle
        if self._src_handle > 0:
            self._release(self._src_handle)
            self._src_handle = 0

        param = _im_handle_param_t(src_w, src_h, src_fmt)
        handle = self._import_va(img_ptr, ctypes.byref(param))
        if handle > 0:
            self._src_handle = handle
            self._last_src_key = key
        return handle

    def _ensure_dst_imported(self):
        """Import destination buffer once. Returns handle."""
        if self._dst_handle > 0:
            return self._dst_handle

        self._dst_buf = np.full(
            (self.target_h, self.target_w, 3), 114, dtype=np.uint8
        )
        param = _im_handle_param_t(self.target_w, self.target_h, self._rk_fmt)
        handle = self._import_va(
            self._dst_buf.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(param),
        )
        if handle > 0:
            self._dst_handle = handle
        return handle

    def _compute_letterbox_params(self, src_w, src_h):
        """Compute letterbox resize parameters."""
        r = min(self.target_w / src_h, self.target_h / src_w)
        # r = min(target_h / src_h, target_w / src_w)
        resized_w = int(round(src_w * r))
        resized_h = int(round(src_h * r))
        dw = (self.target_w - resized_w) / 2
        dh = (self.target_h - resized_h) / 2
        pad_x = int(round(dw - 0.1))
        pad_y = int(round(dh - 0.1))
        return resized_w, resized_h, pad_x, pad_y, r, (dw, dh)

    def process(self, img):
        """Letterbox resize image to target_size using RGA.

        Args:
            img: numpy array, BGR or RGB uint8, shape (H, W, 3).

        Returns:
            (letterboxed_img, ratio, (dw, dh))
            Same interface as CPU letterbox().
        """
        if not self._available or not HAS_CV2:
            return _cpu_letterbox(img, (self.target_h, self.target_w))

        src_h, src_w = img.shape[:2]
        resized_w, resized_h, pad_x, pad_y, r, (dw, dh) = \
            self._compute_letterbox_params(src_w, src_h)

        try:
            # Ensure destination buffer is ready
            dst_handle = self._ensure_dst_imported()
            if dst_handle == 0:
                return _cpu_letterbox(img, (self.target_h, self.target_w))

            # Import source buffer (cached if same size)
            src_handle = self._ensure_src_imported(
                img.ctypes.data_as(ctypes.c_void_p),
                src_w, src_h, self._rk_fmt,
            )
            if src_handle == 0:
                return _cpu_letterbox(img, (self.target_h, self.target_w))

            # Wrap handles (fast, no re-registration)
            src_buf = self._wrap_handle(
                src_handle, src_w, src_h, src_w, src_h, self._rk_fmt
            )
            dst_buf = self._wrap_handle(
                dst_handle, self.target_w, self.target_h,
                self.target_w, self.target_h, self._rk_fmt,
            )

            # Setup rects
            srect = _im_rect(0, 0, src_w, src_h)
            drect = _im_rect(pad_x, pad_y, resized_w, resized_h)
            prect = _im_rect(0, 0, 0, 0)
            pat = _rga_buffer_t()

            # Fill destination with padding color
            self._dst_buf[:] = 114

            # Run RGA improcess
            rc = self._improcess(
                src_buf, dst_buf, pat,
                srect, drect, prect,
                IM_SYNC, None, None, 0,
            )

            if rc == IM_STATUS_SUCCESS:
                return self._dst_buf.copy(), r, (dw, dh)
            else:
                return _cpu_letterbox(img, (self.target_h, self.target_w))

        except Exception:
            return _cpu_letterbox(img, (self.target_h, self.target_w))

    def process_nv12(self, nv12_buf, src_w, src_h):
        """NV12 → RGB/BGR letterbox in a single RGA op.

        Args:
            nv12_buf: numpy array or bytes, NV12 format, size = w*h*3//2.
            src_w: source width.
            src_h: source height.

        Returns:
            (letterboxed_rgb, ratio, (dw, dh))
        """
        if not self._available:
            raise RuntimeError("RGA not available for NV12 processing")

        resized_w, resized_h, pad_x, pad_y, r, (dw, dh) = \
            self._compute_letterbox_params(src_w, src_h)

        # Import NV12 source
        src_ptr = nv12_buf.ctypes.data_as(ctypes.c_void_p) \
            if isinstance(nv12_buf, np.ndarray) \
            else ctypes.c_void_p(nv12_buf)

        src_param = _im_handle_param_t(src_w, src_h, RK_FORMAT_YCbCr_420_SP)
        src_handle = self._import_va(src_ptr, ctypes.byref(src_param))
        if src_handle == 0:
            raise RuntimeError("Failed to import NV12 buffer")

        try:
            dst_handle = self._ensure_dst_imported()
            if dst_handle == 0:
                raise RuntimeError("Failed to import dst buffer")

            src_buf = self._wrap_handle(
                src_handle, src_w, src_h, src_w, src_h, RK_FORMAT_YCbCr_420_SP
            )
            dst_buf = self._wrap_handle(
                dst_handle, self.target_w, self.target_h,
                self.target_w, self.target_h, self._rk_fmt,
            )

            srect = _im_rect(0, 0, src_w, src_h)
            drect = _im_rect(pad_x, pad_y, resized_w, resized_h)
            prect = _im_rect(0, 0, 0, 0)
            pat = _rga_buffer_t()

            self._dst_buf[:] = 114

            rc = self._improcess(
                src_buf, dst_buf, pat,
                srect, drect, prect,
                IM_SYNC, None, None, 0,
            )

            if rc != IM_STATUS_SUCCESS:
                raise RuntimeError(f"RGA improcess failed: {rc}")

            return self._dst_buf.copy(), r, (dw, dh)
        finally:
            self._release(src_handle)

    def release(self):
        """Release all RGA buffer handles."""
        if self._src_handle > 0 and self._release:
            self._release(self._src_handle)
            self._src_handle = 0
        if self._dst_handle > 0 and self._release:
            self._release(self._dst_handle)
            self._dst_handle = 0
        self._last_src_key = None
        self._dst_buf = None

    def __del__(self):
        self.release()
