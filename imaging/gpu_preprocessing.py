"""
GPU-Accelerated Image Preprocessing
====================================

GPU-accelerated image processing operations using CuPy.
Falls back to CPU (NumPy/SciPy) when GPU is not available.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


# ── CUDA Kernels ─────────────────────────────────────────────────────────

_GAUSSIAN_BLUR_3D_KERNEL = r"""
extern "C" __global__
void gaussian_blur_3d(
    const float* input,
    float* output,
    const float* kernel,
    int kernel_size,
    int D, int H, int W
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= W || y >= H || z >= D) return;

    int half_k = kernel_size / 2;
    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int kz = -half_k; kz <= half_k; kz++) {
        for (int ky = -half_k; ky <= half_k; ky++) {
            for (int kx = -half_k; kx <= half_k; kx++) {
                int nz = z + kz;
                int ny = y + ky;
                int nx = x + kx;

                if (nz >= 0 && nz < D && ny >= 0 && ny < H && nx >= 0 && nx < W) {
                    int ki = (kz + half_k) * kernel_size * kernel_size +
                             (ky + half_k) * kernel_size +
                             (kx + half_k);
                    float w = kernel[ki];
                    sum += input[nz * H * W + ny * W + nx] * w;
                    weight_sum += w;
                }
            }
        }
    }

    output[z * H * W + y * W + x] = sum / weight_sum;
}
"""

_MORPHOLOGICAL_KERNEL = r"""
extern "C" __global__
void morphological_op(
    const unsigned char* input,
    unsigned char* output,
    int op_type,  // 0=erosion, 1=dilation
    int kernel_size,
    int H, int W
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    int half_k = kernel_size / 2;

    if (op_type == 0) {
        // Erosion: minimum in neighborhood
        unsigned char min_val = 255;
        for (int ky = -half_k; ky <= half_k; ky++) {
            for (int kx = -half_k; kx <= half_k; kx++) {
                int ny = y + ky;
                int nx = x + kx;
                if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                    unsigned char val = input[ny * W + nx];
                    if (val < min_val) min_val = val;
                }
            }
        }
        output[y * W + x] = min_val;
    } else {
        // Dilation: maximum in neighborhood
        unsigned char max_val = 0;
        for (int ky = -half_k; ky <= half_k; ky++) {
            for (int kx = -half_k; kx <= half_k; kx++) {
                int ny = y + ky;
                int nx = x + kx;
                if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                    unsigned char val = input[ny * W + nx];
                    if (val > max_val) max_val = val;
                }
            }
        }
        output[y * W + x] = max_val;
    }
}
"""


class GPUImageProcessor:
    """
    GPU-accelerated image processing.

    Automatically uses GPU (CuPy) if available, otherwise falls back to CPU.

    Example
    -------
    >>> processor = GPUImageProcessor()
    >>> blurred = processor.gaussian_blur(image, sigma=2.0)
    >>> binary = processor.threshold_otsu(blurred)
    """

    def __init__(self):
        self._has_gpu = False
        self._kernels = {}

        try:
            import cupy as cp
            self._xp = cp
            self._has_gpu = True
            self._compile_kernels()
            log.info("GPU image processing enabled (CuPy)")
        except ImportError:
            self._xp = np
            log.info("GPU not available, using CPU for image processing")

    def _compile_kernels(self):
        """Compile CUDA kernels."""
        try:
            import cupy as cp
            self._kernels['gaussian_3d'] = cp.RawKernel(
                _GAUSSIAN_BLUR_3D_KERNEL, "gaussian_blur_3d"
            )
            self._kernels['morphological'] = cp.RawKernel(
                _MORPHOLOGICAL_KERNEL, "morphological_op"
            )
        except Exception as e:
            log.warning(f"Failed to compile GPU kernels: {e}")

    def _to_gpu(self, arr: np.ndarray) -> np.ndarray:
        """Transfer array to GPU if available."""
        if self._has_gpu:
            return self._xp.asarray(arr)
        return arr

    def _to_cpu(self, arr) -> np.ndarray:
        """Transfer array back to CPU."""
        if self._has_gpu and hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

    def gaussian_blur(
        self,
        image: np.ndarray,
        sigma: Union[float, Tuple[float, ...]] = 1.0,
    ) -> np.ndarray:
        """
        Gaussian blur (smoothing).

        Parameters
        ----------
        image : np.ndarray
            Input image (2D or 3D)
        sigma : float or tuple
            Standard deviation for Gaussian kernel

        Returns
        -------
        np.ndarray
            Blurred image
        """
        if self._has_gpu:
            return self._gaussian_blur_gpu(image, sigma)
        return self._gaussian_blur_cpu(image, sigma)

    def _gaussian_blur_gpu(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """GPU Gaussian blur using custom kernel."""
        import cupy as cp

        # Create Gaussian kernel
        if isinstance(sigma, (int, float)):
            sigma = (sigma,) * image.ndim

        kernel_size = int(6 * max(sigma) + 1) | 1  # Ensure odd

        if image.ndim == 3:
            # 3D Gaussian kernel
            z, y, x = np.ogrid[
                -(kernel_size // 2):(kernel_size // 2) + 1,
                -(kernel_size // 2):(kernel_size // 2) + 1,
                -(kernel_size // 2):(kernel_size // 2) + 1,
            ]
            kernel = np.exp(-(x**2/(2*sigma[2]**2) + y**2/(2*sigma[1]**2) + z**2/(2*sigma[0]**2)))
            kernel = kernel / kernel.sum()
            kernel = kernel.astype(np.float32).ravel()

            # Transfer to GPU
            img_gpu = cp.asarray(image.astype(np.float32))
            out_gpu = cp.zeros_like(img_gpu)
            kernel_gpu = cp.asarray(kernel)

            D, H, W = image.shape
            block = (8, 8, 8)
            grid = ((W + 7) // 8, (H + 7) // 8, (D + 7) // 8)

            self._kernels['gaussian_3d'](
                grid, block,
                (img_gpu, out_gpu, kernel_gpu, np.int32(kernel_size),
                 np.int32(D), np.int32(H), np.int32(W))
            )

            cp.cuda.Stream.null.synchronize()
            return out_gpu.get()

        else:
            # 2D - use scipy.ndimage as fallback
            return self._gaussian_blur_cpu(image, sigma)

    def _gaussian_blur_cpu(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """CPU Gaussian blur using scipy."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(image.astype(np.float32), sigma=sigma)
        except ImportError:
            # Manual convolution
            return self._gaussian_blur_manual(image, sigma)

    def _gaussian_blur_manual(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Manual Gaussian blur implementation."""
        if isinstance(sigma, (int, float)):
            sigma = (sigma,) * image.ndim

        # Create 1D kernels and apply separably
        for dim, s in enumerate(sigma):
            if s > 0:
                kernel_size = int(6 * s + 1) | 1
                x = np.arange(kernel_size) - kernel_size // 2
                kernel = np.exp(-x**2 / (2 * s**2))
                kernel = kernel / kernel.sum()

                # Apply along dimension
                image = self._convolve_1d(image.astype(np.float32), kernel, dim)

        return image

    def _convolve_1d(self, arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
        """1D convolution along axis."""
        # Move axis to end
        arr = np.moveaxis(arr, axis, -1)
        shape = arr.shape
        arr = arr.reshape(-1, shape[-1])

        # Pad and convolve
        pad = len(kernel) // 2
        padded = np.pad(arr, ((0, 0), (pad, pad)), mode='reflect')

        result = np.zeros_like(arr)
        for i in range(len(kernel)):
            result += padded[:, i:i + shape[-1]] * kernel[i]

        result = result.reshape(shape)
        return np.moveaxis(result, -1, axis)

    def threshold_otsu(self, image: np.ndarray) -> np.ndarray:
        """
        Otsu's thresholding.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale image

        Returns
        -------
        np.ndarray
            Binary image
        """
        try:
            from skimage.filters import threshold_otsu as sk_otsu
            thresh = sk_otsu(image)
            return (image > thresh).astype(np.uint8)
        except ImportError:
            return self._threshold_otsu_manual(image)

    def _threshold_otsu_manual(self, image: np.ndarray) -> np.ndarray:
        """Manual Otsu's threshold implementation."""
        hist, bins = np.histogram(image.ravel(), bins=256)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Cumulative sums and means
        w0 = np.cumsum(hist)
        w1 = np.cumsum(hist[::-1])[::-1]
        mu0 = np.cumsum(hist * bin_centers) / np.clip(w0, 1, None)
        mu1 = (np.cumsum((hist * bin_centers)[::-1]) / np.clip(w1[::-1], 1, None))[::-1]

        # Between-class variance
        variance = w0[:-1] * w1[1:] * (mu0[:-1] - mu1[1:]) ** 2
        threshold = bin_centers[np.argmax(variance)]

        return (image > threshold).astype(np.uint8)

    def morphological_ops(
        self,
        image: np.ndarray,
        operation: str = 'opening',
        kernel_size: int = 3,
        iterations: int = 1,
    ) -> np.ndarray:
        """
        Morphological operations.

        Parameters
        ----------
        image : np.ndarray
            Binary input image
        operation : str
            Operation: 'erosion', 'dilation', 'opening', 'closing'
        kernel_size : int
            Kernel size (default: 3)
        iterations : int
            Number of iterations (default: 1)

        Returns
        -------
        np.ndarray
            Processed image
        """
        if self._has_gpu and image.ndim == 2:
            return self._morphological_gpu(image, operation, kernel_size, iterations)
        return self._morphological_cpu(image, operation, kernel_size, iterations)

    def _morphological_gpu(
        self,
        image: np.ndarray,
        operation: str,
        kernel_size: int,
        iterations: int,
    ) -> np.ndarray:
        """GPU morphological operations."""
        import cupy as cp

        img = cp.asarray(image.astype(np.uint8))
        H, W = image.shape
        block = (16, 16)
        grid = ((W + 15) // 16, (H + 15) // 16)

        result = img.copy()

        for _ in range(iterations):
            if operation == 'erosion':
                out = cp.zeros_like(result)
                self._kernels['morphological'](
                    grid, block,
                    (result, out, np.int32(0), np.int32(kernel_size),
                     np.int32(H), np.int32(W))
                )
                result = out
            elif operation == 'dilation':
                out = cp.zeros_like(result)
                self._kernels['morphological'](
                    grid, block,
                    (result, out, np.int32(1), np.int32(kernel_size),
                     np.int32(H), np.int32(W))
                )
                result = out
            elif operation == 'opening':
                # Erosion then dilation
                out = cp.zeros_like(result)
                self._kernels['morphological'](
                    grid, block,
                    (result, out, np.int32(0), np.int32(kernel_size),
                     np.int32(H), np.int32(W))
                )
                result2 = cp.zeros_like(out)
                self._kernels['morphological'](
                    grid, block,
                    (out, result2, np.int32(1), np.int32(kernel_size),
                     np.int32(H), np.int32(W))
                )
                result = result2
            elif operation == 'closing':
                # Dilation then erosion
                out = cp.zeros_like(result)
                self._kernels['morphological'](
                    grid, block,
                    (result, out, np.int32(1), np.int32(kernel_size),
                     np.int32(H), np.int32(W))
                )
                result2 = cp.zeros_like(out)
                self._kernels['morphological'](
                    grid, block,
                    (out, result2, np.int32(0), np.int32(kernel_size),
                     np.int32(H), np.int32(W))
                )
                result = result2

        cp.cuda.Stream.null.synchronize()
        return result.get()

    def _morphological_cpu(
        self,
        image: np.ndarray,
        operation: str,
        kernel_size: int,
        iterations: int,
    ) -> np.ndarray:
        """CPU morphological operations."""
        try:
            from scipy.ndimage import binary_erosion, binary_dilation
            from skimage.morphology import disk, ball

            if image.ndim == 2:
                selem = disk(kernel_size // 2)
            else:
                selem = ball(kernel_size // 2)

            result = image.astype(bool)

            for _ in range(iterations):
                if operation == 'erosion':
                    result = binary_erosion(result, structure=selem)
                elif operation == 'dilation':
                    result = binary_dilation(result, structure=selem)
                elif operation == 'opening':
                    result = binary_erosion(result, structure=selem)
                    result = binary_dilation(result, structure=selem)
                elif operation == 'closing':
                    result = binary_dilation(result, structure=selem)
                    result = binary_erosion(result, structure=selem)

            return result.astype(np.uint8)

        except ImportError:
            log.warning("scipy/skimage not available for morphological ops")
            return image


# ── Convenience Functions ────────────────────────────────────────────────

_processor = None


def _get_processor() -> GPUImageProcessor:
    """Get or create global processor."""
    global _processor
    if _processor is None:
        _processor = GPUImageProcessor()
    return _processor


def gaussian_blur_3d(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Convenience function for 3D Gaussian blur."""
    return _get_processor().gaussian_blur(image, sigma)


def threshold_otsu(image: np.ndarray) -> np.ndarray:
    """Convenience function for Otsu thresholding."""
    return _get_processor().threshold_otsu(image)


def morphological_ops(
    image: np.ndarray,
    operation: str = 'opening',
    **kwargs,
) -> np.ndarray:
    """Convenience function for morphological operations."""
    return _get_processor().morphological_ops(image, operation, **kwargs)
