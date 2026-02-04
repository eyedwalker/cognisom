"""
Image Importers
===============

Import microscopy images from various formats.

Supported formats:
- TIFF (standard and OME-TIFF)
- CZI (Zeiss)
- ND2 (Nikon)
- PNG, JPEG (2D images)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """
    Metadata for an image.

    Attributes
    ----------
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    depth : int
        Number of Z slices (1 for 2D)
    channels : int
        Number of channels
    timepoints : int
        Number of time points
    pixel_size_x : float
        Pixel size in X (um)
    pixel_size_y : float
        Pixel size in Y (um)
    pixel_size_z : float
        Pixel size in Z (um)
    channel_names : List[str]
        Names of channels
    dtype : np.dtype
        Data type
    """
    width: int = 0
    height: int = 0
    depth: int = 1
    channels: int = 1
    timepoints: int = 1
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    pixel_size_z: float = 1.0
    channel_names: List[str] = field(default_factory=list)
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.uint8))


@dataclass
class ImageStack:
    """
    A multi-dimensional image stack.

    Attributes
    ----------
    data : np.ndarray
        Image data with shape depending on dimensions:
        (H, W) for 2D
        (D, H, W) for 3D
        (C, H, W) for multi-channel 2D
        (T, C, D, H, W) for full 5D
    metadata : ImageMetadata
        Image metadata
    """
    data: np.ndarray
    metadata: ImageMetadata = field(default_factory=ImageMetadata)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def get_channel(self, channel: int) -> np.ndarray:
        """Get a specific channel."""
        if self.ndim == 2:
            return self.data
        elif self.ndim == 3:
            if self.metadata.channels > 1:
                return self.data[channel]
            return self.data
        elif self.ndim >= 4:
            return self.data[..., channel, :, :, :]
        return self.data

    def get_slice(self, z: int) -> np.ndarray:
        """Get a specific Z slice."""
        if self.metadata.depth == 1:
            return self.data
        if self.ndim == 3:
            return self.data[z]
        return self.data[..., z, :, :]

    def max_projection(self, axis: int = 0) -> np.ndarray:
        """Maximum intensity projection along axis."""
        if self.metadata.depth == 1:
            return self.data
        return np.max(self.data, axis=axis)


class ImageImporter:
    """
    Import images from various formats.

    Automatically detects format and uses appropriate reader.

    Example
    -------
    >>> importer = ImageImporter()
    >>> stack = importer.load("image.tif")
    >>> print(f"Shape: {stack.shape}")
    """

    def __init__(self):
        self._readers = {
            '.tif': self._load_tiff,
            '.tiff': self._load_tiff,
            '.czi': self._load_czi,
            '.nd2': self._load_nd2,
            '.png': self._load_basic,
            '.jpg': self._load_basic,
            '.jpeg': self._load_basic,
        }

    def load(
        self,
        path: Union[str, Path],
        channels: Optional[List[int]] = None,
        z_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
    ) -> ImageStack:
        """
        Load image from file.

        Parameters
        ----------
        path : str or Path
            Path to image file
        channels : List[int], optional
            Channels to load (all if None)
        z_range : Tuple[int, int], optional
            Z slice range (start, end)
        t_range : Tuple[int, int], optional
            Time point range (start, end)

        Returns
        -------
        ImageStack
            Loaded image stack
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in self._readers:
            return self._readers[suffix](path, channels, z_range, t_range)
        else:
            raise ValueError(f"Unsupported image format: {suffix}")

    def _load_tiff(
        self,
        path: Path,
        channels: Optional[List[int]] = None,
        z_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
    ) -> ImageStack:
        """Load TIFF file."""
        try:
            import tifffile

            with tifffile.TiffFile(path) as tif:
                data = tif.asarray()

                # Try to get metadata
                metadata = ImageMetadata()

                # Shape handling
                if data.ndim == 2:
                    metadata.height, metadata.width = data.shape
                elif data.ndim == 3:
                    # Could be (Z, H, W) or (C, H, W)
                    if data.shape[0] <= 4:
                        metadata.channels = data.shape[0]
                        metadata.height, metadata.width = data.shape[1:]
                    else:
                        metadata.depth = data.shape[0]
                        metadata.height, metadata.width = data.shape[1:]
                elif data.ndim >= 4:
                    # Assume (T, C, Z, H, W) or subset
                    metadata.timepoints = data.shape[0] if data.ndim >= 5 else 1
                    metadata.channels = data.shape[-4] if data.ndim >= 4 else 1
                    metadata.depth = data.shape[-3] if data.ndim >= 3 else 1
                    metadata.height, metadata.width = data.shape[-2:]

                metadata.dtype = data.dtype

                # Try OME metadata
                if tif.ome_metadata:
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(tif.ome_metadata)
                        # Parse pixel sizes etc.
                    except Exception:
                        pass

            return ImageStack(data=data, metadata=metadata)

        except ImportError:
            log.warning("tifffile not available, using basic loader")
            return self._load_basic(path, channels, z_range, t_range)

    def _load_czi(
        self,
        path: Path,
        channels: Optional[List[int]] = None,
        z_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
    ) -> ImageStack:
        """Load Zeiss CZI file."""
        try:
            import aicspylibczi

            czi = aicspylibczi.CziFile(path)
            dims = czi.dims

            # Get dimensions
            metadata = ImageMetadata()
            metadata.width = dims.get('X', 1)
            metadata.height = dims.get('Y', 1)
            metadata.depth = dims.get('Z', 1)
            metadata.channels = dims.get('C', 1)
            metadata.timepoints = dims.get('T', 1)

            # Load data
            data, _ = czi.read_image()

            return ImageStack(data=np.squeeze(data), metadata=metadata)

        except ImportError:
            raise ImportError("aicspylibczi required for CZI files. Install with: pip install aicspylibczi")

    def _load_nd2(
        self,
        path: Path,
        channels: Optional[List[int]] = None,
        z_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
    ) -> ImageStack:
        """Load Nikon ND2 file."""
        try:
            import nd2

            with nd2.ND2File(path) as f:
                data = f.asarray()

                metadata = ImageMetadata()
                metadata.width = f.sizes.get('X', data.shape[-1])
                metadata.height = f.sizes.get('Y', data.shape[-2])
                metadata.depth = f.sizes.get('Z', 1)
                metadata.channels = f.sizes.get('C', 1)
                metadata.timepoints = f.sizes.get('T', 1)

                # Pixel size
                if hasattr(f, 'voxel_size'):
                    metadata.pixel_size_x = f.voxel_size().x
                    metadata.pixel_size_y = f.voxel_size().y
                    metadata.pixel_size_z = f.voxel_size().z

            return ImageStack(data=data, metadata=metadata)

        except ImportError:
            raise ImportError("nd2 required for ND2 files. Install with: pip install nd2")

    def _load_basic(
        self,
        path: Path,
        channels: Optional[List[int]] = None,
        z_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
    ) -> ImageStack:
        """Load basic image formats using PIL/Pillow."""
        try:
            from PIL import Image

            img = Image.open(path)
            data = np.array(img)

            metadata = ImageMetadata()
            if data.ndim == 2:
                metadata.height, metadata.width = data.shape
            elif data.ndim == 3:
                metadata.height, metadata.width = data.shape[:2]
                metadata.channels = data.shape[2]
                # Transpose to (C, H, W)
                data = np.transpose(data, (2, 0, 1))

            metadata.dtype = data.dtype

            return ImageStack(data=data, metadata=metadata)

        except ImportError:
            # Fallback to numpy-only loading
            log.warning("PIL not available, using numpy-only loader")

            # Try to load as raw array
            with open(path, 'rb') as f:
                # This won't work for most formats, but provides a fallback
                raise ImportError("PIL/Pillow required for basic image formats")


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Convenience function to load image as numpy array.

    Parameters
    ----------
    path : str or Path
        Path to image file

    Returns
    -------
    np.ndarray
        Image data
    """
    importer = ImageImporter()
    stack = importer.load(path)
    return stack.data


def load_stack(path: Union[str, Path]) -> ImageStack:
    """
    Convenience function to load image stack.

    Parameters
    ----------
    path : str or Path
        Path to image file

    Returns
    -------
    ImageStack
        Image stack with metadata
    """
    importer = ImageImporter()
    return importer.load(path)
