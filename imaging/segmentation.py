"""
Cell Segmentation
=================

Cell segmentation from microscopy images using various methods.

Supports:
- Watershed segmentation (classic)
- Otsu thresholding
- Marker-controlled watershed
- Integration with external tools (Cellpose, StarDist) when available
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


class SegmentationMethod(Enum):
    """Available segmentation methods."""
    OTSU = auto()           # Simple Otsu thresholding
    WATERSHED = auto()      # Marker-controlled watershed
    CELLPOSE = auto()       # Cellpose deep learning (if available)
    STARDIST = auto()       # StarDist deep learning (if available)


@dataclass
class CellProperties:
    """
    Properties of a segmented cell.

    Attributes
    ----------
    label : int
        Unique cell label
    centroid : Tuple[float, ...]
        Center of mass coordinates (z, y, x) or (y, x)
    area : float
        Cell area (2D) or volume (3D) in pixels/voxels
    bbox : Tuple[int, ...]
        Bounding box: (min_row, min_col, max_row, max_col) for 2D
    equivalent_diameter : float
        Diameter of circle/sphere with same area/volume
    eccentricity : float
        Eccentricity of fitted ellipse (0-1)
    solidity : float
        Ratio of area to convex hull area
    mean_intensity : Optional[float]
        Mean intensity within cell (if intensity image provided)
    """
    label: int
    centroid: Tuple[float, ...]
    area: float
    bbox: Tuple[int, ...]
    equivalent_diameter: float = 0.0
    eccentricity: float = 0.0
    solidity: float = 1.0
    mean_intensity: Optional[float] = None


@dataclass
class SegmentationResult:
    """
    Result of cell segmentation.

    Attributes
    ----------
    labels : np.ndarray
        Label image where each cell has unique integer ID
    n_cells : int
        Number of detected cells
    cell_properties : List[CellProperties]
        Properties for each cell
    method : SegmentationMethod
        Method used for segmentation
    parameters : Dict[str, Any]
        Parameters used
    """
    labels: np.ndarray
    n_cells: int
    cell_properties: List[CellProperties] = field(default_factory=list)
    method: SegmentationMethod = SegmentationMethod.WATERSHED
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_3d(self) -> bool:
        return self.labels.ndim == 3

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.labels.shape


class CellSegmenter:
    """
    Cell segmentation from microscopy images.

    Supports multiple segmentation methods with fallbacks.

    Parameters
    ----------
    method : str or SegmentationMethod
        Segmentation method: 'watershed', 'otsu', 'cellpose', 'stardist'
    min_size : int
        Minimum cell size in pixels (default: 100)
    max_size : int
        Maximum cell size in pixels (default: 50000)

    Example
    -------
    >>> segmenter = CellSegmenter(method='watershed')
    >>> result = segmenter.segment(image, channels=['nuclei'])
    >>> print(f"Found {result.n_cells} cells")
    """

    def __init__(
        self,
        method: Union[str, SegmentationMethod] = 'watershed',
        min_size: int = 100,
        max_size: int = 50000,
    ):
        if isinstance(method, str):
            method_map = {
                'watershed': SegmentationMethod.WATERSHED,
                'otsu': SegmentationMethod.OTSU,
                'cellpose': SegmentationMethod.CELLPOSE,
                'stardist': SegmentationMethod.STARDIST,
            }
            self.method = method_map.get(method.lower(), SegmentationMethod.WATERSHED)
        else:
            self.method = method

        self.min_size = min_size
        self.max_size = max_size

    def segment(
        self,
        image: np.ndarray,
        channels: List[str] = None,
        markers: Optional[np.ndarray] = None,
        **kwargs,
    ) -> SegmentationResult:
        """
        Segment cells in image.

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W) or (D, H, W) or (C, H, W)
        channels : List[str], optional
            Channel names if multi-channel: ['nuclei', 'membrane', 'cytoplasm']
        markers : np.ndarray, optional
            Marker image for seeded segmentation
        **kwargs
            Additional method-specific parameters

        Returns
        -------
        SegmentationResult
            Segmentation result with labels and properties
        """
        if channels is None:
            channels = ['intensity']

        # Prepare image
        if image.ndim == 2:
            img = image
        elif image.ndim == 3:
            # Could be (D, H, W) for 3D or (C, H, W) for multi-channel
            # Assume multi-channel if first dim is small
            if image.shape[0] <= 4:
                img = image[0]  # Use first channel
            else:
                img = image
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

        # Dispatch to method
        if self.method == SegmentationMethod.OTSU:
            labels = self._segment_otsu(img, **kwargs)
        elif self.method == SegmentationMethod.WATERSHED:
            labels = self._segment_watershed(img, markers, **kwargs)
        elif self.method == SegmentationMethod.CELLPOSE:
            labels = self._segment_cellpose(img, **kwargs)
        elif self.method == SegmentationMethod.STARDIST:
            labels = self._segment_stardist(img, **kwargs)
        else:
            labels = self._segment_watershed(img, markers, **kwargs)

        # Filter by size
        labels = self._filter_by_size(labels)

        # Compute properties
        properties = self._compute_properties(labels, img)

        return SegmentationResult(
            labels=labels,
            n_cells=len(properties),
            cell_properties=properties,
            method=self.method,
            parameters={'min_size': self.min_size, 'max_size': self.max_size},
        )

    def _segment_otsu(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simple Otsu thresholding segmentation."""
        try:
            from skimage.filters import threshold_otsu
            from skimage.measure import label
            from skimage.morphology import remove_small_objects

            # Threshold
            thresh = threshold_otsu(image)
            binary = image > thresh

            # Clean up
            binary = remove_small_objects(binary, min_size=self.min_size)

            # Label connected components
            labels = label(binary)

            return labels.astype(np.int32)

        except ImportError:
            log.warning("scikit-image not available, using basic thresholding")
            return self._segment_basic_threshold(image)

    def _segment_basic_threshold(self, image: np.ndarray) -> np.ndarray:
        """Basic thresholding without scipy/skimage."""
        # Otsu's method manually
        hist, bins = np.histogram(image.ravel(), bins=256)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Between-class variance
        w0 = np.cumsum(hist)
        w1 = np.cumsum(hist[::-1])[::-1]
        mu0 = np.cumsum(hist * bin_centers) / w0
        mu1 = (np.cumsum((hist * bin_centers)[::-1]) / w1[::-1])[::-1]

        # Avoid division by zero
        w0 = np.clip(w0, 1, None)
        w1 = np.clip(w1, 1, None)

        variance = w0[:-1] * w1[1:] * (mu0[:-1] - mu1[1:]) ** 2
        threshold = bin_centers[np.argmax(variance)]

        binary = image > threshold

        # Simple connected component labeling
        labels = self._connected_components_2d(binary)

        return labels

    def _connected_components_2d(self, binary: np.ndarray) -> np.ndarray:
        """Simple connected components labeling."""
        labels = np.zeros_like(binary, dtype=np.int32)
        current_label = 0
        visited = np.zeros_like(binary, dtype=bool)

        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j] and not visited[i, j]:
                    current_label += 1
                    # BFS
                    stack = [(i, j)]
                    while stack:
                        ci, cj = stack.pop()
                        if (0 <= ci < binary.shape[0] and
                            0 <= cj < binary.shape[1] and
                            binary[ci, cj] and not visited[ci, cj]):
                            visited[ci, cj] = True
                            labels[ci, cj] = current_label
                            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                stack.append((ci + di, cj + dj))

        return labels

    def _segment_watershed(
        self,
        image: np.ndarray,
        markers: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Watershed segmentation."""
        try:
            from scipy import ndimage
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_max
            from skimage.filters import gaussian

            # Smooth image
            smoothed = gaussian(image, sigma=2)

            # If no markers provided, generate from local maxima
            if markers is None:
                # Distance transform for seeds
                binary = image > np.percentile(image, 50)
                distance = ndimage.distance_transform_edt(binary)

                # Find local maxima as markers
                coords = peak_local_max(
                    distance,
                    min_distance=10,
                    threshold_abs=0.1,
                    labels=binary.astype(int),
                )
                markers = np.zeros_like(binary, dtype=np.int32)
                markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

            # Watershed
            labels = watershed(-smoothed, markers, mask=image > 0)

            return labels.astype(np.int32)

        except ImportError:
            log.warning("scipy/skimage not available, falling back to Otsu")
            return self._segment_otsu(image, **kwargs)

    def _segment_cellpose(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using Cellpose (if available)."""
        try:
            from cellpose import models

            model = models.Cellpose(model_type='cyto')
            masks, _, _, _ = model.eval(image, diameter=30, channels=[0, 0])

            return masks.astype(np.int32)

        except ImportError:
            log.warning("Cellpose not available, falling back to watershed")
            return self._segment_watershed(image, **kwargs)

    def _segment_stardist(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using StarDist (if available)."""
        try:
            from stardist.models import StarDist2D

            model = StarDist2D.from_pretrained('2D_versatile_fluo')
            labels, _ = model.predict_instances(image)

            return labels.astype(np.int32)

        except ImportError:
            log.warning("StarDist not available, falling back to watershed")
            return self._segment_watershed(image, **kwargs)

    def _filter_by_size(self, labels: np.ndarray) -> np.ndarray:
        """Remove objects outside size range."""
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        for label in unique_labels:
            mask = labels == label
            size = np.sum(mask)

            if size < self.min_size or size > self.max_size:
                labels[mask] = 0

        # Relabel to ensure consecutive labels
        return self._relabel_sequential(labels)

    def _relabel_sequential(self, labels: np.ndarray) -> np.ndarray:
        """Relabel to have consecutive integer labels."""
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        new_labels = np.zeros_like(labels)
        for new_label, old_label in enumerate(unique_labels, start=1):
            new_labels[labels == old_label] = new_label

        return new_labels

    def _compute_properties(
        self,
        labels: np.ndarray,
        intensity_image: Optional[np.ndarray] = None,
    ) -> List[CellProperties]:
        """Compute properties for each labeled region."""
        properties = []

        try:
            from skimage.measure import regionprops

            regions = regionprops(labels, intensity_image=intensity_image)

            for region in regions:
                props = CellProperties(
                    label=region.label,
                    centroid=tuple(region.centroid),
                    area=region.area,
                    bbox=region.bbox,
                    equivalent_diameter=region.equivalent_diameter,
                    eccentricity=region.eccentricity if hasattr(region, 'eccentricity') else 0.0,
                    solidity=region.solidity if hasattr(region, 'solidity') else 1.0,
                    mean_intensity=region.mean_intensity if intensity_image is not None else None,
                )
                properties.append(props)

        except ImportError:
            # Manual computation without skimage
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]

            for label in unique_labels:
                mask = labels == label
                coords = np.where(mask)

                # Centroid
                centroid = tuple(np.mean(c) for c in coords)

                # Area
                area = np.sum(mask)

                # Bounding box
                bbox = tuple(
                    [coords[i].min() for i in range(len(coords))] +
                    [coords[i].max() + 1 for i in range(len(coords))]
                )

                # Equivalent diameter
                if labels.ndim == 2:
                    eq_diam = np.sqrt(4 * area / np.pi)
                else:
                    eq_diam = (6 * area / np.pi) ** (1/3)

                props = CellProperties(
                    label=int(label),
                    centroid=centroid,
                    area=float(area),
                    bbox=bbox,
                    equivalent_diameter=float(eq_diam),
                )
                properties.append(props)

        return properties


def segment_cells(
    image: np.ndarray,
    method: str = 'watershed',
    **kwargs,
) -> SegmentationResult:
    """
    Convenience function to segment cells.

    Parameters
    ----------
    image : np.ndarray
        Input image
    method : str
        Segmentation method
    **kwargs
        Additional parameters

    Returns
    -------
    SegmentationResult
        Segmentation result
    """
    segmenter = CellSegmenter(method=method, **kwargs)
    return segmenter.segment(image)
