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
    OTSU = auto()           # Otsu threshold + labelling, via scikit-image
    WATERSHED = auto()      # Marker-controlled watershed, via scikit-image
    CELLPOSE = auto()       # Cellpose deep learning (if available)
    STARDIST = auto()       # StarDist deep learning (if available)
    # What actually runs when scikit-image is absent: a hand-rolled Otsu
    # threshold plus BFS labelling, needing only numpy. It must be
    # nameable, or a result cannot report it truthfully.
    BASIC_THRESHOLD = auto()


class SegmentationUnavailable(RuntimeError):
    """A segmentation method was requested that cannot run here.

    Raised instead of quietly substituting a different algorithm. An Otsu
    threshold is not an approximation of a Cellpose model, and a result
    labelled CELLPOSE that came from thresholding cannot be reviewed,
    reproduced, or compared against a published benchmark.
    """


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
        The method that actually produced these labels -- NOT the one that
        was requested. `fallback_from` carries the request when they differ.
    fallback_from : Optional[SegmentationMethod]
        The requested method, when something else ran. None if the
        requested method ran.
    morphology_measured : bool
        False when eccentricity and solidity are dataclass defaults rather
        than measurements, because scikit-image was unavailable.
    parameters : Dict[str, Any]
        Parameters used
    """
    labels: np.ndarray
    n_cells: int
    cell_properties: List[CellProperties] = field(default_factory=list)
    method: SegmentationMethod = SegmentationMethod.WATERSHED
    fallback_from: Optional[SegmentationMethod] = None
    morphology_measured: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        """True when a method other than the requested one produced these."""
        return self.fallback_from is not None

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

    METHOD_NAMES = {
        'watershed': SegmentationMethod.WATERSHED,
        'otsu': SegmentationMethod.OTSU,
        'cellpose': SegmentationMethod.CELLPOSE,
        'stardist': SegmentationMethod.STARDIST,
        'basic_threshold': SegmentationMethod.BASIC_THRESHOLD,
    }

    #: Methods whose output no other method here approximates. Falling back
    #: from one of these would substitute classical intensity thresholding
    #: for a learned instance segmenter, so it is refused instead. The
    #: classical trio may substitute for each other, because they are the
    #: same kind of algorithm and the result now says which one ran.
    NO_SUBSTITUTE = (SegmentationMethod.CELLPOSE, SegmentationMethod.STARDIST)

    def __init__(
        self,
        method: Union[str, SegmentationMethod] = 'watershed',
        min_size: int = 100,
        max_size: int = 50000,
        allow_fallback: bool = True,
    ):
        if isinstance(method, str):
            try:
                self.method = self.METHOD_NAMES[method.lower()]
            except KeyError:
                # `.get(..., WATERSHED)` turned CellSegmenter('cellpoze')
                # into a silent watershed run: a typo became a different
                # algorithm with no diagnostic anywhere.
                raise ValueError(
                    f"Unknown segmentation method {method!r}. "
                    f"Available: {sorted(self.METHOD_NAMES)}"
                ) from None
        else:
            self.method = method

        self.min_size = min_size
        self.max_size = max_size
        self.allow_fallback = allow_fallback
        #: Set by whichever leaf actually produced the labels; see segment().
        self._ran: Optional[SegmentationMethod] = None
        self._morphology_measured = False

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

        # Each leaf sets self._ran to the algorithm it actually ran, so the
        # result reports that rather than the request. Reset per call, or a
        # reused segmenter carries a stale answer.
        self._ran = None

        if (self.method in self.NO_SUBSTITUTE
                and not self._method_available(self.method)):
            raise SegmentationUnavailable(
                f"{self.method.name} was requested but is not installed, and "
                f"no available method approximates it. Install the imaging "
                f"extras (requirements.imaging.txt), or request "
                f"'otsu'/'watershed' explicitly."
            )
        if not self.allow_fallback and not self._method_available(self.method):
            raise SegmentationUnavailable(
                f"{self.method.name} was requested with allow_fallback=False "
                f"but its dependencies are not installed."
            )

        # Dispatch to method
        if self.method == SegmentationMethod.OTSU:
            labels = self._segment_otsu(img, **kwargs)
        elif self.method == SegmentationMethod.WATERSHED:
            labels = self._segment_watershed(img, markers, **kwargs)
        elif self.method == SegmentationMethod.CELLPOSE:
            labels = self._segment_cellpose(img, **kwargs)
        elif self.method == SegmentationMethod.STARDIST:
            labels = self._segment_stardist(img, **kwargs)
        elif self.method == SegmentationMethod.BASIC_THRESHOLD:
            labels = self._segment_basic_threshold(img)
            self._ran = SegmentationMethod.BASIC_THRESHOLD
        else:
            labels = self._segment_watershed(img, markers, **kwargs)

        ran = self._ran or self.method

        # Filter by size
        labels = self._filter_by_size(labels)

        # Compute properties
        properties = self._compute_properties(labels, img)

        if ran is not self.method:
            log.warning(
                "%s was requested but %s ran; result.method reports %s and "
                "result.fallback_from reports %s",
                self.method.name, ran.name, ran.name, self.method.name,
            )

        return SegmentationResult(
            labels=labels,
            n_cells=len(properties),
            cell_properties=properties,
            method=ran,
            fallback_from=None if ran is self.method else self.method,
            morphology_measured=self._morphology_measured,
            parameters={'min_size': self.min_size, 'max_size': self.max_size},
        )

    @staticmethod
    def _method_available(method: "SegmentationMethod") -> bool:
        """Whether `method` can actually run in this interpreter.

        Checked with find_spec rather than by importing, so refusing a
        request costs nothing and does not depend on reaching the leaf.
        """
        import importlib.util as _u

        def have(mod: str) -> bool:
            try:
                return _u.find_spec(mod) is not None
            except (ImportError, ValueError):
                return False

        if method == SegmentationMethod.CELLPOSE:
            return have("cellpose")
        if method == SegmentationMethod.STARDIST:
            return have("stardist")
        if method == SegmentationMethod.OTSU:
            return have("skimage")
        if method == SegmentationMethod.WATERSHED:
            return have("skimage") and have("scipy")
        return True  # BASIC_THRESHOLD needs only numpy

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

            self._ran = SegmentationMethod.OTSU
            return labels.astype(np.int32)

        except ImportError:
            # Note this path is a *different algorithm* from scikit-image
            # Otsu (manual histogram, no remove_small_objects), so reporting
            # OTSU here was a mislabelling too.
            log.warning("scikit-image not available, using basic thresholding")
            labels = self._segment_basic_threshold(image)
            self._ran = SegmentationMethod.BASIC_THRESHOLD
            return labels

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

            from skimage.filters import threshold_otsu

            # Smooth image
            smoothed = gaussian(image, sigma=2)

            # Foreground by Otsu, not by the 50th percentile. The median splits
            # the histogram in half whatever is in the image, so on a frame that
            # is mostly background it called half the field foreground.
            try:
                binary = smoothed > threshold_otsu(smoothed)
            except ValueError:            # a flat image has no Otsu threshold
                binary = np.zeros(image.shape, dtype=bool)
            binary = ndimage.binary_fill_holes(binary)

            # If no markers provided, generate from local maxima
            if markers is None:
                distance = ndimage.distance_transform_edt(binary)

                # Find local maxima as markers
                coords = peak_local_max(
                    distance,
                    min_distance=10,
                    threshold_abs=0.1,
                    labels=binary.astype(int),
                )
                markers = np.zeros(image.shape, dtype=np.int32)
                markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

            # Flood only the foreground. `mask=image > 0` admitted every pixel
            # the moment an image carried any noise floor, so the watershed
            # partitioned the whole frame and handed background to the nearest
            # object: 5 blobs came back as 15 regions, and on a confocal stack
            # each "nucleus" came back at ~4,300 um^3 spanning the full z-range
            # against a real 200-500 um^3.
            labels = watershed(-smoothed, markers, mask=binary)

            self._ran = SegmentationMethod.WATERSHED
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

            self._ran = SegmentationMethod.CELLPOSE
            return masks.astype(np.int32)

        except ImportError:
            # Unreachable via segment(), which refuses this request up
            # front; kept so a direct call cannot mislabel either.
            raise SegmentationUnavailable(
                "Cellpose was requested but is not installed. Classical "
                "thresholding is not a substitute for a learned instance "
                "segmenter; install the imaging extras, or request "
                "'otsu'/'watershed' explicitly."
            ) from None

    def _segment_stardist(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using StarDist (if available)."""
        try:
            from stardist.models import StarDist2D

            model = StarDist2D.from_pretrained('2D_versatile_fluo')
            labels, _ = model.predict_instances(image)

            self._ran = SegmentationMethod.STARDIST
            return labels.astype(np.int32)

        except ImportError:
            raise SegmentationUnavailable(
                "StarDist was requested but is not installed. Classical "
                "thresholding is not a substitute for a learned instance "
                "segmenter; install the imaging extras, or request "
                "'otsu'/'watershed' explicitly."
            ) from None

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
        # Reset per call so a reused segmenter cannot carry a stale True.
        self._morphology_measured = False

        try:
            from skimage.measure import regionprops

            regions = regionprops(labels, intensity_image=intensity_image)

            # `hasattr` is not a usable guard here. A RegionProperties object
            # HAS `eccentricity` in 3-D -- accessing it is what raises
            # NotImplementedError, so the old hasattr check never fired and a
            # 3-D stack died inside this loop. The segmenter is reachable with
            # 3-D input by design: the importers read z-stacks and
            # MeshGenerator takes (D, H, W).
            unavailable = set()

            def _prop(region, name, default):
                try:
                    return getattr(region, name)
                except (NotImplementedError, AttributeError):
                    unavailable.add(name)
                    return default

            for region in regions:
                # 2-D-only shape descriptors come back NaN rather than 0.0 and
                # 1.0 in 3-D. Those defaults are not neutral: 0.0 eccentricity
                # reads as a perfect circle and 1.0 solidity as a perfectly
                # convex object, so a reader plotting them would see a
                # flawlessly round population that was never measured. NaN
                # cannot be mistaken for a measurement.
                props = CellProperties(
                    label=region.label,
                    centroid=tuple(region.centroid),
                    area=region.area,
                    bbox=region.bbox,
                    equivalent_diameter=_prop(
                        region, 'equivalent_diameter_area',
                        _prop(region, 'equivalent_diameter', float('nan'))),
                    eccentricity=_prop(region, 'eccentricity', float('nan')),
                    solidity=_prop(region, 'solidity', float('nan')),
                    mean_intensity=(region.mean_intensity
                                    if intensity_image is not None else None),
                )
                properties.append(props)

            if unavailable:
                log.warning(
                    "%s not defined for %d-D input; left as NaN rather than a "
                    "default that would read as a measurement",
                    ", ".join(sorted(unavailable)), labels.ndim,
                )
            self._morphology_measured = True

        except ImportError:
            # Manual computation without skimage. eccentricity and solidity
            # are left at their dataclass defaults and are NOT measurements
            # of anything; `morphology_measured` records that. A reader
            # plotting them would otherwise get a column of 0.0 and 1.0 that
            # looks like a perfectly circular, perfectly solid population.
            log.warning(
                "scikit-image absent: eccentricity and solidity are not "
                "computed and remain at their defaults (0.0 and 1.0)"
            )
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
