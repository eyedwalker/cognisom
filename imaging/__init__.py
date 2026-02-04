"""
Imaging Package
===============

Image-to-geometry pipeline for simulation mesh generation.

This package provides:
- Cell segmentation from microscopy images
- Mesh generation from segmentation labels
- Import of various microscopy formats
- GPU-accelerated image preprocessing

VCell Parity Phase 5 - Image-to-geometry pipeline.

Usage::

    from cognisom.imaging import CellSegmenter, MeshGenerator

    # Segment cells from image
    segmenter = CellSegmenter(method='watershed')
    result = segmenter.segment(image, channels=['nuclei', 'membrane'])

    # Generate simulation mesh
    generator = MeshGenerator()
    mesh = generator.labels_to_mesh(result.labels, resolution=0.5)

Supported formats:
- TIFF (standard and OME-TIFF)
- CZI (Zeiss)
- ND2 (Nikon)
- PNG, JPEG (for 2D images)
"""

from .segmentation import (
    CellSegmenter,
    SegmentationResult,
    CellProperties,
    SegmentationMethod,
)
from .mesh_generation import (
    MeshGenerator,
    SimulationMesh,
    Compartment,
    Mesh,
)
from .importers import (
    ImageImporter,
    ImageStack,
    load_image,
    load_stack,
)
from .gpu_preprocessing import (
    GPUImageProcessor,
    gaussian_blur_3d,
    threshold_otsu,
    morphological_ops,
)

__all__ = [
    # Segmentation
    "CellSegmenter",
    "SegmentationResult",
    "CellProperties",
    "SegmentationMethod",
    # Mesh generation
    "MeshGenerator",
    "SimulationMesh",
    "Compartment",
    "Mesh",
    # Importers
    "ImageImporter",
    "ImageStack",
    "load_image",
    "load_stack",
    # GPU preprocessing
    "GPUImageProcessor",
    "gaussian_blur_3d",
    "threshold_otsu",
    "morphological_ops",
]
