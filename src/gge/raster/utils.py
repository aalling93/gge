import rasterio
from typing import Tuple
import numpy as np



def init_transformer(
    filepath: str,
) -> Tuple[rasterio.transform.Affine, rasterio.crs.CRS]:
    """Initialize the spatial transformer and CRS from a raster file.

    Args:
        filepath (str): Path to the raster file.

    Returns:
        Tuple[rasterio.transform.Affine, rasterio.crs.CRS]: The transformer and CRS of the raster file.
    """
    with rasterio.open(filepath) as src:
        if src.gcps[0]:
            gcps, crs = src.gcps
            transformer = rasterio.transform.GCPTransformer(gcps)
        else:
            transformer = src.transform
            crs = src.crs
        return transformer, crs


def load_bands(path_to_band: str) -> np.ndarray:
    with rasterio.open(path_to_band) as src1:
        data = src1.read(1)
    return data
