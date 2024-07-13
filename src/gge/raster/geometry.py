from rasterio.warp import transform
from typing import Tuple, List, Union
import rasterio

Coordinate = Union[Tuple[int, int], List[Tuple[int, int]]]
GeoCoordinate = Union[Tuple[float, float], List[Tuple[float, float]]]


def transform_to_lonlat(transformer, crs, x: int, y: int) -> GeoCoordinate:
    """Transform pixel coordinates to geographic coordinates.

    Args:
        transformer (rasterio.transform.Affine or rasterio.transform.GCPTransformer): Spatial transformer.
        crs (rasterio.crs.CRS): Coordinate Reference System of the raster.
        x (int): X coordinate (pixel column).
        y (int): Y coordinate (pixel row).

    Returns:
        GeoCoordinate: Latitude and longitude.
    """
    if isinstance(transformer, rasterio.transform.GCPTransformer):
        lon, lat = transformer.xy(
            int(x), int(y), offset="center"
        )  # Note the order switch and center offset
    else:
        x, y = transformer * (int(x), int(y))

    if not crs.is_geographic:
        lon, lat = transform(crs, "EPSG:4326", [x], [y])
        return lat[0], lon[0]
    return lon, lat


def transform_to_indices(transformer, crs, lon: float, lat: float) -> Coordinate:
    """Transform geographic coordinates to pixel coordinates.

    Args:
        transformer (rasterio.transform.Affine or rasterio.transform.GCPTransformer): Spatial transformer.
        crs (rasterio.crs.CRS): Coordinate Reference System of the raster.
        lon (float): Longitude.
        lat (float): Latitude.
        

    Returns:
        Coordinate: Pixel row and column.
    """
    if not crs.is_geographic:
        lon, lat = transform("EPSG:4326", crs, [lon], [lat])

    if isinstance(transformer, rasterio.transform.GCPTransformer):
        y, x = transformer.rowcol(lon, lat)
    else:
        raise NotImplementedError(
            "Inverse transformation from lat/lon to pixel indices is not supported for GCP-based datasets."
        )

    return int(y), int(x)
