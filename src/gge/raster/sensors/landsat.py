import logging
import sys
from typing import Tuple, List, Union, Optional
import numpy as np
import rioxarray
from gge.raster.utils import init_transformer
from gge.raster.geometry import transform_to_lonlat, transform_to_indices
from gge.raster.geometry import GeoCoordinate, Coordinate
from shapely.geometry import Polygon
from gge.algorithms.band_math.indices import compute_NDVI, compute_EVI, compute_NDWI


class Landsat:
    def __init__(
        self,
        validate: bool = False,
        verbose: int = 0,
    ):
        """ """
        self.verbose = verbose

        self.dataset = None
        self._is_valid = False

    def load_data(self, data_path: Optional[str] = None):

        self.dataset = rioxarray.open_rasterio(data_path)
        self.transformer, self.crs = init_transformer(data_path)

        self.band_names = {"band_name": self.dataset.long_name, "band_index": self.dataset.band.data.tolist()}

        return None

    def yx(self, rows: Union[int, List[int]], cols: Union[int, List[int]]) -> Union[GeoCoordinate, List[GeoCoordinate]]:
        """Get geographic coordinates (lon, lat) from pixel indices.
        Args:
            row (int): Pixel row index. (y)
            col (int): Pixel column index. (x)
        Returns:
            GeoCoordinate: Latitude and longitude.
        """
        if isinstance(rows, list) and isinstance(cols, list):
            if len(rows) != len(cols):
                raise ValueError("Lists of rows and cols must be of equal length.")
            return [transform_to_lonlat(self.transformer, self.crs, row, col) for row, col in zip(rows, cols)]
        elif isinstance(rows, int) and isinstance(cols, int):
            return transform_to_lonlat(self.transformer, self.crs, rows, cols)
        else:
            raise ValueError("Rows and cols must be both single values or lists of equal length.")

    def lonlat(self, lons: Union[float, List[float]], lats: Union[float, List[float]]) -> Union[Coordinate, List[Coordinate]]:
        """Get pixel indices from geographic coordinates ( lon, lat).

        Args:
            lon (float): Longitude.
            lat (float): Latitude.

        Returns:
            Coordinate: Pixel row and column indices.
        """
        if isinstance(lats, list) and isinstance(lons, list):
            if len(lats) != len(lons):
                raise ValueError("Lists of latitudes and longitudes must be of equal length.")
            return [transform_to_indices(self.transformer, self.crs, lat, lon) for lat, lon in zip(lats, lons)]
        elif isinstance(lats, float) and isinstance(lons, float):
            return transform_to_indices(self.transformer, self.crs, lats, lons)
        else:
            raise ValueError("Latitudes and longitudes must be both single values or lists of equal length.")

    def _raster_extent_to_polygon(self) -> Polygon:
        """
        Converts the extent of a rasterio dataset with GCPs to a geographic polygon.

        Parameters:
        - src: rasterio dataset

        Returns:
        - A shapely Polygon object representing the geographic extent of the raster.
        """
        c, rows, cols = self.shape()
        top_left = self.yx(0, 0)
        top_right = self.yx(0, cols)
        bottom_right = self.yx(rows, cols)
        bottom_left = self.yx(rows, 0)
        geo_coords = [top_left, top_right, bottom_right, bottom_left, top_left]
        polygon = Polygon(geo_coords)
        return polygon

    def _validate(self) -> bool:
        if self.data_path is None:
            logging.error("No data path provided.")
            return False

    @property
    def data_paths(self):
        return self.data_paths

    @data_paths.setter
    def data_paths(self, value: Union[str, list, List] = None) -> None:
        """ """
        self._data_paths = value
        return None

    @property
    def shape(self) -> Union[tuple, Tuple[int, int, int]]:
        return (
            self.dataset.sizes["band"],
            self.dataset.sizes["y"],
            self.dataset.sizes["x"],
        )

    @property
    def size(self) -> Union[tuple, Tuple[int, int, int]]:
        return (
            self.dataset.sizes["band"],
            self.dataset.sizes["y"],
            self.dataset.sizes["x"],
        )

    def data(
        self,
        band: Union[int, None] = None,
        y: Union[tuple, Tuple[int, int], None] = None,
        x: Union[tuple, Tuple[int, int], None] = None,
    ) -> Union[None, np.ndarray]:

        # we want to slice like this self.dataset.sel(band = 1, y = slice(0,100), x=slice(0, 100))
        # depending on the intiuts.. However, it is okay to only slice on the band, or x,..

        # band
        if band is not None and y is None and x is None:
            return self.dataset.sel(band=band).to_array()[0]
        # band and y
        elif band is not None and y is not None and x is not None:
            return self.dataset.sel(band=band, y=slice(y[0], y[1]), x=slice(x[0], x[1])).to_array()[0]
        # band and x and y
        elif band is not None and y is not None and x is None:
            return self.dataset.sel(band=band, y=slice(y[0], y[1])).to_array()[0]
        # band and x
        elif band is not None and y is None and x is not None:
            return self.dataset.sel(band=band, x=slice(x[0], x[1])).to_array()[0]
        # y and x
        elif band is None and y is not None and x is not None:
            return self.dataset.sel(y=slice(y[0], y[1]), x=slice(x[0], x[1])).to_array()[0]
        # y
        elif band is None and y is not None and x is None:
            return self.dataset.sel(y=slice(y[0], y[1])).to_array()[0]
        # x
        elif band is None and y is None and x is not None:
            return self.dataset.sel(x=slice(x[0], x[1])).to_array()[0]
        else:
            return self.dataset.to_array()[0]

    def close_dataset(self) -> None:
        """
        Closes the dataset to free resources.
        """
        if self.dataset:
            self.dataset.close()

    @property
    def item_type(self):
        return self._item_type

    @item_type.setter
    def item_type(self, value):
        valid_bands = [
            "SR_B1",
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B6",
            "SR_B7",
            "SR_QA_AEROSOL" "ST_B10",
            "ST_ATRAN",
            "NDVI",
            "EVI",
            "NDWI",
            "RGB",
        ]
        if value.upper() in valid_bands:
            self._item_type = value
        else:
            raise ValueError("Invalid item type.")

        def __getitem__(self, item):
            img = self.images_data[item]
            metadata = img.get("metadata")
            bands = img.get("image_bands")

            landsat_number = int(metadata["LANDSAT_PRODUCT_ID"][3])

            if self._item_type in ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "SR_B8"]:
                array = bands[self._item_type]
            elif self._item_type.upper() == "NDVI":
                array = compute_NDVI(bands["SR_B4"], bands["SR_B5"])
            elif self._item_type.upper() == "EVI":
                array = compute_EVI(bands["SR_B4"], bands["SR_B5"], bands["SR_B2"])
            elif self._item_type.upper() == "NDWI":
                if landsat_number == 4 or landsat_number == 5:
                    array = compute_NDWI(bands["SR_B3"], bands["SR_B5"])
                elif landsat_number == 7:
                    array = compute_NDWI(bands["SR_B3"], bands["SR_B5"])
                elif landsat_number == 8:
                    array = compute_NDWI(bands["SR_B3"], bands["SR_B5"])
                else:
                    try:
                        array = compute_NDWI(bands["SR_B3"], bands["SR_B4"])
                    except KeyError:
                        self.logger.error(f"Band {self._item_type} not found in the image.")
                        array = None
            elif self._item_type.upper() == "RGB":
                array = self.convert_to_plotable_rgb(
                    {
                        "SR_B4": bands["SR_B4"],
                        "SR_B3": bands["SR_B3"],
                        "SR_B2": bands["SR_B2"],
                    }
                )
            else:
                raise ValueError(f"Band {self._item_type} not found in the image.")

            return array, metadata

    def __repr__(self) -> str:
        return f"""S1(data_path="{self.data_path}", validate= {self.validate}, verbose=0, save_path="{self.save_path}")"""

    def __str__(self) -> str:
        return f"Sentinel-1 Data Handler for {self.data_path}"

    def __sizeof__(self) -> tuple:
        """
        Estimates the size of the S1 instance in bytes, distinguishing between NumPy arrays and other attributes.

        Returns:
            tuple: A tuple where the first element is the total size excluding NumPy arrays, and the second element
                   is the total size of all NumPy arrays.
        """
        base_size = super().__sizeof__()
        non_numpy_size = base_size  # Start with the base object size
        numpy_size = 0  # Initialize numpy arrays size

        attributes = [
            self.verbose,
            self.dataset,
        ]

        for attr in attributes:
            if isinstance(attr, np.ndarray):
                numpy_size += attr.nbytes
            elif hasattr(attr, "__sizeof__"):
                non_numpy_size += attr.__sizeof__()
            else:
                non_numpy_size += sys.getsizeof(attr)

        if self.dataset and not isinstance(self.dataset, np.ndarray):
            non_numpy_size += sys.getsizeof(self.dataset)

        return (non_numpy_size, numpy_size)

    def __bool__(self) -> bool:
        return self._is_valid

    def __len__(self):
        return self.dataset.rio.count if self.dataset else 0

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.dataset.close()
