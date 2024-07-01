import ee
import geopandas as gpd
import json
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Tuple, Union
from gge.util.logger import setup_logging
from gge.util import exception_handler
import sys
import numpy as np
import gc


class SatelliteData(ABC):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
    ):
        self.logger = setup_logging()

        if not ee.data._initialized:
            self.logger.info("Initializing Earth Engine...")
            try:
                ee.Initialize()
            except Exception as e:
                self.logger.error(f"Error initializing Earth Engine: {e}")
                raise e
            self.logger.info("Earth Engine initialized.")
        self.area = area
        self.time_range = time_range
        self.filters = []
        self.images_data = []

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value: Union[Tuple[float, float, float, float], str]):
        """
        area can be tuple (lat_min lon_min lat_max lon_max) or a path to a geojson or shapefile.

        """
        if isinstance(value, tuple) and len(value) == 4:
            self._area = ee.Geometry.Rectangle(value)
        elif isinstance(value, str):
            self._area = self.load_geojson_or_shapefile(value)
        elif value is None:
            self._area = None
        else:
            raise ValueError("Invalid area input.")

    @exception_handler(default_return_value={})
    def load_geojson_or_shapefile(self, filepath):

        if filepath.endswith(".geojson"):
            with open(filepath, "r") as file:
                geojson = json.load(file)

            area = ee.Geometry(geojson["features"][0]["geometry"])
            del geojson
            return area

        elif filepath.endswith(".shp"):
            gdf = gpd.read_file(filepath)
            try:
                gdf.set_crs("EPSG:3413", inplace=True)
                gdf = gdf.to_crs(epsg=4326)
            except Exception as e:
                self.logger.info(f"CRS is {gdf.crs} (with {e})")

            geojson_str = gdf.to_json()
            geojson = json.loads(geojson_str)
            area = ee.Geometry(geojson["features"][0]["geometry"])
            del gdf, geojson_str, geojson
            return area

        else:
            raise ValueError("File format not supported.")

    @property
    def time_range(self):
        return self._time_range

    @exception_handler(default_return_value={})
    @time_range.setter
    def time_range(self, value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            self._time_range = (ee.Date(value[0]), ee.Date(value[1]))
        elif isinstance(value, str):
            start, end = value.split("/")
            self._time_range = (ee.Date(start), ee.Date(end))
        elif value is None:
            self._time_range = None
        else:
            raise ValueError("Invalid time range input.")

    @exception_handler(default_return_value={})
    def add_filter(self, filter_func):
        self.filters.append(filter_func)

    @exception_handler(default_return_value={})
    def apply_filters(self, collection):
        for f in self.filters:
            collection = f(collection)
        return collection

    @exception_handler(default_return_value={})
    def convert_to_plotable_rgb(self, array_dict, scale=255, gamma=1.0, gain=1.0, red=1.0, green=1.0, blue=1.0):
        if gamma > 10:
            self.logger.warning("Gamma value is very high. It may cause overflow errors.")
        if gain > 10:
            self.logger.warning("Gain value is very high. It may cause overflow errors.")
        if red > 10:
            self.logger.warning("Red value is very high. It may cause overflow errors.")
        if green > 10:
            self.logger.warning("Green value is very high. It may cause overflow errors.")
        if blue > 10:
            self.logger.warning("Blue value is very high. It may cause overflow errors.")

        if gamma < 0.0001:
            self.logger.warning("Gamma value is very low. It may cause underflow errors.")
        if gain < 0.0001:
            self.logger.warning("Gain value is very low. It may cause underflow errors.")
        if red < 0.0001:
            self.logger.warning("Red value is very low. It may cause underflow errors.")
        if green < 0.0001:
            self.logger.warning("Green value is very low. It may cause underflow errors.")
        if blue < 0.0001:
            self.logger.warning("Blue value is very low. It may cause underflow errors.")

        bands = np.stack([array_dict[band] for band in sorted(array_dict.keys())], axis=-1)
        if bands.shape[-1] == 1:
            bands = np.dstack((bands, bands, bands))
        # Normalize bands
        min_val, max_val = bands.min(), bands.max()
        norm_bands = (bands - min_val) / (max_val - min_val)
        # Apply gamma correction
        gamma_corrected = np.power(norm_bands, gamma)
        # Apply gain
        gain_applied = gamma_corrected * gain
        # Apply individual color adjustments
        gain_applied[..., 0] *= red
        gain_applied[..., 1] *= green
        if bands.shape[-1] == 2:
            new_channel = (gain_applied[..., 0] + gain_applied[..., 1]) / (gain_applied[..., 0] - gain_applied[..., 1])
            gain_applied = np.dstack((gain_applied, new_channel))
            del new_channel
        gain_applied[..., 2] *= blue

        # Scale bands to the desired range
        scaled_bands = gain_applied * scale
        del bands, norm_bands, gamma_corrected, gain_applied
        return np.clip(scaled_bands, 0, scale).astype(np.uint8)

    @exception_handler(default_return_value={})
    def validate_geometry(self, geo):
        try:
            test = ee.FeatureCollection([ee.Feature(geo)]).size().getInfo()
            del test
            return True
        except ee.EEException as e:
            self.logger.error(f"Invalid geometry: {e}")
            return False

    @exception_handler(default_return_value=None)
    def kill(self):
        """Explicitly unloads the model from memory."""
        if self._model is not None:
            del self._model

        attrs_to_delete = [
            "_area",
            "_time_range",
        ]
        for attr in attrs_to_delete:
            if hasattr(self, attr):
                delattr(self, attr)
        self._model = None
        gc.collect()  # Suggest garbage collection to free up unused memory

    @abstractmethod
    def download_data(self):
        pass

    @abstractmethod
    def convert_data(self, image):
        pass

    @abstractmethod
    def display_rgb(self, data):
        pass

    def __enter__(self):
        # TODO: Do some logging and stuff and timings and stuff...
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._class_names
        gc.collect()
        if exc_type:
            self.logger.error(f"Exception occurred: {exc_val}", exc_info=True)
        return False

    def __repr__(self):
        return f"<{self.__class__.__name__} covering area {self.area} from {self.time_range[0]} to {self.time_range[1]}>"

    def __str__(self):
        return "SatelliteData"

    def __eq__(self, other):
        if not isinstance(other, SatelliteData):
            return NotImplemented

    def __sizeof__(self):
        "size of numpy and the object itself. :"
        total_size = super().__sizeof__()
        for attr in self.__dict__.values():
            if isinstance(attr, np.ndarray):
                total_size += attr.nbytes
            elif isinstance(attr, list):
                total_size += sum(sys.getsizeof(item) for item in attr)
        return (
            total_size,
            object.__sizeof__(self) + sum(sys.getsizeof(v) for v in self.__dict__.values()),
        )

    def __class__(self):
        return "SatelliteData"
