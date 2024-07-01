import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.algorithms.band_math.indices import compute_NDVI, compute_EVI, compute_NDWI
from gge.util import timing_decorator, exception_handler
from typing import Tuple, Union
from datetime import datetime
from gge.util.types import PixelType


class Sentinel2(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        cloud_threshold=10,
    ):
        super().__init__(area, time_range)
        self.cloud_threshold = cloud_threshold
        self.pixel_types = PixelType.DN

    @timing_decorator
    @exception_handler(default_return_value={})
    def download_data(self):
        collections = ["COPERNICUS/S2_SR_HARMONIZED"]

        for collection_id in collections:
            collection = (
                ee.ImageCollection(collection_id)
                .filterBounds(self.area)
                .filterDate(self.time_range[0], self.time_range[1])
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_threshold))
                .sort("system:time_start")
            )

            count = collection.size().getInfo()
            if count == 0:
                print(f"No images found in collection {collection_id} for the given filters.")
                continue

            image_list = collection.toList(count)
            for i in range(count):
                image = ee.Image(image_list.get(i))
                try:
                    self.images_data.append(self.convert_data(image))
                except Exception as e:
                    self.logger.error(f"Error converting image {image.id().getInfo()}: {e}")

    def convert_data(self, image):
        band_names = image.bandNames().getInfo()

        band_data = {}
        for band in band_names:
            sample = image.select(band).sampleRectangle(region=self.area, defaultValue=0)
            band_data[band] = np.array(sample.get(band).getInfo())
        return {"image_bands": band_data, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def dn_to_reflectance(self):
        if self.pixel_types == PixelType.DN:
            for img in self.images_data:
                img["image_bands"] = self.convert_dn_to_reflectance(img["image_bands"], img["metadata"])
            self.pixel_types = PixelType.Reflectance
        else:
            self.logger.info("Data is already in Reflectance.")

    def reflectance_to_dn(self):
        if self.pixel_types == PixelType.Reflectance:
            for img in self.images_data:
                img["image_bands"] = self.convert_reflectance_to_dn(img["image_bands"], img["metadata"])
            self.pixel_types = PixelType.DN
        else:
            self.logger.info("Data is already in DN.")

    def convert_dn_to_reflectance(self, band_data, metadata):
        """ 
        see S2_MSI_Product_Specification page 403..
        """
        reflectance_data = {}
        for band in band_data.keys():
            if band in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]:
                try:
                    #  band_suffix = "".join(filter(str.isdigit, band))
                    reflectance_data[band] = band_data[band] / 10000
                except:
                    self.logger.warning(f"Reflectance scaling factors not found for {band}. Available keys: {list(metadata.keys())}")
                    reflectance_data[band] = band_data[band]
            else:
                reflectance_data[band] = band_data[band]
        return reflectance_data

    def convert_reflectance_to_dn(self, band_data, metadata):
        dn_data = {}
        for band in band_data.keys():
            if band in ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "ST_B6", "SR_B7"]:
                try:
                    #  band_suffix = band[-1]
                    dn_data[band] = (band_data[band] * 10000).astype(np.int32)
                except:
                    self.logger.warning(f"DN conversion factors not found for {band}. Available keys: {list(metadata.keys())}")
                    dn_data[band] = band_data[band]
            else:
                dn_data[band] = band_data[band]
        return dn_data

    @property
    def item_type(self):
        return self._item_type

    @item_type.setter
    def item_type(self, value):
        if value in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12", "QA10", "QA20", "QA60", "NDVI", "EVI"]:
            self._item_type = value
        else:
            raise ValueError("Invalid item type.")

    def __getitem__(self, item):

        img = self.images_data[item]
        metadata = img.get("metadata")
        bands = img.get("image_bands")

        if self._item_type in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12", "QA10", "QA20", "QA60"]:
            try:
                array = bands[self._item_type]
            except KeyError:
                self.logger.warning(f"Band {self._item_type} not found in the image.")
                raise ValueError(f"Band {self._item_type} not found in the image.")
        elif self._item_type == "NDVI":
            array = compute_NDVI(bands["B4"], bands["B8"])
        elif self._item_type == "EVI":
            array = compute_EVI(bands["B4"], bands["B8"], bands["B2"])
        elif self._item_type == "NDWI":
            array = compute_NDWI(bands["B3"], bands["B8"])
        elif self._item_type.upper() == "RGB":
            array = self.convert_to_plotable_rgb({band: bands[band] for band in ["B4", "B3", "B2"]})
        else:
            raise ValueError(f"Band {self._item_type} not found in the image.")

        return array, metadata

    def display_rgb(self, index, bands=["B4", "B3", "B2"], scale=255):
        data = self.images_data[index]
        if data is not None:
            rgb_image = self.convert_to_plotable_rgb({band: data["image_bands"][band] for band in bands}, scale)
            plt.imshow(rgb_image)
            plt.show()

    @staticmethod
    def convert_to_plotable_rgb(array_dict, scale=255):
        bands = np.stack([array_dict[band] for band in sorted(array_dict.keys())], axis=-1)
        min_val, max_val = bands.min(), bands.max()
        return ((bands - min_val) / (max_val - min_val) * scale).astype(np.uint8)

    def __len__(self):
        return len(self.images_data)

    def __class__(self):
        return "Sentinel2"
