import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.algorithms.band_math.indices import compute_NDVI, compute_EVI
import logging


class Sentinel2(SatelliteData):
    def __init__(self, area, time_range, cloud_threshold=10):
        super().__init__(area, time_range)
        self.cloud_threshold = cloud_threshold
        self.loggger = logging.getLogger()

    def download_data(self):
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(self.area)
            .filterDate(self.time_range[0], self.time_range[1])
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_threshold))
            .sort("system:time_start")
        )

        image_list = collection.toList(collection.size())
        for i in range(image_list.size().getInfo()):
            image = ee.Image(image_list.get(i))
            self.images_data.append(self.convert_data(image))

    def convert_data(self, image):
        band_names = image.bandNames().getInfo()

        band_data = {}
        for band in band_names:
            sample = image.select(band).sampleRectangle(region=self.area, defaultValue=0)
            band_data[band] = np.array(sample.get(band).getInfo())
        return {"image_bands": band_data, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

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

        if self._item_type == "NDVI":
            array = compute_NDVI(bands["B4"], bands["B8"])
        if self._item_type == "EVI":
            array = compute_EVI(bands["B4"], bands["B8"], bands["B2"])

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

    def __class__(self):
        return "Sentinel2"
