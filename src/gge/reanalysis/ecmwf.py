import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator
from typing import Tuple, Union
from datetime import datetime


class GlobalClimateData(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        variables: list = ["sea_surface_temperature", "2m_temperature"],
        cloud_threshold=0,  # Placeholder as cloud cover is not typically a factor in ERA5
    ):
        super().__init__(area, time_range)
        self.variables = variables
        self.cloud_threshold = cloud_threshold  # Not used for ERA5, but kept for consistency with Landsat

    @timing_decorator
    def download_data(self):
        collection_id = "ECMWF/ERA5/MONTHLY"
        collection = (
            ee.ImageCollection(collection_id).filterBounds(self.area).filterDate(self.time_range[0], self.time_range[1]).select(self.variables)
        )

        count = collection.size().getInfo()
        if count == 0:
            self.logger.info(f"No images found in collection {collection_id} for the given filters.")
            return

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

    def display_variable(self, index, variable, cmap="coolwarm"):
        data = self.images_data[index]
        if data is not None:
            variable_data = data["image_bands"][variable]
            plt.figure(figsize=(8, 8))
            plt.imshow(variable_data, cmap=cmap)
            plt.colorbar()
            plt.title(f'{variable.capitalize()} at {data["time"]}')
            plt.axis("off")  # Hide axis
            plt.show()

    def __class__(self):
        return "GlobalClimateData"

    def __str__(self):
        return "Global Climate Data"

    def __repr__(self):
        return "GlobalClimateData"

    def __dict__(self):
        return {"area": self.area, "time_range": self.time_range, "variables": self.variables, "cloud_threshold": self.cloud_threshold}
