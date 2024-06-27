import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator
from typing import Tuple, Union
from datetime import datetime


class Sentinel1(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
    ):
        super().__init__(area, time_range)

    @timing_decorator
    def download_data(self):
        collections = ["COPERNICUS/S1_GRD"]  # Ground Range Detected (GRD) products are commonly used

        for collection_id in collections:
            collection = (
                ee.ImageCollection(collection_id)
                .filterBounds(self.area)
                .filterDate(self.time_range[0], self.time_range[1])
                .filter(ee.Filter.eq("instrumentMode", "IW"))  # Interferometric Wide swath mode
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
        # Select the polarizations and apply logarithmic scaling to convert to dB
        band_names = ["VV", "VH"]  # Adjust depending on available polarizations

        band_data = {}
        for band in band_names:
            # Logarithmic scale conversion from linear to dB for better visual interpretation
            db_image = image.select(band)  # .log10().multiply(10.0)
            sample = db_image.sampleRectangle(region=self.area, defaultValue=0)
            band_data[band] = np.array(sample.get(band).getInfo())
        return {"image_bands": band_data, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def display_rgb(self, index, bands=["VV", "VH", "VV"], scale=255, gamma=1.0, gain=1.0, red=1.0, green=1.0, blue=1.0):
        data = self.images_data[index]
        if data is not None:
            rgb_image = self.convert_to_plotable_rgb({band: data["image_bands"][band] for band in bands}, scale, gamma, gain, red, green, blue)
            plt.imshow(rgb_image)
            plt.axis("off")  # Hide axis
            plt.show()

    def __class__(self):
        return "Sentinel1"
