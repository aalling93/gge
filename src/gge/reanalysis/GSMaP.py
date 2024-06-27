import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator, exception_handler
from typing import Tuple, Union
from datetime import datetime


class JAXAGPMData(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
    ):
        super().__init__(area, time_range)

    @timing_decorator
    def download_data(self):
        collection_id = "JAXA/GPM_L3/GSMaP/v6/operational"
        collection = (
            ee.ImageCollection(collection_id).filterBounds(self.area).filterDate(self.time_range[0], self.time_range[1]).select("hourlyPrecipRateGC")
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

    @exception_handler(default_return_value={})
    def convert_data(self, image):
        sample = image.sampleRectangle(region=self.area, defaultValue=0)
        precip_data = np.array(sample.get("hourlyPrecipRateGC").getInfo())
        return {"precip_data": precip_data, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def display_precipitation(self, index):
        data = self.images_data[index]
        if data:
            plt.figure(figsize=(8, 8))
            plt.imshow(data["precip_data"], cmap="Blues")
            plt.colorbar()
            plt.title(f"Hourly Precipitation at {data['time']}")
            plt.axis("off")
            plt.show()

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index):
        return self.images_data[index]

    def __repr__(self):
        return "<JAXAGPMData Data Handler>"

    def __str__(self):
        return "JAXA GPM Data Handler"
