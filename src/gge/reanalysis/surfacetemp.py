import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator, exception_handler
from typing import Tuple, Union
from datetime import datetime


class NCEPRESurfaceTemp(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        cloud_threshold=0,  # Not used in this context
    ):
        super().__init__(area, time_range)
        self._cloud_threshold = cloud_threshold  # Placeholder

    @property
    def cloud_threshold(self):
        return self._cloud_threshold

    @cloud_threshold.setter
    @exception_handler(default_return_value=None)
    def cloud_threshold(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Cloud threshold must be a number.")
        self._cloud_threshold = value

    @timing_decorator
    def download_data(self):
        collection_id = "NCEP_RE/surface_temp"
        collection = ee.ImageCollection(collection_id).filterBounds(self.area).filterDate(self.time_range[0], self.time_range[1])

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
        temp_data = np.array(sample.get("surface_temp").getInfo())
        return {"image_bands": {"surface_temp": temp_data}, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def display_temperature(self, index):
        data = self.images_data[index]
        if data:
            plt.figure(figsize=(8, 8))
            plt.imshow(data["image_bands"]["surface_temp"], cmap="coolwarm")
            plt.colorbar()
            plt.title(f"Surface Temperature at {data['time']}")
            plt.axis("off")
            plt.show()

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index):
        return self.images_data[index]

    def __repr__(self):
        return f"<NCEPRESurfaceTemp covering area {self.area} from {self.time_range[0]} to {self.time_range[1]}>"

    def __str__(self):
        return "NCEP RE Surface Temperature Data Handler"
