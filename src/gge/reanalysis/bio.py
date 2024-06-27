import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator, exception_handler
from typing import Tuple, Union
from datetime import datetime


class WorldClimBio(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
    ):
        super().__init__(area, time_range)

    @timing_decorator
    def download_data(self):
        image_id = "WORLDCLIM/V1/BIO"
        image = ee.Image(image_id).select(["bio01", "bio12"])

        try:
            self.images_data.append(self.convert_data(image))
        except Exception as e:
            self.logger.error(f"Error converting image: {e}")

    @exception_handler(default_return_value={})
    def convert_data(self, image):
        sample = image.sampleRectangle(region=self.area, defaultValue=0)
        band_data = {"bio01": np.array(sample.get("bio01").getInfo()), "bio12": np.array(sample.get("bio12").getInfo())}
        return {"image_bands": band_data, "metadata": image.getInfo()["properties"]}

    def display_data(self, variable):
        if self.images_data:
            data = self.images_data[0]  # Only one image expected
            if variable in data["image_bands"]:
                plt.figure(figsize=(8, 8))
                plt.imshow(data["image_bands"][variable], cmap="viridis")
                plt.colorbar()
                plt.title(f"WorldClim {variable}")
                plt.axis("off")
                plt.show()

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index):
        return self.images_data[index]

    def __repr__(self):
        return "<WorldClimBio Data Handler>"

    def __str__(self):
        return "WorldClim Bio Data Handler"
