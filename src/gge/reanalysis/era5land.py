import numpy as np
import matplotlib.pyplot as plt
import ee
from datetime import datetime
from typing import Tuple, Union
from gge.sensors.SatelliteData import SatelliteData
from gge.util import exception_handler


class ERA5LandHourly(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        variable: Union[str, None] = "total_precipitation",
    ):
        super().__init__(area, time_range)
        self.variable = variable

    @exception_handler()
    def download_data(self):
        collection_id = "ECMWF/ERA5_LAND/HOURLY"
        if self._variable is None:
            collection = ee.ImageCollection(collection_id).filterBounds(self.area).filterDate(self.time_range[0], self.time_range[1])
        else:
            collection = (
                ee.ImageCollection(collection_id).filterBounds(self.area).filterDate(self.time_range[0], self.time_range[1]).select(self._variable)
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

    @exception_handler()
    def convert_data(self, image):
        sample = image.sampleRectangle(region=self.area, defaultValue=0)
        band_data = np.array(sample.get(self.variable).getInfo())
        return {"image_bands": {self.variable: band_data}, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def plot_time_series(self):
        data = self.images_data
        if data is None:
            self.logger.warning("No data available to plot.")
            return

        time_series = [datetime.strptime(d["time"], "%Y-%m-%dT%H:%M:%S") for d in data]
        values = [d["image_bands"][self.variable].mean() for d in data]

        plt.figure(figsize=(12, 6))
        plt.plot(time_series, values)
        plt.title(f"{self.variable} Time Series")
        plt.xlabel("Time")
        plt.ylabel(self.variable)
        plt.show()

    def display_rgb(self, index):
        raise NotImplementedError

    def __class__(self):
        return "ERA5LandHourly"

    @property
    def variable(self):
        return self._variable

    @variable.setter
    def variable(self, value):
        assert value in [
            "dewpoint_temperature_2m",
            "temperature_2m",
            "skin_temperature",
            "soil_temperature_level_1",
            "soil_temperature_level_2",
            "soil_temperature_level_3",
            "soil_temperature_level_4",
            "snow_cover",
            "surface_latent_heat_flux",
            "surface_net_solar_radiation",
            "total_precipitation",
            "total_evaporation_hourly",
            "total_precipitation_hourly",
        ]
        self._variable = value
