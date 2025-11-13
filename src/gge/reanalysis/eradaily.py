import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator, exception_handler
from typing import Tuple, Union, List
from datetime import datetime


class ERA5Daily(SatelliteData):
    """
    Handles ERA5 Daily reanalysis data from the ECMWF/ERA5/DAILY collection.

    Example
    -------
    >>> area = (-10.0, 35.0, 10.0, 45.0)
    >>> time_range = (datetime(2022, 1, 1), datetime(2022, 1, 3))
    >>> era = ERA5Daily(area, time_range)  # defaults to all available variables
    >>> era.download_data()
    >>> data = era[0]
    >>> print(data["image_bands"].keys())  # all available variables
    >>> era.display_data(0, "mean_2m_air_temperature")
    """

    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        variables: Union[List[str], None] = None,
    ):
        super().__init__(area, time_range)
        self.variables = variables  # None means: all available variables

    @timing_decorator
    def download_data(self):
        """Download ERA5 Daily images as NumPy arrays for each selected variable."""
        collection_id = "ECMWF/ERA5/DAILY"
        collection = ee.ImageCollection(collection_id).filterBounds(self.area).filterDate(self.time_range[0], self.time_range[1])

        # Select specific variables if provided
        if self.variables:
            collection = collection.select(self.variables)

        count = collection.size().getInfo()
        if count == 0:
            self.logger.info(f"No images found in collection {collection_id} for the given filters.")
            return

        image_list = collection.toList(count)
        for i in range(count):
            image = ee.Image(image_list.get(i))
            try:
                data = self.convert_data(image)
                if data:
                    self.images_data.append(data)
            except Exception as e:
                self.logger.error(f"Error converting image {image.id().getInfo()}: {e}")

    @exception_handler(default_return_value={})
    def convert_data(self, image):
        """Convert a single ERA5 image into NumPy arrays per variable."""
        sample = image.sampleRectangle(region=self.area, defaultValue=0).getInfo()
        image_props = image.getInfo()["properties"]
        time_str = image.date().format().getInfo()

        # Fetch all available variables if none were specified
        if not self.variables:
            variables = list(sample.keys())
        else:
            variables = self.variables

        band_data = {}
        for var in variables:
            if var in sample:
                arr = np.array(sample[var])
                # Handle potential empty or scalar returns
                if arr.size > 0:
                    band_data[var] = arr
        return {"image_bands": band_data, "time": time_str, "metadata": image_props}

    def display_data(self, index: int, variable: str):
        """Display one variable from a given image as a map."""
        data = self.images_data[index]
        if variable not in data["image_bands"]:
            self.logger.warning(f"Variable {variable} not found in image {index}.")
            return
        plt.figure(figsize=(8, 6))
        plt.imshow(data["image_bands"][variable], cmap="viridis")
        plt.colorbar(label=variable)
        plt.title(f"{variable} at {data['time']}")
        plt.axis("off")
        plt.show()

    def __getitem__(self, index: int):
        """Access image data by index."""
        return self.images_data[index]

    def __len__(self):
        return len(self.images_data)

    def __repr__(self):
        return f"<ERA5Daily covering {self.area} from {self.time_range[0]} to {self.time_range[1]}>"

    def __str__(self):
        return "ERA5 Daily Data Handler"

    def display_rgb(self, index):
        raise NotImplementedError("ERA5 data is not RGB imagery.")


class ERA5Daily2(SatelliteData):
    """
    # Usage example:
    area = (-10.0, 35.0, 10.0, 45.0)  # Example European coverage
    time_range = (datetime(2022, 1, 1), datetime(2022, 1, 31))  # Example time range for January 2022
    era5_daily = ERA5Daily(area, time_range, variables=["2m_temperature"])
    era5_daily.download_data()
    era5_daily.display_data(0, "2m_temperature")  # Display the temperature map for the first image


    """

    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        variables: list = ["2m_temperature"],  # Default variable
    ):
        super().__init__(area, time_range)
        self.variables = variables

    @timing_decorator
    def download_data(self):
        collection_id = "ECMWF/ERA5/DAILY"
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

    @exception_handler(default_return_value={})
    def convert_data(self, image):
        sample = image.sampleRectangle(region=self.area, defaultValue=0)
        band_data = {}
        for var in self.variables:
            band_data[var] = np.array(sample.get(var).getInfo())
        return {"image_bands": band_data, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def display_data(self, index, variable):
        data = self.images_data[index]
        if data:
            plt.figure(figsize=(8, 8))
            plt.imshow(data["image_bands"][variable], cmap="viridis")
            plt.colorbar()
            plt.title(f"{variable} at {data['time']}")
            plt.axis("off")
            plt.show()

    def __len__(self):
        return len(self.images_data)

    def display_rgb(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.images_data[index]

    def __repr__(self):
        return f"<ERA5Daily covering area {self.area} from {self.time_range[0]} to {self.time_range[1]}>"

    def __str__(self):
        return "ERA5 Daily Data Handler"
