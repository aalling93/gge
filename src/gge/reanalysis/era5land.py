import numpy as np
import matplotlib.pyplot as plt
import ee
from datetime import datetime
from typing import Tuple, Union, List
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator, exception_handler


class ERA5LandHourly(SatelliteData):
    """
    ERA5-Land Hourly Reanalysis Data Handler

    Retrieves all available ERA5-Land hourly weather variables as 2D NumPy arrays
    for a given area and time range.

    Example
    -------
    >>> area = (-10.0, 35.0, 10.0, 45.0)
    >>> time_range = ("2025-11-01", "2025-11-02")
    >>> era = ERA5LandHourly(area, time_range)
    >>> era.download_data()
    >>> print(list(era[0]["image_bands"].keys())[:10])   # shows available variables
    >>> era.display_data(0, "temperature_2m")
    """

    COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"

    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        variables: Union[List[str], None] = None,
    ):
        super().__init__(area, time_range)
        self.variables = variables  # None → all available bands

    # -------------------------------------------------------------------------
    # Main data retrieval
    # -------------------------------------------------------------------------
    @timing_decorator
    def download_data(self):
        """Download all ERA5-Land hourly data for given area/time."""
        collection = (
            ee.ImageCollection(self.COLLECTION_ID)
            .filterBounds(self.area)
            .filterDate(self.time_range[0], self.time_range[1])
        )

        if self.variables:
            collection = collection.select(self.variables)

        count = collection.size().getInfo()
        if count == 0:
            self.logger.info(f"No images found in collection {self.COLLECTION_ID} for the given filters.")
            return

        self.logger.info(f"Found {count} hourly images in {self.COLLECTION_ID}. Downloading...")

        image_list = collection.toList(count)
        for i in range(count):
            image = ee.Image(image_list.get(i))
            try:
                data = self.convert_data(image)
                if data:
                    self.images_data.append(data)
            except Exception as e:
                self.logger.error(f"Error converting image {i}: {e}")

    # -------------------------------------------------------------------------
    # Conversion from EE to NumPy arrays
    # -------------------------------------------------------------------------
    @exception_handler(default_return_value={})
    def convert_data(self, image):
        """Convert a single ERA5 image into NumPy arrays for all selected variables."""
        sample = image.sampleRectangle(region=self.area, defaultValue=0).getInfo()
        image_props = image.getInfo()["properties"]
        time_str = image.date().format().getInfo()

        # Get all variable names automatically
        variables = self.variables or list(sample.keys())

        band_data = {}
        for var in variables:
            if var in sample:
                arr = np.array(sample[var])
                if arr.size > 0:
                    band_data[var] = arr
        return {"image_bands": band_data, "time": time_str, "metadata": image_props}

    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------
    def display_data(self, index: int, variable: str):
        """Display one variable (2D field) for a given timestamp."""
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

    # -------------------------------------------------------------------------
    # Convenience accessors
    # -------------------------------------------------------------------------
    def __getitem__(self, index: int):
        return self.images_data[index]

    def __len__(self):
        return len(self.images_data)

    def __repr__(self):
        return f"<ERA5LandHourly covering {self.area} from {self.time_range[0]} to {self.time_range[1]}>"

    def __str__(self):
        return "ERA5-Land Hourly Data Handler"

    def display_rgb(self, index):
        raise NotImplementedError("ERA5-Land data is not RGB imagery.")
