import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator
from typing import Tuple, Union, List
from datetime import datetime


class Sentinel1(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        year_range: Union[Tuple[int, int], None] = None,
        month_range: Union[Tuple[int, int], None] = None,
        hour_range: Union[Tuple[int, int], None] = None,
        day_range: Union[Tuple[int, int], None] = None,
    ):
        super().__init__(area, time_range)
        self.year_range = year_range
        self.month_range = month_range
        self.hour_range = hour_range
        self.day_range = day_range

    @timing_decorator
    def download_data(self, bands: Union[list, List, str] = ["VV", "VH"]):

        self.bands2dwl = [s.upper() for s in bands]

        collections = ["COPERNICUS/S1_GRD"]  # Ground Range Detected (GRD) products are commonly used

        for collection_id in collections:
            collection = (
                ee.ImageCollection(collection_id)
                .filterBounds(self.area)
                .filterDate(self.time_range[0], self.time_range[1])
                .filter(ee.Filter.eq("instrumentMode", "IW"))  # Interferometric Wide swath mode
                .sort("system:time_start")
            )

            # Apply optional calendar filters
            if self.year_range:
                collection = collection.filter(ee.Filter.calendarRange(self.year_range[0], self.year_range[1], "year"))
            if self.month_range:
                collection = collection.filter(ee.Filter.calendarRange(self.month_range[0], self.month_range[1], "month"))
            if self.hour_range:
                collection = collection.filter(ee.Filter.calendarRange(self.hour_range[0], self.hour_range[1], "hour"))
            if self.day_range:
                collection = collection.filter(ee.Filter.calendarRange(self.day_range[0], self.day_range[1], "day"))

            count = collection.size().getInfo()
            if count == 0:
                print(f"No images found in collection {collection_id} for the given filters.")
                continue
            # help me here. 
            # 

            # Define a fixed grid with a specific resolution and CRS (pretty improtant to make sure that the HxW pixels of all subsets are the same)
            # can change the projection and scale though...
            target_projection = "EPSG:4326"
            target_scale = 10
            fixed_grid = self.area.bounds()

            # Process each image
            image_list = collection.toList(count)
            for i in range(count):
                image = ee.Image(image_list.get(i))
                try:
                    # Align the image to the fixed grid
                    aligned_image = (
                        image.reproject(crs=target_projection, scale=target_scale)
                        .resample("bilinear")  # Resample to align pixels
                        .clip(fixed_grid)  # Clip to the fixed grid
                    )

                    # Convert the processed image
                    self.images_data.append(self.convert_data(aligned_image))
                except Exception as e:
                    self.logger.error(f"Error processing image {image.id().getInfo()}: {e}")

    def convert_data(self, image):
        # Select the polarizations and apply logarithmic scaling to convert to dB
        band_data = {}
        for band in self.bands2dwl:
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
