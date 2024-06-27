import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator
from typing import Tuple, Union
from datetime import datetime
import time
from googleapiclient.errors import HttpError
import random


class Sentinel3(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        cloud_threshold=0.1,
    ):
        super().__init__(area, time_range)
        self.cloud_threshold = cloud_threshold

    @timing_decorator
    def download_data(self):
        collections = ["COPERNICUS/S3/OLCI"]  # Modify as needed for additional collections

        for collection_id in collections:
            retry_count = 0
            while retry_count < 5:
                try:
                    collection = (
                        ee.ImageCollection(collection_id)
                        .filterBounds(self.area)
                        .filterDate(self.time_range[0], self.time_range[1])
                        .sort("system:time_start")
                        # Uncomment the next line if you need to filter by cloud cover
                        # .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_threshold))
                    )

                    count = collection.size().getInfo()
                    if count == 0:
                        print(f"No images found in collection {collection_id} for the given filters.")
                        break

                    image_list = collection.toList(count)
                    for i in range(count):
                        image = ee.Image(image_list.get(i))
                        self.images_data.append(self.convert_data(image))
                    break  # Break from the retry loop on success

                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504]:
                        sleep_time = (2**retry_count) + (random.random() * 0.5)
                        print(f"Retrying... {retry_count + 1}/5 after {sleep_time:.2f}s due to server error")
                        time.sleep(sleep_time)
                        retry_count += 1
                    else:
                        raise Exception("An error occurred that was not related to server instability") from e
                except Exception as e:
                    self.logger.error(f"Error converting image: {e}")
                    break  # Exit on any other type of error

            if retry_count == 5:
                print("Failed after 5 retries.")

    def convert_data(self, image):
        band_names = image.bandNames().getInfo()  # Example bands from OLCI and SLSTR

        band_data = {}
        for band in band_names:
            sample = image.select(band).sampleRectangle(region=self.area, defaultValue=0)
            band_data[band] = np.array(sample.get(band).getInfo())
        return {"image_bands": band_data, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def convert_radiance_to_temperature(self, band_data, metadata):
        temperature_data = {}
        # Example conversion for SLSTR bands for temperature
        for band in ["S7", "S8"]:  # Hypothetical thermal bands in SLSTR
            if band in band_data:
                # Placeholder for actual conversion formula
                # Typically involves calibration offsets and coefficients which are sensor-specific
                calibration_offset = metadata.get(f"{band}_offset", 0)
                calibration_scale = metadata.get(f"{band}_scale", 1)
                temperature_data[band] = band_data[band] * calibration_scale + calibration_offset
            else:
                temperature_data[band] = band_data[band]
        return temperature_data

    def display_rgb(
        self, index, bands=["Oa08_radiance", "Oa06_radiance", "Oa04_radiance"], scale=255, gamma=1.0, gain=1.0, red=1.0, green=1.0, blue=1.0
    ):
        data = self.images_data[index]
        if data is not None:
            rgb_image = self.convert_to_plotable_rgb({band: data["image_bands"][band] for band in bands}, scale, gamma, gain, red, green, blue)
            plt.imshow(rgb_image)
            plt.axis("off")  # Hide axis
            plt.show()

    def __len__(self):
        return len(self.images_data)

    def __class__(self):
        return "Sentinel3"
