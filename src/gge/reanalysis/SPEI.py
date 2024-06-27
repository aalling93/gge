import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator, exception_handler
from typing import Tuple, Union
from datetime import datetime


class SPEIbaseData(SatelliteData):
    """
    # Usage example:
    area = (-180, -90, 180, 90)  # Global coverage
    time_range = (datetime(2020, 1, 1), datetime(2020, 12, 31))  # Example time range
    spei_data = SPEIbaseData(area, time_range)
    spei_data.download_data()
    spei_data.display_spei(0)  # Display the SPEI map for the first image


    """

    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        scale_index: int = 1,  # Default scale index for SPEI (e.g., 1 month)
    ):
        super().__init__(area, time_range)
        self.scale_index = scale_index

    @property
    def scale_index(self):
        return self._scale_index

    @scale_index.setter
    @exception_handler(default_return_value=None)
    def scale_index(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Scale index must be a positive integer.")
        self._scale_index = value

    @timing_decorator
    def download_data(self):
        collection_id = "projects/sat-io/open-datasets/SPEIbase_v28"
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(self.area)
            .filterDate(self.time_range[0], self.time_range[1])
            .select(f"SPEI_{self.scale_index}")
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
        spei_data = np.array(sample.get(f"SPEI_{self.scale_index}").getInfo())
        return {"spei_data": spei_data, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def display_spei(self, index):
        data = self.images_data[index]
        if data:
            plt.figure(figsize=(8, 8))
            plt.imshow(data["spei_data"], cmap="viridis")
            plt.colorbar()
            plt.title(f"SPEI Index at {data['time']}")
            plt.axis("off")
            plt.show()

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index):
        return self.images_data[index]

    def __repr__(self):
        return f"<SPEIbaseData covering area {self.area} from {self.time_range[0]} to {self.time_range[1]}>"

    def __str__(self):
        return "SPEIbase v2.8 Data Handler"
