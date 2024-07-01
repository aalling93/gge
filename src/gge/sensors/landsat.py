import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.algorithms.band_math.indices import compute_NDVI, compute_EVI, compute_NDWI
from gge.util import timing_decorator
from typing import Tuple, Union
from datetime import datetime
from gge.util.types import PixelType


class Landsat(SatelliteData):
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        cloud_threshold=10,
    ):
        super().__init__(area, time_range)
        self.cloud_threshold = cloud_threshold
        self.pixel_types = PixelType.DN

    @timing_decorator
    def download_data(self):
        collections = [
            "LANDSAT/LT04/C02/T1_L2",
            "LANDSAT/LT05/C02/T1_L2",
            "LANDSAT/LE07/C02/T1_L2",
            "LANDSAT/LC08/C02/T1_L2",
            "LANDSAT/LC09/C02/T1_L2",
            "LANDSAT/LC09/C02/T2_L2",
        ]

        for collection_id in collections:
            collection = (
                ee.ImageCollection(collection_id)
                .filterBounds(self.area)
                .filterDate(self.time_range[0], self.time_range[1])
                .filter(ee.Filter.lt("CLOUD_COVER", self.cloud_threshold))
                .sort("system:time_start")
            )

            count = collection.size().getInfo()
            if count == 0:
                print(f"No images found in collection {collection_id} for the given filters.")
                continue

            image_list = collection.toList(count)
            if count > 0:
                for i in range(count):
                    image = ee.Image(image_list.get(i))
                    try:
                        self.images_data.append(self.convert_data(image))
                    except Exception as e:
                        self.logger.error(f"Error converting image {image.id().getInfo()}: {e}")
            else:
                self.logger.info(f"No images found in collection {collection_id} for the given filters.")

    def convert_data(self, image):
        band_names = image.bandNames().getInfo()

        band_data = {}
        for band in band_names:
            sample = image.select(band).sampleRectangle(region=self.area, defaultValue=0)
            band_data[band] = np.array(sample.get(band).getInfo())
        return {"image_bands": band_data, "time": image.date().format().getInfo(), "metadata": image.getInfo()["properties"]}

    def dn_to_reflectance(self):
        if self.pixel_types == PixelType.DN:
            for img in self.images_data:
                img["image_bands"] = self.convert_dn_to_reflectance(img["image_bands"], img["metadata"])
            self.pixel_types = PixelType.Reflectance
        else:
            self.logger.info("Data is already in Reflectance.")

    def reflectance_to_dn(self):
        if self.pixel_types == PixelType.Reflectance:
            for img in self.images_data:
                img["image_bands"] = self.convert_reflectance_to_dn(img["image_bands"], img["metadata"])
            self.pixel_types = PixelType.DN
        else:
            self.logger.info("Data is already in DN.")

    def convert_dn_to_reflectance(self, band_data, metadata):
        reflectance_data = {}
        for band in band_data.keys():
            if band in [
                "SR_B1",
                "SR_B2",
                "SR_B3",
                "SR_B4",
                "SR_B5",
                "ST_B6",
                "SR_B7",
            ]:
                band_suffix = band[-1]  # Extracts 'B1' from 'SR_B1' etc

                mult_key = f"REFLECTANCE_MULT_BAND_{band_suffix}"  # Forms 'BAND_1' from 'B1' etc
                add_key = f"REFLECTANCE_ADD_BAND_{band_suffix}"
                if mult_key in metadata and add_key in metadata:
                    scale = float(metadata[mult_key])
                    offset = float(metadata[add_key])
                    reflectance_data[band] = (band_data[band] * scale) + offset
                else:
                    self.logger.warning(f"Reflectance scaling factors not found for {band}. Available keys: {list(metadata.keys())}")
                    reflectance_data[band] = band_data[band]
            else:
                reflectance_data[band] = band_data[band]
        return reflectance_data

    def convert_reflectance_to_dn(self, band_data, metadata):
        dn_data = {}
        for band in band_data.keys():
            if band in ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "ST_B6", "SR_B7"]:
                band_suffix = band[-1]

                mult_key = f"REFLECTANCE_MULT_BAND_{band_suffix}"
                add_key = f"REFLECTANCE_ADD_BAND_{band_suffix}"
                if mult_key in metadata and add_key in metadata:
                    scale = float(metadata[mult_key])
                    offset = float(metadata[add_key])
                    dn_data[band] = ((band_data[band] - offset) / scale).astype(np.int32)
                else:
                    self.logger.warning(f"DN conversion factors not found for {band}. Available keys: {list(metadata.keys())}")
                    dn_data[band] = band_data[band]
            else:
                dn_data[band] = band_data[band]
        return dn_data

    @property
    def item_type(self):
        return self._item_type

    @item_type.setter
    def item_type(self, value):
        valid_bands = [
            "SR_B1",
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B6",
            "SR_B7",
            "SR_B8",
            "B8A",
            "B9",
            "B10",
            "B11",
            "B12",
            "SR_B1",
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B6",
            "SR_B7",
            "ST_B10",
            "QA_PIXEL",
            "QA_RADSAT",
            "NDVI",
            "EVI",
            "NDWI",
            "RGB"
        ]
        if value.upper() in valid_bands:
            self._item_type = value
        else:
            raise ValueError("Invalid item type.")

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, item):
        img = self.images_data[item]
        metadata = img.get("metadata")
        bands = img.get("image_bands")

        landsat_number = int(metadata["LANDSAT_PRODUCT_ID"][3])

        if self._item_type in ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "SR_B8"]:
            array = bands[self._item_type]
        elif self._item_type.upper() == "NDVI":
            array = compute_NDVI(bands["SR_B4"], bands["SR_B5"])
        elif self._item_type.upper() == "EVI":
            array = compute_EVI(bands["SR_B4"], bands["SR_B5"], bands["SR_B2"])
        elif self._item_type.upper() == "NDWI":
            if landsat_number == 4 or landsat_number == 5:
                array = compute_NDWI(bands["SR_B3"], bands["SR_B5"])
            elif landsat_number == 7:
                array = compute_NDWI(bands["SR_B3"], bands["SR_B5"])
            elif landsat_number == 8:
                array = compute_NDWI(bands["SR_B3"], bands["SR_B5"])
            else:
                try:
                    array = compute_NDWI(bands["SR_B3"], bands["SR_B4"])
                except KeyError:
                    self.logger.error(f"Band {self._item_type} not found in the image.")
                    array = None
        elif self._item_type.upper() == "RGB":
            array = self.convert_to_plotable_rgb(
                {
                    "SR_B4": bands["SR_B4"],
                    "SR_B3": bands["SR_B3"],
                    "SR_B2": bands["SR_B2"],
                }
            )
        else:
            raise ValueError(f"Band {self._item_type} not found in the image.")

        return array, metadata

    def display_rgb(self, index, bands=["SR_B4", "SR_B3", "SR_B2"], scale=255, gamma=1.0, gain=1.0, red=1.0, green=1.0, blue=1.0):
        data = self.images_data[index]
        if data is not None:
            rgb_image = self.convert_to_plotable_rgb({band: data["image_bands"][band] for band in bands}, scale, gamma, gain, red, green, blue)
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.show()

    def __class__(self):
        return "Landsat"
