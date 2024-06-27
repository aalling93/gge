import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator, exception_handler
from typing import Tuple, Union
from matplotlib.colors import ListedColormap, Normalize


class CGLSLandCover100(SatelliteData):
    def __init__(self, area: Union[Tuple[float, float, float, float], str, None] = None):
        super().__init__(area)

    @timing_decorator
    def download_data(self):
        collection_id = "COPERNICUS/Landcover/100m/Proba-V-C3/Global"
        collection = ee.ImageCollection(collection_id).filterBounds(self.area)
        cgls_image = collection.first()
        if cgls_image:
            self.images_data = self.convert_data(cgls_image)

    @exception_handler(default_return_value={})
    def convert_data(self, image):
        cgls_clip = image.clip(self.area)
        cgls_clip = cgls_clip.sampleRectangle(region=self.area, defaultValue=0).getInfo()

        self.mapping_list = []
        self.geometry = cgls_clip["geometry"]

        # Dynamically create the mapping list for different classification types
        for key in cgls_clip["properties"]:
            if key.endswith("_class_values"):
                class_type = key.replace("_class_values", "")
                values = cgls_clip["properties"][key]
                names = cgls_clip["properties"].get(f"{class_type}_class_names", [])
                colors = cgls_clip["properties"].get(f"{class_type}_class_palette", [])
                self.mapping_list.extend([{"value": value, "name": name, "color": color} for value, name, color in zip(values, names, colors)])

        return {"world_cover": cgls_clip["properties"].get("discrete_classification", []), "geometry": self.geometry}

    def display_world_cover(self):
        if not self.images_data:
            self.logger.warning("No world cover data available to display.")
            return

        land_cover_data = self.images_data["world_cover"]

        if not land_cover_data:
            print("No land cover data found.")
            return

        # Create mappings from the dynamically built mapping_list
        values = [item["value"] for item in self.mapping_list]
        names = [item["name"] for item in self.mapping_list]
        colors = ["#" + item["color"] for item in self.mapping_list]

        # Ensure only values present in the data are used for the colormap
        unique_values = np.unique(land_cover_data)
        mask = np.isin(values, unique_values)
        filtered_values = np.array(values)[mask]
        filtered_names = np.array(names)[mask]
        filtered_colors = np.array(colors)[mask]

        # Create colormap and normalization
        cmap = ListedColormap(filtered_colors)
        norm = Normalize(vmin=np.min(filtered_values), vmax=np.max(filtered_values))

        # Plotting
        plt.figure(figsize=(8, 8))
        im = plt.imshow(land_cover_data, cmap=cmap, norm=norm)

        # Create a colorbar with labels at correct positions
        cbar = plt.colorbar(im, ticks=filtered_values, drawedges=True)
        cbar.ax.set_yticklabels(filtered_names)  # Set the labels

        plt.title("CGLS Land Cover 100m Global")
        plt.axis("off")
        plt.show()

    def __repr__(self):
        return "<Sentinel2WorldCover Data Handler>"

    def __str__(self):
        return "Sentinel-2 World Cover Data Handler"

    def display_rgb(self, index, bands=["red", "green", "blue"], scale=255, gamma=1.0, gain=1.0, red=1.0, green=1.0, blue=1.0):
        # Placeholder implementation if no RGB data is available or not applicable
        raise NotImplementedError("RGB display is not supported for Sentinel2WorldCover Land Cover data.")
