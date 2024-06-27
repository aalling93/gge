import numpy as np
import matplotlib.pyplot as plt
import ee
from gge.sensors.SatelliteData import SatelliteData
from gge.util import timing_decorator, exception_handler
from typing import Tuple, Union
from matplotlib.colors import ListedColormap, Normalize


class ESAWorldCover(SatelliteData):
    def __init__(self, area: Union[Tuple[float, float, float, float], str, None] = None):
        super().__init__(area)

    @timing_decorator
    def download_data(self):
        image_id = "ESA/WorldCover/v100/2020"
        world_cover_image = ee.Image(image_id).clip(self.area)
        if world_cover_image:
            self.images_data = self.convert_data(world_cover_image)

    @exception_handler(default_return_value={})
    def convert_data(self, image):
        world_cover_clip = image.clip(self.area)
        world_cover_clip = world_cover_clip.sampleRectangle(region=self.area, defaultValue=0).getInfo()

        landcovers = world_cover_clip["properties"]["Map"]
        landcover_class_names = world_cover_clip["properties"]["Map_class_names"]
        landcover_class_palette = world_cover_clip["properties"]["Map_class_palette"]
        landcover_class_values = world_cover_clip["properties"]["Map_class_values"]
        geometry = world_cover_clip["geometry"]

        self.mapping_list = [
            {"value": value, "name": name, "color": color}
            for value, name, color in zip(landcover_class_values, landcover_class_names, landcover_class_palette)
        ]
        self.geometry = geometry
        return {"world_cover": landcovers, "worldcover_class_names": landcover_class_names, "geometry": geometry}

    def display_world_cover(self):
        if not self.images_data:
            self.logger.warning("No worldcover data available to display.")
            return

        land_cover_data = self.images_data["world_cover"]

        values = [item["value"] for item in self.mapping_list]
        names = [item["name"] for item in self.mapping_list]
        colors = ["#" + item["color"] for item in self.mapping_list]

        unique_values = np.unique(land_cover_data)

        mask = np.isin(values, unique_values)
        filtered_values = np.array(values)[mask].tolist()
        filtered_names = np.array(names)[mask].tolist()
        filtered_colors = np.array(colors)[mask].tolist()

        cmap = ListedColormap(filtered_colors)
        norm = Normalize(vmin=min(filtered_values), vmax=max(filtered_values))
        plt.figure(figsize=(8, 8))
        im = plt.imshow(land_cover_data, cmap=cmap, norm=norm)
        cbar = plt.colorbar(im, ticks=filtered_values)
        cbar.ax.set_yticklabels(filtered_names)  # Set tick labels

        plt.axis("off")
        plt.show()

    def display_rgb(self, index, bands=["red", "green", "blue"], scale=255, gamma=1.0, gain=1.0, red=1.0, green=1.0, blue=1.0):
        # Placeholder implementation if no RGB data is available or not applicable
        raise NotImplementedError("RGB display is not supported for CORINE Land Cover data.")

    def __repr__(self):
        return "<ESAWorldCover Data Handler>"

    def __str__(self):
        return "ESA World Cover Data Handler"
