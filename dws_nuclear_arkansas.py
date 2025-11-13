from gge import sensors

import numpy as np
import json
from shapely.geometry import mapping
import os
import matplotlib.pyplot as plt

from datetime import datetime

os.environ["google_drive_folder"] = "EarthEngineImages"


def default_converter(o):
    if hasattr(o, "__geo_interface__"):  # shapely geometry
        return mapping(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):  # datetime
        return o.isoformat()
    return str(o)


def merge_to_multichannel(arrays):
    """
    Resize all single-channel arrays to the largest shape using bilinear
    interpolation implemented with NumPy only, then merge into one array.

    Parameters
    ----------
    arrays : list[np.ndarray]
        List of 2D NumPy arrays (float or int).

    Returns
    -------
    np.ndarray
        3D array of shape (N, H, W), where N = len(arrays)
        and (H, W) = max dimensions among all arrays.
    """
    max_h = max(a.shape[0] for a in arrays)
    max_w = max(a.shape[1] for a in arrays)

    # target coordinate grid
    y_target = np.linspace(0, 1, max_h)
    x_target = np.linspace(0, 1, max_w)

    resized = []
    for a in arrays:
        h, w = a.shape
        y_src = np.linspace(0, 1, h)
        x_src = np.linspace(0, 1, w)

        # interpolate along x (each row)
        a_interp_x = np.array([np.interp(x_target, x_src, row) for row in a])
        # interpolate along y (each column)
        a_interp_xy = np.array([np.interp(y_target, y_src, a_interp_x[:, j]) for j in range(max_w)]).T

        resized.append(a_interp_xy)

    merged = np.stack(resized, axis=0)
    return merged


area = (12.1, 55.60, 12.2, 55.7)  # Define the area

area = "/Users/kaaso/Documents/phd/coding/gge/data/location/bunaeset/POLYGON.shp"
area = "/Users/kaaso/Documents/phd/coding/gge/Notebooks/arkansas_Nuclear_one.geojson"


start_year = 2021
end_year = datetime.now().year

for year in range(start_year, end_year + 1):
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01" if year < end_year else datetime.now().strftime("%Y-%m-%d")

    print(f"\n=== Processing Sentinel-2 data for {year} ({start_date} to {end_date}) ===")

    sentinel2 = sensors.Sentinel2(area, (start_date, end_date))
    sentinel2.item_type = "ALL"

    data_folder = f"data/Sentinel2/Nuclear/Sentinel2_{year}"
    os.makedirs(data_folder, exist_ok=True)

    sentinel2.download_data()

    for i in range(len(sentinel2)):
        image, metadata = sentinel2[i]
        name = metadata["PRODUCT_ID"]
        metadata["area"] = sentinel2.area.toGeoJSON()

        array, bands = [], []
        for k in image.keys():
            array.append(image[k])
            bands.append(k)

        im_folder = os.path.join(data_folder, name)
        os.makedirs(im_folder, exist_ok=True)
        try:
            merged = merge_to_multichannel(array)
            metadata["bands"] = bands

            np.save(os.path.join(im_folder, "image.npy"), merged)
        except:
            pass
            pass
        with open(os.path.join(im_folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4, default=default_converter)

        rgb = sentinel2.display_rgb(index=i, gamma=0.6, gain=1.6)
        rgb.figure.savefig(os.path.join(im_folder, "preview.png"))
        plt.close()

    print(f"Finished year {year}: saved to {data_folder}")
