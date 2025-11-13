import ee
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Union, List
from gge.util import timing_decorator, exception_handler
from gge.sensors.SatelliteData import SatelliteData


import ee
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Union, List
from gge.util import timing_decorator, exception_handler
from gge.sensors.SatelliteData import SatelliteData



class ERA5Environment(SatelliteData):
    """
    ERA5-Land Hourly Reanalysis Handler
    Handles:
      - Small AOI → sampleRectangle() → full 2D NumPy arrays
      - Large AOI → reduceRegion(mean) → scalars
    """

    COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"

    # EXACT BAND NAMES FROM YOUR getInfo() DUMP
    ERA5_LAND_VARIABLES = [
        # Temperature
        "temperature_2m",
        "dewpoint_temperature_2m",
        "skin_temperature",

        # Soil temperature
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",

        # Soil moisture
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",

        # Pressure
        "surface_pressure",

        # Wind
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",

        # Water cycle
        "total_precipitation",
        "total_evaporation_hourly",
        "runoff",

        # Radiation
        "surface_solar_radiation_downwards",
        "surface_net_solar_radiation",
        "surface_thermal_radiation_downwards",
        "surface_net_thermal_radiation",

        # Flux
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",

        # Cryosphere
        "snow_cover",
        "snow_depth",
    ]

    # -------------------------------------------------------------------------
    def __init__(
        self,
        area=None,
        time_range=None,
        variables=None,
        sampling="auto",
        grid_scale_m=9000,
    ):
        super().__init__(area, time_range)
        self.variables = variables or self.ERA5_LAND_VARIABLES
        self.sampling = sampling.lower()
        self.grid_scale_m = int(grid_scale_m)
        self._use_reduce_region = None

    # -------------------------------------------------------------------------
    def _decide_strategy_from_geometry(self) -> bool:
        """True → reduceRegion, False → sampleRectangle."""
        if self.sampling == "mean":
            self.logger.info("sampling=mean → reduceRegion")
            return True
        if self.sampling == "grid":
            self.logger.info("sampling=grid → sampleRectangle")
            return False

        # AUTO MODE
        coords = self.area.bounds().coordinates().get(0).getInfo()
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        use_reduce = (width > 5) or (height > 5)

        self.logger.info(
            f"sampling=auto; AOI={width:.4f}°x{height:.4f}° → "
            + ("reduceRegion" if use_reduce else "sampleRectangle")
        )

        return use_reduce

    def _ensure_strategy(self):
        if self._use_reduce_region is None:
            self._use_reduce_region = self._decide_strategy_from_geometry()

    # -------------------------------------------------------------------------
    @timing_decorator
    def download_data(self):
        """Fetch all ERA5-Land data into images_data[]."""
        if self.area is None or self.time_range is None:
            raise ValueError("area and time_range must be set")

        self._ensure_strategy()

        collection = (
            ee.ImageCollection(self.COLLECTION_ID)
            .filterBounds(self.area)
            .filterDate(self.time_range[0], self.time_range[1])
            .select(self.variables)
        )

        count = collection.size().getInfo()
        if count == 0:
            self.logger.info("No images found for filters.")
            return

        self.logger.info(f"Found {count} hourly ERA5 images. Downloading…")
        imgs = collection.toList(count)

        self.data = []

        for i in range(count):
            img = ee.Image(imgs.get(i))
            self.data.append(img)
            try:
                rec = self.convert_data(img)
                if rec and rec.get("image_bands"):
                    self.images_data.append(rec)
            except Exception as e:
                self.logger.error(f"Error converting image {i}: {e}")

    # -------------------------------------------------------------------------
    @exception_handler(default_return_value={})
    def convert_data(self, image):
        """Convert one ERA5-Land image: scalars (region mean) or 2D NumPy arrays."""
        props = image.getInfo()["properties"]
        tstr = image.date().format().getInfo()

        # ============================================================
        # MODE 1: ReduceRegion (for big AOIs)
        # ============================================================
        if self._use_reduce_region:
            reduced = (
                image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=self.area,
                    scale=max(self.grid_scale_m, 1000),
                    maxPixels=self.max_pixels,
                ).getInfo()
                or {}
            )
            return {"image_bands": reduced, "time": tstr, "metadata": props}

        # ============================================================
        # MODE 2: sampleRectangle (small AOIs)
        # ============================================================
        FILL = -9999

        # IMPORTANT: ERA5 native resolution is 0.1 deg. DO NOT reproject lower.
        img2 = image.unmask(FILL)

        rect = img2.sampleRectangle(region=self.area, defaultValue=FILL).getInfo()

        if rect is None:
            return {"image_bands": {}, "time": tstr, "metadata": props}

        band_data = {}
        for var in self.variables:
            arr = rect.get(var)
            if arr is None:
                continue
            if not isinstance(arr, list):
                continue

            np_arr = np.array(arr, dtype=float)
            np_arr[np_arr == FILL] = np.nan
            band_data[var] = np_arr

        return {"image_bands": band_data, "time": tstr, "metadata": props}

    # -------------------------------------------------------------------------
    def display_data(self, index, variable, cmap="viridis"):
        rec = self.images_data[index]
        arr = rec["image_bands"].get(variable)

        if arr is None:
            self.logger.warning(f"{variable} not found at index {index}")
            return

        if np.isscalar(arr):
            print(f"{variable}={arr:.3f} @ {rec['time']}")
            return

        plt.imshow(arr, cmap=cmap)
        plt.colorbar()
        plt.title(f"{variable} @ {rec['time']}")
        plt.axis("off")
        plt.show()

    

    # -------------------------------------------------------------------------
    def compute_wind_speed(self, index):
        rec = self.images_data[index]["image_bands"]
        u = rec.get("u_component_of_wind_10m")
        v = rec.get("v_component_of_wind_10m")
        if u is None or v is None:
            return np.nan
        return np.sqrt(np.array(u) ** 2 + np.array(v) ** 2)

    # -------------------------------------------------------------------------
    def compute_daily_mean(self):
        if not self.images_data:
            return {}

        out = {}
        for rec in self.images_data:
            day = rec["time"][:10]
            for k, v in rec["image_bands"].items():
                val = np.nanmean(v) if hasattr(v, "mean") else float(v)
                out.setdefault(k, {}).setdefault(day, []).append(val)

        for k in out:
            for d in out[k]:
                out[k][d] = float(np.nanmean(out[k][d]))

        return out

    # -------------------------------------------------------------------------
    def __getitem__(self, index: int):
        return self.images_data[index]

    def __len__(self):
        return len(self.images_data)

    def __repr__(self):
        return f"<ERA5Environment AOI={self.area} time={self.time_range}>"

    def __str__(self):
        return "ERA5-Land Environmental Data Handler"

    def display_rgb(self, index):
        raise NotImplementedError("ERA5 is not RGB imagery.")




class ERA5Environment3(SatelliteData):
    """
    ERA5-Land Hourly Reanalysis Handler
    -----------------------------------
    - Automatically handles small AOI grids (sampleRectangle)
    - Falls back to regional means for large AOIs (reduceRegion)
    - All variable names VALID for ECMWF/ERA5_LAND/HOURLY
    """

    COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"

    ERA5_LAND_VARIABLES = [
        # Temperature
        "temperature_2m",
        "dewpoint_temperature_2m",
        "skin_temperature",
        # Soil temperature
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
        # Soil moisture
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
        # Surface pressure
        "surface_pressure",
        # Wind
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        # Water cycle (VALID!)
        "total_precipitation",
        "total_evaporation_hourly",
        "runoff",
        # Radiation (VALID!)
        "surface_solar_radiation_downwards",
        "surface_net_solar_radiation",
        "surface_thermal_radiation_downwards",
        "surface_net_thermal_radiation",
        # Fluxes (VALID!)
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",
        # Cryosphere (ERA5-land names)
        "snow_cover",  # Snow cover
        "snow_depth",  # Snow depth
    ]

    # -------------------------------------------------------------------------
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        variables: Union[List[str], None] = None,
        sampling: str = "auto",
        grid_scale_m: int = 9000,
    ):
        super().__init__(area, time_range)
        self.variables = variables or self.ERA5_LAND_VARIABLES
        self.sampling = sampling.lower()
        self.grid_scale_m = int(grid_scale_m)
        self._use_reduce_region = None

    # -------------------------------------------------------------------------
    def _decide_strategy_from_geometry(self) -> bool:
        """True → reduceRegion (scalar means), False → sampleRectangle (2D grid)."""
        if self.sampling == "mean":
            self.logger.info("sampling=mean → using reduceRegion()")
            return True

        if self.sampling == "grid":
            self.logger.info("sampling=grid → using sampleRectangle()")
            return False

        # automatic mode: choose based on AOI size
        coords = self.area.bounds().coordinates().get(0).getInfo()
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        use_reduce = (width > 5) or (height > 5)

        self.logger.info(f"sampling=auto; AOI {width:.4f}°×{height:.4f}° → " + ("reduceRegion()" if use_reduce else "sampleRectangle()"))
        return use_reduce

    def _ensure_strategy(self):
        if self.area is None:
            raise ValueError("Area must be set.")
        if self._use_reduce_region is None:
            self._use_reduce_region = self._decide_strategy_from_geometry()

    # -------------------------------------------------------------------------
    @timing_decorator
    def download_data(self):
        """Fetch ERA5-Land hourly data and store into images_data[]"""
        if self.area is None or self.time_range is None:
            raise ValueError("area and time_range must be set")

        self._ensure_strategy()

        collection = (
            ee.ImageCollection(self.COLLECTION_ID).filterBounds(self.area).filterDate(self.time_range[0], self.time_range[1]).select(self.variables)
        )

        count = collection.size().getInfo()
        if count == 0:
            self.logger.info("No ERA5-Land images found.")
            return

        self.logger.info(f"Found {count} hourly ERA5 images. Starting download...")
        imgs = collection.toList(count)

        self.data =[]

        for i in range(count):
            img = ee.Image(imgs.get(i))
            self.data.append(img)
            try:
                rec = self.convert_data(img)
                if rec and rec.get("image_bands"):
                    self.images_data.append(rec)
            except Exception as e:
                self.logger.error(f"Error converting image {i}: {e}")

    # -------------------------------------------------------------------------
    @exception_handler(default_return_value={})
    def convert_data(self, image):
        """Convert one ERA5-land image into np arrays or scalars."""
        props = image.getInfo()["properties"]
        tstr = image.date().format().getInfo()

        # ------------------------------------------------------------------
        # MODE 1: Large AOI → mean values only
        # ------------------------------------------------------------------
        if self._use_reduce_region:
            reduced = (
                image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=self.area,
                    scale=max(self.grid_scale_m, 1000),
                    maxPixels=self.max_pixels,
                ).getInfo()
                or {}
            )
            return {"image_bands": reduced, "time": tstr, "metadata": props}

        # ------------------------------------------------------------------
        # MODE 2: Small AOI → full 2D grid
        # ------------------------------------------------------------------
        FILL = -9999

        img2 = image.unmask(FILL).reproject(crs="EPSG:4326", scale=self.grid_scale_m)

        rect = img2.sampleRectangle(region=self.area, defaultValue=FILL).getInfo() or {}

        band_data = {}
        for var in self.variables:
            arr = rect.get(var, None)
            if isinstance(arr, list) and arr:
                np_arr = np.array(arr, dtype=float)
                np_arr[np_arr == FILL] = np.nan
                band_data[var] = np_arr

        return {"image_bands": band_data, "time": tstr, "metadata": props}

    # -------------------------------------------------------------------------
    def display_data(self, index: int, variable: str, cmap: str = "viridis"):
        rec = self.images_data[index]
        val = rec["image_bands"].get(variable)

        if val is None:
            self.logger.warning(f"{variable} not found in record.")
            return

        if np.isscalar(val):
            print(f"{variable} = {float(val):.3f} @ {rec['time']}")
            return

        plt.figure(figsize=(8, 6))
        plt.imshow(val, cmap=cmap)
        plt.colorbar(label=variable)
        plt.title(f"{variable} @ {rec['time']}")
        plt.axis("off")
        plt.show()

    # -------------------------------------------------------------------------
    def compute_wind_speed(self, index: int):
        d = self.images_data[index]["image_bands"]
        u = d.get("u_component_of_wind_10m")
        v = d.get("v_component_of_wind_10m")

        if u is None or v is None:
            return np.nan

        return np.sqrt(np.array(u) ** 2 + np.array(v) ** 2)

    # -------------------------------------------------------------------------
    def compute_daily_mean(self):
        if not self.images_data:
            return {}

        out = {}
        for rec in self.images_data:
            day = rec["time"][:10]
            for k, v in rec["image_bands"].items():
                val = np.nanmean(v) if hasattr(v, "mean") else float(v)
                out.setdefault(k, {}).setdefault(day, []).append(val)

        for k in out:
            for day in out[k]:
                out[k][day] = float(np.nanmean(out[k][day]))

        return out

    # -------------------------------------------------------------------------
    def __getitem__(self, index: int):
        return self.images_data[index]

    def __len__(self):
        return len(self.images_data)

    def __repr__(self):
        return f"<ERA5Environment AOI={self.area} time={self.time_range}>"

    def __str__(self):
        return "ERA5-Land Environmental Data Handler"

    def display_rgb(self, index):
        raise NotImplementedError("ERA5 is not RGB imagery.")


class ERA5Environment1(SatelliteData):
    """
    ERA5-Land Hourly Reanalysis Data Handler (Environmental and Weather Variables)
    ------------------------------------------------------------------------------

    Fetches ERA5-Land hourly data from Google Earth Engine, including:
      - Temperature (2 m, skin, soil)
      - Wind (U/V components at 10 m)
      - Pressure
      - Radiation (solar, thermal)
      - Precipitation and evaporation
      - Snow, soil moisture

    Automatically selects the best extraction strategy:
      - For small AOIs (<5° width/height): downloads full 2D pixel arrays
      - For large AOIs: returns regional mean per variable

    Example
    -------
    >>> area = (-10, 35, 10, 45)
    >>> time_range = ("2025-11-01", "2025-11-02")
    >>> era = ERA5Environment(area, time_range)
    >>> era.download_data()
    >>> era.display_data(0, "temperature_2m")
    """

    COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"

    DEFAULT_VARIABLES = [
        # Temperature
        "temperature_2m",
        "dewpoint_temperature_2m",
        "skin_temperature",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
        # Pressure
        "surface_pressure",
        # Wind
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        # Water cycle
        "total_precipitation",
        "total_evaporation_hourly",
        "evaporation_from_open_water_surfaces_excluding_oceans",
        # Radiation
        "surface_net_solar_radiation",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "surface_net_thermal_radiation",
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",
        # Cryosphere / soil
        "snow_cover",
        "snow_depth",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
    ]

    # -------------------------------------------------------------------------
    def __init__(
        self,
        area: Union[Tuple[float, float, float, float], str, None] = None,
        time_range: Union[Tuple[Union[str, datetime], Union[str, datetime]], str, None] = None,
        variables: Union[List[str], None] = None,
    ):
        super().__init__(area, time_range)
        self.variables = variables or self.DEFAULT_VARIABLES
        self._use_reduce_region = self._determine_sampling_strategy()

    # -------------------------------------------------------------------------
    def _determine_sampling_strategy(self) -> bool:
        """Automatically decide whether to use reduceRegion or sampleRectangle."""
        if self.area is None:
            return True

        coords = self.area.bounds().coordinates().get(0).getInfo()
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width > 5 or height > 5:
            self.logger.info(f"AOI too large ({width:.2f}°×{height:.2f}°) → using mean values via reduceRegion.")
            return True
        else:
            self.logger.info(f"AOI small ({width:.2f}°×{height:.2f}°) → sampling full grid via sampleRectangle.")
            return False

    # -------------------------------------------------------------------------
    @timing_decorator
    def download_data(self):
        """Download all selected ERA5-Land variables as NumPy arrays."""
        collection = (
            ee.ImageCollection(self.COLLECTION_ID).filterBounds(self.area).filterDate(self.time_range[0], self.time_range[1]).select(self.variables)
        )

        count = collection.size().getInfo()
        if count == 0:
            self.logger.info(f"No images found in {self.COLLECTION_ID} for the given filters.")
            return

        self.logger.info(f"Found {count} hourly ERA5 images. Starting download...")

        image_list = collection.toList(count)
        for i in range(count):
            image = ee.Image(image_list.get(i))
            try:
                data = self.convert_data(image)
                if data and data["image_bands"]:
                    self.images_data.append(data)
            except Exception as e:
                self.logger.error(f"Error converting image {i}: {e}")

    # -------------------------------------------------------------------------
    @exception_handler(default_return_value={})
    def convert_data(self, image):
        """Convert Earth Engine image to NumPy arrays per variable."""
        image_props = image.getInfo()["properties"]
        time_str = image.date().format().getInfo()

        # Option 1: Large AOI → mean values
        if self._use_reduce_region:
            reduced = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=self.area, scale=self.scale, maxPixels=self.max_pixels).getInfo()
            if not reduced:
                self.logger.warning(f"No data returned for {time_str}")
                return {}
            return {"image_bands": reduced, "time": time_str, "metadata": image_props}

        # Option 2: Small AOI → full grid
        sample = image.sampleRectangle(region=self.area, defaultValue=np.nan).getInfo()
        if not sample:
            self.logger.warning(f"No 2D data returned for {time_str}")
            return {}

        band_data = {}
        for var in self.variables:
            if var in sample:
                arr = np.array(sample[var])
                if arr.size > 0:
                    band_data[var] = arr
        return {"image_bands": band_data, "time": time_str, "metadata": image_props}

    # -------------------------------------------------------------------------
    def display_data(self, index: int, variable: str, cmap: str = "viridis"):
        """Display one variable as a 2D field or single value."""
        data = self.images_data[index]
        val = data["image_bands"].get(variable)

        if val is None:
            self.logger.warning(f"Variable {variable} not found at index {index}.")
            return

        if np.isscalar(val) or not hasattr(val, "shape"):
            print(f"{variable} = {val:.3f} ({data['time']})")
            return

        plt.figure(figsize=(8, 6))
        plt.imshow(val, cmap=cmap)
        plt.colorbar(label=variable)
        plt.title(f"{variable} at {data['time']}")
        plt.axis("off")
        plt.show()

    # -------------------------------------------------------------------------
    def compute_wind_speed(self, index: int) -> Union[float, np.ndarray]:
        """Compute wind speed magnitude (m/s) from U and V components."""
        data = self.images_data[index]["image_bands"]
        u = data.get("u_component_of_wind_10m")
        v = data.get("v_component_of_wind_10m")

        if u is None or v is None:
            self.logger.warning("Wind components not found for this record.")
            return np.nan

        return np.sqrt(np.array(u) ** 2 + np.array(v) ** 2)

    # -------------------------------------------------------------------------
    def compute_daily_mean(self):
        """Compute daily mean for each variable (aggregated over AOI)."""
        if not self.images_data:
            self.logger.warning("No images loaded.")
            return {}

        means = {}
        for d in self.images_data:
            date = d["time"][:10]
            for var, arr in d["image_bands"].items():
                val = np.nanmean(arr) if hasattr(arr, "mean") else arr
                means.setdefault(var, {}).setdefault(date, []).append(val)

        for var in means:
            for date in means[var]:
                means[var][date] = float(np.nanmean(means[var][date]))
        return means

    # -------------------------------------------------------------------------
    def __getitem__(self, index: int):
        return self.images_data[index]

    def __len__(self):
        return len(self.images_data)

    def __repr__(self):
        return f"<ERA5Environment covering {self.area} from {self.time_range[0]} to {self.time_range[1]}>"

    def __str__(self):
        return "ERA5-Land Environmental Data Handler"

    def display_rgb(self, index):
        raise NotImplementedError("ERA5 data is not RGB imagery.")
