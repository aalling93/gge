from gge.algorithms.band_math.INCICES_CONF import EVI_G, EVI_C1, EVI_C2, EVI_L


def compute_ATSAVI(RED, NIR):

    # Adjusted transformed soil-adjusted VI (requires red, NIR, and a soil brightness correction factor)
    # ATSAVI = (B8 - a * B4 - b) / (a * B8 + B4 + b) * (1 + a^2)
    a = 0.08  # soil brightness correction factor (user-defined, example value)
    b = 0.2  # another user-defined coefficient
    return (NIR - a * RED - b) / ((a * NIR + RED + b) * (1 + a**2) + 1e-9)


def compute_NDVI(RED, NIR):
    # Normalized Difference Vegetation Index (requires red and NIR)
    return (NIR - RED) / ((NIR + RED) + 1e-9)


def compute_NDWI(NIR, SWIR):
    """
    The NDWI is used to monitor changes related to water content in water bodies.
    As water bodies strongly absorb light in visible to infrared electromagnetic spectrum,
    NDWI uses green and near infrared bands to highlight water bodies.
    It is sensitive to built-up land and can result in over-estimation of water bodies. The index was proposed by McFeeters, 1996.


    Values description:
        -1 to 0 - surface with no vegetation or water content
        +1 - represent water content
            Index values greater than 0.5 usually correspond to water bodies.
            Vegetation usually corresponds to much smaller values and built-up areas to values between zero and 0.2.
    """
    return (NIR - SWIR) / ((NIR + SWIR) + 1e-9)


def compute_GNDVI(nir, green):
    # Green Normalized Difference Vegetation Index (requires green and NIR)
    return (nir - green) / ((green + nir) + 1e-9)


def compute_OSAVI(red, nir):
    # Optimized Soil Adjusted Vegetation Index (requires red and NIR)
    return (1.16 * (nir - red)) / ((nir + red + 0.16) + 1e-9)


def compute_SAVI(red, nir):
    # Soil Adjusted Vegetation Index (requires red and NIR)
    L = 0.5  # soil brightness correction factor (typical value)
    return ((nir - red) / ((nir + red + L) + 1e-9)) * (1 + L)


def compute_EVI(blue, red, nir):
    # Enhanced Vegetation Index (requires blue, red, and NIR)

    return EVI_G * ((nir - red) / (nir + EVI_C1 * red - EVI_C2 * blue + 1e-9) + EVI_L)
