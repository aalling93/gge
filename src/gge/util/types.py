from enum import Enum, auto


class PixelType(Enum):
    DN = auto()  # Digital Number
    Reflectance = auto()  # Reflectance
    Radiance = auto()  # Radiance
    Ampltitude = auto()  # Amplitude
    Intensity = auto()
