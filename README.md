
# Google Earth Engine Python Package (GGE)
[![DOI](https://zenodo.org/badge/820947329.svg)](https://zenodo.org/doi/10.5281/zenodo.12607034)
## License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE). So you better cite it.


## Introduction

Welcome to the Google Earth Engine (GGE) Python package, a powerful toolkit designed to simplify the interaction with the Google Earth Engine API. Our focus is on satellite data analysis, providing comprehensive tools to handle data from various satellite sensors, perform advanced band mathematics, and visualize the results.

Harness the power of satellite imagery to explore and analyze the Earth's surface with ease. Whether you're conducting research, working on environmental projects, or developing innovative solutions, GGE offers the tools you need to achieve your goals.

I have not implemeted all fully, but parts works in each folder, i.e., landsat, landcover, etc..
## Prerequisites

Before using this package, ensure you have the following installed:

- Python 3.8 or higher
- Google Earth Engine Python API
- Need a project that has access to Google Earth Engine, see https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api

## Installation

To install the package, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/aalling93/gge.git
   ```

2. Navigate to the package directory:
   ```bash
   cd gge
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or even better
   ```bash
   pip install -e .
   ```




The GGE package provides a simple and intuitive interface for working with satellite data. Here are some examples:

#### Downloading and Displaying RGB Images

```python
from gge.sensors.landsat import Landsat

# Define the area and time range
area = (35.6895, 139.6917, 35.6895, 139.6917)  # Tokyo coordinates
time_range = ('2020-01-01', '2020-12-31')

# Initialize Landsat object
landsat = Landsat(area, time_range)

# Download data
landsat.download_data()

# Display RGB image
landsat.display_rgb(2)
```

#### Accessing Specific Bands and Metadata

```python
# Set the item type to Surface Reflectance Band 4
landsat.item_type = "SR_B4"

# Access the first item
img, meta = landsat[0]


```
Likewise you can do 

```python
# Set the item type to Surface Reflectance Band 4
landsat.item_type = "EVI"

# Access the first item
img, meta = landsat[0]


```
and then you get the EVI for image 0... 



# models
I have added a single model, for change detection/anomaly detection. It is the SOM model.

It takes a 3D numpy array of, e.g., NDWI images from a region. It then traing a SOM model and finds where the anomlaies are. 



Happy analyzing!

