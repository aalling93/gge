from setuptools import setup, find_packages

setup(
    name="gge",
    version="0.0.0.2",
    description="gge python. ",
    long_description="",
    url="https://github.com/aalling93",
    author="Aalling93",
    author_email="kaaso@space.dtu.dk",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "earthengine-api>=0.1",
        "geopandas>=1.0.0",
        "google_api_python_client>=2.1",
        "matplotlib>=3.6",
        "numpy>=2.0.0",
        "tqdm>=4.5"
    ],
)
