import json
import time
from abc import ABC
from collections import Counter

import numpy as np
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from matplotlib import pyplot as plt

from src.dataset.cnn_meteorological_dataset import CNN_MeteorologicalDataset
from src.enum.biome_type import BiomeType
from src.enum.storm_type import StormType
from src.object.geo_rect import GeoRect
from src.object.geo_xr import GeoXr
from src.script.script import Script
import xarray as xr


class StormClassificationScript(Script, ABC):
    """
    A script for classifying storms based on meteorological data.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        land_xr = xr.load_dataarray(self.service.land_sea_mask_path)

        self.geos = [GeoXr('LAND', land_xr),
                     GeoXr('SEA', 1 - land_xr),
                     GeoRect('PACIFIC', 120, 240, 35, 65),
                     GeoRect('NORTH_ATLANTIC', 300, 350, -60, 60),
                     GeoRect('SOUTH_HEMISPHERE', 0, 360, -90, 0),
                     GeoRect('NORTH_HEMISPHERE', 0, 360, 0, 90),
                     GeoRect('MEDITERRANEAN', 0, 40, 30, 50),
                     ]

    def create_datamodule(self):
        pass

    def create_architecture(self, datamodule):
        pass

    def create_trainer(self, callbacks: list):
        pass

    def extract_geos_lst(self, cyclone_data):
        """
        Extract the geos that a cyclone is in.
        :param cyclone_data: The data for a cyclone.
        :return: A list of geos that the cyclone is in.
        """
        # Extract relevant parameters
        lat = cyclone_data[self.service.get_parameter_index('lat')].mean()  # Latitude
        lon = cyclone_data[self.service.get_parameter_index('lon')].mean()  # Longitude

        return [geo for geo in self.geos if geo.is_in(lon, lat)]

    def __call__(self):
        """
        This method orchestrates the training process.
        It creates the data module, architecture, callbacks, and trainer,
        and then fits the model using the trainer.
        """

        dataset = CNN_MeteorologicalDataset(self.service, self.service.config['DATA']['PATH'], self.service.data_years,
                                            self.service.data_cache)

        storm_clf_dict = {}
        for i in range(len(dataset)):
            storm = dataset[i]
            storm_geos = self.extract_geos_lst(storm.cpu().numpy())
            storm_geos_names = [geo.name for geo in storm_geos]
            storm_clf_dict[dataset.index_files[i]] = storm_geos_names

        with open('stats/STORM_CLASSIFICATION.json', 'w') as f:
            json.dump(storm_clf_dict, f)

