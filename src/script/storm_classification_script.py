import json
from abc import ABC

import xarray as xr

from src.dataset.cnn_meteorological_dataset import CNN_MeteorologicalDataset
from src.object.geo_rect import GeoRect
from src.object.geo_xr import GeoXr
from src.script.script import Script


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
        lat_tensor = cyclone_data[self.service.get_parameter_index('lat')]  # Latitude
        lon_tensor = cyclone_data[self.service.get_parameter_index('lon')]  # Longitude

        if len(lat_tensor.shape) == 3 and len(lon_tensor.shape) == 3:
            # TIMES x HEIGHT x WIDTH
            _, HEIGHT, WIDTH = lat_tensor.shape
            lat_tensor = lat_tensor[:, HEIGHT // 2, WIDTH // 2]
            lon_tensor = lon_tensor[:, HEIGHT // 2, WIDTH // 2]
        elif len(lat_tensor.shape) == 1 and len(lon_tensor.shape) == 1:
            # TIMES
            pass
        else:
            raise NotImplementedError(f'lat_tensor.shape: {lat_tensor.shape}, lon_tensor.shape: {lon_tensor.shape}')

        assert len(lat_tensor.shape) == 1 and len(lon_tensor.shape) == 1, "lat_tensor and lon_tensor must be 1D tensors"
        assert lat_tensor.shape[0] == lon_tensor.shape[0], "time dimension mismatch"

        geos = []
        for t in range(len(lat_tensor)):
            lat = lat_tensor[t].item()
            lon = lon_tensor[t].item()
            for geo in self.geos:
                if geo.is_in(lon, lat):
                    geos.append(geo)

        return geos

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
