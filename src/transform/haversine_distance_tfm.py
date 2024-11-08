from typing import Callable

import torch
import math


class HaversineDistanceTfm(Callable):
    def __init__(self, service, drop_first_time=True):
        self.tfm_name = f'haversine_distance{"_drop_first_time" if drop_first_time else ""}'
        self.service = service
        self.lon_param = f'lon#haversine'
        self.lat_param = f'lat#haversine'
        self.data_parameters = service.data_parameters.copy()
        self.drop_first_time = drop_first_time

    def __call__(self, x):
        lon_param_i, lat_param_i = self.service.get_parameter_index(
            self.lon_param), self.service.get_parameter_index(self.lat_param)

        x[lat_param_i] = ((x[lon_param_i] + 180) % 360) - 180

        assert len(x.shape) in [2,
                                4], f"shape of x must be (PARAMS x TIMES x HEIGHT x WIDTH) or (PARAMS x TIMES) instead got {x.shape}"
        if len(x.shape) == 4:
            # PARAMS x TIMES x HEIGHT x WIDTH
            lon_tensor_flat = x[lon_param_i][..., x.shape[-2] // 2, x.shape[-1] // 2]
            lat_tensor_flat = x[lat_param_i][..., x.shape[-2] // 2, x.shape[-1] // 2]
        else:
            # PARAMS x TIMES
            lon_tensor_flat = x[lon_param_i]
            lat_tensor_flat = x[lat_param_i]

        lat_distance_tensor, lon_distance_tensor = torch.zeros([2] + list(x[lat_param_i].shape))
        for t in range(1, lon_tensor_flat.shape[0]):
            _, lat_distance, lon_distance = self.haversine_with_components(lon_tensor_flat[t - 1],
                                                                           lat_tensor_flat[t - 1],
                                                                           lon_tensor_flat[t],
                                                                           lat_tensor_flat[t])
            lat_distance_tensor[t] = lat_distance
            lon_distance_tensor[t] = lon_distance

        x[lon_param_i] = lon_distance_tensor
        x[lat_param_i] = lat_distance_tensor

        if self.drop_first_time:
            x = x[:, 1:]

        return x

    @staticmethod
    def haversine_with_components(lon1, lat1, lon2, lat2):
        """
        Calculate the great-circle distance between two points on the Earth's surface using their
        latitude and longitude values. This function returns the total distance and the approximate
        distance components along the x (longitude) and y (latitude) axes.

        Parameters:
        lon1, lat1 : float
            Longitude and latitude of the first point in degrees.
        lon2, lat2 : float
            Longitude and latitude of the second point in degrees.

        Returns:
        total_distance : float
            The total great-circle distance between the two points in kilometers.
        lat_distance : float
            The distance component along the y-axis (latitude) in kilometers.
        lon_distance : float
            The distance component along the x-axis (longitude) in kilometers.

        Note:
        - Latitude values must be in the range [-90, 90].
        - Longitude values must be in the range [-180, 180].
        - Earth's radius is approximated as 6,371 km.
        """

        # Earth's radius in kilometers
        radius_of_earth_km = 6371.0

        # Convert latitude and longitude from degrees to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # Differences between the points
        dlon = lon2 - lon1  # Difference in longitude
        dlat = lat2 - lat1  # Difference in latitude

        # Total Haversine formula for the great-circle distance
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        total_distance = radius_of_earth_km * c

        # Latitude (north-south) distance
        lat_distance = radius_of_earth_km * dlat

        # Longitude (east-west) distance: Adjust for Earth's curvature
        avg_lat = (lat1 + lat2) / 2  # Average latitude
        lon_distance = radius_of_earth_km * math.cos(avg_lat) * dlon

        return total_distance, lat_distance, lon_distance

    @staticmethod
    def inverse_haversine(lon1, lat1, lat_distance, lon_distance):
        """
        Calculate lon2 and lat2 based on lon1, lat1, lat_distance, and lon_distance.

        Parameters:
        lon1, lat1 : float
            Longitude and latitude of the first point in degrees.
        lat_distance : float
            The north-south distance (latitude) in kilometers.
        lon_distance : float
            The east-west distance (longitude) in kilometers.

        Returns:
        lon2, lat2 : float
            The calculated longitude and latitude of the second point in degrees.
        """
        # Earth's radius in kilometers
        radius_of_earth_km = 6371.0

        # Convert latitude and longitude from degrees to radians
        lon1_rad = math.radians(lon1)
        lat1_rad = math.radians(lat1)

        # Calculate the change in latitude (north-south direction)
        delta_lat = lat_distance / radius_of_earth_km
        lat2_rad = lat1_rad + delta_lat

        # Calculate the change in longitude (east-west direction)
        delta_lon = lon_distance / (radius_of_earth_km * math.cos(lat1_rad))
        lon2_rad = lon1_rad + delta_lon

        # Convert back from radians to degrees
        lon2 = math.degrees(lon2_rad)
        lat2 = math.degrees(lat2_rad)

        return lon2, lat2
