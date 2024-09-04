from src.object.geo import Geo


class GeoRect(Geo):
    def __init__(self, name: str, min_lon, max_lon, min_lat, max_lat):
        super().__init__(name)
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        assert min_lon < max_lon
        assert min_lat < max_lat
        assert 0 <= min_lon <= 360
        assert 0 <= max_lon <= 360
        assert -90 <= min_lat <= 90
        assert -90 <= max_lat <= 90

    def is_in(self, lon: float, lat: float):
        """
        Check if a given point is within the bounds of the Geo object.
        :param lon:
        :param lat:
        :return:
        """
        return self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat
