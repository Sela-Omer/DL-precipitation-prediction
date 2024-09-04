from src.object.geo import Geo
import xarray as xr


class GeoXr(Geo):
    def __init__(self, name: str, xr: xr.DataArray):
        super().__init__(name)
        self.xr = xr

    def is_in(self, lon: float, lat: float):
        """
        Check if a given point is within the bounds of the Geo object.
        :param lon:
        :param lat:
        :return:
        """
        return bool(self.xr.sel(indexers={'longitude': lon, 'latitude': lat}, method='nearest') > 0.5)
