from abc import ABC, abstractmethod


class Geo(ABC):
    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def is_in(self, lon, lat):
        """
        Check if a given point is within the bounds of the Geo object.
        :param lon:
        :param lat:
        :return:
        """
        pass
