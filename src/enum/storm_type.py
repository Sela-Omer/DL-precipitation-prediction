from enum import Enum, auto

class StormType(Enum):
    HURRICANE_TYPHOON = auto()
    TROPICAL_CYCLONE = auto()
    EXTRATROPICAL_CYCLONE = auto()
    SUBTROPICAL_CYCLONE = auto()
    POLAR_LOW = auto()
    MEDICANE = auto()
    MONSOON_DEPRESSION = auto()
    LAND_BASED_CYCLONE = auto()
    COASTAL_CYCLONE = auto()
    OCEANIC_CYCLONE = auto()
    UNKNOWN_CYCLONE_TYPE = auto()