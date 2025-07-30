from abc import ABC, abstractmethod

class Constants:
    VON_KARMAN = 0.41
    EARTH_ROTATION = 7.29e-5


class WindProfileBase(ABC):
    @abstractmethod
    def along_wind_velocity_profile(self, z):
        pass
    
    @property
    @abstractmethod
    def profile_name(self) -> str:
        pass


