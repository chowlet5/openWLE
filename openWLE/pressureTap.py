import numpy as np
import pandas as pd
import math


from openWLE.extreme_blue import extreme_blue

class PressureTap:
    
    '''
    Pressure Tap Class
    Description: Contains the methods for each tap using in the High Frequency Pressure Integration Test.

    General Variables:

        tap_id = numerical value or name of the tap (typically the one using the model). Example: 701, 1616
        tap_index = the current taps location/index within the main cp array obtained from the wind tunnel.
        home_wall = the name of the wall that the current tap is located on. Example: 'north', 'south'
        x = the x coordinate of the pressure tap. The center of the wall is consider zero which means x can be negative. The coordinates of full scale values.
        z = the z coordinate of the pressure tap/ the height in which the tap is located. The bottom of the wall is considered zero. The coordinates of full scale values.
        z_index = the index/location of the current tap's height within the height matrix.
        surface_width = the width dimension of the current home surface.
        surface_height = the height dimension of the current home wall.
        tap_cp = the cp time history for the tap.

    Additional Variables


        left = the tap id of the pressure tap located to the left of the current tap. If there is no tap to the left, the variable is set as "wall_left".
        right = the tap id of the pressure tap located to the right of the current tap. If there is no tap to the right, the variable is set as "wall_right".
        above = the tap id of the pressure tap located above of the current tap. If there is no tap above, the variable is set as "top".
        below = the tap id of the pressure tap located below of the current tap. If there is no tap below, the variable is set as "ground".

        width = the width dimension of the tributary area of the tap.
        height = the height dimension of the tributary area of the tap. This value only considered the pressure taps and is different than the floor height of the structure
        r = moment arm of the pressure tap from the center of the wall. Measured from the center of the wall to the center of the tap. 
            The value can be either negative (clockwise rotation/negative moment) or positive (counterclockwise rotation/ positive moment).

        

    Class Methods:

    __init__ (self, tap_id, tap_index, x, z, z_index = None, home_wall = None, surface_width = None, surface_height=None)
        Import the information required to create the pressure_tap instances. Required Information: tap_id, tap_index, x, z. The other values are set to None by default and can be set later on.

    surrounding_taps(self,tap_array,x_coor,z_coor)
        Copies the tap_id of the surrounding taps (above, left, below, right).The copied ids are placed in the corresponding variable (i.e. the tap id to the left is placed in the variable left.) 
            tap_array is an array of the tap_ids. x_coor is a list of the different x coordinates used for the current surface. z_coor is a list of the different z coordinates used on the current surface.

    tap_trib_dim_nc(self,pressure_taps)
        Determines the dimensions used to calculate the tributary area (also r). This method is used when the origin is bottom left corner of the surface. To use this method,
            the x and z coordinates provided in the __init__ method must be calcuated using the bottom left corner of the surface. pressure_taps is a dictionary of pressure_tap instances.
    tap_trib_dim_nc(self,pressure_taps)
        Determines the dimensions used to calculate the tributary area (also r). This method is used when the origin is the geometric center of the build. 
            pressure_taps is a dictionary of pressure_tap instances.
    trib_area(self,floor_height=None)
        Calcuates and returns the tributary area of the current tap. If a floor height is provide, the area will be calculated using the tributry width and the floor height.
            If a floor height is not provide, the area will be calculated using the tributry width and height.
    find_tap_cp (self, cp_data)
        Finds and stores the cp time history of the current tap. cp_data is the full array contains all the cp time histories from the wind tunnel.
            Returns the cp time history.
    cp_fix (self,pressure_taps)
        This method should be called current pressure tap has been identified as missing. If the tap is located inbetween two taps (along the x axis/ same height), the cp is estimated by taking an average between the two adjacent taps.
            If the tap is along one of the edges (if the left or right variable is set as "wall_left" or "wall_right" respectively), the cp is estimated using the adjacent tap.  
            pressure_taps is a dictionary of pressure taps instances. Returns the estimated cp time history.
    cp_fix_vertical(self,pressure_taps)
        This method should be called current pressure tap has been identified as missing and the adjacent taps are also missing (Typically used if there is a ring of bad taps/ broken scanner). 
            The cp is estimated using an average of cp values from the tap above and below the current tap.  
            pressure_taps is a dictionary of pressure taps instances. Returns the estimated cp time history.
    '''

    def __init__(self, tap_id:int, tap_index:int, x:float, y:float, z:float):

        self.tap_id=tap_id                          
        self.tap_index= tap_index           
        self.x = x
        self.y = y
        self.z = z
        self.cp = None
        self.sample_rate = None
        self.corresponding_surface = None
        self.tributary_area = None

    def __hash__(self):
        return hash(self.tap_id)
    
    def __lt__(self, other):
        if not isinstance(other, PressureTap):
            return NotImplemented
        return self.tap_id < other.tap_id
    
    def __eq__(self,other):
        if not isinstance(other,PressureTap):
            return NotImplemented
        return self.tap_id == other.tap_id
    
    def __call__(self):
        return self.tap_id

    @property
    def coords(self) -> np.array:
        return np.array([self.x,self.y,self.z])
    
    @property
    def surface(self) -> str:
        return self.corresponding_surface
    
    @property
    def area(self) -> float:
        if self.tributary_area is None:
            raise ValueError('Tributary Area missing. Please calculate the tributary area polygon first.')
        return self.tributary_area.area
    
    @surface.setter
    def surface(self, surface:str):
        self.corresponding_surface = surface

    @property
    def tap_cp(self) -> np.array:
        if self.cp is None:
            return 0
        return self.cp
    
    @tap_cp.setter
    def tap_cp(self, cp:np.array):
        self.cp = cp

    def filter_cp(self, cp:np.array) -> np.array:
        """
        Filter the cp time history using a moving average filter.
        """

    def force_calculation(self, reference_velocity: float,  density:float = 1.225, area:float = None,) -> float:

        if area is None:
            if self.tributary_area is None:
                raise ValueError('Area missing. Please provide an area value')
            else:
                area = self.area

        return -0.5*density*reference_velocity**2*area * self.tap_cp

    def max_peak_pressure(self, non_exceedance:list = [0.8], epoches:int = 16, start_index:int = 0, duration_ratio = None) -> list:

        
        segments = np.array_split(self.tap_cp[start_index:],epoches)
        min_len = min([len(x) for x in segments])
        segments = np.array([x[:min_len] for x in segments])
        peaks = np.max(segments,axis=1)        

        value, mu, sigma = extreme_blue(peaks, non_exceedance, 'max', duration_ratio)
        return value

    def min_peak_pressure(self, non_exceedance:list = [0.8], epoches:int = 16, start_index:int = 0, duration_ratio = None) -> list:

        segments = np.array_split(self.tap_cp[start_index:],epoches)
        min_len = min([len(x) for x in segments])
        segments = np.array([x[:min_len] for x in segments])
        peaks = np.min(segments,axis=1)         

        value, mu, sigma = extreme_blue(peaks, non_exceedance, 'min', duration_ratio)
        return value

    def get_data_type(self, data_type:str, starting_index:int = 0,
                       non_exceedance:list = [0.8],
                       peak_epoches:int = 16,
                       duration_ratio:float = None) ->float|np.ndarray:

        tap_cp = self.tap_cp[starting_index:] if starting_index < len(self.tap_cp) else self.tap_cp
        match data_type.lower():
            case 'mean':
                return np.mean(tap_cp)
            case 'max':
                return np.max(tap_cp)
            case 'min':
                return np.min(tap_cp)
            case 'std':
                return np.std(tap_cp)
            case 'peak_max':
                return self.max_peak_pressure(non_exceedance, peak_epoches, starting_index, duration_ratio)[0]
            case 'peak_min':
                return self.min_peak_pressure(non_exceedance, peak_epoches, starting_index, duration_ratio)[0]
            case 'peak_abs':
                max_peak = self.max_peak_pressure(non_exceedance, peak_epoches, starting_index, duration_ratio)[0]
                min_peak = self.min_peak_pressure(non_exceedance, peak_epoches, starting_index, duration_ratio)[0]
    
                return max((max_peak,min_peak), key=abs)
            case 'time_history':
                return tap_cp
            case _:
                raise ValueError(f'Invalid data type: {data_type}. Valid options are: mean, max, min, std, peak_max, peak_min, peak, time_history.')


