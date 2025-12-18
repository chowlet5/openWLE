import time
import datetime
import math

import pandas as pd
import numpy as np
from scipy import stats
from dateutil import rrule
from geopy.distance import distance

import openWLE.extreme_blue as extreme_blue
from openWLE.exception import InputError
from openWLE.windProfile import WindProfile

class ClimateStudy:

    '''A class used to import, organize, and analyze climate data, specifically wind data.

    Attributes
    ----------

    '''

    def __init__(self, use_ECCC_data:bool, automatic_station_selection:bool, config:dict = None):

        self.config = config
        site_location = None
        search_radius = None
        station_ids = None
        start_dates = None
        end_dates = None
        csv_path = None

        self.climate_data = None

        if self.config is not None:
            if config.__contains__('site_location'):
                site_location = config['site_location'] 
            if config.__contains__('search_radius'):
                site_location = config['search_radius'] 
            if config.__contains__('stations_ids'):
                site_location = config['stations_ids'] 
            if config.__contains__('start_dates'):
                site_location = config['start_dates'] 
            if config.__contains__('end_dates'):
                site_location = config['end_dates'] 
            if config.__contains__('csv_path'):
                site_location = config['csv_path'] 

        if use_ECCC_data:
            self.eccc_collector = ECCCCollector(site_location,station_ids,start_dates,end_dates)
            self.run_station_selection(automatic_station_selection, search_radius)
        else:
            self.read_csv(csv_path)

    def run_station_selection(self, automatic_station_selection:bool, search_radius:float = 10) -> None:
        """Setup ECCC data collection based on user station_selection type. If automatic_station_selection is True,
        the site location is required. If automatic_station_selection is False, the station_ids list is required.
        Once conditions have been check, the data collection process begins.

        Args:
            automatic_station_selection (bool): Determines if the weather stations will be selected using the 
                                                default station selection criteria within OpenWLE. If false,
                                                the user manually selects the stations to obtain data from.
            search_radius (float, optional): The search radius used for the automatic weather station search. Defaults to 10.

        Raises:
            InputError: climateStudy.site_location is None. site_location is required for automatic station selection.
            InputError: climateStudy.station_ids is None. station_ids is required for manual station selection.
            
        """

        if automatic_station_selection:
            if self.eccc_collector.site_location is None:
                raise InputError("climateStudy.site_location = None",
                                "ClimateStudy Module: site_location is required for automatic station selection."
                                +" Please provide the design site latitude and longitude in decimal degrees.")
        
            self.eccc_collector.find_weather_stations(search_radius)
            self.collect_eccc_data()
        else:
            if self.eccc_collector.station_ids is None:
                raise InputError("climateStudy.station_ids = None",
                                "ClimateStudy Module: station_ids are required for manual station selection."
                                +" Please provide the station_ids to collect data from.")
            
            self.collect_eccc_data()
            


    def read_csv(self, file_path:str) -> pd.DataFrame:
        """Imports climate data from a csv file. This method assumed imported data is from a single weather station.


        Args:
            file_path (str): Filepath to the csv file which contains the climate data.

        Returns:
            pd.DataFrame: A pandas dataframe containing the imported climate data.
        """
        data = pd.read_csv(file_path, header = 0 )
        data['Date/Time'] = pd.to_datetime(self.data['Date/Time'])

        self.climate_data = {'00000':data}


    def collect_eccc_data(self):
        """Imports processed climate data for selected weather stations found in the 
        eccc_collector. 
        """

        self.climate_data = self.eccc_collector.get_selected_station_data()

    

    def weibull_distribution(self, wind_data:pd.DataFrame) -> tuple:

        weibull_parameters = stats.weibull_min.fit(wind_data['Wind Spd (m/s)'], floc=0)
        weibull = stats.weibull_min(*weibull_parameters)
        return weibull


    def change_site_exposure(self, wind_speed:float,
                             height: float,
                             original_exposure_parameters:dict,
                             new_exposure_parameters:dict,
                             new_height:float = None) -> float:
        """Adjusts the wind speed for change in exposure and reference height.

        Args:
            wind_speed (float): Original wind speed.
            height (float): Height of the original wind speed.
            original_exposure_parameters (dict): The exposure parameters which describe the original exposure terrain.
            new_exposure_parameters (dict): The exposure parameters which describe the new exposure terrain.
            new_height (float, optional): A new reference height to determine the new wind speed. 
                                          If None, the new wind speed will be evaluated at the original height. 
                                          Defaults to None.

        Raises:
            InputError: climateStudy.new_exposure_parameters.profile_type must be either logarithmic or power law

        Returns:
            float: Wind speed for the new exposure and new reference height if selected.
        """
        if new_height is None:
            new_height = height

        original_exposure_parameters['u_ref'] = wind_speed
        original_exposure_parameters['z_ref'] = height

        original_exposure = WindProfile(**original_exposure_parameters)

        match new_exposure_parameters['profile_type'].lower():

            case 'logarithmic':
                zg_original = original_exposure.gradient_height
                zg_new = new_exposure_parameters['gradient_height']

                roughness_length_original = original_exposure.profile.roughness_length
                roughness_length_new = new_exposure_parameters['roughness_length']

                zg_roughness_original = zg_original/roughness_length_original
                zg_roughness_new = zg_new/roughness_length_new

                new_u_star = original_exposure.profile.u_star 
                new_u_star *= math.log(zg_roughness_original) 
                new_u_star /= math.log(zg_roughness_new)

                new_exposure_parameters['u_star'] = new_u_star

                new_exposure = WindProfile(**new_exposure_parameters)
                new_wind_speed = new_exposure.along_wind_velocity_profile(new_height)

            case 'power_law':

                gradient_speed = original_exposure.gradient_wind_speed()
                zg_new = new_exposure_parameters['gradient_height']
                alpha_new = new_exposure_parameters['alpha']
                new_wind_speed = gradient_speed * (new_height/zg_new)**alpha_new



        return new_wind_speed




    def save_climate_data(self, HDF5_file:object) -> None: #TODO Complete save function

        dgroup = HDF5_file.create_group('Climate_data')

        climate_dtypes = [()]     


class ECCCCollector:
    '''A class used to export data from the Environment and Climate Change Canada database.

    Attributes
    ----------

    base_url: str
        the base url for the climate database hosted by Environment and Climate Change Canada
    
    '''

    base_url = "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    station_list_url = "https://collaboration.cmc.ec.gc.ca/cmc/climate/Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv"

    def __init__(self, site_location:tuple = None,
                 station_ids:list = None,
                 start_dates:list = None,
                 end_dates:list = None):
        
        self.site_location = site_location
        self.station_ids = station_ids
        self.start_dates = start_dates
        self.end_dates = end_dates

        self.station_names = None
        self.station_locations = None


        self.keep_columns = ['Date/Time (LST)','Year','Month','Wind Dir (10s deg)',
                             'Wind Dir Flag','Wind Spd (km/h)','Wind Spd Flag']
        self.all_station_list = self.get_station_list()


    

    def get_station_list(self) -> pd.DataFrame:
        """Get the station list from the data dictionary.

        Returns:
            pd.DataFrame: _description_
        """
        station_list = pd.read_csv(self.station_list_url, header=0, skiprows=[0,1,2])
        station_list = station_list[station_list['Latitude (Decimal Degrees)'].notna()]
        station_list = station_list[station_list['Longitude (Decimal Degrees)'].notna()]
        
        
        return station_list

    def geo_distance(self, row: dict, latitude:float, longitude:float) -> float: 
        """Calculates the geographical distance between the a given weather station and 
        a location defined by a latitude and longitude.

        Args:
            row (dict): the data for a single weather station obtained from the ECCC database
            latitude (float): The latitude for the design site location in decimal degrees.
            longitude (float): The longitude for the design site location in decimal degrees.

        Returns:
            float: The computed distance in kilometers
        """

        return distance((latitude, longitude), (row['Latitude (Decimal Degrees)'], row['Longitude (Decimal Degrees)'])).km

    def calc_station_distance(self, latitude:float = None,  longitude:float = None,) -> None:
        """Computes the distance to all available weather station hosted by ECCC 
        for a given longitude and latitude. Calculated distances are stored within the 
        all_station_list dict.

        Args:
            latitude (float, optional): The latitude of a given site location in decimal degrees. Defaults to None.
            longitude (float, optional): The longitude of a given site location in decimal degrees. Defaults to None.
            
        """
        station_list = self.all_station_list

        if longitude is None or latitude is None:
            latitude = self.site_location[0]
            longitude = self.site_location[1]

        self.all_station_list['distance'] = station_list.apply(self.geo_distance, 
                                                               axis=1,
                                                               latitude=latitude,
                                                               longitude=longitude)


    def get_closest_station(self, search_radius:float, 
                            wind_speed_only:bool = True, 
                            assume_airport = True) -> list:
        """Returns the closest station to the site location within the specified distance (in km)

        Args:
            search_radius (float): The radius (in km) around the site location to be considered.
            wind_speed_only (bool, optional): Only return weather stations which record wind data. 
                                              Defaults to True.
            assume_airport (bool, optional): Only return weather stations located at airports. 
                                             Defaults to True.

        Returns:
            list: A list of all the weather stations that satisfy the given criteria
        """

        station_list = self.all_station_list
        closest_station = station_list[station_list['distance'] < search_radius]

        if wind_speed_only:
            closest_station = closest_station[closest_station['TC ID'].notna()]

        if assume_airport:
            closest_station = closest_station[closest_station['Name'].str.contains(r"\b a\b|airport",case = False)]

        return closest_station.sort_values('distance')

    def find_weather_stations(self, search_radius:float) -> None:
        """Search for each weather station within a search radius within the design site. 
        Collect wind data from each weather station located at an airport. 
        Stores station id, name, location and date range data.

        Args:
            search_radius (float): The radius (in km) around the site location to be considered.
        """
        station_list = self.get_closest_station(search_radius,True,True)

        self.station_ids = station_list['Station ID'].values
        self.station_names = station_list['Name'].values
        self.station_locations = station_list[['Latitude (Decimal Degrees)',
                                               'Longitude (Decimal Degrees)']].values
        if self.start_dates is None:
            self.start_dates = station_list['HLY First Year'].values
        if self.end_dates is None:
            self.start_dates = station_list['HLY Last Year'].values


    def get_hourly_station_start_end_date(self, station_id:str ) -> tuple:
        """Find and returns the start and end dates which contained recorded data for a given station id.

        Args:
            station_id (str): The station id of interest.

        Returns:
            tuple: _description_
        """

        try:
            station_id_index = self.all_station_list['Station ID'] == station_id
            start_date = int(self.all_station_list[station_id_index]['HLY First Year'].values[0])
            end_date = int(self.all_station_list[station_id_index]['HLY Last Year'].values[0])
            return (start_date, end_date)
        except KeyError:
            print(f"An invalid station_id was provided: {station_id}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    def process_ECCC_climate_data(self, climate_data:pd.DataFrame) -> pd.DataFrame:
        """Read data from ECCC databases and performs unit conversions and removes 
        columns not related to wind engineering.

        Args:
            climate_data (pd.DataFrame): The dataframe which is extracted from ECCC databases.

        Returns:
            pd.DataFrame: Returns the processed climate dataframe
        """
        
        climate_data['Date/Time (LST)'] = pd.to_datetime(climate_data['Date/Time (LST)'])
        climate_data['Wind Spd (m/s)'] = climate_data['Wind Spd (km/h)'] * 1000 / 3600
        climate_data['Wind Dir'] = climate_data['Wind Dir (10s deg)']*10
        climate_data.drop(columns = ['Wind Spd (km/h)','Wind Dir (10s deg)'], inplace = True)
        climate_data.rename(columns = {'Date/Time (LST)':'Date/Time'}, inplace = True)
        climate_data.dropna(how = 'all',subset=['Wind Spd (m/s)','Wind Dir Flag'], inplace = True)
        return climate_data


    def get_selected_station_data(self) -> dict:
        """Loops through all the station id found in the station_ids variable and collects the data 
        from the ECCC database. Collected data is processed for wind engineering applications and stored in 
        the station_data variable and return as well

        Returns:
            dict: processed wind data dictionary for each station in the station_ids variable.
        """
        station_data = {}
        for index, station in enumerate(self.station_ids):

            if (self.start_dates is None) or ( self.end_dates is None):
                start_date, end_date = self.get_hourly_station_start_end_date(station)
                start_date = f'Jan{start_date}'
                end_date = f'Dec{end_date}'

            start_date = self.start_dates[index] if self.start_dates is not None else start_date
            end_date = self.end_dates[index] if self.end_dates is not None else end_date

            data = self.get_ECCC_data(station, start_date, end_date)
            data = self.process_ECCC_climate_data(data)

            station_data[station] = data

        
        return station_data 

    def get_ECCC_data(self, station_id:str, start_date:str, end_date:str) -> pd.DataFrame:
        """Collect hourly climate data from the ECCC database for a given station_id. 
        This method loops through each month defined by the start and end dates. 

        Args:
            station_id (str): The id of the station where the data is located at.
            start_date (str): The date to start collecting data from. String format follows an abbreviated month and year format '%b%Y'
            end_date (str): The date to stop collecting data from. String format follows an abbreviated month and year format '%b%Y'

        Returns:
            pd.DataFrame: The raw climate data recieved from the ECCC database.
        """
        start_date = datetime.datetime.strptime(start_date, "%b%Y")
        end_date = datetime.datetime.strptime(end_date, "%b%Y")

        data = []
        for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
            year = dt.year
            month = dt.month

            df = self.get_hourly_data(station_id, year, month)

            data.append(df[self.keep_columns])

        hourly_data = pd.concat(data)

        return hourly_data

    def get_hourly_data(self, station_id:str, year:int, month:int) -> pd.DataFrame:
        """Collect hourly data from the ECCC database for a given month and year at the station with the matching id.

        Args:
            station_id (str): The id for the station of interest.
            year (int): 
            month (int): _description_

        Returns:
            pd.DataFrame: _description_
        """

        query_string = f"format=csv&stationID={station_id}&Year={year}&Month={month}"
        query_string += "&timeframe=1&submit=Download+Data"

        data_endpoint = self.base_url + query_string
        counter = 0
        data = pd.DataFrame()
        while counter < 10:
            try:
                data = pd.read_csv(data_endpoint)
                break
            except Exception:
                print(f"Failed to get data for {station_id} - {year} - {month}. Attempting again in 10s")
                time.sleep(10)
                counter += 1
        
        return data   


class ClimateDataAnalyzer:

    def __init__(self):
        pass

    
    def calc_yearly_peaks(self, data:pd.DataFrame, wind_speed_field:str = 'Wind Spd (m/s)', overlap_date:int = 4) -> pd.DataFrame:
        """
        Function to calculate the yearly peaks from the data
        """
        temp_data = data.copy()
        wind_speeds = []
        years = sorted(list(set(data['Date/Time'].dt.year)))
        for year in years:
            temp = temp_data[temp_data['Date/Time'].dt.year==year]
            max_value = temp[wind_speed_field].max()
            date = temp[temp[wind_speed_field]==max_value]['Date/Time'].values[-1]
            
            temp_data = temp_data[~(temp_data['Date/Time']< (date+np.timedelta64(overlap_date,'D')))]
            wind_speeds.append((date,max_value))
            

        # yearly_peaks = data.groupby(data['Date/Time'].dt.year).max(numeric_only=True)
        yearly_peaks = pd.DataFrame(wind_speeds,columns=['Date/Time',wind_speed_field])
        yearly_peaks['Date/Time'] = pd.to_datetime(yearly_peaks['Date/Time'])
        return yearly_peaks
    
    def calc_yearly_peaks_directionality(self, data:pd.DataFrame, 
                                         wind_speed_field:str = 'Wind Spd (m/s)', 
                                         overlap_date:int = 4) -> pd.DataFrame:
        """
        Function to calculate the yearly peak directionality from the data
        """
        directionality_data = []
        temp_data = data.copy()

        years = sorted(list(set(data['Date/Time'].dt.year)))
        columns = sorted(list(set(data['Wind Dir'][data['Wind Dir'].notna()])))[1:]
        
        for wind_dir in columns:
            temp = temp_data.where(data['Wind Dir'] == wind_dir)

            directional_peaks = 

        
        for year in years:
            temp = temp_data[temp_data['Date/Time'].dt.year==year]
            wind_speeds = []
            dates = []
            for wind_dir in columns:
                hold = temp.where(data['Wind Dir'] == wind_dir)
                max_value = hold[wind_speed_field].max()
                wind_speeds.append(max_value)
                dates.append(hold[hold[wind_speed_field]==max_value]['Date/Time'].values[-1])
            
            temp_data = temp_data[~(temp_data['Date/Time']< (max(dates)+np.timedelta64(overlap_date,'D')))]

            directionality_data.append(tuple([max(dates).astype('datetime64[Y]')]+wind_speeds))

        new_columns = ['Date/Time'] + [int(c) for c in columns]
        yearly_peak_directionality = pd.DataFrame(directionality_data, columns= new_columns)

        return yearly_peak_directionality

    
    
class ExtremeWindSpeedEstimation:

    def __init__(self):
        pass

    def extreme_wind_speed_estimation(self, annual_peak_wind_speeds:np.ndarray, return_period:float|list = 50) -> np.ndarray:
        annual_peak_wind_speeds = annual_peak_wind_speeds.copy() 
        if annual_peak_wind_speeds.shape[0] == 1:
            annual_peak_wind_speeds = np.expand_dims(annual_peak_wind_speeds, axis=0)
        
        if isinstance(return_period,list):
            probablity_exceedence = [(1-1/t) for t in return_period]
        else:
            probablity_exceedence = [1-1/return_period]


        wind_speed, _, _ = extreme_blue.extreme_blue(annual_peak_wind_speeds, probablity_exceedence,duration_ratio=1)

        return np.array(wind_speed)
    
    def get_nondirectional_peaks(self, directional_annual_peaks:np.ndarray):

        return np.max(directional_annual_peaks.copy(),axis = 0)
    
    def sector_based_wind_speed_estimation(self, annual_peak_wind_speeds:np.ndarray, return_period:float|list = 50, max_reduction:float = 0.3, non_directional_scale:bool = True, non_directional_wind_speed: float|list = None):

        if isinstance(non_directional_wind_speed, list):
            if len(return_period) != len(non_directional_wind_speed):
                raise InputError('len(return_period) == len(non_directional_wind_speed)','Number of non directional wind speed does not match number of return periods.')

        directional_wind_speed = []

        for dir_wind_peaks in annual_peak_wind_speeds:
            dir_wind_speeds = self.extreme_wind_speed_estimation(dir_wind_peaks,return_period)
            directional_wind_speed.append(dir_wind_speeds)

        directional_wind_speed = np.array(directional_wind_speed)


        if non_directional_scale:
            if not non_directional_wind_speed:
                non_directional_peak = self.get_nondirectional_peaks(annual_peak_wind_speeds)
                non_directional_wind_speed = self.extreme_wind_speed_estimation(non_directional_peak,return_period)
            
            scaled_wind_speed = [wind_speed*x/max(x) for wind_speed,x in zip(non_directional_wind_speed,directional_wind_speed.T)]

            directional_wind_speed = np.array(scaled_wind_speed).T

        else:
            if max_reduction:
                if not non_directional_wind_speed:
                    non_directional_peak = self.get_nondirectional_peaks(annual_peak_wind_speeds)
                    non_directional_wind_speed = self.extreme_wind_speed_estimation(non_directional_peak,return_period)
                
                min_wind_speed = (1-max_reduction) * non_directional_wind_speed

                modified_wind_speed = []
                for dir_wind_speed, min_speed in zip(directional_wind_speed.T, min_wind_speed):
                    temp = dir_wind_speed
                    temp[temp<min_speed] = min_speed
                    modified_wind_speed.append(temp)
                
                directional_wind_speed = np.array(modified_wind_speed).T
           
        return directional_wind_speed
