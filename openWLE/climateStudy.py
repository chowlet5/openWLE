from abc import ABC, abstractmethod
import time 
import pandas as pd
import numpy as np
import math 
import scipy.stats as stats
from dateutil import rrule
import datetime
from geopy.distance import distance
import matplotlib.pyplot as plt

import openWLE.extreme_blue as extreme_blue
from openWLE.exception import InputError


class ClimateStudy:

    base_url = "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    station_list_url = "https://collaboration.cmc.ec.gc.ca/cmc/climate/Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv"


    def __init__(self, config:dict = None) -> None:
        
        
        self.config = config
        if self.config is not None:
            self.site_location = config['site_location'] if 'site_location' in config.keys() else None
            self.station_radius = config['station_radius'] if 'station_radius' in config.keys() else None
            self.stations_ids = config['stations_ids'] if 'stations_ids' in config.keys() else None
            
            self.start_dates = config['start_dates'] if 'start_dates' in config.keys() else None
            self.end_dates = config['end_dates'] if 'end_dates' in config.keys() else None

            

        self.keep_columns = ['Date/Time (LST)','Year','Month','Wind Dir (10s deg)','Wind Dir Flag','Wind Spd (km/h)','Wind Spd Flag']
        self.all_station_list = self.get_station_list()
        
        if self.stations_ids:
            self.station_names = self.all_station_list[self.all_station_list['Station ID'].isin(self.stations_ids)]['Name'].to_list()

    def run_station_selection(self):

        try:
            method = self.config['select_method'].lower()
        except KeyError:
            raise InputError("self.config['select_method'].lower()", "select_method not provided. Please include the selection method in the configuration file")
        match method:
            case 'manual':
                self.check_manual_station_parameters()
            case 'automatic':
                if self.check_automatic_station_parameters():
                    self.run_automatic_station_selection()
            case _:
                raise InputError(f"self.climate_study.config['select_method] = {method}", f"{method} is not an avaliable option. Please select either manual or automatic.")


    def check_manual_station_parameters(self):

        if self.stations_ids == None:
            raise InputError("self.stations_ids == None", "No station ids were provided. Please provided station ids and re-run.")

        return True
        
    def check_automatic_station_parameters(self):

        if self.site_location == None:
            raise InputError("self.site_location == None", "site location was not provided. Please provided the site location coordinates and re-run.")

        if self.station_radius == None:
            raise InputError("self.site_location == None", "site radius was not provided. Please provided the site location coordinates and re-run.")

        return True

    def run_manual_station_selection(self):

        self.get_station_data()

        if self.config['analysis']:

            pass

        
    def run_automatic_station_selection(self):

        station_list = self.get_closest_station(self.station_radius,True,True)

        self.station_ids = station_list['Station ID'].values
        self.station_names = station_list['Name'].values
        self.station_locations = station_list[['Latitude (Decimal Degrees)','Longitude (Decimal Degrees)']].values
        if self.start_dates == None:
            self.start_dates = station_list['HLY First Year'].values
        if self.end_dates == None:
            self.start_dates = station_list['HLY Last Year'].values


    def get_station_list(self) -> pd.DataFrame:
        """
        Function to get the station list from the data dictionary
        """
        station_list = pd.read_csv(self.station_list_url, header=0, skiprows=[0,1,2])
        station_list = station_list[station_list['Latitude (Decimal Degrees)'].notna()]
        station_list = station_list[station_list['Longitude (Decimal Degrees)'].notna()]
        
        
        return station_list
    
    def geo_distance(self, row, latitude, longitude):
       
        return distance((latitude, longitude), (row['Latitude (Decimal Degrees)'], row['Longitude (Decimal Degrees)'])).km

    def calc_station_distance(self, longitude:float = None, latitude:float = None) -> None:
        station_list = self.all_station_list

        if longitude is None or latitude is None:
            latitude = self.site_location[0]
            longitude = self.site_location[1]

        
        self.all_station_list['distance'] =station_list.apply(self.geo_distance, axis=1, latitude=latitude, longitude=longitude)

        
    def get_closest_station(self, distance:float, wind_speed_only:bool = True, assume_airport = True) -> str:
        """
        Returns the closest station to the site location within the specified distance (in km)
        """

        station_list = self.all_station_list
        closest_station = station_list[station_list['distance'] < distance]

        if wind_speed_only:

            closest_station = closest_station[closest_station['TC ID'].notna()]

        if assume_airport:
            closest_station = closest_station[closest_station['Name'].str.contains(r"\b a\b|airport",case = False)]

        return closest_station.sort_values('distance')
    
    def get_hourly_station_start_end_date(self, stationID:str ) -> tuple:

        start_date = int(self.all_station_list[self.all_station_list['Station ID'] == stationID]['HLY First Year'].values[0])
        end_date = int(self.all_station_list[self.all_station_list['Station ID'] == stationID]['HLY Last Year'].values[0])

        return start_date, end_date

    def read_csv(self, file_path:str) -> pd.DataFrame:
        data = pd.read_csv(file_path, header = 0 )
        data['Date/Time'] = pd.to_datetime(data['Date/Time'])
        return data

    def get_station_data(self) -> dict:

        station_data = {}
        for index, station in enumerate(self.stations_ids):

            if self.start_dates is None or self.end_dates is None:
                start_date, end_date = self.get_hourly_station_start_end_date(station)
                start_date = f'Jan{start_date}'
                end_date = f'Dec{end_date}'

            start_date = self.start_dates[index] if self.start_dates is not None else start_date
            end_date = self.end_dates[index] if self.end_dates is not None else end_date

            data = self.get_climate_canada_data(station, start_date, end_date)
            data = self.wind_data_processing(data)

            station_data[station] = data

        self.station_data = station_data
        return station_data 

    
    def get_climate_canada_data(self, stationID:str, start_date:str, end_date:str) -> pd.DataFrame:
        """
        Function to get the data from the data dictionary
        """
        
        start_date = datetime.datetime.strptime(start_date, "%b%Y")
        end_date = datetime.datetime.strptime(end_date, "%b%Y")

        data = []
        for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
        
            year = dt.year
            month = dt.month

            df = self.get_hourly_data(stationID, year, month)

            data.append(df[self.keep_columns])

        hourly_data = pd.concat(data)

        return hourly_data

    def wind_data_processing(self, data:pd.DataFrame) -> pd.DataFrame:

        data['Date/Time (LST)'] = pd.to_datetime(data['Date/Time (LST)'])
        data['Wind Spd (m/s)'] = data['Wind Spd (km/h)'] * 1000 / 3600
        data['Wind Dir'] = data['Wind Dir (10s deg)']*10
        data.drop(columns = ['Wind Spd (km/h)','Wind Dir (10s deg)'], inplace = True)
        data.rename(columns = {'Date/Time (LST)':'Date/Time'}, inplace = True)
        data.dropna(how = 'all',subset=['Wind Spd (m/s)','Wind Dir Flag'], inplace = True)
        return data


    def get_hourly_data(self, stationID:str, year:int, month:int) -> pd.DataFrame:
        """
        Function to get the hourly data from the data dictionary
        """
        
        query_string = f"format=csv&stationID={stationID}&Year={year}&Month={month}&timeframe=1&submit=Download+Data"

        data_endpoint = self.base_url + query_string
        counter = 0
        data = pd.DataFrame()
        while counter < 10:
            try:
                data = pd.read_csv(data_endpoint)
                break
            except:
                print(f"Failed to get data for {stationID} - {year} - {month}. Attempting again in 10s")
                time.sleep(10)
                counter += 1
        
        
        return data    

    def calc_yearly_peaks(self, data:pd.DataFrame,wind_speed_field:str = 'Wind Spd (m/s)', overlap_date:int = 4) -> pd.DataFrame:
        """
        Function to calculate the yearly peaks from the data
        """
        temp_data = data.copy()
        wind_speeds = []
        years = sorted(list(set(data['Date/Time'].dt.year)))
        for year  in years:
            temp = temp_data[temp_data['Date/Time'].dt.year==year]
            max_value = temp[wind_speed_field].max()
            date = temp[temp[wind_speed_field]==max_value]['Date/Time'].values[-1]
            
            temp_data = temp_data[~(temp_data['Date/Time']< (date+np.timedelta64(overlap_date,'D')))]
            wind_speeds.append((date,max_value))
            

        # yearly_peaks = data.groupby(data['Date/Time'].dt.year).max(numeric_only=True)
        yearly_peaks = pd.DataFrame(wind_speeds,columns=['Date/Time',wind_speed_field])
        yearly_peaks['Date/Time'] = pd.to_datetime(yearly_peaks['Date/Time'])
        return yearly_peaks
    
    def calc_yearly_peaks_directionality(self, data:pd.DataFrame, wind_speed_field:str = 'Wind Spd (m/s)', overlap_date:int = 4) -> pd.DataFrame:
        """
        Function to calculate the yearly peak directionality from the data
        """
        directionality_data = []
        temp_data = data.copy()

        years = sorted(list(set(data['Date/Time'].dt.year)))
        columns = sorted(list(set(data['Wind Dir'][data['Wind Dir'].notna()])))[1:]
        
        
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


    def weibull_distribution(self, wind_data:pd.DataFrame) -> tuple:

        weibull_parameters = stats.weibull_min.fit(wind_data['Wind Spd (m/s)'], floc=0)
        weibull = stats.weibull_min(*weibull_parameters)
        return weibull


    def wind_speed_histogram(self, wind_data:pd.DataFrame, predefined_speeds:list = None) -> pd.DataFrame:

        if predefined_speeds is None:
            
            max_speed = math.ceil(wind_data['Wind Spd (m/s)'].max())
            max_speed += (5 * round(0.5*max_speed/5)) # Add half the max speed (rounded to nearest 5) to the max speed for visualization purposes
            predefined_speeds = np.arange(0,max_speed+1,1)
        

        wind_speeds = wind_data['Wind Spd (m/s)'].to_numpy()
        
        histogram = np.histogram(wind_speeds, bins=predefined_speeds)
        histogram = (histogram[0]/sum(histogram[0]),histogram[1])

        return histogram, predefined_speeds


    def calc_wind_speed_occurance(self, wind_records:np.ndarray,present_wind_speeds:list = None) -> np.ndarray:

        present_wind_speeds = np.sort(present_wind_speeds)
        count = []
        for i in range(len(present_wind_speeds[:-1])):
            
            low_speed = present_wind_speeds[i]
            high_speed = present_wind_speeds[i+1]
            logic = np.logical_and(low_speed <= wind_records,wind_records < high_speed)
            count.append(np.sum(logic))
        
        count.append(np.sum(wind_records > present_wind_speeds[-1]))
        
        return np.array(count)

    def wind_speed_legend(self,predefined_speeds:list) ->list:
        index = []
        for i in range(len(predefined_speeds)-1):
            index.append(f'{predefined_speeds[i]}≤V<{predefined_speeds[i+1]}')
        index.append(f'V≥{predefined_speeds[-1]}')

        return index

    def calc_wind_rose_data(self, wind_data:pd.DataFrame, directions:list, predefined_speeds:list = None, frequency_values:list = None) -> pd.DataFrame:

        if predefined_speeds is None:
            num_speeds = 5
            max_speed = wind_data['Wind Spd (m/s)'].max()
            predefined_speeds = np.linspace(0, max_speed, num_speeds)
        
        total_elements = len(wind_data['Wind Dir'])
        wind_occurance = {}
        for direction in directions:
            wind_speeds = wind_data[wind_data['Wind Dir'] == direction]['Wind Spd (m/s)'].to_numpy()
            
            wind_occurance[direction] = self.calc_wind_speed_occurance(wind_speeds, predefined_speeds)/total_elements

        index = self.wind_speed_legend(predefined_speeds)
        
        wind_occurance = pd.DataFrame(wind_occurance,index=index)

        if frequency_values is None:
            
            max_freq = wind_occurance.max().max()
            max_freq = (math.ceil(max_freq*100 / 2.) * 2)/100

            frequency_values = np.linspace(0,max_freq,5)
        
        return wind_occurance,index,frequency_values



    def save_climate_data(self, HDF5_file:object) -> None: #TODO Complete save function

        dgroup = HDF5_file.create_group('Climate_data')

        

        climate_dtypes = [()]     

    






# class ExtremeAnalysisTemplate(ABC):

#     @abstractmethod
#     def extreme_wind_speed_estimation(self, annual_peak_wind_speeds:np.ndarray, return_period:float = 50) -> float:
#         pass



class ExtremeWindSpeedEstimation:

    def __init__(self) -> None:
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

            
            


        

            
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    config = {
        'select_method': 'automatic',
        'site_location': (43.65, -79.53),
        'station_radius': 20
    }

    # config = {
    #     'select_method': 'automatic',
    #     'site_location': (43.65, -79.53),
    #     'station_radius'
    #     'stations_ids': [5097,51459],
    #     'start_dates': ['Jan1953','May2013'],
    #     'end_dates': ['Mar1953','Jul2013']
    # }

    # site_location = (43.65, -79.53)
    climate = ClimateStudy(config)
    climate.calc_station_distance()
    climate.run_station_selection()
    print(climate)
    # stations = climate.get_closest_station(100)
    # print(stations)
    #print(climate.get_station_data())
    #climate.get_station_list()
    #climate.calc_station_distance()
    #closest_station = climate.get_closest_station(10)


    
    # data_1 = climate.get_climate_canada_data(5097,'Jan1953', 'Jun2013')  
    # data_2 = climate.get_climate_canada_data(51459,'Jan2013', 'Dec2022')  
    # data_2 = climate.wind_data_processing(data_2)

    # data_2 = climate.read_csv('climate_data.csv')


    # # yearly_peaks = climate.calc_yearly_peaks(data_2)['Wind Spd (m/s)'].to_numpy()
    # yearly_peaks = climate.calc_yearly_peaks(data_2)
    # yearly_directional = climate.calc_yearly_peaks_directionality(data_2)
    # yearly_directional_peaks =  yearly_directional.to_numpy()
    # yearly_directional_peaks = yearly_directional_peaks[:,1:].T
    # extreme_value = ExtremeWindSpeedEstimation()

    # peaks = extreme_value.sector_based_wind_speed_estimation(yearly_directional_peaks,50,0.3,False,None)
    # print(peaks)
    # print(extreme_value.extreme_wind_speed_estimation(yearly_peaks['Wind Spd (m/s)'].to_numpy(),50)*0.7)
    # #print(len(data_1), '_' , len(data_2))

    # data = pd.concat([data_1, data_2])

    #print(data[['Date/Time','Wind Spd (m/s)', 'Wind Dir']].head(5))
    # data.to_csv('testing.csv',index = False)






