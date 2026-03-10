from ipaddress import collapse_addresses
import sys

from matplotlib.pyplot import clim
from requests import head
sys.path.append("../openWLE/")

import random
import numpy as np
import pandas as pd
from openWLE import climateStudy
import pytest



class TestClimateStudyClass:

    test_csv_data_location = 'tests/test_data/test_climate_data.csv'

    def test_import_csv(self):
        
        
        config = {'csv_path': self.test_csv_data_location}
        
        climate_analyzer = climateStudy.ClimateStudy(use_eccc_data=False, 
                                                     automatic_station_selection=False, 
                                                     config=config)
        
        assert isinstance(climate_analyzer.climate_data,dict)
        assert isinstance(climate_analyzer.climate_data['00000'],pd.DataFrame)

class TestECCCCollector:

    test_site_location = (43.003810, -81.276669)
    test_closest_station_id = 4791
    test_closest_station_name = 'london sharon drive'
    test_closest_airport_station_id = 50093
    test_closest_airport_station_name = "LONDON A"
    test_closest_airport_start_year = 2012
    test_closest_airport_end_year = 2026


    def load_test_data(self):

        df = pd.read_csv('tests/test_data/test_climate_data.csv')
        return df
    
    
    
    def test_get_station_list(self):

        collector = climateStudy.ECCCCollector()
        assert isinstance(collector.get_station_list(),pd.DataFrame)
        
        assert 'Latitude (Decimal Degrees)' in collector.all_station_list.columns
        assert 'Longitude (Decimal Degrees)' in collector.all_station_list.columns

    def test_get_closest_station(self):

        collector = climateStudy.ECCCCollector(self.test_site_location)

        closest_station = collector.get_closest_station(10,wind_speed_only=False,assume_airport=False)
        
        closest_station = closest_station.iloc[0]

        assert closest_station['Name'].lower() == self.test_closest_station_name
        assert closest_station['Station ID'] == self.test_closest_station_id

    def test_get_closest_wind_station(self):

        collector = climateStudy.ECCCCollector(self.test_site_location)

        closest_station = collector.get_closest_station(20,wind_speed_only=True,assume_airport=False)
        
        closest_station = closest_station.iloc[0]

        assert closest_station['Name'].upper() == self.test_closest_airport_station_name
        assert closest_station['Station ID'] == self.test_closest_airport_station_id

    def test_get_closest_airport_station(self):

        collector = climateStudy.ECCCCollector(self.test_site_location)

        closest_station = collector.get_closest_station(20,wind_speed_only=True,assume_airport=True)
        
        closest_station = closest_station.iloc[0]

        assert closest_station['Name'].upper() == self.test_closest_airport_station_name
        assert closest_station['Station ID'] == self.test_closest_airport_station_id

    def test_find_weather_stations(self):

        collector = climateStudy.ECCCCollector(self.test_site_location)

        collector.find_all_airport_stations(20)

        assert len(collector.station_names) == 2
        assert collector.station_names[0].upper() == self.test_closest_airport_station_name
        assert collector.station_ids[0] == self.test_closest_airport_station_id
        assert collector.start_dates[0] == self.test_closest_airport_start_year
        assert collector.end_dates[0] == self.test_closest_airport_end_year
    
    def test_hourly_station_start_end_date(self):

        collector = climateStudy.ECCCCollector(self.test_site_location)

        dates = collector.get_hourly_station_start_end_date(self.test_closest_airport_station_id)

        assert dates[0] == self.test_closest_airport_start_year
        assert dates[1] == self.test_closest_airport_end_year


    def test_get_ECCC_data(self):

        collector = climateStudy.ECCCCollector()
        hourly_data = collector.get_ECCC_data(self.test_closest_airport_station_id,
                                              'Jan2013', 'Mar2013')

        saved_data = self.load_test_data()

        print(hourly_data.head())
        assert isinstance(hourly_data, pd.DataFrame)
        assert hourly_data.iloc[0]['Wind Spd (km/h)'] == 15
        assert hourly_data.head(10).equals(saved_data.head(10))

class TestClimateDataAnalyzer:
    pass

class TestExtremeWindSpeedEstimation:
    pass