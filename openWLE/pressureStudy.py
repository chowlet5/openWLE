import os
import warnings
import numpy as np
import pandas as pd
import pathlib
import re
import glob
import h5py

from openWLE.pressureTap import PressureTap
from openWLE.buildingClass import Building
from openWLE.geometry import vector_rotation

try:
    from readPSSfile import readPSSfile
except ImportError:
    warnings.warn("readPSSfile is not included. This is a proprietary script and cannot be shared public")


def list_tap_layout_helper(df:pd.DataFrame, tap_id:int, surface:str):
    """ Helper function for tap_layout_list"""
    data = df[df['ID'] == tap_id]
    if len(data) > 1:
        raise ValueError(f'There are multiple taps with the value: {tap_id} in the tap layout file.')
    else:
        data = data.iloc[0].to_dict()
        pressure_tap = PressureTap(data['ID'], surface, data['X'], data['Y'],data['Z'])
        return pressure_tap



class ReadOpenFOAMPressure:

    def __init__(self,config:dict):

        self.config = config
        self.air_density = self.config['raw_data_config']['raw_data_ref_density']
        self.reference_velocity = self.config['raw_data_config']['raw_data_ref_velocity']
        self.static_pressure = self.config['raw_data_config']['raw_data_ref_pressure']

        self.pressure = None
        
        self.cp_data = None
        self.time = None
        self.probe_locations = None
    
    def get_directory_number(self,directory:str) -> float|None:

        try:
            return float(directory)
        except:
            return None
    
    def check_numerical_value(self,directory:str) -> float|None:

        try:
            float(directory)
        except:
            return False
        return True

    def process_probe_location(self, string:str) -> tuple:

            locations = string.split('(')[1].split(')')[0].split(' ')

            return (float(locations[0]), float(locations[1]), float(locations[2]))
    

    def process_probe_data(self, string:str) -> tuple:

        string = string.replace('+','')
        data = re.sub(r'\s+', ' ', string)
        data = data.split(' ')[:-1]
        time = float(data[0])

        pressure = np.array(data[1:],dtype=float)
        return time, pressure

    def process_pressure_file(self, lines:list) -> tuple:

        probe_locations = {}
        time = np.array([]) 
    
        for l in lines:
            if '# Probe' in l:
                probe_number = int(l.split(' ')[2])
                probe_locations[probe_number] = self.process_probe_location(l)

            elif '# Time' in l:
                pressure = np.empty((0, len(probe_locations)))
            else:

                time_step, p = self.process_probe_data(l)
                time = np.append(time, time_step)
                pressure = np.vstack((pressure, p))

        return time, pressure,  probe_locations
    

    

    def read_pressure_directory(self, directory_path:str) -> tuple:

        directories = glob.glob(f'{directory_path}/*/')
        directories = [os.path.normpath(d) for d in directories]
        base_names = [os.path.basename(d) for d in directories]
        base_names = [d if self.check_numerical_value(d) else '-1' for d in base_names]
        base_names.sort(key = float)
        # numeric_directories = [self.get_directory_number(d.split('/')[-2]) for d in directories]

        time = None
        pressure = None
        probe_locations = None

        for index, direct in enumerate(base_names):
            if direct == "-1":
                continue


            file = os.path.join(directory_path,direct,'p')

            with open(file, 'r') as f:
                lines = f.readlines()
            
            temp_time, temp_pressure, temp_probe_locations = self.process_pressure_file(lines)
            if time is None:
                time = temp_time
                pressure = temp_pressure
                probe_locations = temp_probe_locations
            else:
                check = time<min(temp_time)
                time = np.append(time[check], temp_time)
                pressure = np.vstack((pressure[check,:], temp_pressure))
        
        return pressure, time, probe_locations


    def read_pressure_data(self, directory_path:str) -> None:

        pressure, time, probe_locations = self.read_pressure_directory(directory_path)

        new_pressure, new_time = self.resample_data(time, pressure.T, self.config['scanner_sampling_rate'])

        self.time = new_time
        self.probe_locations = probe_locations  
        self.pressure = new_pressure

        if isinstance(self.static_pressure, str):
            base_name = os.path.basename(directory_path)
            self.static_pressure = self.get_reference_pressure(base_name)
            

    def read_cp_data(self, file_path:str) -> None:

        self.read_pressure_directory(file_path)

        self.cp_data = self.pressure
        self.pressure = None
        
    def resample_data(self, time:np.ndarray, data:np.ndarray,sampling_rate:float) -> np.ndarray:

        """ Resamples the data to match the time vector. """

        new_time = np.arange(time[0], time[-1], 1/sampling_rate)

        if data.ndim == 1:
            data = np.expand_dims(data,axis = 0)
        new_data = np.zeros((data.shape[0],len(new_time)))

        for i in range(data.shape[0]):
            new_data[i,:] = np.interp(new_time, time, data[i,:])

        return new_data, new_time
    
    def resample_reference_data(self, time:np.ndarray, data:np.ndarray, new_time:np.ndarray) -> np.ndarray:

        """ Resamples the data to match the time vector. """

        new_data = np.interp(new_time, time, data)

        return new_data
    
    def read_reference_pressure_probe(self, file_path:str, probe_index:int) -> None:
        """ Reads the reference pressure from the file. """

        with open(file_path, 'r') as f:
            lines = f.readlines()
    
        pressure, probes, time = self.process_pressure_file(lines)
        
        new_pressure, new_time = self.resample_data(time, pressure, self.config['scanner_sampling_rate'])

        self.static_pressure = new_pressure[:,probe_index]
        self.time = new_time
        self.probe_locations = probes


    def get_reference_pressure(self, base_name:str) -> np.ndarray:
        """ Check is static pressure is file or """

        
        probe_index = self.config['raw_data_config']['raw_data_ref_pressure_index'] 
        static_pressure, time, _ = self.read_pressure_directory(f'{self.static_pressure}/{base_name}')
        static_pressure = static_pressure[:,probe_index]
        new_static = self.resample_reference_data(time, static_pressure,self.time)

        return new_static
        

        
    

class PressureStudy:
    """
    High Frequency Pressure Intergration Test Class

    Contains the methods for processing the pressure data from the wind tunnel test.
    Uses the Surface and PressureTap classes to store the data.

    Attributes:

    config: A dictionary containing the configuration data for the test. Imported from YAML config file.
    starting_tap: The tap number that the test will start at. This maybe used to skip taps that are not used in the test or used in another aspect of the test not included in the pressure integration.
    missing_taps: A list of tap numbers that are missing from the test. Commonly these taps are either diconnected or broken while the model is being install in the wind tunnel.
    broken_scanner: A list of taps that are not functioning correctly. These taps are replaced using interprolation methods.
    tap_input_method: The method used to input the tap layout data. Valid options are 'dxf', 'list', and 'excel'.
    tap_layout_filepath: The file path to the tap layout data. The file path is used to read the tap layout data and generate the PressureTap objects.
    pressure_tap_list: A list of PressureTap objects that contain data associated with each pressure tap. Refer to the PressureTap class for more information.
    
    Methods:

    tap_layout_dxf: This method will be added in the future. It will read the autodesk tap layout file and returns a list of PressureTap objects.
    tap_layout_list: Reads the tap layout text/csv file and returns a list of PressureTap objects.
    tap_layout_excel: Reads the tap layout excel file and returns a list of PressureTap objects.

    """

    def __init__(self, config, building:Building, wind_direction:float = 0.0) -> None:

        self.config = config
        self.cp_data = None
        self.raw_data = None


        self.wind_angle = wind_direction
        self.starting_tap = self.config['starting_tap']
        self.missing_taps = self.config['missing_taps']
        self.broken_scanner = self.config['problem_scanner']
        self.scanner_type = self.config['scanner_type']
        self.scanner_channel = self.config['scanner_channel']
        self.scanner_sampling_rate = self.config['scanner_sampling_rate']
        tap_input_method = self.config['tap_input_method']
        tap_layout_filepath = self.config['tap_layout_file']
        self.raw_data_config = self.config['raw_data_config']
        self.data_type = self.config['raw_data_config']['raw_data_type']
        self.cp_import_method = self.config['raw_data_config']['raw_data_import_method']
        self.HDF_field_list = self.config['raw_data_config']['hdf5_field_path'] if self.config['raw_data_config'].__contains__('hdf5_field_path') else None
        self.recorded_reference_height = self.config['recorded_reference_height']
        self.new_reference_height = self.config['new_reference_height']
        

        self.building = building

        self.tap_import(tap_input_method, tap_layout_filepath)

    def file_extension(self, data_file_name:str,) -> str:
        """ Imports the data file and tap layout file. """
        data_import_method = pathlib.Path(data_file_name).suffix
        if data_import_method == '.csv':
            return 'csv'
        elif data_import_method == '.pssr':
            return 'BLWTL'
        else:
            raise ValueError('Invalid data file format. Valid options are csv and pssr.')
        
    def tap_import(self,tap_input_method:str, tap_layout_filepath:str) -> None:
        

        if tap_input_method.lower() == 'list':
            self.tap_layout_list(tap_layout_filepath)
        # elif tap_input_method == 'dxf':
        #     self.tap_layout_list(tap_layout_filepath)
        # elif tap_input_method == 'excel':
        #     self.tap_layout_excel(tap_layout_filepath)
        else:
            raise ValueError('Invalid tap input method. Valid options are list.')

        self.create_tap_id_list()
        self.assign_tap_index()


    def calculate_wind_vector(self, angle:float, axis:str = 'z'):

        self.wind_vector = vector_rotation(self.wind_direction_axis, angle, axis)

    def assign_cp_data(self) -> None:
        """ Assigns the cp data to the pressureTap objects. """

        for tap in self.pressure_tap_list:
            tap.tap_cp = self.cp_data[tap.index]
            tap.sampling_rate = self.scanner_sampling_rate

        self.estimate_missing_taps()

    def get_data(self, file_path:str) -> None:

        """Determine which method to use and call the appropriate method."""
        match self.data_type.lower():
            case 'pressure':
                self.get_pressure_data(file_path)
            case 'cp':
                self.get_cp_data(file_path)
            case _:
                raise ValueError('Invalid raw data type. Valid options are "pressure" and "cp".')
        
    def get_cp_data(self,file_path:str) -> None:
        """ Imports the pressure data and stores it in the test object. """

        match self.cp_import_method.lower():
            case 'openfoam':
                openfoam = ReadOpenFOAMPressure(self.config)
                self.cp_data = openfoam.read_cp_data(file_path)
            case 'csv':
                self.read_cp_csv_data(file_path)
            case 'blwtl':
                self.read_BLWTL_cp(file_path, file_path.replace('.pssr','.pssd'))
            case 'hdf5':
                if self.HDF_field_list is None:
                    raise KeyError('HDF_field_list not provide for hdf5 import method.')
                self.read_HDF5(file_path)
            case _:
                raise ValueError('Invalid data import method. Valid options are "csv", "BLWTL", and "openFOAM".')
       

    def get_pressure_data(self, file_path:str) -> None:
        """ Imports the cp data and stores it in the test object. """
        match self.cp_import_method.lower():
            case 'openfoam':
                openFoamReader = ReadOpenFOAMPressure(self.config)
                openFoamReader.read_pressure_data(file_path)
                self.cp_data = self.pressure_2_cp(openFoamReader.pressure, openFoamReader.reference_velocity, openFoamReader.static_pressure, openFoamReader.air_density)
            case 'csv':
                self.read_pressure_csv_data(file_path)
            case _:
                raise ValueError('Invalid data import method. Valid options are csv and openFOAM.')

    
    def pressure_2_cp(self, pressure_data:np.array, reference_velocity, static_pressure = 0, air_density = 1.225) -> np.ndarray:
        """ Converts the pressure data to cp data. """
        
        cp_data = (pressure_data - static_pressure)/(0.5*air_density*reference_velocity**2)
        return cp_data
       

    def re_reference_cp_data(self, velocity_ratio: float) -> None:
        """ Converts the cp data to a new reference height. """
        self.raw_data = self.cp_data.copy()
        self.velocity_conversion_ratio = velocity_ratio**2
        self.cp_data = self.cp_data*(velocity_ratio**2)

    def read_cp_csv_data(self, file_path:str, delimiter:str = ',') ->None:
        """ Reads the cp data from the wind tunnel and stores it in the test object. """

        #Assume the tap data is oriented in the columns
        self.cp_data = np.loadtxt(file_path, delimiter=delimiter).T

    def read_pressure_csv_data(self, file_path:str, delimiter:str = ',') ->None:
        """ Reads the cp data from the wind tunnel and stores it in the test object. """

        #Assume the tap data is oriented in the columns
        pressure = np.loadtxt(file_path, delimiter=delimiter).T
        self.cp_data = self.pressure_2_cp(pressure)

    def read_BLWTL_cp(self, pssr_file_path:str, pssd_file_path:str) -> None:

        [cp_data,analog,header]=readPSSfile(pssr_file_path,pssd_file_path)
        self.cp_data = cp_data.T

    def read_HDF5(self, HDF_file_path:str, HDF_fields:list) -> None:
        
        f = h5py.File(HDF_file_path)
        for field in HDF_fields:
            if field in list(f.keys()):
                f = f[field]
            else:
                raise KeyError(f'"{field}" not found in HDF5. Select a valid path.')

        self.cp_data = np.array(f).T

    def tap_layout_list(self, file_path) -> None:
        """ Reads the tap layout text/csv file and returns a list of PressureTap objects."""
        
        self.pressure_tap_list = []
        data = pd.read_csv(file_path, header=0, delimiter=',')
        unique_surface = data['SURFACE'].unique()

        for surface in unique_surface:
            surface_data = data[data['SURFACE'] == surface]
            tap_id_list = surface_data['ID'].values.flatten().tolist()
            tap_list = list(map(lambda id: list_tap_layout_helper(surface_data,id,surface), tap_id_list))

            self.building.assign_surface_taps(surface, tap_list)
            
            self.pressure_tap_list.extend(tap_list)

        
        self.pressure_tap_list.sort(key=lambda x: x.tap_id)
        self.building.assign_floor_taps()
        
    def create_tap_id_list(self) -> None:
        """ Create a list of the index of the tap data in the cp data. """
        scanner_channel_count= self.scanner_channel
        max_pressure_tap = max(self.pressure_tap_list)
        max_scanner = int(str(max_pressure_tap.tap_id)[:-2])
        
        min_scanner = int(str(self.starting_tap)[:-2])

        self.tap_id_list = []
        for scanner in range(min_scanner, max_scanner+1):
            for channel in range(1,scanner_channel_count+1):
                self.tap_id_list.append(int(f'{scanner}{channel:02d}'))
    
    def tap_index(self, tap_id: int) -> None:
        """ Returns the index of the tap data in the cp data. """
        return self.tap_id_list.index(tap_id)
    
    def assign_tap_index(self) -> None:
        """ Assigns the index of the tap data in the cp data to the PressureTap object. """
        for tap in self.pressure_tap_list:
            tap.index = self.tap_index(tap.tap_id)

    def estimate_missing_taps(self) -> None:

        for surface in self.building.surfaces.values():
            surface_missing_taps = [tap for tap in self.missing_taps if tap in surface.tap_id_list]
            if surface_missing_taps:
                surface.estimate_missing_taps(surface_missing_taps)

    def save_raw_cp_data(self, HDF5_file:object) -> None:
        """ Saves the raw data to a HDF5 file. """

        dgroup = HDF5_file.create_group('Raw_cp_pressure_data')
        dset = dgroup.create_dataset('raw_cp_data', data=self.raw_data.T)
        dset.attrs['Description'] = 'Raw pressure coefficient data'
        dset.attrs['Reference_velocity_height'] = self.recorded_reference_height
        dset.attrs['Wind_angle_of_attack'] = self.wind_angle
        dset.attrs['Raw_data_type'] = self.data_type
        dset.attrs['Raw_data_import_method'] = self.cp_import_method
        dset.attrs['Reference_velocity'] = self.raw_data_config['raw_data_ref_velocity']
        dset.attrs['Reference_air_density'] = self.raw_data_config['raw_data_ref_density']
        dset.attrs['Reference_pressure'] = self.raw_data_config['raw_data_ref_pressure']

    def save_cp_data(self, HDF5_file:object) -> None:
        """ Saves the cp data to a HDF5 file. """

        dgroup = HDF5_file.create_group('Cp_data')
        dset = dgroup.create_dataset('cp_data', data=self.cp_data.T)
        dset.attrs['Description'] = 'Pressure coefficient data'
        dset.attrs['Reference_velocity_height'] = self.new_reference_height
        dset.attrs['Wind_angle_of_attack'] = self.wind_angle
        dset.attrs['velocity_ratio'] = self.velocity_conversion_ratio

    def save_pressure_tap_data(self, HDF5_file:object) -> None:

        # get pressure tap location
        tap_dtype = [('tap_id','i8'),('X','f8'),('Y','f8'),('Z','f8')]
        tributary_area_dtype = [('tap_id','i8'),('tributary_area','f8')]
        tap_data = []
        tributary_area_data = []
        for tap in self.pressure_tap_list:
            tap_data.append((tap.tap_id, tap.x, tap.y, tap.z))
            tributary_area_data.append((tap.tap_id, tap.tributary_area.area))

        
        tap_data = np.array(tap_data, dtype=tap_dtype)

        dgroup = HDF5_file.create_group('Pressure_tap_data')
        dset = dgroup.create_dataset('Pressure_tap_locations', data=tap_data)
        dset.attrs['Description'] = 'Pressure tap locations'
        dset.attrs['scanner_type'] = self.scanner_type
        dset.attrs['scanner_channel_count'] = self.scanner_channel
        dset.attrs['broken_scanner'] = self.broken_scanner
        dset.attrs['sampling_frequency'] = self.scanner_sampling_rate
        dset = dgroup.create_dataset('Missing_taps_id',self.missing_taps)
        
        # get tributary area data
        tributary_area_data = np.array(tributary_area_data,dtype = tributary_area_dtype)
        dset = dgroup.create_dataset('Tributary_area', data = tributary_area_data)
        dset.attrs['Description'] = 'Tributary area of the pressure taps'
        dset.attrs['Units'] = 'm^2'


        # get surface tap data
        surface_group = dgroup.create_group('Surface_tap_ids')
        for surface in self.building.surfaces.values():

            dset = surface_group.create_dataset(f'{surface.name}_tap_list',data = np.array(surface.tap_id_list).T)
            dset.attrs['Description'] = 'Surface Pressure tap ids'

        
