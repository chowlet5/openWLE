import os
import pickle        
import warnings     
import datetime
import pickle
import math
from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import numpy as np
from dataclasses import dataclass

from openWLE.exception import InputError  
from openWLE.climateStudy import ClimateStudy
from openWLE.buildingClass import Building, BuildingSurface
from openWLE.pressureTap import PressureTap
from openWLE.funcs import number_to_list, GeneralFunctions
from openWLE.geometry import plane_intersection, sort_points_clockwise
from openWLE.extreme_blue import extreme_blue



@dataclass
class Visualization:
    label_font_size:int = 20
    title_font_size:int = 28
    line_style:str = 'k'
    line_width:int = 1
    

    def __init__(self) -> None:
         
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 24
        plt.rcParams["text.usetex"] = True

        self.default_data_types = ['pressure_coefficient', 'floor_shear', 'floor_moment', 'base_shear', 'normalized_base_shear', 'base_moment', 'normalized_base_moment', 'displacement', 'acceleration']


    

class VisualizationProcessor:

    default_plot_types = ['settings','contour', 'spectra','time_history', 'line', 'ring', 'climate_bar', 'climate_wind_rose', 'directional']
    default_data_types = ['pressure_coefficient', 'floor_shear', 'floor_moment', 'base_shear', 'base_moment', 'displacement', 'acceleration']
    default_plot_settings = {
        'overwrite': False,
        'save_location': '.',
        'create_save_directory': False,
        'pickle': False
    }
    default_contour_titles = {
        'peak_max':'Peak $C_p$',
        'peak_min':'Peak $C_p$',
        'peak_abs':'Peak $C_p$',
        'mean':'Mean $C_p$',
        'rms':'RMS $C_p$',
    }
    peak_options = ['peak_max', 'peak_min', 'peak_abs']
    def __init__(self,config:list):

        self._wind_AOA = None

        self.settings = []
        self.contour_plots = []
        self.spectra_plots = []
        self.time_history_plots = []
        self.line_plots = []
        self.ring_plots = []
        self.climate_bar_plots = []
        self.climate_wind_rose_plots = []

        self.directional_plots = {}
        self.directional_processor = {}
        # self.group_spectra_plots = {}
        # self.group_time_history_plots = {}
        # self.group_line_plots = {}

        self.overwrite = None
        self.save_location = None
        self.create_save_directory = None
        self.pickle_bool = None

        directional_task_number = 0
        for task in config:

            plot_type = list(task.keys())[0].lower()

            if not plot_type in self.default_plot_types:
                raise InputError(f'"{plot_type}" in self.default_plot_types', f'"{plot_type}" is not an option. Please select on of the following: '+", ".join(self.default_plot_types))

            match plot_type:
                case 'settings':
                    self.overwrite = task['settings']['overwrite'] if 'overwrite' in task['settings'].keys() else self.default_plot_settings['overwrite']
                    self.save_location = task['settings']['save_location'] if 'save_location' in task['settings'].keys() else self.default_plot_settings['save_location']
                    self.create_save_directory = task['settings']['create_save_directory'] if 'create_save_directory' in task['settings'].keys() else self.default_plot_settings['create_save_directory']
                    self.pickle_bool = task['settings']['pickle'] if 'pickle' in task['settings'].keys() else self.default_plot_settings['pickle']
                case 'contour':
                    self.contour_plots.append(list(task.values())[0])
                case 'spectra':
                    self.spectra_plots.append(list(task.values())[0])
                case 'time_history':
                    self.time_history_plots.append(list(task.values())[0])
                case 'line':
                    self.line_plots.append(list(task.values())[0])
                case 'ring':
                    self.ring_plots.append(list(task.values())[0])
                case 'directional':
                    self.directional_plots[directional_task_number] =list(task.values())[0]
                    directional_task_number += 1
                case 'climate_bar':
                    self.climate_bar_plots.append(list(task.values())[0])
                case 'climate_wind_rose':
                    self.climate_wind_rose_plots.append(list(task.values())[0])

        if not self.overwrite:
            self.overwrite = self.default_plot_settings['overwrite']
        if not self.save_location:
            self.save_location = self.default_plot_settings['save_location']
        if not self.create_save_directory:
            self.create_save_directory = self.default_plot_settings['create_save_directory']
        if not self.pickle_bool:
            self.pickle_bool = self.default_plot_settings['pickle']
    
    @property
    def wind_AOA(self) -> float:
        return self._wind_AOA

    @wind_AOA.setter
    def wind_AOA(self,value):
        self._wind_AOA = float(value)

    def get_process_data(self, building_instance:Building,
                           options:dict) -> tuple:

        
        data_type = options['data_type'] if options.__contains__('data_type') else None
        function_type = options['function'] if options.__contains__('function') else None

        time, data = self.get_data_type(building_instance, data_type, options)
        
        if function_type:
            return self.process_data_by_function(data, function_type, options)
        else:
            return time, data


    def get_data_type(self, building_instance:Building, data_type:str, options:dict) -> tuple:
        
        direction = options['direction'].lower() if options.__contains__('direction') else None

        match data_type:
            case 'pressure_coefficient':
                tap_id = options['tap_id'] if options.__contains__('tap_id') else None
                if tap_id:
                    tap_id = [int(tap_id)] if not isinstance(tap_id, list) else tap_id
                taps =  building_instance.get_taps_by_ids(tap_id)
                value = np.array([tap.cp for tap in taps])
                time = np.array([np.arange(len(tap.cp)*tap.sample_frequency) for tap in taps])
            case 'floor_shear':
                
                direction_index = {'x' : 0, 'y' : 1, 'z' : 2}
                floors = options['floors'] if options.__contains__('floors') else None

                if isinstance(floors,str):
                    if floors.lower() == 'top':
                        top_floor_num = max(building_instance.floor_force.keys())
                        floor_shear = building_instance.floor_force[top_floor_num]
                    else:
                        InputError("floors.lower() == 'top'", "Incorrect input. 'top' is the only valid string entry")
                elif floors == None:
                    floor_shear = np.array(building_instance.floor_force.values())

                else:
                    floor_shear = np.array([building_instance.floor_force[floor] for floor in floors])
                
                value = floor_shear[direction_index[direction]]
                time = np.arange(value.shape[-1])*building_instance.force_sampling_frequency

            case 'floor_moment':
                direction_index = {'x' : 0, 'y' : 1, 'z' : 2}
                floors = options['floors'] if options.__contains__('floors') else None

                if isinstance(floors,str):
                    if floors.lower() == 'top':
                        top_floor_num = max(building_instance.floor_moment.keys())
                        floor_moment = building_instance.floor_moment[top_floor_num]
                    else:
                        InputError("floors.lower() == 'top'", "Incorrect input. 'top' is the only valid string entry")
                elif floors == None:
                    floor_moment = np.array(building_instance.floor_moment.values())

                else:
                    floor_moment = np.array([building_instance.floor_moment[floor] for floor in floors])
                
                value = floor_moment[direction_index[direction]]
                time = np.arange(value.shape[-1])*building_instance.force_sampling_frequency

            case 'base_shear':
                match direction:
                    case 'x':
                        value = building_instance.force_x
                    case 'y':
                        value = building_instance.force_y
                    case 'z':
                        value = building_instance.force_z
                    case _:
                        raise InputError(f'direction = {direction}', f'"{direction}" is not a valid direction. Please select one of the following options: x, y, z')
                
                time = np.arange(len(value))*building_instance.force_sampling_frequency
            
            case 'base_moment':
                match direction:
                    case 'x':
                        value = building_instance.moment_x
                    case 'y':
                        value = building_instance.moment_y
                    case 'z':
                        value = building_instance.moment_z
                    case _:
                        raise InputError(f'direction = {direction}', f'"{direction}" is not a valid direction. Please select one of the following options: x, y, z')
                time = np.arange(len(value))*building_instance.force_sampling_frequency
            
            case 'displacement':
                floors = options['floors'] if options.__contains__('floors') else None
                direction_index = {'x' : 0, 'y' : 1, 'z' : 2}
                floor_displacement = []
                if isinstance(floors,str):
                    if floors.lower() == 'top':
                        floor_displacement = building_instance.displacement_time_history[-1]
                    else:
                        InputError("floors.lower() == 'top'", "Incorrect input. 'top' is the only valid string entry")
                elif floors == None:
                    floor_displacement = building_instance.displacement_time_history
                else:
                    floor_displacement = building_instance.displacement_time_history[floors]
                value = floor_displacement[direction_index[direction]] 
                time = np.arange(value.shape[-1])/building_instance.force_sampling_frequency
            case 'acceleration':
                            
                floors = options['floors'] if options.__contains__('floors') else None
                direction_index = {'x' : 0, 'y' : 1, 'z' : 2}
                if isinstance(floors, str):
                    if floors.lower() == 'top':
                        floor_acceleration = building_instance.acceleration_time_history[-1]
                    else:
                        InputError("floors.lower() == 'top'", "Incorrect input. 'top' is the only valid string entry")
                elif floors == None:
                    floor_acceleration = building_instance.acceleration_time_history
                else:
                    floor_acceleration = building_instance.acceleration_time_history[floors]

                value = floor_acceleration[direction_index[direction]]
                time = np.arange(value.shape[-1])/building_instance.force_sampling_frequency
            case _:
                raise InputError(f'datatype = {data_type}',f'"{data_type}" is not a valid option. Please provide a valid data_type: cp, floor_shear, floor_moment, base_shear, base_moment, displacement, acceleration' )
            
        return time, value

    def process_extreme_peak(self, data:np.ndarray,
                                probablity_non_exceedence:list,
                                max_peak:bool = True,
                                peak_epoches:int = 10,
                                duration_ratio:float = None) -> np.ndarray:
        duration = int(len(data) - math.remainder(len(data),peak_epoches))
        segments = np.array_split(data[:duration],peak_epoches)
        if max_peak:
            annual_maxima = np.max(segments,axis=1)
            peak_value, _, _ = extreme_blue(annual_maxima, probablity_non_exceedence, 'max', duration_ratio)
        else:
            annual_maxima = np.min(segments,axis=1)
            peak_value, _, _ = extreme_blue(annual_maxima, probablity_non_exceedence, 'min', duration_ratio) 
       
        return peak_value

    def process_data_by_function(self, data:np.ndarray, function:str, options:dict) -> np.ndarray:

        if options.__contains__('starting_index'):
            starting_index = int(options['starting_index'])
            data = data[...,starting_index:]

        if options.__contains__('MRI'):
            mri = np.array(number_to_list(options['MRI']))
            probablity_non_exceedence = 1 - 1/mri
        else:
            probablity_non_exceedence = 0.98

        match function:
            case 'mean':
                return np.mean(data, axis = -1)
            case 'std':
                return np.std(data, axis = -1)
            case 'max':
                return np.max(data, axis = -1)
            case 'min':
                return np.min(data, axis = -1)
            case 'peak_max':
                return self.process_extreme_peak(data,probablity_non_exceedence,True, **options)
            case 'peak_min':
                return self.process_extreme_peak(data,probablity_non_exceedence,False, **options)
            case 'peak_abs':
                max_values = self.process_extreme_peak(data,probablity_non_exceedence,True, **options)
                min_values = self.process_extreme_peak(data,probablity_non_exceedence,False, **options)

                return np.where(np.abs(max_values) >= np.abs(min_values), max_values, min_values)
            case _:
                raise ValueError(f'Invalid function: {function}. Valid options are: mean, max, min, std, peak_max, peak_min, peak.')


    def process_output(self,
                       fig:plt.Figure, 
                       ax:plt.Axes, 
                       file_path:str,
                       pickle_bool:bool,
                       save_fig) -> None|tuple:
        """
        Saves the figure and axes to a file.
        """

        if pickle_bool:

            pickle.dump((fig,ax),open(f'{file_path}.pickle','wb'))

        if save_fig:
            fig.savefig(f'{file_path}.png',dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        
        return fig, ax


    def process_file_save(self, plot_name:str, name_suffix:str, task_config:dict) -> str:

        if  task_config.__contains__('save_directory'):
            save_directory_path = os.path.join(self.save_location, task_config['save_directory'])
            # save_directory_path = f'{self.save_location}/{task_config["save_directory"]}'
        else:
            save_directory_path = os.path.join(self.save_location,'')
            # save_directory_path = f'{self.save_location}/'
        if self.create_save_directory:

            if os.path.isdir(save_directory_path) and self.overwrite:
                os.makedirs(save_directory_path,exist_ok=True)
            elif not os.path.isdir(save_directory_path):
                os.makedirs(save_directory_path,exist_ok=True)
            else:
                pass
        
        file_path = f'{save_directory_path}/{plot_name}_{name_suffix}'

        return file_path

    
    def process_contour_plots(self, name_suffix:str,
                              building_instance:Building,
                              task_config:dict, 
                              save_fig:bool = True) -> None:
        
        if task_config.__contains__('wind_AOA'):
            accepted_AOA = number_to_list(task_config['wind_AOA'])
            if self.wind_AOA not in accepted_AOA:
                return
        contour_plot = SurfaceBaseVisualization()
        if task_config.__contains__('plot_options'):
            contour_plot.update_settings(**task_config['plot_options'])

        contour_func = task_config['function']
        colorbar_label = self.default_contour_titles[contour_func.lower()]

        surface_list = contour_plot.get_surfaces_by_name(building_instance, task_config)

        analysis_options = {}
        if task_config.__contains__('starting_index'):
            analysis_options['starting_index'] = int(task_config['starting_index'])

        if contour_func in self.peak_options:
            
            if task_config.__contains__('peak_mri'):
                analysis_options['mri'] = float(task_config['peak_mri'])
            if task_config.__contains__('peak_epoches'):
                analysis_options['peak_epoches'] = float(task_config['peak_epoches'])
            if task_config.__contains__('duration_ratio'):
                analysis_options['duration_ratio'] = float(task_config['duration_ratio'])
            if task_config.__contains__('grid_x'):
                analysis_options['grid_x'] = float(task_config['grid_x'])
            if task_config.__contains__('grid_y'):
                analysis_options['grid_y'] = float(task_config['grid_y'])
            
        fig, ax = contour_plot.plot_contour(surface_list,contour_func,analysis_options,colorbar_label)

        file_path = self.process_file_save('contour_', name_suffix, task_config)
        pickle_bool = task_config['pickle'] if task_config.__contains__('pickle') else self.pickle_bool
        return self.process_output(fig, ax, file_path, pickle_bool, save_fig)




    def process_ring_plots(self, name_suffix:str,
                              building_instance:Building,
                              task_config:dict,
                              save_fig:bool = True
                              ) -> None|tuple:
    
        if task_config.__contains__('wind_AOA'):
            accepted_AOA = number_to_list(task_config['wind_AOA'])
            if self.wind_AOA not in accepted_AOA:
                return

        if task_config.__contains__('function'):
            function = task_config['function']
        else:
            raise InputError("task_config.__contains__('function')", '"function" option missing for ring plot. Please include the "function" option within the configuration file')


        if function in self.peak_options:
            if task_config.__contains__('peak_mri'):
                if not isinstance(task_config['peak_mri'], list):
                    task_config['non_exceedance'] = [1 - 1/float(task_config['peak_mri'])]
                else:
                    task_config['non_exceedance'] = [1 - 1/float(non_exceed) for non_exceed in task_config['peak_mri']]

        
        ring_plot = HorizontalRingBaseVisualization()
        
        name, ring_position, ring_data = ring_plot.get_ring_data(building_instance,**task_config)

        name = 'ring_plot_' + name


        if task_config.__contains__('plot_options'):
            fig, ax = ring_plot.plot_ring_data(ring_position, ring_data, **task_config['plot_options'])
        else:
            fig, ax = ring_plot.plot_ring_data(ring_position, ring_data)

        file_path = self.process_file_save(name, name_suffix,task_config)
        pickle_bool = task_config['pickle'] if task_config.__contains__('pickle') else self.pickle_bool

        return self.process_output(fig, ax, file_path, pickle_bool, save_fig)

    def spectra_plot_helper(self, spectra:np.ndarray, frequency:np.ndarray, task_config:dict) -> tuple:

        point_plot = PointBaseVisualization()

        if task_config.__contains__('x_axis_factor'):
            frequency = frequency * task_config['x_axis_factor'] 

        if task_config.__contains__('plot_options'):
            fig,ax = point_plot.spectra_plot(spectra,frequency,**task_config['plot_options'])
        else:
            fig,ax = point_plot.spectra_plot(spectra,frequency)

        return fig, ax
        
    def process_spectra_plots(self, name_suffix:str, 
                              building_instance:Building, 
                              task_config:dict, 
                              save_fig:bool = True) -> None|tuple:

        if task_config.__contains__('wind_AOA'):
            accepted_AOA = number_to_list(task_config['wind_AOA'])
            if self.wind_AOA not in accepted_AOA:
                return

        if not task_config.__contains__('data_type'):
            raise InputError("task_config.__contains__('data_type')", '"data_type" option missing for spectra plot. Please include the "data_type" option within the configuration file')
        else:
            data_type = task_config['data_type'].lower()


        point_plots = PointBaseVisualization()

        name, time, data, sampling_freq = point_plots.get_point_data(building_instance, data_type, task_config)
        name = 'spectra_' + name

        
        spectra, f = GeneralFunctions().spectral_density(data,sampling_freq)

        if task_config.__contains__('reduced_spectra') and task_config['reduced_spectra']:
            spectra = f*spectra/np.var(data)
        
        fig , ax = self.spectra_plot_helper(spectra, f, task_config)

        file_path = self.process_file_save(name, name_suffix,task_config)
        pickle_bool = task_config['pickle'] if task_config.__contains__('pickle') else self.pickle_bool

        return self.process_output(fig, ax, file_path, pickle_bool, save_fig)

    def time_history_plot_helper(self, time_history:np.ndarray, time:np.ndarray, task_config:dict,) -> tuple:
        

        point_plot = PointBaseVisualization()
        if task_config.__contains__('plot_options'):
            fig,ax= point_plot.time_history_plot(time_history,time,None,**task_config['plot_options'])
        else:
            fig,ax= point_plot.time_history_plot(time_history,time,None,)

        return fig, ax
        

    def process_time_history_plots(self, name_suffix:str, 
                                   building_instance:Building, 
                                   task_config:dict,
                                   save_fig:bool = True  ) -> None|tuple:
        
        if task_config.__contains__('wind_AOA'):
            accepted_AOA = number_to_list(task_config['wind_AOA'])
            if self.wind_AOA not in accepted_AOA:
                return

        if not task_config.__contains__('data_type'):
            raise InputError("task_config.__contains__('data_type')", '"data_type" option missing for time history plot. Please include the "data_type" option within the configuration file')
        data_type = task_config['data_type'].lower()

        point_plot = PointBaseVisualization()
        name, time, time_history, _ = point_plot.get_point_data(building_instance, data_type, task_config)

        name = 'time_history_' + name

        file_path = self.process_file_save(name, name_suffix, task_config)

        fig, ax = self.time_history_plot_helper(time_history, time, task_config)

        pickle_bool = task_config['pickle'] if task_config.__contains__('pickle') else self.pickle_bool
        return self.process_output(fig, ax, file_path, pickle_bool, save_fig)


    def process_directional_plots(self,name_suffix:str,
                                  building_instance:Building,
                                  task_id: int,
                                  task_config:dict,
                                  final_data_point:bool = False,
                                  save_fig:bool = True) -> None|tuple:
        
        if task_id not in self.directional_processor.keys():
            self.directional_processor[task_id] = DirectionalBaseVisualization()

        if task_config.__contains__('skip_AOA'):
            skip_AOA = number_to_list(task_config['skip_AOA'])
            if self.wind_AOA not in skip_AOA:
                data = self.get_process_data(building_instance, task_config)
                self.directional_processor[task_id].store_data(self.wind_AOA, data)
        else:
            data = self.get_process_data(building_instance, task_config)
            self.directional_processor[task_id].store_data(self.wind_AOA, data)
        
        if final_data_point:
            direction, data = self.directional_processor[task_id].get_data()
            plot_type = task_config['plot_type'].lower() if task_config.__contains__('plot_type') else None
            plot_options = task_config['plot_options'] if task_config.__contains__('plot_options') else {}
            match plot_type:
                case 'polar':
                    fig, ax = self.directional_processor[task_id].plot_polar(direction, data, **plot_options)
                # case 'bar': #TODO: Add bar plot options
                #     fig, ax = self.directional_processor[task_id].plot_bar()

                case _: # default option is line plot
                    fig, ax = self.directional_processor[task_id].plot_line(direction, data, **plot_options)
                

            name = f'directional_plot_{task_id}'

            file_path = self.process_file_save(name, name_suffix, task_config)
            pickle_bool = task_config['pickle'] if task_config.__contains__('pickle') else self.pickle_bool
            return self.process_output(fig, ax, file_path, pickle_bool, save_fig)


        


class ClimateVisualizationProcessor(Visualization):

    def __init__(self, config:dict, save_location:str, overwrite:bool, climate_study:ClimateStudy):
        self.config = config
        self.save_location = save_location
        self.overwrite = overwrite
        self.climate_study = climate_study

        os.makedirs(save_location, exist_ok = overwrite)

        for key, value in config.items():
            os.makedirs(f"{save_location}/{key}")

            match value['type'].lower():

                case 'time_history':
                    self.time_history_config(value)
                case 'wind_rose':
                    self.wind_rose_plot(value)

    def id_check(self,id_config:list|str):

        if isinstance(id_config,str):
            if id_config.lower() != 'all':
                warnings.warn("Climate visualisation id list include str value not equal to all. Other str entries are unavaliable. All will be used.")
            return self.climate_study.stations_ids
        else:
            unknown_station_id = [str(station_id) for station_id in id_config if not station_id in self.climate_study.stations_ids]
            unknown_station = ','.join(unknown_station_id)
            warnings.warn(f"{unknown_station} were not processed by the climate study class. Include these stations in the yml configuration file.")

            return [station_id for station_id in id_config if station_id in self.climate_study.stations_ids]
            
    def time_history_config(self,config:dict) -> None:
        
        # required parameters
        self.processing_ids = self.id_check(config['id'])

        
        if 'start_date' in config.keys():
            self.start_date = [datetime.strptime(config['start_date'],'%Y-%m-%d')]*len(self.processing_ids)
        else:
             self.start_date = self.climate_study.start_dates
        if 'end_date' in config.keys():
             self.end_date = [datetime.strptime(config['end_date'],'%Y-%m-%d')]*len(self.processing_ids)
        else:
            self.end_date = self.climate_study.end_dates
        
        self.plot_time_history()


    def plot_time_history(self):
        pass

    def wind_rose_config(self,config:dict) -> None:
        self.processing_ids = self.id_check(config['id'])

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


# Point Base Vis 

class PointBaseVisualization(Visualization):     

    def __init__(self):
        super().__init__()

        
    def get_point_data(self, building_instance:Building, 
                       data_type:str, 
                       task_config:dict) -> tuple:
    

        direction = task_config['direction'].lower() if task_config.__contains__('direction') else None

        normalize_factor = task_config['normalize_factor'] if task_config.__contains__('normalize_factor') else 1

        # data = self.get_data_type(building_instance, data_type, task_config) # New centralized data collection/processing system

        #TODO Move data collection to new system
        match data_type:
            case 'pressure_coefficient': #TODO allow for multiple tap ids
                tap_id = number_to_list(task_config['tap_id'])
                for tap in building_instance.get_taps_by_ids(tap_id):
                    if tap:
                        name = f'tap_{tap.tap_id}'
                        time = np.arange(len(tap.cp))/tap.sampling_rate
                        return (name, time, tap.cp, tap.sampling_rate)
                        

            case 'floor_shear':
                floor = task_config['floor'].lower() if task_config.__contains__('floor') else -1
                match direction:
                    case 'x':
                        force = building_instance.floor_forces[floor][0]
                    case 'y':
                        force = building_instance.floor_forces[floor][1]
                    case 'z':
                        force = building_instance.floor_forces[floor][2]
                name = f'floor_{floor}_direction_{direction}'
                time = np.arange(len(force))/building_instance.force_sampling_frequency
                return (name, time, force, building_instance.force_sampling_frequency)
            case 'floor_moment':
                floor = task_config['floor'].lower() if task_config.__contains__('floor') else -1
                
                match direction:
                    case 'x':
                        force = building_instance.floor_moments[floor][0]
                    case 'y':
                        force = building_instance.floor_moments[floor][1]
                    case 'z':
                        force = building_instance.floor_moments[floor][2]
                name = f'floor_{floor}_direction_{direction}'
                time = np.arange(len(force))/building_instance.force_sampling_frequency
                return (name, time, force, building_instance.force_sampling_frequency)

            case 'base_shear':

                match direction:
                    case 'x':
                        force = building_instance.force_x
                    case 'y':
                        force = building_instance.force_y
                    case 'z':
                        force = building_instance.force_z
                    case _:
                        raise InputError(f'direction = {direction}', f'"{direction}" is not a valid direction. Please select one of the following options: x, y, z')
                        
                name = f'base_shear_{direction}'
                time = np.arange(len(force))/building_instance.force_sampling_frequency

                return (name, time, force, building_instance.force_sampling_frequency)

            case 'normalized_base_shear':
                direction = task_config['direction'].lower()
                
                match direction:
                    case 'x':
                        force = building_instance.force_x
                    case 'y':
                        force = building_instance.force_y
                    case 'z':
                        force = building_instance.force_z
                    case _:
                        raise InputError(f'direction = {direction}', f'"{direction}" is not a valid direction. Please select one of the following options: x, y, z')
                        
                name = f'normalized_base_shear_{direction}'
                time = np.arange(len(force))/building_instance.force_sampling_frequency

                return (name, time, force/normalize_factor, building_instance.force_sampling_frequency)
                
            case 'base_moment':
                match direction:
                    case 'x':
                        moment = building_instance.moment_x
                    case 'y':
                        moment = building_instance.moment_y
                    case 'z':
                        moment = building_instance.moment_z
                    case _:
                        raise InputError(f'direction = {direction}', f'"{direction}" is not a valid direction. Please select one of the following options: x, y, z')
                        
                name = f'base_moment_{direction}'
                time = np.arange(len(moment))/building_instance.force_sampling_frequency

                return (name, time, moment, building_instance.force_sampling_frequency)
            
            case 'normalized_base_moment':
                match direction:
                    case 'x':
                        moment = building_instance.moment_x
                    case 'y':
                        moment = building_instance.moment_y
                    case 'z':
                        moment = building_instance.moment_z
                    case _:
                        raise InputError(f'direction = {direction}', f'"{direction}" is not a valid direction. Please select one of the following options: x, y, z')
                
                name = f'normalized_base_moment_{direction}'
                time = np.arange(len(moment))/building_instance.force_sampling_frequency
                return (name, time, moment/normalize_factor, building_instance.force_sampling_frequency)
            
            case 'displacement':
                floors = task_config['floors'] if task_config.__contains__('floors') else 'top'
                x_axis_factor = task_config['x_axis_factor'] if task_config.__contains__('x_axis_factor') else 1
                radius_gyration = task_config['radius_gyration'] if task_config.__contains__('radius_gyration') else 1
                starting_index = task_config['starting_index'] if task_config.__contains__('starting_index') else 0
                direction_index = {'x' : 0, 'y' : 1, 'z' : 2}
                if isinstance(floors,str):
                    if floors.lower() == 'top':
                        floor_displacement = building_instance.displacement_time_history[-1]
                        name = f'top_displacement_{direction}'
                    elif floors == 'all':
                        floor_displacement = building_instance.displacement_time_history
                        name = f'displacement_{direction}'
                    else:
                        InputError("floors.lower() == 'top'", "Incorrect input. 'top' is the only valid string entry")

                else:
                    floor_displacement = building_instance.displacement_time_history[floors]
                    floor_string = '_'.join([str(f) for f in floors])
                    name = f'displacement_floor_{floor_string}_{direction}'

                
                directional_displacement = floor_displacement[direction_index[direction]] * radius_gyration
                time = np.arange(directional_displacement.shape[-1]) * building_instance.structural_analysis_time_step * x_axis_factor
                directional_displacement = directional_displacement[..., starting_index:]
                time = time[...,starting_index:]
                sampling_frequency = 1/building_instance.structural_analysis_time_step
                
                return (name, time, directional_displacement, sampling_frequency)

            case 'acceleration':
                floors = task_config['floors'] if task_config.__contains__('floors') else 'top'
                x_axis_factor = task_config['x_axis_factor'] if task_config.__contains__('x_axis_factor') else 1
                starting_index = task_config['starting_index'] if task_config.__contains__('starting_index') else 0
                radius_gyration = task_config['radius_gyration'] if task_config.__contains__('radius_gyration') else 1
                direction_index = {'x' : 0, 'y' : 1, 'z' : 2}
                if isinstance(floors, str):
                    if floors.lower() == 'top':
                        floor_acceleration = building_instance.acceleration_time_history[-1]
                        name = f'top_floor_acceleration_{direction}'
                    elif floors == 'all':
                        floor_acceleration = building_instance.acceleration_time_history
                        name = f'acceleration_{direction}'
                    else:
                        InputError("floors.lower() == 'top'", "Incorrect input. 'top' is the only valid string entry")
                else:
                    floor_acceleration = building_instance.acceleration_time_history[floors]
                    floor_string = '_'.join([str(f) for f in floors])
                    name = f'acceleration_floor_{floor_string}_{direction}'

                directional_acceleration = floor_acceleration[direction_index[direction]] * radius_gyration
                time = np.arange(directional_acceleration.shape[-1]) * building_instance.structural_analysis_time_step * x_axis_factor
                directional_acceleration = directional_acceleration[..., starting_index:]
                time = time[...,starting_index:]
                sampling_frequency = 1/building_instance.structural_analysis_time_step
                
                return (name, time, directional_acceleration, sampling_frequency)

                
            
            case _:
                raise InputError(f'"{data_type}" in self.default_data_types', f'"{data_type}" is not a valid option. Please select one of the following options: ' + ", ".join(self.default_data_types) )
        
    
    def spectra_label(self, series_symbol:str, normalized:bool = False):
        if normalized:
            return r'$\frac{fS_{}(f)} {C_{\sigma {p}}}$'.format(series_symbol)
        else:
            return r'$fS_{}(f)$'.format(series_symbol)
        
    def frequency_label(self, dimension_symbol:str, reduced:bool = False):
        
        if reduced:
            return r'$f{}/\overline{U}_{H}$'.format(dimension_symbol)
        else:
            return r'$f$'

    def spectra_plot(self, 
                     spectra:np.ndarray,
                     frequency:np.ndarray,
                     line_style:str = 'k-',
                     fig_size = (6,6),
                     value_lims:tuple = None,
                     xlabel:str = None,
                     ylabel:str = None
                     ) -> plt.figure:
    
        plt.close('all')
        fig, ax = plt.subplots(1,1,figsize = fig_size)
        ax.loglog(frequency,spectra,line_style)
        ax.grid(which='both', linestyle=':')

        if value_lims:
            ax.set_xlim(value_lims[0])
            ax.set_ylim(value_lims[1])

        if xlabel:
            ax.set_xlabel(xlabel,fontsize = self.label_font_size)
        if ylabel:
            ax.set_ylabel(ylabel,fontsize = self.label_font_size)

        return fig, ax


    def time_history_plot(self, time_history:np.ndarray,
                          time:np.ndarray = None,
                          sampling_rate:float = None,
                          fig_size:tuple = (10,4), 
                          line_style:str = 'k-',
                          value_lims:tuple = None,
                          xlabel:str = None, 
                          ylabel:str = None ) -> tuple:
        
        if time is None:
            time = np.arange(len(time_history))
            if sampling_rate != None:
                time /= sampling_rate

        fig, ax = plt.subplots(1,1,figsize = fig_size)

        ax.plot(time, time_history,line_style)
        ax.grid()

        if value_lims:
            ax.set_xlim(value_lims[0])
            ax.set_ylim(value_lims[1])

        if xlabel:
            ax.set_xlabel(xlabel, fontsize = self.label_font_size)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize = self.label_font_size)

        ax.tick_params(axis='x', length = 10, width = 2, labelsize=self.label_font_size)
        ax.tick_params(axis='y', length = 10, width = 2, labelsize=self.label_font_size)

        return fig, ax



# class WindDirectionBaseVisualization(Visualization):

class HorizontalRingBaseVisualization(Visualization):

    
    def plot_ring_data(self, position:np.ndarray, 
                       data:np.ndarray,
                       fig_size:tuple = (10,6),
                       line_style = '-k',
                       line_label = None,
                       value_lims:tuple = None,
                       xlabel:str = None, 
                       ylabel:str = None,
                       alphabetical_x_axis = False,
                       **_  ) -> tuple:

        fig, ax = plt.subplots(1,1,figsize = fig_size)

        ax.plot(position, data, line_style, label = line_label)
        ax.grid()

        if value_lims:
            if value_lims[0]:
                ax.set_xlim(value_lims[0])
            else:
                ax.set_xlim(min(position), max(position))
            ax.set_ylim(value_lims[1])
        else:
            ax.set_xlim(min(position), max(position))

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        ax.tick_params(axis='x', length = 10, width = 2, labelsize=self.label_font_size)
        ax.tick_params(axis='y', length = 10, width = 2, labelsize=self.label_font_size)

        if alphabetical_x_axis:
            self.set_alphabetical_x_axis(ax)

        return fig, ax



    def sort_taps(self, taps:List[PressureTap], clockwise:bool = True) -> List[PressureTap]:

        points = [tap.coords for tap in taps]
        points = np.array(points)[:,0:-1]

        c = points.mean(axis=0)
        angles = np.arctan2(points[:,1] - c[1], points[:,0] - c[0])
        sorted_taps = list(np.array(taps)[np.argsort(angles)])
        
        if clockwise:
            sorted_taps.reverse()
        
        return sorted_taps

    def get_taps_by_elevation(self, 
                              building_instance:Building, 
                              elevation:float, 
                              tolerance:float = 0.01) -> List[PressureTap]:
        """
        Get taps by elevation within a specified tolerance.
        """
        taps = []
        for tap in building_instance.pressure_tap_list:
            if abs(tap.z - elevation) <= tolerance:
                taps.append(tap)
        
        return taps
    
    def get_taps_by_ring_number(self,
                                building_instance:Building,
                                ring_number:int,
                                tolerance:float = 0.01) -> List[PressureTap]:

        """
        Gets all the pressure taps associated with a specific horizontal ring of pressure taps
        """

        rings = building_instance.get_horizontal_tap_ring_elevations(tolerance)

        elevation = rings[ring_number-1] if ring_number-1 < len(rings) else None

        if elevation is None:
            raise InputError(f'ring_number = {ring_number}', f'No horizontal ring with number {ring_number} found. Please select a ring number between 1 and {len(rings)}')
        
        taps = self.get_taps_by_elevation(building_instance, elevation, tolerance)

        return taps, elevation 

    def organize_tap_order(self, tap_list:List[PressureTap],
                           starting_surface:str = None, 
                           direction:str = 'clockwise') -> List[PressureTap]:
        """
        Organizes the taps in a clockwise or counter-clockwise order starting with a specific surface.
        """
        if tap_list is None:
            raise InputError('tap_list = None', 'tap_list must be provided to organize taps')
        
        if direction.lower() not in ['clockwise', 'counter-clockwise']:
            raise InputError(f'direction = {direction}', 'direction must be either "clockwise" or "counter-clockwise"')

        sorted_taps = self.sort_taps(tap_list)
        
        if starting_surface: 
            while (sorted_taps[-1].surface == starting_surface.lower() or ( not sorted_taps[0].surface == starting_surface.lower()) ):
                sorted_taps = sorted_taps[-1:] + sorted_taps[:-1]

        return sorted_taps
        
    def get_surface_bounds(self, elevation:float, surface:BuildingSurface) -> tuple:

        elevation_plane = (0, 0, 1, -elevation)

        intersection_line =  plane_intersection(surface.polygon.plane_equation(), elevation_plane)
        points = surface.polygon.intersection([intersection_line])


        points = np.array(surface.polygon.local_to_global(points))[:,0:-1]

        points = sort_points_clockwise(points,(0,0))

        return points
    
    def get_point_distance(self, point:np.ndarray,
                                start_point:np.ndarray) -> float:

        return np.sqrt((point[1] - start_point[1])**2 + (point[0] - start_point[0])**2)

    def set_alphabetical_x_axis(self, ax:plt.Axes) -> plt.Axes:
        
        corner_position = []
        corner_name = []
        position = 0
        for index, surface in enumerate(self.surfaces):
            corner_name.append(chr(index+65))
            corner_position.append(position)
            line = self.get_surface_bounds(self.elevation, surface)
            distance = self.get_point_distance(line[1],line[0])
            position += distance

        corner_name.append('A')
        corner_position.append(position)
        ax.set_xticks(corner_position,corner_name)
        ax.set_xlim((min(corner_position), max(corner_position)))

    def get_tap_positions(self, tap_list:List[PressureTap], 
                          building_instance:Building, 
                          elevation:float,
                          starting_surface:str = None) -> np.ndarray:
        """
        Gets the tap positions along surfaces
        """
        line = self.get_surface_bounds(elevation,building_instance.surfaces[starting_surface])
        starting_distance = 0
        tap_positions = np.zeros((len(tap_list)))
        for index, tap in enumerate(tap_list):
            if not (tap.surface == starting_surface):
                starting_surface = tap.surface
                starting_distance += self.get_point_distance(line[1],line[0])
                line = self.get_surface_bounds(elevation,building_instance.surfaces[starting_surface])
            tap_pos = tap.coords[0:-1]
            tap_distance = self.get_point_distance(tap_pos,line[0])
            tap_positions[index] = tap_distance + starting_distance

        return tap_positions
    
    def get_tap_data(self, tap_list:List[PressureTap],
                     function:str,
                     starting_index:int = 0,
                     non_exceedance:list = [0.8],
                     peak_epoches:int = 16,
                     duration_ratio = None,
                     **_) -> np.ndarray:

        tap_data = np.zeros(len(tap_list))

        for index, tap in enumerate(tap_list):
            tap_data[index] = tap.get_data_type(function, starting_index=starting_index,non_exceedance = non_exceedance, peak_epoches = peak_epoches, duration_ratio = duration_ratio)
        
        return tap_data


    def get_ring_data(self, 
                        building_instance:Building, 
                        function:str,
                        elevation:float = None,
                        ring_number:int = None,
                        starting_index:int = 0,
                        non_exceedance:list = [0.8],
                        peak_epoches:int = 16,
                        starting_surface:str = None,
                        duration_ratio = None,
                        **_) -> tuple:

        if ring_number is None and elevation is None:
            raise InputError('ring_number = None and elevation = None', 'Either ring_number or elevation must be provided to get ring data')
        
        if elevation:
            taps, elevation = self.get_taps_by_elevation(building_instance, ring_number)
        else:
            taps, elevation = self.get_taps_by_ring_number(building_instance, ring_number)
        taps = self.organize_tap_order(taps,starting_surface)

        tap_positions = self.get_tap_positions(taps, building_instance, elevation, starting_surface)


        tap_data = self.get_tap_data(taps,function,starting_index,non_exceedance,peak_epoches,duration_ratio)

        
        surfaces = [building_instance.surfaces[tap.surface] for tap in taps]
        self.surfaces = list(dict.fromkeys(surfaces)) 
        self.elevation = elevation
        
        return f'{function.lower()}_{np.round(elevation,2)}', tap_positions, tap_data
        
        

class DirectionalBaseVisualization(Visualization):

    def __init__(self):
        super().__init__()
        self.directional_data = {}
        self.plot_type = None


    def store_data(self,direction:float, data:float|list) -> None:
        self.directional_data[direction] = data

    def get_data(self) -> tuple:

        direction = list(self.directional_data.keys())
        direction.sort()
        data = np.array([self.directional_data[key] for key in direction])

        return np.array(direction), data

    def get_directional_data(self, building_instance:Building,
                             task_config:dict) -> tuple:

        if task_config.__contains__('function'):
            function = task_config['function'].lower()
        else:
            raise ValueError(f'Missing function type. Please provide a valid function type: mean, max, min, std, peak_max, peak_min, peak_abs, peak_gust.')

        if task_config.__contains__('data_type'):
            data_type = task_config['data_type'].lower()
        else:
            raise ValueError(f'Missing data_type. Please provide a valid data_type: cp, floor_shear, floor_moment, base_shear, base_moment, displacement, acceleration')


    def plot_polar(self,direction:np.ndarray,
                        data:np.ndarray,
                        fig_size:tuple = (6,6),
                        line_style ='-k',
                        line_label = None,
                        clockwise_direction:bool = True,
                        start_direction:str = 'W',
                        inner_circle_size:float = 0.2,
                        value_lims:tuple = None,
                        y_tick_number = 5,
                        xlabel:str = None,
                        ylabel:str = None,) -> tuple:

        theta_rads = np.deg2rad(direction)
        fig, ax = plt.subplots(1,1,figsize = fig_size, subplot_kw={'projection': 'polar'})

        ax.plot(theta_rads, data, line_style, label = line_label)
        if clockwise_direction:
            ax.set_theta_direction(-1)

        ax.set_theta_zero_location(start_direction)

        

        data_size = None

        if value_lims:
            if isinstance(value_lims,str):
                ax.set_thetalim(thetamin = min(direction),thetamax = max(direction))
            else:
                ax.set_thetalim(thetamin = value_lims[0][0], thetamax = value_lims[0][1] )
                ax.set_rlim(value_lims[1])
                if value_lims[1][0] and value_lims[1][1]:
                    data_size = max(value_lims[1]) - min(value_lims[1])
        
        if data_size == None:
            data_size = max(data) - min(data)

        ax.set_rorigin(-data_size*inner_circle_size)
        
        ticks = ax.get_yticks()
        ticks = np.linspace(min(ticks),max(ticks),y_tick_number,endpoint=True)
        ax.set_yticks(ticks)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize = self.label_font_size, labelpad = 18)
            ax.set_rlabel_position(90)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize = self.label_font_size )

        ax.set_thetagrids(direction)
        
        return fig, ax
        
    def plot_bar(self) -> tuple:
        pass

    def plot_line(self,direction:np.ndarray,
                        data:np.ndarray,
                        fig_size:tuple = (6,6),
                        line_style ='-k',
                        line_label = None,
                        value_lims:tuple = None,
                        xlabel:str = None,
                        ylabel:str = None,) -> tuple:


        
        fig, ax = plt.subplots(1,1,figsize = fig_size,subplot_kw={'projection': 'polar'})

        
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize = self.label_font_size)
            
        if ylabel:
            ax.set_ylabel(ylabel, fontsize = self.label_font_size)

        ax.grid()
        ax.plot(direction, data, line_style, label = line_label)

        return fig, ax


        

        



        




class LineBaseVisualization(Visualization):


    def __init__(self, tap_list:List[PressureTap], 
                 line_data_type:str = 'mean',
                 line_coordinates: List[tuple] = None,
                 line_position_axis = 'y',
                 tap_position: list = None) -> None:

        self.data = None
        self.line_data_type = line_data_type


        if tap_position == None:
            plane = define_line_plane()
            tap_position

    
    def get_intersection_line(self, intersection_plane, surface_list:List[BuildingSurface]) -> tuple:

        pass
    # def tap_position(self, plane_direction) -> np.ndarray:

    #     if self.line_postion_axis.lower() =='x':
    #         tap_coord = self.tap_coordinates[:,0]
    #         line_coords = np.array(line_coordinates)[:,0]
    #     elif self.line_postion_axis.lower() =='y':
    #         tap_coord = self.tap_coordinates[:,1]
    #         line_coords = np.array(line_coordinates)[:,1]
    #     elif self.line_postion_axis.lower =='z':
    #         tap_coord = self.tap_coordinates[:,2]
    #         line_coords = np.array(line_coordinates)[:,2]
    #     else:
    #         InputError('line_postion_axis == x,y,z', 'line_potion_axis must be x, y or z')

    def define_line_plane(self) -> str:

        line_coords = np.array(self.line_coordinates)

        check_x = line_coords[0,:] - line_coords[0,:].mean() 
        check_y = line_coords[1,:] - line_coords[1,:].mean() 
        check_z = line_coords[2,:] - line_coords[2,:].mean() 

        if np.all(check_x):
            return 'yz'
        elif np.all(check_y):
            return 'xz'
        elif np.all(check_z): 
            return 'xy'
        else:
            InputError('define_line_plane()',' line_cooridnates should plane' )


    def get_line_data(self) -> None:

        self.collected_data = 0





# Surface Base Vis 

class SurfaceBaseVisualization(Visualization):

    def __init__(self):
        super().__init__()
        self.fig_size = (20,15)
        self.contour_lines = True
        self.plot_taps = True
        self.print_tap_id = False
        self.tap_font_size = 5
        self.apply_surface_title = True
        self.contour_number = 20
        self.contour_line_number = 5
        self.contour_line_color = 'k'
        self.contour_font_size = 22
        self.contour_color_map = cm.jet
        self.value_lims = None



    def get_surfaces_by_name(self, building_instance:Building, task_config:dict) -> List[BuildingSurface]:

        if not task_config.__contains__('surface_list'):
            surface_list = []
            for surface in building_instance.surfaces.values():
                if np.all(np.abs(np.cross(surface.normal_vector, np.array((0,0,1))))< 1e-5):
                    continue
                else:
                    surface_list.append(surface)
        else:
            surface_list = []
            for surface_name in task_config['surface_list']:
                if surface_name in building_instance.surface_list:
                    surface_list.append(building_instance.surfaces[surface_name])
                else:
                    raise InputError(f'"{surface_name}" in building_inst.surfaces', f'"{surface_name} not in surface list')
        
        return surface_list
    
    
    def update_settings(self, **kwargs) -> None:
        
        self.fig_size = kwargs.get('fig_size',self.fig_size)
        self.contour_lines = kwargs.get('contour_lines',self.contour_lines)
        self.plot_taps = kwargs.get('plot_taps',self.plot_taps)
        self.print_tap_id = kwargs.get('print_tap_id',self.print_tap_id)
        self.tap_font_size = kwargs.get('tap_font_size',self.tap_font_size)
        self.apply_surface_title = kwargs.get('apply_surface_title',self.apply_surface_title)
        self.contour_number = kwargs.get('contour_number',self.contour_number)
        self.contour_line_number = kwargs.get('contour_line_number',self.contour_line_number)
        self.contour_line_color = kwargs.get('contour_line_color',self.contour_line_color)
        self.contour_font_size = kwargs.get('contour_font_size',self.contour_font_size)
        self.contour_color_map = kwargs.get('contour_color_map',self.contour_color_map)
        self.value_lims = kwargs.get('value_lims',self.value_lims)

    def plot_contour(self, building_surfaces: List[BuildingSurface], 
                     func:str, func_options:dict = {},
                     colorbar_label:str = None) -> None:

            options = {}

            if func_options.__contains__('grid_x'):
                options['num_x'] = func_options['grid_x']
                del func_options['grid_x']
            if func_options.__contains__('grid_y'):
                options['num_y'] = func_options['grid_y']
                del func_options['grid_y']

            if not self.value_lims:
                data_total = []
                
                for surface in building_surfaces:
                    data_total.extend(surface.get_surface_data(func,[],**func_options)[2])
                data_total = np.array(data_total)
                min_data = min(data_total)
                max_data = max(data_total)
            else:
                min_data, max_data = self.value_lims

            
            widths = np.array([surface.width for surface in building_surfaces])
            widths /= min(widths)
            fig, ax = plt.subplots(1,len(building_surfaces),figsize=self.fig_size,gridspec_kw={'width_ratios':widths})
            
            for index,surface in enumerate(building_surfaces):
                
                
                dim_1, dim_2, data = surface.get_surface_data(func,[], **func_options)
                surface_dims = surface.bounding_dims
                grid_x, grid_y, data = surface.point_data_to_grid((dim_1,dim_2),data,surface_dims,True, **options)

                
                if self.contour_lines:
                    cs = ax[index].contour(grid_x, grid_y,data,self.contour_line_number, colors=self.contour_line_color,negative_linestyles = 'solid')
                    
                    ax[index].clabel(cs, cs.levels[0:-1:2], inline=True, fontsize=self.contour_font_size)

                ax[index].contourf(grid_x, grid_y, data,levels=self.contour_number,cmap=self.contour_color_map,vmin= min_data,vmax=max_data)
                if self.apply_surface_title:
                    ax[index].set_title(f'{surface.name.capitalize()}',fontsize= self.title_font_size)

                if index !=0:
                    ax[index].axes.get_yaxis().set_ticks([])
                    
                
                if self.plot_taps:
                    tap_id, tap_coords = surface.tap_positions
                    tap_coords = np.array(tap_coords)


                    for text, x, y in zip(tap_id,tap_coords[:,0],tap_coords[:,1]):
                        ax[index].scatter(x,y,4,c='k')
                        if self.print_tap_id:
                            ax[index].text(x,y+1,text, rotation = 'vertical', fontweight = 'bold', fontsize = self.tap_font_size)
            
                ax[index].tick_params(axis='x', length = 10, width = 2, labelsize=self.label_font_size)
                ax[index].tick_params(axis='y', length = 10, width = 2, labelsize=self.label_font_size)
                ax[index].set_aspect(1)

            colour_map = matplotlib.colors.Normalize(vmin = min_data, vmax = max_data)
            CB = cm.ScalarMappable(colour_map,self.contour_color_map)
            cb = fig.colorbar(CB, ax= ax, shrink=0.8, extend='neither')
            cb.ax.tick_params(labelsize=self.label_font_size)
            if colorbar_label:
                cb.ax.set_ylabel(colorbar_label,fontsize = self.label_font_size + 4)
            return fig,ax

      