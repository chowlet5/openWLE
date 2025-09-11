import sys, os
sys.path.append('/home/chowlet5/github/Wind_Tunnel_Testing')
import warnings
import glob
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.signal import welch
import scipy.signal as signal
import math
from collections.abc import Iterable
from abc import ABC, abstractmethod
from typing import List

from openWLE.funcs import GeneralFunctions
from openWLE.exception import InputError
from openWLE.buildingClass import Building, BuildingFloor
from openWLE.extreme_blue import extreme_blue

# Common Functions

def displacement_participation_factor(mode_shape:np.ndarray) -> float:
    if mode_shape.ndim == 1:
        mode_shape = mode_shape[np.newaxis,:]
        return mode_shape
    else:
        return mode_shape

def acceleration_participation_factor(mode_shape:np.ndarray, natural_frequecy:np.ndarray) -> float:
    return mode_shape * (natural_frequecy**2)

def background_response(psd:np.ndarray, frequencies: np.ndarray, gen_stiffness:float) -> float:
    
    return np.trapz(psd,frequencies)/gen_stiffness**2

def resonant_response(natural_frequency:float, spectra_value:float, damping_ratio:float, gen_stiffness:float) -> float:
        
        return natural_frequency*spectra_value * (np.pi/4*0.02) *(1/gen_stiffness**2)

def response_sigma(psd:np.ndarray, frequencies:np.ndarray,damping_ratio:np.ndarray,gen_stiffness:np.ndarray) -> float:
    
    background = background_response(psd,frequencies,gen_stiffness)
    resonant = resonant_response(psd,frequencies,damping_ratio,gen_stiffness)
    return math.sqrt(background**2 + resonant**2)

def mechanical_admittance(natural_frequency:float, damping_ratio:float, frequency:np.ndarray) -> float:
    return 1/((1 - np.power(frequency/natural_frequency,2))**2+ (4*damping_ratio**2)*(frequency/natural_frequency)**2)


class StructuralAnalysisFunctions:


    def displacement_participation_factor(self) -> np.ndarray:
        return self.mode_shapes
    
    def acceleration_participation_factor(self) -> np.ndarray:

        return self.mode_shapes * (self.natural_circular_frequency**2)[:, None, None]
    
    def complex_freq_response_amplitude(self,natfreq:float|np.ndarray, f:np.ndarray) -> np.ndarray:
        beta = f/natfreq[:,None]
        denominator = np.power(1 - np.power(beta,2),2) + np.power(self.damping_ratio[:,None] * beta * 2,2)
        denominator = np.power((self.gen_mass[:,None]*natfreq[:,None]**2),2)*denominator
        return 1/denominator 
    
    def mean_response(self, particiation_factor:np.ndarray) -> np.ndarray:
        return np.sum(np.moveaxis(particiation_factor,0,-1) * np.mean(self.gen_force,axis=1)/self.gen_stiffness,axis = -1)

    def calc_gust_factor(self,mean_zero_upcrossing_rate:np.ndarray|float, time_period:float) -> np.ndarray|float:
        
        sqrt_root = np.sqrt(2*np.log(mean_zero_upcrossing_rate*time_period))
        gust_factor = sqrt_root + 0.5772/sqrt_root
        return gust_factor

    def calc_mean_zero_upcrossing_rate(self, response_spectra:np.ndarray, f:np.ndarray) -> float:
        
        numerator = np.trapz(np.multiply(response_spectra,np.power(f,2)),f)
        denominator = np.trapz(response_spectra,f)
        return np.sqrt(numerator/denominator)


class StructuralStudy:

    def __init__(self, config:dict, building:Building = None) -> None:

        self.__analysis_options = ['time_domain', 'frequency_domain']
        self.__loading_options = ['floor_loads','base_loads']
        self.__directions = ['X','Y','Theta']
        self.__default_file_names ={
            'mode_shapes': 'mode_shape',
            'mass_distribution':'mass_distribution',
            'mass_height': 'mass_heights',
            'mode_frequency': 'mode_frequencies',
            'mode_damping': 'damping_ratio'
        }

        self.config = config
        self.building = building
        self.time_analysis = None
        self.frequency_analysis = None

        # Required Inputs

        self._analysis_type = config['analysis_type']
        self._loading_type = config['loading_type']

        structural_properties_dir  = config['structural_properties_directory']
        properties_delimiter = config['delimiter']

        self.mass_height = self.import_data(structural_properties_dir,properties_delimiter,'mass_height')
        self.mass_distribution = self.import_data(structural_properties_dir,properties_delimiter,'mass_distribution')      
        self.mode_frequency =  self.import_data(structural_properties_dir,properties_delimiter,'mode_frequency') 
        self.mode_damping_ratio = self.import_data(structural_properties_dir,properties_delimiter,'mode_damping')       
        self.mode_shapes = self.import_mode_shapes(structural_properties_dir,properties_delimiter, self.number_of_modes)

        self.lumped_mass_system = LumpedMass(self.mass_height,self.mass_distribution,self.mode_shapes,
                                             self.mode_frequency,self.mode_damping_ratio) 

     

    @property
    def analysis_type(self) -> str:
        return self._analysis_type
    
    @analysis_type.setter
    def analysis_type(self,analysis_type:str) -> None:
        if not analysis_type.lower() in self.__analysis_options:
            raise InputError(f'analysis_type = {analysis_type}',f'Analysis type must be one of the following options: {self.__analysis_options}')
        else:
            self._analysis_type = analysis_type.lower()

    @property
    def loading_type(self) -> str:
        return self._loading_type
    
    @loading_type.setter
    def loading_type(self,loading_type:str) -> None:
        if not loading_type.lower() in self.__loading_options:
            raise InputError(f'analysis_type = {loading_type}',f'Analysis type must be one of the following options: {self.__loading_options}')
        else:
            self._loading_type = loading_type.lower()

    @property
    def number_of_modes(self) -> int:
        return self.mode_frequency.shape[0]

    @property
    def direction_index(self) -> dict:

        direction_index = {
            'X': (0,self.directional_mode_shapes['X'].shape[1]),
            'Y': (self.directional_mode_shapes['X'].shape[1],self.directional_mode_shapes['Y'].shape[1]),
            'Theta': (self.directional_mode_shapes['Y'].shape[1],self.directional_mode_shapes['Theta'].shape[1])
        }

        return direction_index

    def import_data(self, directory_path:str, delimiter:str, data_type:str) -> np.ndarray:

        file_path = self.__default_file_names[data_type]

        if len(glob.glob(f'{directory_path}/{file_path}.*')):
                file = glob.glob(f'{directory_path}/{file_path}.*')[0]
                return  np.loadtxt(file,delimiter = delimiter)

        else:
            raise InputError(f"len(glob.glob(f'{directory_path}/{file_path}.*'))", f"{file_path} file is missing from structural properties directory." )

    def import_directional_mode_shapes(self, directory_path:str, delimiter:str) -> dict:

        mode_shape_files = self.__default_file_names['mode_shapes']

        directional_mode_shapes = {}

        for direction, file_path in zip(self.__directions,mode_shape_files):
            
            if len(glob.glob(f'{directory_path}/{file_path}.*')):
                file = glob.glob(f'{directory_path}/{file_path}.*')[0]
                directional_mode_shapes[direction] = np.loadtxt(file,delimiter = delimiter)
            else:
                raise InputError(f"len(glob.glob(f'{directory_path}/{file_path}.*'))", f"{file_path} file is missing from structural properties directory." )
        
        return directional_mode_shapes
    
    def import_mode_shapes(self, directory_path:str, delimiter:str, number_of_modes:int) -> np.ndarray:

        mode_shape_files = self.__default_file_names['mode_shapes']

        mode_shapes = []

        for mode_number in range(number_of_modes):
            
            if glob.glob(f'{directory_path}/{mode_shape_files}_{mode_number+1}.*'):
                file = glob.glob(f'{directory_path}/{mode_shape_files}_{mode_number+1}.*')[0]
                mode_shapes.append(np.loadtxt(file,delimiter = delimiter))
            else:
                raise InputError(f"{directory_path}/{mode_shape_files}_{mode_number+1}.*",f"{mode_shape_files}_{mode_number+1} is missing from structural properties directory." )    

        mode_shapes = np.array(mode_shapes)
        return mode_shapes

    def check_analysis_config(self):

        if self.analysis_type == 'time_domain':
            pass

    def assign_base_loading(self, moments:dict) -> None:

        self.lumped_mass_system.add_base_moments(moments)

    def assign_floor_loading(self, forces:dict) -> None:
        
        self.lumped_mass_system.get_mass_tributary_height()
        floors = self.building.floors
        self.lumped_mass_system.find_mass_floors(floors)

        self.lumped_mass_system.add_floor_forces(forces)

    def assign_loads(self):

        match self.loading_type:

            case 'floor_loads':
                forces = self.building.get_lumped_mass_floor_forces()
                self.assign_floor_loading(forces)
            case 'base_loads':
                moments = self.building.get_base_forces()
                self.assign_base_loading(moments)
            case _:
                pass

    def process_time_domain(self, config:dict) -> None:
        
        mode_shapes = self.mode_shapes
        self.building.displacement_time_history = self.time_analysis.displacement_response(mode_shapes)
        circular_frequencies = self.lumped_mass_system.mode_circular_frequency
        self.building.acceleration_time_history = self.time_analysis.acceleration_response(mode_shapes,circular_frequencies)
        self.building.structural_analysis_time_step = self.time_analysis.time_step
        if config.__contains__('results'):
            self.structural_results = {}
            for result in config['results']:
                key = list(result.keys())[0]
                items = list(result.values())[0]
                match key.lower():
                    case 'displacement':
                        if items['mean_value']:
                            value = self.time_analysis.get_displacement_data(mode_shapes,'mean')
                            self.structural_results['displacement_mean'] = value
                        if items['std_value']:
                            value = self.time_analysis.get_displacement_data(mode_shapes,'std')
                            self.structural_results['displacement_std'] = value
                        if items['peak_value']:
                            if not items.__contains__('MRI'):
                                raise InputError('items.__contains__(\'MRI\')', 'Missing MRI value for peak analysis')
                            peak_type = items['peak_type'] if items.__contains__('peak_type') else 'abs'
                            non_probability = 1 - 1/items['MRI']
                            match peak_type.lower():
                                case 'max':
                                    value = self.time_analysis.get_displacement_data(mode_shapes,'peak_max',non_probability)
                                case 'min':
                                    value = self.time_analysis.get_displacement_data(mode_shapes,'peak_min',non_probability)
                                case 'abs':
                                    value = self.time_analysis.get_displacement_data(mode_shapes,'peak_abs',non_probability)
                                case _:
                                    raise InputError(f'peak_type = {peak_type}', 'peak_type must be one of the following options: max, min, abs')
                            self.structural_results['displacement_peak'] = value
                        
                    case 'acceleration':
                        if items['std_value']:
                            value = self.time_analysis.get_acceleration_data(mode_shapes, circular_frequencies, 'std')
                            self.structural_results['acceleration_std'] = value
                        if items['peak_value']:
                            if not items.__contains__('MRI'):
                                raise InputError('items.__contains__(\'MRI\')', 'Missing MRI value for peak analysis')
                            peak_type = items['peak_type'] if items.__contains__('peak_type') else 'abs'
                            non_probability = 1 - 1/items['MRI']
                            match peak_type.lower():
                                case 'max':
                                    value = self.time_analysis.get_acceleration_data(mode_shapes, circular_frequencies,'peak_max',non_probability)
                                case 'min':
                                    value = self.time_analysis.get_acceleration_data(mode_shapes, circular_frequencies, 'peak_min',non_probability)
                                case 'abs':
                                    value = self.time_analysis.get_acceleration_data(mode_shapes, circular_frequencies, 'peak_abs',non_probability)
                                case _:
                                    raise InputError(f'peak_type = {peak_type}', 'peak_type must be one of the following options: max, min, abs')
                            self.structural_results['acceleration_peak'] = value

            
    def run_time_domain(self, time_domain_config:dict) -> None:

        gen_mass = self.lumped_mass_system.gen_mass
        gen_stiffness = self.lumped_mass_system.gen_stiffness
        gen_force = self.lumped_mass_system.gen_force
        if time_domain_config.__contains__('starting_index'):
            starting_index = time_domain_config['starting_index']
            gen_force = gen_force[:,starting_index:]
        damping_ratio = self.lumped_mass_system.gen_damping
        beta = time_domain_config['beta']
        gamma = time_domain_config['gamma']
        time_step = time_domain_config['time_step']
        time_domain = TimeDomainAnalysis(gen_mass, gen_stiffness, damping_ratio, beta, gamma, time_step)
        if (not 'time_step' in time_domain_config.keys()) or (time_domain_config['time_step'] == None):
            time_domain.time_step = time_domain.get_max_timestep(self.lumped_mass_system.mode_frequency)
            
        q,dot_q,ddot_q = time_domain.run_newmark_method(gen_force)
        self.time_analysis = time_domain
        self.q = q
        self.dot_q = dot_q
        self.ddot_q = ddot_q
        
        self.process_time_domain(time_domain_config)

    def process_frequency_analysis(self, config:dict):
        
        combination_method = config['combination_method']
        if config.__contains__('results'):
            self.structural_results = {}
            for result in config['results']:
                key = list(result.keys())[0]
                items = list(result.values())[0]
                match key.lower():
                    case 'displacement':
                        if items['response_value']:
                            mean_value, sigma = self.frequency_analysis.calc_response('displacement',combination_method)
                            self.structural_results['displacement_mean'] = mean_value
                            self.structural_results['displacement_std'] = sigma
                        if items['peak_analysis']:
                            parameter = {}
                            if items.__contains__('gust_factor'):
                                parameter['background_gust_factor'] = items['gust_factor']
                            if items.__contains__('background_gust_factor'):
                                parameter['background_gust_factor'] = items['background_gust_factor'] 
                            if items.__contains__('resonant_gust_factor'):
                                parameter['resonant_gust_factor'] = items['resonant_gust_factor'] 
                            if items.__contains__('peak_duration'):
                                parameter['peak_duration'] = items['peak_duration']
                            postitive_peak, negative_peak = self.frequency_analysis.calc_peak_response('displacement', combination_method, **parameter)
                            peak_type = items['peak_type'] if items.__contains__('peak_type') else 'peak_abs'
                            match peak_type:
                                case 'peak_positive':
                                    self.structural_results['displacement_positive_peak'] = postitive_peak
                                case 'peak_negative':
                                    self.structural_results['displacement_negative_peak'] = negative_peak
                                case 'peak_abs':
                                    self.structural_results['displacement_abs_peak'] = np.where(np.abs(postitive_peak) >= np.abs(negative_peak), postitive_peak, negative_peak)
                            
                    case 'acceleration':
                        if items['response_value']:
                            _, sigma = self.frequency_analysis.calc_response('acceleration',combination_method)
                            self.structural_results['acceleration_std'] = sigma
                        if items['peak_analysis']:
                            parameter = {}
                            if items.__contains__('gust_factor'):
                                parameter['background_gust_factor'] = items['gust_factor']
                            if items.__contains__('background_gust_factor'):
                                parameter['background_gust_factor'] = items['background_gust_factor'] 
                            if items.__contains__('resonant_gust_factor'):
                                parameter['resonant_gust_factor'] = items['resonant_gust_factor'] 
                            if items.__contains__('peak_duration'):
                                parameter['peak_duration'] = items['peak_duration']
                            postitive_peak , negative_peak = self.frequency_analysis.calc_peak_response('acceleration', combination_method, **parameter)
                            
                            peak_type = items['peak_type'] if items.__contains__('peak_type') else 'peak_abs'
                            match peak_type:
                                case 'peak_positive':
                                    self.structural_results['acceleration_positive_peak'] = postitive_peak
                                case 'peak_negative':
                                    self.structural_results['acceleration_negative_peak'] = negative_peak
                                case 'peak_abs':
                                    self.structural_results['acceleration_abs_peak'] = np.where(np.abs(postitive_peak) >= np.abs(negative_peak), postitive_peak, negative_peak)
                            
        print(self.structural_results)
    def run_frequency_domain(self, frequency_domain_config:dict) -> None:
        
        gen_mass = self.lumped_mass_system.gen_mass
        gen_stiffness = self.lumped_mass_system.gen_stiffness
        natural_frequency = self.lumped_mass_system.mode_circular_frequency
        damping_ratio = self.lumped_mass_system.damping_ratio
        mode_shapes = self.mode_shapes
        self.frequency_analysis = FrequencyDomainAnalysis(gen_mass,damping_ratio,gen_stiffness,
                                                   mode_shapes,natural_frequency)

        gen_force = self.lumped_mass_system.gen_force
        sampling_frequency = self.building.force_sampling_frequency
        self.frequency_analysis.set_generalized_force(gen_force,sampling_frequency)

        self.process_frequency_analysis(frequency_domain_config)
        

    def run_analysis(self) -> None:

        match self.analysis_type:
            case 'time_domain':
                self.run_time_domain(self.config['time_domain_analysis'])
                
            case 'frequency_domain':
                self.run_frequency_domain(self.config['frequency_domain_analysis'])

    @property
    def displacement_time_history(self,direction:str = None) -> np.ndarray:

        if self.time_analysis:
            response = self.time_analysis.displacement_response(self.mode_shapes)
            if direction in self.__directions:
                direction_index = self.direction_index()['direction']
                response = response[:,direction_index[0]:direction_index[1]]

            return response
        else:
            raise ValueError('Time domain analysis was not performed. Re-run analysis including a time domain analysis')

    @property
    def acceleration(self,direction:str = None) -> np.ndarray:
        if self.time_analysis:
            circular_frequencies = self.lumped_mass_system.mode_circular_frequency
            response = self.time_analysis.acceleration_response(self.mode_shapes,circular_frequencies)
            if direction in self.__directions:
                direction_index = self.direction_index()['direction']
                response = response[:,direction_index[0]:direction_index[1]]

            return response
        else:
            raise ValueError('Time domain analysis was not performed. Re-run analysis including a time domain analysis')


    def save_structural_loading(self,HDF5_file:object):

        dgroup = HDF5_file.create_group('Structural_Loading') 
        dset = dgroup.create_dataset('Design_velocity',data = self.building.design_velocity)
        dset.attrs['Description'] = 'Design velocity for force computation with units m/s'
        dset = dgroup.create_dataset('Design_air_design',data = self.building.design_air_density)
        dset.attrs['Description'] = 'Design velocity for force computation with units kg/m^3'
        floor_group = None
        if self.building.base_shear_run:
            x = self.building.force_x
            y = self.building.force_y
            z = self.building.force_z

            data = np.vstack((x,y,z)).T
            
            dset = dgroup.create_dataset('Base_shear_forces',data = data)
            dset.attrs['Description'] = 'Base shear force in the x-, y-, z-axis'
            dset.attrs['x_axis'] = self.building.x_axis
            dset.attrs['y_axis'] = self.building.y_axis
            dset.attrs['z_axis'] = self.building.z_axis

        if self.building.base_moment_run:
            x = self.building.force_x
            y = self.building.force_y
            z = self.building.force_z

            data = np.vstack((x,y,z)).T
            dset = dgroup.create_dataset('Base_moment_forces',data = data)
            dset.attrs['Description'] = 'Base moment about the x-, y-, z-axis'
            dset.attrs['x_axis'] = self.building.x_axis
            dset.attrs['y_axis'] = self.building.y_axis
            dset.attrs['z_axis'] = self.building.z_axis


        if self.building.floor_shear_run:
            floor_id = []
            x = []
            y = []
            z = []
            for floor_num, values in self.building.floor_forces.items():
                floor_id.append((floor_num,self.building.floors[floor_num].elevation))
                x.append(values[0])
                y.append(values[1])
                z.append(values[2])
            floor_group = dgroup.create_group('Floor_based_loads')
            dtype = [('floor_id','i8'),('elevation','f8')]
            floor_id = np.array(floor_id,dtype = np.dtype(dtype))
            dset = floor_group.create_dataset('Floor_numbers',data = floor_id)
            dset.attrs['Description'] = 'Floor Number IDs'
            dset = floor_group.create_dataset('Floor_shear_forces_x_axis',data = np.array(x).T)
            dset.attrs['Description'] = 'Floor shear force along the x-axis'
            dset.attrs['axis_description'] = self.building.x_axis
            dset = floor_group.create_dataset('Floor_shear_forces_y_axis',data = np.array(y).T)
            dset.attrs['Description'] = 'Floor shear force along the y-axis'
            dset.attrs['axis_description'] = self.building.y_axis
            dset = floor_group.create_dataset('Floor_shear_forces_z_axis',data = np.array(z).T)
            dset.attrs['Description'] = 'Floor shear force along the z-axis'
            dset.attrs['axis_description'] = self.building.z_axis
            
        if self.building.floor_moment_run:

            floor_id = []
            x = []
            y = []
            z = []
            for floor_num, values in self.building.floor_moments.items():
                floor_id.append([floor_num,self.building.floors[floor_num].elevation])
                x.append(values[0])
                y.append(values[1])
                z.append(values[2])

            if not floor_group:
                floor_group = dgroup.create_group('Floor_based_loads')
            
            if not self.building.floor_shear_run:

                dset = floor_group.create_dataset('Floor_numbers',data = np.array(floor_id),dtype=(int,float))
                dset.attrs['Description'] = 'Floor Number IDs'
    

            dset = floor_group.create_dataset('Floor_moment_x_axis',data = np.array(x).T)
            dset.attrs['Description'] = 'Floor moment about the x-axis'
            dset.attrs['axis_description'] = self.building.x_axis
            dset = floor_group.create_dataset('Floor_moment_y_axis',data = np.array(y).T)
            dset.attrs['Description'] = 'Floor moment aboutthe y-axis'
            dset.attrs['axis_description'] = self.building.y_axis
            dset = floor_group.create_dataset('Floor_moment_z_axis',data = np.array(z).T)
            dset.attrs['Description'] = 'Floor moment about the z-axis'
            dset.attrs['axis_description'] = self.building.z_axis


        self.building.base_shear_run = False
        self.building.base_moment_run = False
        self.building.floor_shear_run = False
        self.building.floor_moment_run = False

    def save_time_domain_analysis(self, HDF5_group:object):
        
        HDF5_group.attrs['method'] = 'Newmark-beta'
        HDF5_group.attrs['beta'] = self.time_analysis.beta
        HDF5_group.attrs['gamma'] = self.time_analysis.gamma
        HDF5_group.attrs['time_step'] = self.time_analysis.time_step

        
        for direction in range(self.building.displacement_time_history.shape[1]):
            HDF5_group.create_dataset(f'Displacement_{direction}',data = self.building.displacement_time_history[:,direction,:])
        
        for direction in range(self.building.acceleration_time_history.shape[1]):
            HDF5_group.create_dataset(f'Acceleration_{direction}',data = self.building.acceleration_time_history[:,direction,:])
        
        for key in self.structural_results:

            HDF5_group.create_dataset(key,data = self.structural_results[key])

    def save_frequency_domain_analysis(self, HDF5_group:object):
        
        HDF5_group.attrs['combination_method'] = self.frequency_analysis.combination_obj()

        for key in self.structural_results:

            HDF5_group.create_dataset(key,data = self.structural_results[key])

    def save_structural_analysis(self,HDF5_file:object):
        
        match self.analysis_type:
            case 'time_domain':
                dgroup = HDF5_file.create_group('time_domain_analysis')
                self.save_time_domain_analysis(dgroup)
            case 'frequency_domain':
                pass
        

        

class LumpedMass(GeneralFunctions):
    """
    Lumped Mass Model 
    Description: This class is used to create a idealized lumped mass model for a structure. The model is used to calculate the
    the generalized mass and generalized force for a given direction. The generalized mass and generalized force are used to calculate
    the acceleration response of the structure. 

    Attributes:
    mass_height: list of the height of each lump mass in the structure
    mode_shapes: list of the mode shapes of the structure
    mass_distribution: list of the mass distribution(percentage of total mass) of the structure
    moment_inertia: float value of the polar moment of inertia of the structure
    """
    def __init__(self,mass_height:list,
                 mass_distribution:list,
                 mode_shapes: np.ndarray|str,
                 mode_frequency:np.ndarray,
                 damping_ratio: np.ndarray) -> None:
        
        self.__directions = ['X','Y','Theta']
        self.__gen_force = None
        self.mass_height = np.sort(np.array(mass_height))
        self.mass_distribution = mass_distribution
        self.base_moments = {}
        self.mass_forces = {}
        
        # Assign mode shapes and mass distribution to class attributes
        if isinstance(mode_shapes,str):
            if mode_shapes.lower() != 'ideal':
                warnings.warn('Unknown string entry for mode shapes, assuming idealized mode shapes')
            
            mode_shapes = self.add_idealized_mode_shapes()
            self.directional_mode_shapes = {key:np.array(value) for key,value in mode_shapes.items()}
            self.mode_shapes = self.directional2array_mode_shapes(self.directional_mode_shapes)
        else:
            self.mode_shapes = mode_shapes
        
        self.mode_frequency = np.array(mode_frequency)
        self.mode_circular_frequency  = 2*math.pi*self.mode_frequency
        self.damping_ratio = damping_ratio
        
        self.gen_mass = []
        self.gen_stiffness = []
        self.gen_damping = []
        for mode_shape in self.mode_shapes:
            self.gen_mass.append(self.cal_gen_mass(mass_distribution,mode_shape))

        
        self.gen_mass = np.array(self.gen_mass).T
        
        self.gen_stiffness = self.cal_gen_stiffness(self.gen_mass,self.mode_circular_frequency)
        self.gen_damping = self.cal_gen_damping(damping_ratio,self.gen_mass,self.mode_circular_frequency)
        
    @property
    def idealized_directional_mode_shapes(self, torsional_mode_constant:float = 0.7) -> dict:
        number_of_masses = len(self.mass_height)
        zeros = np.zeros(number_of_masses)
        mode_shape_x = [self.mass_height/self.mass_height.max(),zeros,zeros]
        mode_shape_y = [zeros, self.mass_height/self.mass_height.max(), zeros]
        mode_shape_theta = [zeros,zeros,np.ones(number_of_masses)*torsional_mode_constant]
        mode_shapes = {
            'X': np.array(mode_shape_x),
            'Y': np.array(mode_shape_y),
            'Theta': np.array(mode_shape_theta)
        }

        return mode_shapes

    def directional2array_mode_shapes(self,directional_mode_shape:dict) -> np.ndarray:
        mode_shape = []
        for mode in range(directional_mode_shape['X'].shape[0]):
            mode_shape_x = directional_mode_shape['X'][mode,:]
            mode_shape_y = directional_mode_shape['Y'][mode,:]
            mode_shape_theta = directional_mode_shape['Theta'][mode,:]
            mode_shape.append(np.vstack((mode_shape_x,mode_shape_y,mode_shape_theta)).flatten())
            
        return np.array(mode_shape)
    

    def get_mass_tributary_height(self) -> None:

        mass_height = self.mass_height.copy()
        
        mass_height = np.append([0],mass_height,)
        mass_height = np.append(mass_height,[mass_height[-1]])

        height_diff = np.diff(mass_height)

        mass_tributary_heights = []
        for index in range(len(mass_height)-2):

            if index == 0:
                lower_height = height_diff[0]
                upper_height = height_diff[1]/2
            else:
                lower_height = height_diff[index]/2
                upper_height = height_diff[index + 1]/2

            mass_tributary_heights.append([lower_height,upper_height])

        self.mass_tirbutary_height = mass_tributary_heights

    def find_mass_floors(self, floors:dict) -> None:

        lower_tributary_height = np.array([h-trib_h[0] for h,trib_h in zip(self.mass_height,self.mass_tirbutary_height)])
        upper_tributary_height = np.array([h+trib_h[1] for h,trib_h in zip(self.mass_height,self.mass_tirbutary_height)])

        mass_floor_ids = {}
        for floor in floors.values():

            floor_trib_height = floor.tributary_height[1] + floor.tributary_height[0]
            min_floor_elev = floor.elevation - floor.tributary_height[0]
            max_floor_elev = floor.elevation + floor.tributary_height[1]
            check_1 = upper_tributary_height - min_floor_elev
            check_2 = lower_tributary_height - max_floor_elev

            check = np.sign(check_1 * check_2) == -1

            for index, mass in enumerate(check):
                
                if mass:
                    
                    temp_max = min(max_floor_elev, upper_tributary_height[index])
                    temp_min = max(min_floor_elev, lower_tributary_height[index])
                    height =  temp_max - temp_min

                    if not index in mass_floor_ids.keys():
                        mass_floor_ids[index] = []

                    mass_floor_ids[index].append([int(floor.floor_num),round(height/floor_trib_height,4)])

        self.mass_floor_ids = mass_floor_ids

    def add_floor_forces(self,floor_forces:np.ndarray) -> None:
        
        
            floor_data = floor_forces
            mass_forces = []
            for mass in self.mass_floor_ids.values():
                force = []
                for floor_id,factor in mass:
                    temp = np.array([floor_data[self.__directions[0]][floor_id],
                            floor_data[self.__directions[1]][floor_id],
                            floor_data[self.__directions[2]][floor_id]])
                    if len(force) == 0:
                        
                        force = temp * factor
                    else:
                        force += temp * factor
                
                mass_forces.append(force)
            mass_forces = np.array(mass_forces)
            self.mass_forces = mass_forces
    
    @property
    def gen_force(self) -> np.ndarray:

        if self.__gen_force == None:
            self.__gen_force =  self.cal_gen_force(self.mass_forces)
        return self.__gen_force

    @gen_force.setter
    def gen_force(self, value):
        self.__gen_force = np.array(value)

    def cal_gen_force(self,forces:np.ndarray,) -> np.ndarray: 
        gen_force = []    
        
        mode_shape = self.mode_shapes
        for mode_index in range(mode_shape.shape[0]):
            mode = mode_shape[mode_index]
            temp = []
            for floor in range(mode.shape[0]):
                temp.append(np.sum(mode[floor] * forces[floor].T,axis=1))
            gen_force.append(np.sum(temp,axis=0))
            
        return np.array(gen_force)
            
    def cal_gen_mass(self, mass:np.ndarray, mode_shape:np.ndarray) -> np.ndarray:
        
        return np.sum(np.multiply(mass[:,None],np.power(mode_shape,2)))
    
    def cal_gen_stiffness(self,gen_mass:np.ndarray, mode_circular_frequency:np.ndarray) -> np.ndarray:

        return gen_mass * np.power(mode_circular_frequency,2)

    def cal_gen_damping(self,damping_ratio:np.ndarray,gen_mass:np.ndarray, mode_circular_frequency:np.ndarray) -> np.ndarray:

        return 2 * (damping_ratio * gen_mass * mode_circular_frequency)

    def save_lumped_mass_data(self, HDF5_file:object) -> None:

        dgroup = HDF5_file.create_group('Lumped_mass_system')
        data = np.concatenate((self.mass_height[:,np.newaxis],self.mass_distribution),axis=1)
        dset = dgroup.create_dataset('mass_distribution', data = data)
        dset.attrs['Description'] = 'The vertical mass distribution for the DOFs along the x-axis, y-axis and the rotational'
        dset.attrs['Column_Header'] = 'Height [m], M_x [kg], M_y [kg], M_theta [kg*m^2]'

        dset = dgroup.create_dataset('Natural_frequencies',data =self.mode_frequency)
        dset.attrs['Description'] = 'Natural frequencies associated with each mode in Hz'
        dset = dgroup.create_dataset('Damping_ratio',data = self.damping_ratio)
        dset.attrs['Description'] = 'Damping ratios associated with each mode'

        mode_group = dgroup.create_group('Mode_shapes')
        mode_group.attrs['Description'] = 'Mode shape for the x, y and theta directions'
        mode_group.attrs['Column_Header'] = 'X, Y, Theta'
        for index, mode in enumerate(self.mode_shapes):
            
            dset = mode_group.create_dataset(f'Mode_{index}', data = mode)

        # Store generalized data
        subgroup = dgroup.create_group('Generalized_data')
        dset = subgroup.create_dataset('Generalized_mass',data = self.gen_mass.T)
        dset.attrs['Description'] = 'Computed generalized mass associated with each mode'
        dset = subgroup.create_dataset('Generalized_damping',data = self.gen_damping.T)
        dset.attrs['Description'] = 'Computed generalized damping associated with each mode'
        dset = subgroup.create_dataset('Generalized_stiffness',data = self.gen_stiffness.T)
        dset.attrs['Description'] = 'Computed generalized stiffness associated with each mode' 

    
class FrequencyDomainAnalysis(GeneralFunctions, StructuralAnalysisFunctions):

    def __init__(self,
                 gen_mass:np.ndarray,
                 damping_ratio:np.ndarray,
                 gen_stiffness:np.ndarray, 
                 mode_shapes:np.ndarray,
                 natural_frequency:np.ndarray) -> None:
        
        self.gen_mass = gen_mass
        self.damping_ratio = damping_ratio
        self.gen_stiffness = gen_stiffness
        self.mode_shapes = mode_shapes
        self.natural_frequency = natural_frequency

    def set_generalized_force(self, gen_force:np.ndarray, sampling_frequency:float) -> None:
        self.gen_force = gen_force
        self.sampling_frequency = sampling_frequency
    
    def set_modal_combination(self,combination_method:str) -> None:
        
        params = {}
        params['mode_shapes'] = self.mode_shapes
        params['natural_circular_frequency'] = self.natural_frequency
        params['damping_ratio'] = self.damping_ratio
        params['gen_mass'] = self.gen_mass
        params['gen_stiffness'] = self.gen_stiffness
        params['gen_force'] = self.gen_force
        params['sampling_frequency'] = self.sampling_frequency

        if combination_method.upper() == 'SRSS':
            self.combination_obj = SRSS(**params)
        elif combination_method.upper() == 'CQC':
            self.combination_obj = ChenKareem2005CQC(**params)

    def calc_response(self, response_type:str = 'displacement',
                           combination_method:str = 'SRSS') -> np.ndarray:
        
        if combination_method not in ['SRSS','CQC']:
            raise ValueError('Invalid combination method. Expected values are "SRSS" or "CQC"')
        else:
            self.set_modal_combination(combination_method)

        return self.combination_obj.get_response(response_type)
    
    def calc_peak_response(self, response_type:str = 'displacement',
                           combination_method:str = 'SRSS',
                           background_gust_factor:float = 3.5,
                           resonant_gust_factor:np.ndarray = None,
                           peak_duration:float = 3600) -> np.ndarray:
        
        if combination_method not in ['SRSS','CQC']:
            raise ValueError('Invalid combination method. Expected values are "SRSS" or "CQC"')
        else:
            self.set_modal_combination(combination_method)

        return self.combination_obj.get_peak_response(response_type,background_gust_factor, resonant_gust_factor, peak_duration)
           

########################################################################################################################
# Modal Combination Rules
######################################################################################################################## 

class CombinationRules(ABC):

    __name__ = None
    
    
    def __call__(self):
        return self.__name__

    @property
    def name(self):
        return self.__name__
    
    @name.setter
    def name(self, value):
        self.__name__ = value
    

    def get_response(self,response_type:str) -> np.ndarray:
        match response_type.lower():
            case 'displacement':
                mean_value, sigma = self.calculate_displacement_response()
            case 'acceleration':
                mean_value, sigma = self.calculate_acceleration_response()
        return mean_value, sigma
    
    def get_peak_response(self,response_type:str,
                          background_gust_factor:float = 3.5,
                          resonant_gust: np.ndarray = None,
                          peak_duration:float = 3600) -> np.ndarray:
        parameters = {
            'peak_factor': background_gust_factor,
            'background_gust_factor':background_gust_factor,
            'resonant_gust': resonant_gust,
            'peak_duration':peak_duration
        }
        match response_type.lower():
            case 'displacement':
                mean_value, sigma = self.calculate_peak_displacement(**parameters)
            case 'acceleration':
                mean_value, sigma = self.calculate_peak_acceleration(**parameters)
                
        return mean_value, sigma

    @abstractmethod
    def calculate_displacement_response():
        pass

    @abstractmethod
    def calculate_acceleration_response():
        pass  

    @abstractmethod
    def calculate_peak_displacement():
        pass

    @abstractmethod
    def calculate_peak_acceleration():
        pass  

    
class SRSS(GeneralFunctions, CombinationRules, StructuralAnalysisFunctions):


    def __init__(self, 
                 gen_mass: np.ndarray, 
                 gen_stiffness: np.ndarray,
                 damping_ratio: np.ndarray, 
                 gen_force: np.ndarray,
                 mode_shapes:np.ndarray, 
                 natural_circular_frequency:np.ndarray, 
                 sampling_frequency:float):
        
        self.name = 'SRSS'
        self.mode_shapes = mode_shapes
        self.natural_circular_frequency = natural_circular_frequency
        self.damping_ratio = damping_ratio
        self.gen_stiffness = gen_stiffness
        self.gen_mass = gen_mass
        self.sampling_frequency = sampling_frequency
        self.gen_force = gen_force
   

    def calc_structural_response_deviation(self) -> np.ndarray:
        spectra, f = self.spectral_density(self.gen_force,self.sampling_frequency)
        mechanical_admittance = self.complex_freq_response_amplitude(self.natural_circular_frequency,f)
        response_var = integrate.trapezoid(mechanical_admittance * spectra, f)
        self.structural_response = np.sqrt(response_var)
        return self.structural_response

    def calculate_response(self, participation_factor) -> np.ndarray:

        sigma = self.calc_structural_response_deviation()
        return np.sqrt(np.sum((np.moveaxis(participation_factor,0,-1)**2) * sigma ** 2 , axis= -1))

    def calculate_displacement_response(self,) -> np.ndarray:

        participation_factor = self.displacement_participation_factor()
        mean_value = self.mean_response(participation_factor)
        sigma = self.calculate_response(participation_factor)
        return mean_value, sigma

    def calculate_acceleration_response(self) -> np.ndarray:

        participation_factor = self.acceleration_participation_factor()
        sigma = self.calculate_response(participation_factor)
        mean_value = np.zeros_like(sigma)
        return  mean_value, sigma 

    def calculate_upcrossing_rate(self) -> float:
        spectra, f = self.spectral_density(self.gen_force,self.sampling_frequency)
        mechanical_admittance = self.complex_freq_response_amplitude(self.natural_circular_frequency,f)
        response_spectra =  mechanical_admittance * spectra 

        numerator = np.nansum(np.trapz(np.multiply(response_spectra, np.power(f,2))))
        denominator = np.nansum(np.trapz(response_spectra,f))
        return np.sqrt(numerator/denominator)
    
    def calculate_peak_response(self, participation_factor:np.ndarray, peak_factor:float = None, peak_duration:float = 3600) -> np.ndarray:
        
        mean_value = self.mean_response(participation_factor)
        response = self.calculate_response(participation_factor)

        if peak_factor == None:
            upcrossing_rate = self.calculate_upcrossing_rate()
            peak_factor = self.calc_gust_factor(upcrossing_rate, peak_duration)

        positive_peak = mean_value + peak_factor * response
        negative_peak = mean_value - peak_factor * response

        return positive_peak, negative_peak

    def calculate_peak_displacement(self, peak_factor:float, peak_duration:float = 3600, **kwargs) -> np.ndarray:
        
        participation_factor = self.displacement_participation_factor()
        return self.calculate_peak_response(participation_factor, peak_factor, peak_duration)

    def calculate_peak_acceleration(self, peak_factor:float, peak_duration:float = 3600, **kwargs) -> np.ndarray:

        participation_factor = self.acceleration_participation_factor()
        return self.calculate_peak_response(participation_factor, peak_factor, peak_duration)
    

class ChenKareem2005CQC(GeneralFunctions, CombinationRules, StructuralAnalysisFunctions):

    def __init__(self, 
                 gen_mass: np.ndarray, 
                 gen_stiffness: np.ndarray,
                 damping_ratio: np.ndarray, 
                 gen_force: np.ndarray,
                 mode_shapes:np.ndarray, 
                 natural_circular_frequency:np.ndarray, 
                 sampling_frequency:float) -> None:
        
        self.name = 'CQC'
        self.mode_shapes = mode_shapes
        self.natural_circular_frequency = natural_circular_frequency
        self.damping_ratio = damping_ratio
        self.gen_stiffness = gen_stiffness
        self.gen_mass = gen_mass
        self.sampling_frequency = sampling_frequency
        self.gen_force = gen_force
           

    def gen_force_std(self, index:int) -> float:
        return self.background_force(self.gen_force[index])
    
    def background_response_variance(self) -> np.ndarray:
        spectra,frequencies = self.spectral_density(self.gen_force,self.sampling_frequency)
        variance = np.trapz(spectra,frequencies)/(self.gen_stiffness**2)
        return variance
    
    def resonant_response_variance(self) -> np.ndarray:
            
        spectra,frequency = self.spectral_density(self.gen_force,self.sampling_frequency)
        spectra_value = self.spectra_value(spectra,frequency,self.natural_circular_frequency)
        resonant_variance = spectra_value*self.natural_circular_frequency*(np.pi/4*self.damping_ratio)*(1/(self.gen_stiffness**2))
        return resonant_variance
    
    def background_cross_correlation(self,mode_i_index:int, mode_j_index:int) -> float:

        gen_force_i = self.gen_force[mode_i_index]
        gen_force_j = self.gen_force[mode_j_index]
        sampling_freq = self.sampling_frequency

        response_i = self.spectra_intergration(gen_force_i, sampling_freq)
        response_j = self.spectra_intergration(gen_force_j, sampling_freq)

        cross_spectra, f_cross = self.cross_spectral_density(gen_force_i,gen_force_j, sampling_freq)

        variance_i_j = integrate.trapezoid(np.real(cross_spectra),f_cross)

        return variance_i_j/(response_i*response_j)

    
    def der_kiureghian_correlation_parameter(self,mode_i_index:int,mode_j_index:int):
        if (not isinstance(mode_i_index,int)) or (not isinstance(mode_j_index,int)):
            raise TypeError('Mode index must be an integer.')
        
        i_damping_ratio = self.damping_ratio[mode_i_index]
        j_damping_ratio = self.damping_ratio[mode_j_index]

        nat_freq_i = self.natural_circular_frequency[mode_i_index]
        nat_freq_j = self.natural_circular_frequency[mode_j_index]

        beta = nat_freq_i/nat_freq_j

        numerator = 8*math.sqrt(i_damping_ratio*j_damping_ratio)*(beta*i_damping_ratio + j_damping_ratio)*beta**(3/2)
        demoninator = (1-beta**2)**2 + (4*(i_damping_ratio*j_damping_ratio*beta)*(1+beta**2)) + 4*(i_damping_ratio**2+j_damping_ratio**2)*(beta**2)

        return numerator/demoninator
    
    def alpha_i_j(self,i_index:int,j_index:int):

        spectra_i,f_i = self.spectral_density(self.gen_force[i_index],self.sampling_frequency)
        spectra_j,f_j = self.spectral_density(self.gen_force[j_index],self.sampling_frequency)
        cross_spectra,f_cross = self.cross_spectral_density(self.gen_force[i_index],self.gen_force[j_index],self.sampling_frequency)

        f = self.natural_circular_frequency[i_index]

        spectra_value_i = self.spectra_value(spectra_i,f_i,f)
        spectra_value_j = self.spectra_value(spectra_j,f_j,f)
        cross_spectra_value = self.spectra_value(np.real(cross_spectra),f_cross,f)

        return cross_spectra_value/np.sqrt(spectra_value_i*spectra_value_j)


    def calc_response(self, participation_factor:np.ndarray) -> float:
        
        background_response = self.background_response_variance()
        resonant_response = self.resonant_response_variance()
        combined_response = np.sqrt(background_response + resonant_response)
        
        for i in range(len(self.gen_force)):
            for j in range(len(self.gen_force)):
                
                if i == j:
                    
                    modal_response = (combined_response[i]**2)*participation_factor[i]**2
                
                else:

                    background_correlation_i_j = self.background_cross_correlation(i,j)
                    
                    alpha = self.alpha_i_j(i,j)
                    der_kiureghian = self.der_kiureghian_correlation_parameter(i,j)

                    resonant_correlation_i_j = alpha * der_kiureghian

                    background_response_i = math.sqrt(background_response[i])
                    resonant_response_i = math.sqrt(resonant_response[i])
                    
                    background_response_j = math.sqrt(background_response[j])
                    resonant_response_j = math.sqrt(resonant_response[j])

                    correlation_i_j = background_correlation_i_j * background_response_i * background_response_j
                    correlation_i_j += resonant_correlation_i_j * resonant_response_i * resonant_response_j
                    correlation_i_j /= combined_response[i] * combined_response[j]

                    modal_response = participation_factor[i] * participation_factor[j] * correlation_i_j * combined_response[i] * combined_response[j]

                if (i == 0) and (j == 0):
                    sigma = modal_response
                else:
                sigma += modal_response
        return np.sqrt(sigma)
    
    def calculate_displacement_response(self) -> np.ndarray:
        participation_factor = self.displacement_participation_factor()
        mean_value = self.mean_response(participation_factor)
        sigma = self.calc_response(participation_factor)
        return mean_value, sigma

    def calculate_acceleration_response(self) -> np.ndarray:
        participation_factor = self.acceleration_participation_factor()
        mean_value = 0
        sigma = self.calc_response(participation_factor)
        return mean_value, sigma

    def calc_peak_response(self, participation_factor:np.ndarray, 
                           background_gust_factor:float = 3.5,
                           resonant_gust_factor: np.ndarray = None,
                           peak_duration:float = 3600, ) -> np.ndarray:
        
        background_response = self.background_response_variance()
        resonant_response = self.resonant_response_variance()
        
        if background_gust_factor == None:
            upcrossing_rate = self.calculate_upcrossing_rate()
            background_gust_factor = self.calc_gust_factor(upcrossing_rate, peak_duration)

        if isinstance(resonant_gust_factor, float|int):
            resonant_gust_factor = np.ones_like(self.natural_circular_frequency) * resonant_gust_factor
        
        if resonant_gust_factor == None:
            resonant_gust_factor = self.calc_gust_factor(self.natural_circular_frequency, peak_duration)

        
        for i in range(len(self.gen_force)):
            for j in range(len(self.gen_force)):
                
                if i == j:
                    
                    modal_response = (background_gust_factor ** 2) * background_response[i]
                    modal_response += (resonant_gust_factor[i] ** 2) * resonant_response[i]
                    modal_response *= participation_factor[i]**2
                
                else:

                    background_correlation_i_j = self.background_cross_correlation(i,j)
                    alpha = self.alpha_i_j(i,j)
                    der_kiureghian = self.der_kiureghian_correlation_parameter(i,j)

                    resonant_correlation_i_j = alpha * der_kiureghian

                    background_response_i = math.sqrt(background_response[i])
                    resonant_response_i = math.sqrt(resonant_response[i])
                    
                    background_response_j = math.sqrt(background_response[j])
                    resonant_response_j = math.sqrt(resonant_response[j])

                    response_i_j = background_gust_factor * background_correlation_i_j * background_response_i * background_response_j
                    response_i_j += resonant_gust_factor[i] * resonant_gust_factor[j] * resonant_correlation_i_j * resonant_response_i * resonant_response_j
                    
                    modal_response = participation_factor[i] * participation_factor[j] * response_i_j 

                if (i == 0) and (j == 0):
                    sigma = modal_response
                else:
                    sigma += modal_response

        return np.sqrt(sigma)
    
    def calculate_peak_displacement(self,background_gust_factor:float = 3.5,
                           resonant_gust_factor: np.ndarray = None,
                           peak_duration:float = 3600, **kwargs) -> float:
        participation_factor = self.displacement_participation_factor()
        mean_value = self.mean_response(participation_factor)
        sigma = self.calc_peak_response(participation_factor,background_gust_factor,resonant_gust_factor, peak_duration)

        positive_peak = mean_value + sigma
        negative_peak = mean_value - sigma

        return positive_peak, negative_peak
    
    def calculate_peak_acceleration(self,background_gust_factor:float = 3.5,
                           resonant_gust_factor: np.ndarray = None,
                           peak_duration:float = 3600, **kwargs) -> float:
        participation_factor = self.acceleration_participation_factor()
        mean_value = 0
        sigma = self.calc_peak_response(participation_factor,background_gust_factor,resonant_gust_factor, peak_duration)

        positive_peak = mean_value + sigma
        negative_peak = mean_value - sigma

        return positive_peak, negative_peak
        
class HuangEtAl2009CQC(GeneralFunctions):
    #TODO To be implemented
    def __init__(self):
        self.mode_shapes = None
        self.natural_frequency = None
        self.gen_damping = None
        self.gen_mass = None
        self.gen_force = None
        self.sampling_frequency = None


    def alpha_i_j(self,i_index:int,j_index:int):
        numerator = 2*(self.damping_ratio[j_index]*self.natural_frequency[i_index] -self.damping_ratio[i_index]*self.natural_frequency[j_index])
        denominator = self.natural_frequency[i_index]*self.natural_frequency[j_index]
        return numerator/denominator
    
    def beta_i_j(self,i_index:int,j_index:int):
        numerator = 2*(self.damping_ratio[j_index]*self.natural_frequency[i_index] -self.damping_ratio[i_index]*self.natural_frequency[j_index])
        denominator = (self.natural_frequency[i_index]**2)*(self.natural_frequency[j_index]**2)
        return numerator/denominator
    
    def gamma_i_j(self,i_index:int,j_index:int):
        numerator = 4*self.damping_ratio[i_index]*self.damping_ratio[j_index]*self.natural_frequency[i_index]*self.natural_frequency[j_index]
        numerator = numerator - self.natural_frequency[i_index]**2 - self.natural_frequency[j_index]**2
        denominator = (self.natural_frequency[i_index]**2)*(self.natural_frequency[j_index]**2)
        return numerator/denominator
    
    def epsilon_i_j(self,i_index:int,j_index:int):
        numerator = 1
        denominator = (self.natural_frequency[i_index]**2)*(self.natural_frequency[j_index]**2)
        return numerator/denominator

    def lambda_ii(self,i_index:int):
        spectra_density, f = self.spectral_density(self.gen_force[i_index],self.sampling_frequency)
        complex_response = mechanical_admittance(f,self.natural_frequency[i_index],self.damping_ratio[i_index])
        
        return np.trapz(complex_response*spectra_density,f)
    
    def lambda_ij(self,i_index:int,j_index:int,spectral_moment_num:int):
        
        cross_spectra, f_cross = self.cross_spectral_density(self.gen_force[i_index],self.gen_force[j_index],self.sampling_frequency)
        complex_response_i = mechanical_admittance(f_cross,self.natural_frequency[i_index],self.damping_ratio[i_index])
        complex_response_j = mechanical_admittance(f_cross,self.natural_frequency[j_index],self.damping_ratio[j_index])
        if spectral_moment_num not in [1,3]:
            lambda_jk = np.trapz(f_cross*complex_response_i*complex_response_j*np.imag(np.trapz(cross_spectra,f_cross)),f_cross)

        elif spectral_moment_num not in [0,2,4]:
            lambda_jk = np.trapz(f_cross*complex_response_i*complex_response_j*np.real(np.trapz(cross_spectra,f_cross)),f_cross)

        else: 
            raise ValueError('Spectral moment number must be 0,1,2,3,4')

        return lambda_jk
    
    def modal_displacement_standard_deviation(self,mode_index:int):
        spectra,f = self.spectral_density(self.gen_force[mode_index],self.sampling_frequency)
        complex_response = mechanical_admittance(f,self.natural_frequency[mode_index],self.damping_ratio[mode_index])
        sigma = np.trapz(complex_response*spectra,f)/(self.gen_mass[mode_index*self.natural_frequency[mode_index]]**2)**2
        return sigma

    def calc_cross_correlation(self,j_index:int,k_index:int):

        lambda_ii = self.lambda_ii(j_index)
        lambda_jj = self.lambda_ii(k_index)

        spectral_moments = self.lambda_ij(j_index,k_index,0)
        spectral_moments += self.gamma_i_j(j_index,k_index)*self.lambda_ij(j_index,k_index,2)
        spectral_moments += self.epsilon_i_j(j_index,k_index)*self.lambda_ij(j_index,k_index,4)
        spectral_moments += self.alpha_i_j(j_index,k_index)*self.lambda_ij(j_index,k_index,1)
        spectral_moments += self.beta_i_j(j_index,k_index)*self.lambda_ij(j_index,k_index,3)

        return (1/math.sqrt(lambda_ii*lambda_jj))*spectral_moments
    
    def modal_displacement_variance(self,mode_index:int):
        spectra,f = self.spectral_density(self.gen_force[mode_index],self.sampling_frequency)
        complex_response = mechanical_admittance(f,self.natural_frequency[mode_index],self.damping_ratio[mode_index])
        sigma = np.trapz(complex_response*spectra,f)/(self.gen_mass[mode_index]*self.natural_frequency[mode_index]**2)**2
        return sigma
    
    def calc_standard_deviation(self):
        sigma = 0
        for i in range(len(self.gen_force)):
            for j in range(len(self.gen_force)):
                
                if i == j:
                    sigma_ii = self.modal_displacement_variance(i)
                    sigma_ii = sigma_ii*self.mode_shapes[i]**2
                    sigma += sigma_ii 

                else:
                    sigma_ii = self.modal_displacement_variance(i)
                    sigma_jj = self.modal_displacement_variance(j)
                    sigma_ij = self.calc_cross_correlation(i,j)*math.sqrt(sigma_ii)*math.sqrt(sigma_jj)
                    sigma += 2*self.mode_shapes[i]*self.mode_shapes[j]*sigma_ij

        return math.sqrt(sigma)

class TimeDomainAnalysis:

    def __init__(self, mass:np.ndarray, stiffness:np.ndarray, damping:np.ndarray, beta:float = 0.25,gamma:float = 0.5, time_step:float = None) -> None:
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.beta = beta
        self.gamma = gamma
        self.time_step = time_step

    def get_max_timestep(self,natural_frequency:float|list):
        try:
            natural_period = 1/natural_frequency
            time_step = natural_period/(math.sqrt(2)*np.pi*math.sqrt(self.gamma - 2*self.beta))
            if isinstance(natural_frequency,float):
                return time_step
            else:
                return min(time_step)
        except ZeroDivisionError:
            return np.inf
        
    def __set_initial_conditions(self,force_shape:tuple,initial_disp:float = 0, initial_vel:float = 0, initial_force:float = 0,):
        
        new_shape = (force_shape[0],force_shape[1] + 1)

        self.q = np.empty(new_shape)
        self.q[:,0] = initial_disp
        self.dot_q = np.empty(new_shape)
        self.dot_q[:,0] = initial_vel
        self.p_force = np.empty(new_shape)
        self.p_force[:,0] = initial_force

        self.ddot_q = np.empty(new_shape)
        self.ddot_q[:,0] = np.divide(self.p_force[:,0] - self.dot_q[:,0]*self.damping - self.q[:,0]*self.stiffness,self.mass)
        
    def __calc_newmark_parameters(self):
        self.A1 = 0
        self.A1 += self.mass/(self.beta*self.time_step**2)
        self.A1 += (self.damping*self.gamma)/(self.beta*self.time_step)

        self.A2 = 0
        self.A2 += self.mass/(self.beta*self.time_step)
        self.A2 += (self.gamma/self.beta - 1)*self.damping

        self.A3 = 0
        self.A3 += (1/(2*self.beta) - 1)*self.mass 
        self.A3 += self.time_step*(self.gamma/(2*self.beta) - 1)*self.damping

        self.newmark_stiffness = self.stiffness + self.A1

    
    def run_newmark_method(self,force_time_history:np.ndarray, q_start:float =0, dot_q_start:float = 0, force_start:float = 0) -> np.ndarray:

        self.__calc_newmark_parameters()
        force_len = force_time_history.shape
        self.__set_initial_conditions(force_len,q_start,dot_q_start,force_start)

        q_now = self.q[:,0]
        dot_q_now = self.dot_q[:,0]
        ddot_q_now = self.ddot_q[:,0]

        for index,force in enumerate(force_time_history.T):

            newmark_force_next = force + self.A1*q_now + self.A2*dot_q_now + self.A3*ddot_q_now

            q_next = newmark_force_next/self.newmark_stiffness

            dot_q_next = (self.gamma/(self.beta*self.time_step))*(q_next-q_now)
            dot_q_next += (1-(self.gamma/self.beta))*dot_q_now
            dot_q_next += (1 - (self.gamma/(2*self.beta)))*self.time_step*ddot_q_now

            ddot_q_next = (1/(self.beta*self.time_step**2))*(q_next-q_now)
            ddot_q_next -= (1/(self.beta*self.time_step))*(dot_q_now)
            ddot_q_next -= ((1/(2*self.beta)) - 1)*ddot_q_now

            

            self.q[:,index+1] = q_next
            self.dot_q[:,index+1] = dot_q_next
            self.ddot_q[:,index+1]= ddot_q_next

            q_now = q_next
            dot_q_now = dot_q_next
            ddot_q_now = ddot_q_next

        return self.q.copy(),self.dot_q.copy(),self.ddot_q.copy()

    def displacement_response(self,mode_shapes:np.ndarray):
        
        displacement_total = None
        for mode_shape in mode_shapes:
            disp_factor = displacement_participation_factor(mode_shape).T
            displacement = np.array([np.multiply(q_value[:,np.newaxis],factor[np.newaxis,:]) for q_value,factor in zip(self.q,disp_factor)])
            if displacement_total is None:
                displacement_total = displacement
            else:
                displacement_total += displacement
        
        displacement_total = np.moveaxis(displacement_total,[1,2],[-1,0]) # Axis 0 = Floor, Axis 1 = Direction, Axis 2 = Time
        
        return displacement_total

    def get_peak_values(self, data:np.ndarray, probablity_non_exceedence: float|list, max_peak:bool = True) -> float|list:
         
        if probablity_non_exceedence is None:
                    raise ValueError('MRI value must be provided for peak calculation')

        if isinstance(probablity_non_exceedence, (int, float)):
            probablity_non_exceedence = [probablity_non_exceedence]
        epoches = 10
        peaks = []
        for floor in range(len(data)):
            floor_peak = []
            for direction in range(len(data[floor])):
                
                temp_disp = data[floor][direction]

                duration = int(len(temp_disp) - math.remainder(len(temp_disp),epoches))
                segments = np.array_split(temp_disp[:duration],epoches)
                if max_peak:
                    annual_maxima = np.max(segments,axis=1)
                    peak_value, _, _ = extreme_blue(annual_maxima, probablity_non_exceedence, 'max')
                else:
                    annual_maxima = np.min(segments,axis=1)
                    peak_value, _, _ = extreme_blue(annual_maxima, probablity_non_exceedence, 'min')
                floor_peak.append(peak_value)
            peaks.append(floor_peak)
        return np.array(peaks)


    def get_displacement_data(self, mode_shapes:np.ndarray, method:str, probablity_non_exceedence: float|list = None) -> float|list:

        displacement = self.displacement_response(mode_shapes)

        match method.lower():

            case 'max':
                return np.max(displacement,axis = 2)
            case 'min':
                return np.min(displacement,axis = 2)
            case 'mean':
                return np.mean(displacement,axis = 2)
            case 'std':
                return np.std(displacement,axis = 2)
            case 'peak_max':
                return self.get_peak_values(displacement, probablity_non_exceedence, max_peak = True)
            case 'peak_min':
                return self.get_peak_values(displacement, probablity_non_exceedence, max_peak = False)
            case 'peak_abs':
                max_value = self.get_peak_values(displacement, probablity_non_exceedence, max_peak = True)
                min_value = self.get_peak_values(displacement, probablity_non_exceedence, max_peak = False)

                abs_max = np.where(np.abs(max_value) >= np.abs(min_value), max_value, min_value)
                return abs_max


    def get_top_floor_displacement_data(self, mode_shapes:np.ndarray, method:str, probablity_non_exceedence: float|list = None) -> float|list:
        """
        Get the top floor displacement data for a given method.
        """
        return self.get_displacement_data(mode_shapes[:,-1], method, probablity_non_exceedence)

    def get_floor_displacement_data(self, floor_number:int, mode_shapes:np.ndarray, method:str, probablity__non_exceedence: float|list = None) -> float|list:
        """
        Get the floor displacement data for a given method.
        """
        return self.get_displacement_data(mode_shapes[:,floor_number], method, probablity__non_exceedence)

    def acceleration_response(self,mode_shapes:np.ndarray, natural_circular_frequency:np.ndarray) -> np.ndarray:
        
        acceleration_total = None
        for mode_shape in mode_shapes:
            accel_factor = displacement_participation_factor(mode_shape).T
            acceleration = np.array([np.multiply(q_value[:,np.newaxis],factor[np.newaxis,:]) for q_value,factor in zip(self.ddot_q,accel_factor)])
            if acceleration_total is None:
                acceleration_total = acceleration
            else:
                acceleration_total += acceleration

        acceleration_total = np.moveaxis(acceleration_total,[1,2],[-1,0]) # Axis 0 = Floor, Axis 1 = Direction, Axis 2 = Time
        
        return acceleration_total


    def get_acceleration_data(self, mode_shapes:np.ndarray, natural_circular_frequency:np.ndarray, method:str, probablity_non_exceedence: float|list = None) -> float|list:

        acceleration = self.acceleration_response(mode_shapes, natural_circular_frequency)

        match method.lower():

            case 'max':
                return np.max(acceleration,axis = 2)
            case 'min':
                return np.min(acceleration,axis = 2)
            case 'std':
                return np.std(acceleration,axis = 2)
            case 'peak_max':
                return self.get_peak_values(acceleration, probablity_non_exceedence, max_peak = True)
            case 'peak_min':
                return self.get_peak_values(acceleration, probablity_non_exceedence, max_peak = False)
            case 'peak_abs':
                max_value = self.get_peak_values(acceleration, probablity_non_exceedence, max_peak = True)
                min_value = self.get_peak_values(acceleration, probablity_non_exceedence, max_peak = False)

                abs_max = np.where(np.abs(max_value) >= np.abs(min_value), max_value, min_value)
                return abs_max
            

    def get_top_floor_acceleration_data(self, mode_shapes:np.ndarray, natural_circular_frequency:np.ndarray, method:str, probablity__non_exceedence: float|list = None) -> float|list:
        """
        Get the top floor acceleration data for a given method.
        """
        return self.get_acceleration_data(mode_shapes[:,-1], natural_circular_frequency,  method, probablity__non_exceedence)

    def get_floor_acceleration_data(self, floor_number:int, mode_shapes:np.ndarray, natural_circular_frequency:np.ndarray, method:str, probablity__non_exceedence: float|list = None) -> float|list:
        """
        Get the top floor acceleration data for a given method.
        """
        return self.get_acceleration_data(mode_shapes[:,floor_number], natural_circular_frequency, method, probablity__non_exceedence)
