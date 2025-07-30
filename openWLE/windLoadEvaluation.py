import os

import numpy as np
import pandas as pd
import math
import yaml
import warnings
import glob

from dataclasses import dataclass

from openWLE.exception import InputError,DataClassInputError
from openWLE.buildingClass import Building
from openWLE.pressureStudy import PressureStudy
from openWLE.climateStudy import ClimateStudy
from openWLE.windProfile import WindProfile
from openWLE.funcs import InputFileNaming
from openWLE.structuralStudy import StructuralStudy
from openWLE.visualization import VisualizationProcessor
from openWLE.save import SaveClass


@dataclass
class GeneralDetails(DataClassInputError):
    facility_name:str
    facility_location:str
    test_description:str
    test_date:str
    author:str
    author_email:str
    test_wind_angle:float
    test_geometric_scale:str
    test_time_scale:str
    raw_data_directory:str
    raw_data_extension:str
    test_type:str
    notes:str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@dataclass
class StudyDetails(DataClassInputError):
    study_name:str
    description:str
    notes:str


class windLoadEvaluation:

    def __init__(self, config_file):

        self.angle = None
        self.name = None
        self.alpha = None
        self.exposure = None
        self.velocity_conversion_factor = None
        
        config_file = self.split_config(config_file)
        self.setup(config_file)

    
    def split_config(self, config_file:str) -> dict:

        processed_config = {}
        for group in config_file:
            if group.keys() in list(processed_config.keys()):
                raise InputError('Duplicate configuration found in config file')
            else:
                group_name = list(group.keys())[0]
                processed_config[group_name] = list(group.values())[0]

        return processed_config


    def setup(self, config_file):


        # Check config files to determine type test
        if 'general_details' not in config_file.keys():
            raise InputError('General details missing in config file')
        self.general_details(config_file['general_details'])

        if 'name_config' not in config_file.keys():
            raise InputError('Naming configuration missing in config file')
        self.naming_config_setup(config_file['name_config'])

        if 'climate_config' in config_file.keys(): 
            self.climate_study_setup(config_file['climate_config'])
        else:
            self.climate_study = None

        if 'visualization_config' in config_file.keys(): 
            self.visualization_setup(config_file['visualization_config'])
        else:
            self.visualizer = None

        if 'wind_profile' not in config_file.keys():
            raise InputError('Wind profile missing in config file')
        self.wind_profile_setup(config_file['wind_profile'])

        if 'study_config' not in config_file.keys():
            raise InputError('Study configuration missing in config file')
        
        if 'save_config' in config_file.keys():
            self.output_config = config_file['save_config']
        else:
            self.output_config = None

        self.study_config_dict = {}
        study_config = self.split_config(config_file['study_config'])
        for dic in study_config.keys():
            if dic in self.study_config_dict.keys():
                raise InputError('Duplicate study configuration found in config file')
            else:
                temp = {}
                pressure_study = False
                structural_loading = False
                structural_analysis = False
                output_data = True
            group_config = self.split_config(study_config[dic])
            for key in group_config.keys():
                if key in temp.keys():
                    raise InputError(f'Duplicate key found in study configuration: {dic}')
                
                match key:
                    case 'study_details':
                        temp[key] = self.study_details_setup(dic,group_config['study_details'])
                    case 'building_config':
                        data = group_config['building_config']
                        temp[key] = self.building_setup(data)
                    case 'pressure_study':
                        pressure_study = True
                    case 'structural_loading':
                        temp[key] = group_config['structural_loading']
                    case 'structural_analysis':
                        structural_analysis = True

                    case _:
                        warnings.warn(f'Invalid key found in study configuration: {dic}')

                # Buillding config is required for pressure study, structural loading and structural analysis
            if pressure_study:
                data = group_config['pressure_study']
                temp['pressure_study'] = self.pressure_study_setup(data,temp['building_config'])
            if structural_loading:
                data = group_config['structural_loading']
                temp['structural_loading'] = self.structural_loading_setup(data)
            if structural_analysis:
                data = group_config['structural_analysis']
                temp['structural_analysis'] = self.structural_analysis_setup(data,temp['building_config'])



            self.study_config_dict[dic] = temp


        self.run(self.general_details.test_type)

        
    def general_details(self, config:dict) -> None:
        self.general_details = GeneralDetails(**config)


    def naming_config_setup(self, config:dict) -> None:

        """ Setup the naming configuration for the study. File are assumed to following the following naming convention:"""
        self.naming = InputFileNaming(config)
    
    def climate_study_setup(self, config:dict) -> None:
            
        self.climate_study = ClimateStudy(config)

    def visualization_setup(self, config:dict):

        self.visualizer = VisualizationProcessor(config)

    def study_details_setup(self, study_name:str, config:dict) -> StudyDetails:
        return StudyDetails(study_name,**config)


    def building_setup(self, config:dict) -> Building:

        return Building(config)

    def wind_profile_setup(self, config:dict) -> None:

        self.wind_profile_dict = {}
        wind_profile_config = self.split_config(config)
        for key in wind_profile_config.keys():
            if key in self.wind_profile_dict.keys():
                raise InputError(f'Duplicate key found in wind profile configuration: {key}')
            else:
                temp_config = self.split_config(wind_profile_config[key])
                self.wind_profile_dict[key] = WindProfile(**temp_config)

    def pressure_study_setup(self, config:dict, building:Building) -> PressureStudy:
        
        return PressureStudy(config, building)


    def structural_analysis_setup(self, config:dict, building:Building) -> None:
       
       return StructuralStudy(config,building)      
    
    def format_input_file(self, file:str) -> dict:

        return self.naming.extract(file)


    def import_WLE_data(self, data_file:str) -> dict:

            _, file_name = os.path.split(data_file)
            extracted_name = self.naming.extract(file_name)
            self.building_config = extracted_name['building_config']
            self.exposure = extracted_name['exposure']
            self.angle = extracted_name['angle']
            try :
                self.study_config_dict[self.building_config]['pressure_study']
            except KeyError:
                raise InputError(f'Building configuration {self.building_config} not found in study configuration')
            
            try:
                self.wind_profile_dict[self.exposure]
            except KeyError:
                raise InputError(f'Exposure {self.exposure} not found in wind profile configuration')               

            return self.building_config, self.exposure

            
    def run(self,test_type:str) -> None:
        
        match test_type.lower():

            case 'pressurestudy':
                data_files = glob.glob(f'{self.general_details.raw_data_directory}/*{self.general_details.raw_data_extension}')
                data_length = len(data_files)
                for index, data_file in enumerate(data_files):
                    building_config, wind_profile  = self.import_WLE_data(data_file)
                    self.run_pressure_study(data_file, building_config, wind_profile)
                    
                    if 'structural_loading' in self.study_config_dict[building_config].keys():
                        self.run_structural_loading(self.study_config_dict[building_config]['structural_loading'], self.study_config_dict[building_config]['building_config'])
                        
                    if 'structural_analysis' in self.study_config_dict[building_config].keys():
                        if 'structural_loading' not in self.study_config_dict[building_config].keys():
                            raise RuntimeError('Structural loading not found. Structural loading is required for strucutral analysis')
                        structural_study = self.study_config_dict[building_config]['structural_analysis']
                        self.run_structural_analysis(structural_study)
                    
                    if self.visualizer:
                        self.visualizer.wind_AOA = self.angle
                        final_plot = True if index == (data_length - 1) else False
                        self.run_visualization(final_plot)
                    if self.output_config:
                        self.save()
            case _:
                raise InputError(f'Invalid test type {test_type} found in config file. Test type must be pressureStudy currently. New test types to be introduced.')


    def run_climate_study(self) -> None:
        pass


    def run_pressure_study(self, file_path, building_config:str, exposure:str) -> None:
        
        self.study_config_dict[building_config]['pressure_study'].get_data(file_path) 
        z_original_ref = self.study_config_dict[building_config]['pressure_study'].recorded_reference_height
        z_new_ref = self.study_config_dict[building_config]['pressure_study'].new_reference_height
        velocity_ratio = self.wind_profile_dict[exposure].change_reference(z_original_ref, z_new_ref)
        self.velocity_conversion_factor = velocity_ratio**2
        self.study_config_dict[building_config]['pressure_study'].re_reference_cp_data(velocity_ratio)
        self.study_config_dict[building_config]['pressure_study'].assign_cp_data()

    def run_structural_loading(self, config:dict, building:Building) -> None:

        velocity = config['design_velocity']
        if 'design_air_density' in config.keys():
            air_density = config['design_air_density']
        else:
            air_density = 1.225

        building.design_velocity = velocity
        building.design_air_density = air_density

        if config['base_loading']:
            building.calculate_base_shears(velocity, air_density)
            building.calculate_base_moments(velocity, air_density)

        if config['floor_loading']:
            building.calculate_floor_forces(velocity, air_density)
            building.calculate_floor_moments(velocity, air_density)

        
    def run_structural_analysis(self, structural_study:StructuralStudy) -> None:
        
        structural_study.assign_loads()
        structural_study.run_analysis()
        
    def run_visualization(self, final_plot) -> None:
        
        print(self.angle)
        name_suffix = self.get_save_naming()
        building_instances = self.study_config_dict[self.building_config]['building_config']

        for task in self.visualizer.contour_plots:
            self.visualizer.process_contour_plots(name_suffix,building_instances,task)
        for task in self.visualizer.time_history_plots:
            self.visualizer.process_time_history_plots(name_suffix,building_instances,task)
        for task in self.visualizer.spectra_plots:
            self.visualizer.process_spectra_plots(name_suffix,building_instances,task)
        for task in self.visualizer.ring_plots:
            self.visualizer.process_ring_plots(name_suffix,building_instances,task)
        for task_id, task in self.visualizer.directional_plots.items():
            self.visualizer.process_directional_plots(name_suffix,building_instances,task_id,task, final_plot)



    def get_save_naming(self, base_name:str = None) -> str:

        if base_name:
            return f'{base_name}_{self.building_config}_{self.exposure}_{self.angle}'
        else:
            return f'{self.building_config}_{self.exposure}_{self.angle}'

    def save(self) -> None:
        
        name = self.get_save_naming(self.output_config['base_name'])
        output_folder = name
        directory = self.output_config['directory']

        saveData = SaveClass(name,output_folder,directory)

        saveData.create_H5DF_file()
        saveData.meta_information(self.general_details)
        
        if self.output_config['results']['raw_data']:
            self.study_config_dict[self.building_config]['pressure_study'].save_raw_cp_data(saveData.file)
        if self.output_config['results']['climate_data']:
            saveData.save_climate_data(self.climate_study.datetime,self.climate_study.wind_speed,self.climate_study.wind_direction)
        if self.output_config['results']['wind_profile']:
            self.wind_profile_dict[self.exposure].save_wind_profile_data(saveData.file)
        if self.output_config['results']['pressure_data']:
            self.study_config_dict[self.building_config]['pressure_study'].save_cp_data(saveData.file)
            self.study_config_dict[self.building_config]['pressure_study'].save_pressure_tap_data(saveData.file)
        if self.output_config['results']['structural_loading']:
            self.study_config_dict[self.building_config]['structural_analysis'].save_structural_loading(saveData.file)
        if self.output_config['results']['lumped_mass']:
            self.study_config_dict[self.building_config]['structural_analysis'].lumped_mass_system.save_lumped_mass_data(saveData.file)
        if self.output_config['results']['structural_analysis']:
            self.study_config_dict[self.building_config]['structural_analysis'].save_structural_analysis(saveData.file)
        saveData.close_file()

            
    

