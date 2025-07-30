import os
import h5py
import numpy as np


class SaveClass:
    def __init__(self,name:str,  output_folder:str = '', output_directory:str = ''):
        
        self.output_directory = output_directory
        if output_folder != '':
            self.output_directory = os.path.join(self.output_directory, output_folder)
        
        self.name = name

    @property
    def save_path(self):
        return f'{self.output_directory}/{self.name}.h5'
    
    def convert_datetime_array(self,datetime_array:np.ndarray):
        datetime_array = [np.datetime_as_string(i,timezone='UTC') for i in datetime_array]
        datetime_array = np.array(datetime_array,dtype='S30')
        return datetime_array

    def create_H5DF_file(self, overwrite:bool = True):

        if os.path.isfile(self.save_path) and (not overwrite):
            raise ValueError('File already exists. Please change the name of the file.')

        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

        self.file = h5py.File(self.save_path,'w')

    def meta_information(self, meta_data:dict):

        dset = self.file
        dset.attrs['facility_name'] = meta_data.facility_name
        dset.attrs['facility_location'] = meta_data.facility_location
        dset.attrs['test_date'] = meta_data.test_date
        dset.attrs['author'] = meta_data.author
        dset.attrs['author_email'] = meta_data.author_email
        dset.attrs['test_description'] = meta_data.test_description
        dset.attrs['test_wind_angle'] = meta_data.test_wind_angle
        dset.attrs['test_geometric_scale'] = meta_data.test_geometric_scale
        dset.attrs['test_time_scale'] = meta_data.test_time_scale
        dset.attrs['notes'] = meta_data.notes if meta_data else ''

    

    def save_climate_data(self,datatime:np.ndarray,wind_speed:np.ndarray,wind_direction:np.ndarray):

        dgroup = self.file.create_group('Climate_data')
        dset = dgroup.create_dataset('datetime',data = self.convert_datetime_array(datatime))
        dset.attrs['Description'] = 'Date and time (LST)'
        dset = dgroup.create_dataset('wind_speed',data = wind_speed)
        dset.attrs['Description'] = 'Wind speed (m/s)'
        dset = dgroup.create_dataset('wind_direction',data = wind_direction)
        dset.attrs['Description'] = 'Wind direction'

    def save_raw_pressure_data(self,raw_pressure_data:np.ndarray,reference_height:float,wind_angle:float):
        dgroup = self.file.create_group('Raw_cp_data')
        dset = dgroup.create_dataset('cp_data',data = raw_pressure_data,)
        dset.attrs['Description'] = 'Raw pressure coefficient data'
        dset.attrs['Reference_Velocity_Height'] = reference_height
        dset.attrs['Wind Angle of Attack'] = wind_angle
        dset = dgroup.create_dataset('pressure_taps',data = reference_height)

    def save_pressure_data(self,cp_data:np.ndarray,reference_height:float,wind_angle:float):
        
        dgroup = self.file.create_group('Pressure_data')
        dset = dgroup.create_dataset('cp_data',data = cp_data,)
        dset.attrs['Description'] = 'Pressure coefficient data'
        dset.attrs['Reference_Velocity_Height'] = reference_height
        dset.attrs['Wind Angle of Attack'] = wind_angle
        dset = dgroup.create_dataset('pressure_taps',data = reference_height)

    def save_base_forces_data(self,direction:str, base_moment_data:np.ndarray,reference_height:float,reference_velocity:float):
            
            dgroup = self.file.create_group('Base_force_coefficient')
            dset = dgroup.create_dataset(f'Base_force_coefficient',data = base_moment_data)
            dset.attrs['Description'] = 'Base moment coefficient'
            dset.attrs['Direction'] = direction
            dset.attrs['Reference_Velocity_Height'] = reference_height
            dset.attrs['Reference_Velocity'] = reference_velocity
            dset.attrs['Normalizing_Factors'] = ['0.5*rho*U^2*B*H',
                                                 '0.5*rho*U^2*D*H',
                                                 '0.5*rho*U^2*B*D',
                                                 '0.5*rho*U^2*D*H^2',
                                                 '0.5*rho*U^2*B*H^2',
                                                 '0.5*rho*U^2*B*D*H']
            
    def save_floor_forces_data(self,direction:str, floor_moment_data:np.ndarray,reference_height:float,reference_velocity:float):
                
                dgroup = self.file.create_group('Floor_force_coefficient')
                for floor_number in range(floor_moment_data.shape[0]):
                    dset = dgroup.create_dataset(f'Floor_force_coefficient_{floor_number}',data = floor_moment_data[floor_number])
                    dset.attrs['Description'] = 'Floor moment coefficient'
                    dset.attrs['Direction'] = direction
                    dset.attrs['Reference_Velocity_Height'] = reference_height
                    dset.attrs['Reference_Velocity'] = reference_velocity
                    dset.attrs['Normalizing_Factors'] = ['0.5*rho*U^2*B*H',
                                                        '0.5*rho*U^2*D*H',
                                                        '0.5*rho*U^2*B*D',
                                                        '0.5*rho*U^2*D*H^2',
                                                        '0.5*rho*U^2*B*H^2',
                                                        '0.5*rho*U^2*B*D*H']
       
    def save_lumped_mass_data(self,direction:str,mode_shapes:np.ndarray,natural_frequencies:np.ndarray,
                                   distributed_mass:np.ndarray, damping_ratio:np.ndarray,
                                   structural_responses:dict):

        if 'Structural_data' in self.file.keys():
            dgroup = self.file['Structural_data']
        else:
            dgroup = self.file.create_group('Structural_data')

        dgroup = dgroup.create_group(direction)
        
        dset = dgroup.create_dataset('mode_shapes',data = mode_shapes)
        dset.attrs['Description'] = 'Mode shapes'
        dset = dgroup.create_dataset('natural_frequencies',data = natural_frequencies)
        dset.attrs['Description'] = 'Natural frequencies'
        dset = dgroup.create_dataset('distributed_mass',data = distributed_mass)
        dset.attrs['Description'] = 'Distributed mass'
        dset = dgroup.create_dataset('structural_damping_ratio',data = damping_ratio)
        dset.attrs['Description'] = 'Structural damping ratio'

        dgroup = dgroup.create_group('structural_response')

        for key,value in structural_responses.items():
            dset = dgroup.create_dataset(key,data = value)
            dset.attrs['Description'] = key



    def close_file(self):
        self.file.close()