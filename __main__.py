import glob,os

import pandas as pd
import numpy as np
import time
import h5py
import yaml
import classes




'''
File Name Format:

MATTI_18p1E01R001P001a

MATTI_18 - Base Name
p1 - Test Configuration (i.e. different building heights)
E01 - Testing Exposure (01 - first exposure, 02 - second exposure)
R001 - Currently Unknown
P001a - Testing Angle
'''


def test_angle(string):
    ''' Converts the wind tunnel naming for testing angle to a readable string'''
    return int(''.join(c for c in string if c.isdigit()))*10-10

def name_format(wind_tunnel_name, name_config):
    
    test_config = name_config['name_config']['test_building_config']
    test_exposure = name_config['name_config']['test_exposure']



    '''
    Testing Dictionaries

    This section holds the conversion data from the wind tunnel naming scheme to a naming scheme which is more readable.
    '''
    
    code_config=wind_tunnel_name[0:2]
    code_exposure=wind_tunnel_name[2:5]
    code_angle=wind_tunnel_name[-5:]
    
    return '{}_{}_Deg_{}'.format(test_config[code_config],test_exposure[code_exposure],test_angle(code_angle)), int(code_config.replace('p','')), test_exposure[code_exposure], test_angle(code_angle)

def main():

    current_dir=str(os.getcwd())
    folder_data_pressure= r'data\pressure'
    
    folder_data_output=r'data\output_datasets'


    # Import the config file
    
    with open(r'WTT_input_parameters.yml','r') as stream:
        config=yaml.load(stream)

        name_config=config[0]

        test_config=config[1:]


    if not os.path.exists(folder_data_output):
        os.makedirs(folder_data_output)

    
  
    os.chdir(folder_data_pressure)


    for file in glob.glob("*.pssr"):
        test_naming=file.replace(file[:-19],'').replace('.pssr','')
        name, test_config_index, test_exposure_current, test_angle_current = name_format(test_naming, name_config)
        
        test_config_current=test_config[test_config_index-1]
        test_class_current = classes.wind_tunnel_test(test_config_current)
        test_class_current.name = name
        test_class_current.exposure = test_exposure_current      
        test_class_current.angle = test_angle_current
        test_class_current.alpha = name_config['name_config']['alpha'][test_naming[2:5]]
        
        #Determine files to be used to extract the cp data
        
        file_pssr=file
        file_pssd=file[:-1]+"d"

        # Extracts the cp data from the pssr/pssd file provided by the wind tunnel
        test_class_current.cp_extraction(file_pssr,file_pssd)
        test_class_current.taps_per_floor()
        test_class_current.wind_loading()
        
        test_class_current.save(r'F:\Documents\School\Wind Tunnel Testing\Sample Test\datasets')

        
        #test_class_current.print_contour_simple()
        
        #test_class_current.print_contour_bokeh()
             
if __name__ == "__main__":
    
    main()
    
