import numpy as np
import warnings

from openWLE.standard_values import Constants, WindProfileBase

class WindProfile(Constants):

    def __init__(self, profile_type:str, **kwargs):

        self.gradient_height = kwargs['gradient_height']

        if profile_type.lower() == 'logarithmic':
            roughness_length = kwargs['roughness_length']
            latitude = kwargs['latitude']
            
            if 'zero_plane_displacement' not in kwargs:
                warnings.warn('Zero plane displacement not provided. Assuming zero.')
                zero_plane_displacement = 0
            else:
                zero_plane_displacement = kwargs['zero_plane_displacement']
            
            u_star = kwargs['u_star'] if 'u_star' in kwargs else None
            u_ref = kwargs['u_ref'] if 'u_ref' in kwargs else None
            z_ref = kwargs['z_ref'] if 'z_ref' in kwargs else None
            

            self.profile = LogarithmicWindProfile(roughness_length=roughness_length,zero_plane_displacement=zero_plane_displacement
                                                  ,latitude=latitude,u_star=u_star,u_ref=u_ref,z_ref=z_ref)
        elif profile_type.lower() == 'power_law':
            u_ref = kwargs['u_ref']
            z_ref = kwargs['z_ref']
            alpha = kwargs['alpha']
            self.profile = PowerLawWindProfile(u_ref=u_ref,z_ref=z_ref,alpha=alpha)
        else:
            raise ValueError('Invalid profile type. Choose from logarithmic or power_law')
        
    def along_wind_velocity_profile(self, z):
        return self.profile.along_wind_velocity_profile(z)
    
    def gradient_wind_speed(self):
        return self.profile.along_wind_velocity_profile(self.gradient_height)

    def change_reference(self, z_original_ref:float, z_new_ref:float) -> None:

        u_original_ref = self.profile.along_wind_velocity_profile(z_original_ref)
        u_new_ref = self.profile.along_wind_velocity_profile(z_new_ref)

        velocity_ratio = u_original_ref/u_new_ref

        return velocity_ratio
    
    def save_wind_profile_data(self, HDF5_file:object) -> None:

        dgroup = HDF5_file.create_group('Wind_Profile_Details')
        dgroup.attrs['profile_type'] = self.profile.profile_name.lower()
        heights = np.hstack((np.linspace(1,50,98), np.linspace(50,1500,600)))
        velocity_profile = self.profile.along_wind_velocity_profile(heights)
        velocity_profile = np.vstack((heights,velocity_profile)).T
        dset = dgroup.create_dataset('Velocity_profile', data = velocity_profile)
        dset.attrs['Description'] = 'Full scale velocity profile used during analysis'
        dset.attrs['profile_type'] = self.profile.profile_name.lower()
        dset.attrs['Column_header'] = 'Column Header = Height [m], Velocity [m/s]'
        if self.profile.profile_name.lower() == 'logarithmic':
            
            dset.attrs['roughness_length'] = self.profile.roughness_length
            dset.attrs['u_star'] = self.profile.u_star
            dset.attrs['zero_plane_displacement'] = self.profile.zero_plane_displacement
            if self.profile.u_ref:
                dset.attrs['u_ref'] = self.profile.u_ref
            if self.profile.z_ref:
                dset.attrs['z_ref'] = self.profile.z_ref
        else:
            dset.attrs['alpha'] = self.profile.roughness_length
            dset.attrs['u_ref'] = self.profile.u_ref
            dset.attrs['z_ref'] = self.profile.z_ref




class LogarithmicWindProfile(WindProfileBase,Constants):
    
    def __init__(self, roughness_length:float, latitude:float, zero_plane_displacement:float = 0, u_star:float = None,
                  u_ref:float = None, z_ref:float = None, ):
        
        self.roughness_length = roughness_length
        self.zero_plane_displacement = zero_plane_displacement
        self.latitude = latitude

        if u_star is not None:
            self.u_star = u_star
        elif all([u_ref is not None, z_ref is not None]):
            self.u_ref = u_ref
            self.z_ref = z_ref
            self.calc_u_star(u_ref, z_ref)
        else:
            raise ValueError('Either u_star or u_ref and z_ref must be provided')
        
    @property
    def profile_name(self) -> str:
        return 'Logarithmic'

    def calc_u_star(self, u_ref:float, z_ref:float) -> None:
        f = 2*self.EARTH_ROTATION*np.sin(np.radians(self.latitude))
        self.u_star = (u_ref - (f*z_ref*34.5/self.VON_KARMAN))/(np.log(z_ref/self.roughness_length)/self.VON_KARMAN) # Based on Cook 1997 and ESDU 
        
    def along_wind_velocity_profile(self, z):

        f = 2*self.EARTH_ROTATION*np.sin(np.radians(self.latitude))
        return self.u_star/self.VON_KARMAN*np.log(z/self.roughness_length) + 34.5*f*z/self.VON_KARMAN # Based on Cook 1997 and ESDU
    

class PowerLawWindProfile(WindProfileBase,Constants):

    def __init__(self, u_ref, z_ref, alpha):
        self.u_ref = u_ref
        self.z_ref = z_ref
        self.alpha = alpha

    @property
    def profile_name(self) -> str:
        return 'Power_law'

    def along_wind_velocity_profile(self, z):
        return self.u_ref*(z/self.z_ref)**self.alpha
    
