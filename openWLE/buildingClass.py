import numpy as np
import itertools
from shapely import Polygon as PolyShape
from scipy.interpolate import griddata, bisplrep, bisplev, RBFInterpolator

from openWLE.geometry import plane_intersection, WLEPolygon, moment_arm_scalar
from openWLE.pressureTap import PressureTap



class Building:
    """
    Building Class
    Description: Contains the methods to determine the global forces on a building structre under wind loading.

    Attributes:


    Methods:

    
    """
    def __init__(self,config: dict) -> None:
        
        self.config = config
        self.origin = config['building_origin']
        self.x_axis = config['x_axis']
        self.y_axis = config['y_axis']
        self.z_axis = config['z_axis']
        self.building_height = config['building_height']
        self.number_of_floors = config['number_of_floors']
        self.first_floor_height = config['first_floor_height']

        self.pressure_tap_list = None

        self.surfaces = {}
        self.floors = {}

        if 'surfaces' in config.keys():
            for surface in config['surfaces']:
                name = list(surface.keys())[0]
                data = surface[name]
                self.add_surface(name,data)

        self.generate_floors()


        # Initializes check if structual loading has run
        self.design_velocity = None
        self.design_air_density = None
        self.floor_shear_run = False
        self.floor_moment_run = False
        self.base_shear_run = False
        self.base_moment_run = False


        self._displacement_time_history = None
        self._acceleration_time_history = None
        self.structural_analysis_time_step = 1

        self._force_sampling_frequency = None

    @property
    def surface_list(self) -> list:
        return list(self.surfaces.keys())

    @property
    def force_sampling_frequency(self) -> float:
        """
        Return the force sampling frequency
        """
        if self._force_sampling_frequency:
            return self._force_sampling_frequency
        else:
            for tap in self.pressure_tap_list:
                if tap.tap_cp is not None:
                    self._force_sampling_frequency = tap.sampling_rate
                    return self._force_sampling_frequency
        
    @property
    def displacement_time_history(self) -> np.ndarray:
        return self._displacement_time_history
    
    @displacement_time_history.setter
    def displacement_time_history(self, value:np.ndarray) -> None:
        self._displacement_time_history = value

    @property
    def acceleration_time_history(self) -> np.ndarray:
        return self._acceleration_time_history
    
    @acceleration_time_history.setter
    def acceleration_time_history(self, value:np.ndarray) -> None:
        self._acceleration_time_history = value

    def get_taps_by_ids(self, tap_id_list:list) -> list:

        taps = []
        for tap_id in tap_id_list:
            for pressure_tap in self.pressure_tap_list:
                if pressure_tap.tap_id == tap_id:
                    taps.append(pressure_tap)
                else:
                    taps.append(None)

        return taps

    def get_horizontal_tap_ring_elevations(self, tolerance: float = 0.01) -> list:

        """
        Get unique horizontal tap elevations
        """

        elevations = []
        for tap in self.pressure_tap_list:
            if tap is not None:
                if 1 - np.abs(np.dot(tap.normal_vector, self.z_axis)) <= tolerance:
                    continue 
                if np.round(tap.z,5) not in [np.round(elev,5) for elev in elevations]:
                    elevations.append(tap.z)

        # Remove taps that are within the tolerance
        unique_elevations = []
        for elev in elevations:
            if not any([abs(elev - unique_elev) < tolerance for unique_elev in unique_elevations]):
                unique_elevations.append(elev)
        unique_elevations.sort()
        return unique_elevations
        

    def generate_floors(self) -> None:
        """
          Calculate the data to generate the floor objects
        """

        if self.first_floor_height is None:

            tributary_height = [(self.building_height/self.number_of_floors)/2]*2
            # Ground Floor
            self.add_floor(0,[0,tributary_height[1]],0)
            # Add Intermediate Floors
            for floor_num in range(1,self.number_of_floors):
                self.add_floor(floor_num,tributary_height,self.tributary_height*floor_num)
            # Roof Floor
            self.add_floor(self.number_of_floors,[self.tributary_height/2,0],self.building_height)
        else:

            tributary_height = (self.building_height-self.first_floor_height)/(self.number_of_floors - 1)
            # Ground Floor
            self.add_floor(0,[0,self.first_floor_height/2],0)
            # Add First Floor
            self.add_floor(1,[self.first_floor_height/2,tributary_height/2],self.first_floor_height)
            # Add Intermediate Floors
            for floor_num in range(2,self.number_of_floors):
                self.add_floor(floor_num,[tributary_height/2]*2, self.first_floor_height + (floor_num-1)*tributary_height)    
            # Roof Floor
            self.add_floor(self.number_of_floors,[tributary_height/2,0],self.building_height)

    def add_floor(self, floor_num: int, tributary_height: list, floor_elev: float) -> None:
        """
        Add a floor to the building
        """

        origin = (self.origin[0],self.origin[1],floor_elev)
        self.floors[floor_num] = BuildingFloor(floor_num,tributary_height,floor_elev,origin)

    def add_surface(self,name: str, data: dict) -> None:
        """
        Add a surface to the building
        """

        self.surfaces[name] = BuildingSurface(name,**data)

    def assign_floor_taps(self) -> None:
        """
        Assign taps to the surfaces
        """

        for floor in self.floors.values():
            floor.find_floor_taps(list(self.surfaces.values()))


    def assign_surface_taps(self, surface_name: str, tap_list: list ) -> None:
        """
        Assign taps to the surfaces
        """ 
        
        self.surfaces[surface_name].taps = tap_list.copy()
        self.surfaces[surface_name].set_tap_parameters(self.z_axis)
        
        if self.pressure_tap_list == None:
            self.pressure_tap_list = tap_list
        else:
            self.pressure_tap_list += tap_list


    def calculate_base_shears(self, velocity: float, air_density: float) -> np.ndarray:
        """
        Calculate the base shear of the building
        """
        self.force_x = 0
        self.force_y = 0
        self.force_z = 0

        for surface in self.surfaces.values():
            force_x, force_y, force_z = surface.force_calculation(velocity, air_density, self.x_axis, self.y_axis, self.z_axis)
            self.force_x += force_x
            self.force_y += force_y
            self.force_z += force_z

        self.force_x = np.round(self.force_x,5)
        self.force_y = np.round(self.force_y,5)
        self.force_z = np.round(self.force_z,5)
        
        self.base_shear_run = True

        return self.force_x, self.force_y, self.force_z

    def calculate_base_moments(self,  velocity: float, air_density: float) -> None:
        """
        Calculate the base moment of the building
        """
        
        self.moment_x = 0
        self.moment_y = 0
        self.moment_z = 0

        for surface in self.surfaces.values():
            moment_x, moment_y, moment_z = surface.moment_calculation(velocity, air_density, self.x_axis, self.y_axis, self.z_axis)
            self.moment_x += moment_x
            self.moment_y += moment_y
            self.moment_z += moment_z

        self.moment_x = np.round(self.moment_x,5)
        self.moment_y = np.round(self.moment_y,5)
        self.moment_z = np.round(self.moment_z,5)

        self.base_moment_run = True
        return self.moment_x, self.moment_y, self.moment_z
    

    def calculate_floor_forces(self, velocity: float, air_density: float) -> dict:
        """
        Calculate the forces on each floor of the building
        """

        self.floor_forces = {}

        for floor in self.floors.values():
            force_x, force_y, force_z = floor.forces_calculation(velocity, air_density, self.x_axis, self.y_axis, self.z_axis)
            self.floor_forces[floor.floor_num] = [force_x,force_y,force_z]

        self.floor_shear_run = True
        return self.floor_forces
    
    def calculate_floor_moments(self, velocity: float, air_density: float) -> dict:
        """
        Calculate the moments on each floor of the building
        """

        self.floor_moments = {}
        for floor in self.floors.values():
            moment_x, moment_y, moment_z = floor.moment_calculation(velocity, air_density, self.x_axis, self.y_axis, self.z_axis)
            self.floor_moments[floor.floor_num] = [moment_x,moment_y,moment_z]
        self.floor_moment_run = True
        return self.floor_moments


    def get_lumped_mass_floor_forces(self) -> dict:
        
        force_x = []
        force_y = []
        moment_z = []

        for forces,moments in zip(self.floor_forces.values(),self.floor_moments.values()):
            force_x.append(forces[0])
            force_y.append(forces[1])
            moment_z.append(moments[2])

        floor_loads = {
            'X':force_x,
            'Y': force_y,
            'Theta': moment_z
        }

        return floor_loads

    def get_base_forces(self) -> dict:

        base_moments = {
            'X': self.moment_x,
            'Y': self.moment_y,
            'Theta': self.moment_z}
        
        return base_moments


    def save_building_data(self, HDF5file: object) -> None:

        dgroup = HDF5file.create_group('Building_data')
        dset = dgroup.create_dataset('Building_origin',data = self.origin)
        dset.attrs['Description'] = 'Building origin'
        dset = dgroup.create_dataset('Building_x_axis',data = self.x_axis)
        dset.attrs['Description'] = 'Building x-axis'
        dset = dgroup.create_dataset('Building_y_axis',data = self.y_axis)
        dset.attrs['Description'] = 'Building y-axis'
        dset = dgroup.create_dataset('Building_z_axis',data = self.y_axis)
        dset.attrs['Description'] = 'Building z-axis'
        dset = dgroup.create_dataset('Building_height',data = self.building_height)
        dset.attrs['Description'] = 'Building height'

        floors = []
        for floor in self.floors.values():
            floor_data = [floor.floor_num, floor.elevation, floor.tributary_height]
            floors.append(floor_data) 
        dset = dgroup.create_dataset('Floors',data = np.array(floors))
        dset.attrs['Description'] = 'Floor information'
        dset.attrs['Column_desciption'] = 'Floor Number, Floor Elevation[m], Tributary Height[m]'
 

class BuildingSurface():
    
    """
    Surface Class
    Description: Contains the methods to perform surface based calculations such as generating pressure contours

    Attributes:

    name: Name associated with the surface (ie. roof, west_wall).
    taps: List of all PressureTap Objects associated with this surface.
    width: Width of the surface.
    height: Height of the surface.
    tangent_vector_1: Vector that is tangent to the surface.
    tangent_vector_2: Vector that is tangent to the surface.
    origin: Origin of the surface.


    Methods:
    grid_data: Take data from pressure taps and generate a grid of data for contour plots.

    """


    def __init__(self, name: str, bounding_vertices: list, tangent_vector_1: tuple, tangent_vector_2: tuple, origin: tuple):
        
        self.name = name
        self.bounding_vertices = np.array(bounding_vertices)
        self.tangent_vector_1 = tangent_vector_1
        self.tangent_vector_2 = tangent_vector_2
        self.origin = origin

        self.taps = None
        self.polygon = WLEPolygon(bounding_vertices,origin,tangent_vector_1,tangent_vector_2)
        self.polygon.local_coordinate_parameters()

    
    @property
    def normal_vector(self) -> np.array:
        return self.polygon.normal_vector
    

    @property
    def bounding_dims(self) -> list:

        data =  self.polygon.bounding_dims
        min_x = data[0]
        max_x = data[2]
        min_y = data[1]
        max_y = data[3]

        return (min_x,max_x,min_y,max_y)
    
    @property
    def width(self) -> float:
        return self.polygon.width
    
    @property
    def tap_id_list(self) -> list:
        return [tap.tap_id for tap in self.taps]
    
    @property
    def tap_positions(self) -> tuple:

        tap_id = [tap.tap_id for tap in self.taps]
        coords = [tap.coords for tap in self.taps]

        local_coords = self.polygon.global_to_local(global_coords=coords)

        return (tap_id, local_coords)

    def set_tap_parameters(self, z_axis: tuple) -> None:
        """
        Assign taps parameter associated with the surface
        """

        tap_coords = [tap.coords for tap in self.taps]
        area_polygons = self.polygon.voronoi_polygons(tap_coords)

        for area,tap in zip(area_polygons,self.taps):
            tap.tributary_area = area
            tap.normal_vector = self.normal_vector
            tap.moment_arm = self.get_moment_arm(tap,z_axis)
            tap.surface = self.name
                
    def set_tap_normal_vector(self) -> None:
        """
        Set the normal vector of the surface
        """
        
        for tap in self.taps:
            tap.normal_vector = self.normal_vector

    def set_tap_moment_arm(self) -> None:

        for tap in self.taps:
            tap.moment_arm = self.get_moment_arm(tap)

    def get_moment_arm(self,tap: PressureTap, z_axis: tuple) -> float:

        tap_vector = tap.coords - self.origin
        if np.dot(tap_vector,z_axis) == 0:
            moment_arm = 0
        else:
            moment_arm = moment_arm_scalar(tap_vector, self.normal_vector, np.array([0,0,1]))
        return moment_arm

    def force_calculation(self,velocity: float, air_density: float,  x_axis: tuple, y_axis: tuple, z_axis:tuple ) -> np.ndarray:

        self.force_x = 0
        self.force_y = 0
        self.force_z = 0

        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)
        z_axis = z_axis/np.linalg.norm(z_axis)
                              
        
        for tap in self.taps:
            force = tap.force_calculation(velocity, air_density)

            self.force_x += force*np.dot(tap.normal_vector,x_axis)
            self.force_y += force*np.dot(tap.normal_vector,y_axis)
            self.force_z += force*np.dot(tap.normal_vector,z_axis)

        return self.force_x, self.force_y, self.force_z
    
    def moment_calculation(self, velocity: float, air_density: float, x_axis: tuple, y_axis: tuple, z_axis:tuple ) -> np.ndarray:

        self.moment_x = 0
        self.moment_y = 0
        self.moment_z = 0

        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)
        z_axis = z_axis/np.linalg.norm(z_axis)

        for tap in self.taps:
            force = tap.force_calculation(velocity, air_density)
            self.moment_y += force*np.dot(tap.normal_vector,x_axis) * (tap.z - self.origin[2])
            self.moment_x += force*np.dot(tap.normal_vector,y_axis) * (tap.z - self.origin[2])
            self.moment_z += force * -tap.moment_arm 

        return self.moment_x, self.moment_y, self.moment_z
    
    def get_surface_data(self, method:str, skip_taps:int|list, 
                         starting_index:int=0, 
                         mri: float = 50, 
                         peak_epoches: int = 10, 
                         duration_ratio: float = None) -> tuple:
        """
        Return the method to generate surface data
        """        
        if isinstance(skip_taps,int):
            skip_taps = [skip_taps]

        dim_1 = []
        dim_2 = []
        data = []
        for tap in self.taps:
            if tap.tap_id in skip_taps:
                continue
            
            tap_local_position = self.polygon.global_to_local([tap.coords])[0]
            dim_1.append(tap_local_position[0])
            dim_2.append(tap_local_position[1])
            

            match method.lower(): #TODO Move this logic outside of loop
                case 'peak_max':
                    probabily_non_exceedence = 1 - 1/mri
                    value = tap.max_peak_pressure([probabily_non_exceedence],peak_epoches)
                    data.append(value[0])
                case 'peak_min':
                    
                    probabily_non_exceedence = 1 - 1/mri
                    value = tap.min_peak_pressure([probabily_non_exceedence],peak_epoches)
                    data.append(value[0])  
                    
                case 'peak_abs':
                    probabily_non_exceedence = 1 - 1/mri
                    max_value = tap.max_peak_pressure([probabily_non_exceedence],peak_epoches,starting_index, duration_ratio)
                    min_value = tap.min_peak_pressure([probabily_non_exceedence],peak_epoches,starting_index, duration_ratio)
                    
                    data.append(max([max_value[0],min_value[0]], key=abs))
                case 'max':
                    data.append(np.max(tap.tap_cp[starting_index:]))
                case 'min':
                    data.append(np.min(tap.tap_cp[starting_index:]))
                case 'mean':
                    data.append(np.mean(tap.tap_cp[starting_index:]))
                case 'rms':
                    data.append(np.std(tap.tap_cp[starting_index:]))
                case 'time_history':
                    data.append(tap.tap_cp[starting_index:])

        return (dim_1, dim_2, np.array(data))
    
    def estimate_missing_taps(self, missing_taps: list,) -> None:

        
        
        dim_1, dim_2, data = self.get_surface_data('time_history',missing_taps)
        tap_id_list = self.tap_id_list
        temp_missing_taps = [self.taps[tap_id_list.index(tap)] for tap in (missing_taps)]
        local_coords = np.array([np.array(self.polygon.global_to_local(global_coords=[tap.coords])[0]) for tap in temp_missing_taps])
        tap_data = self.interpolate_extrapolate_data_birep([dim_1,dim_2],data,local_coords)
        # tap_data = self.interpolate_extrapolate_data_rbf([dim_1,dim_2],data,local_coords)
        for index,tap in enumerate(temp_missing_taps):
            tap.tap_cp = tap_data[:,index]

        # for tap in missing_taps:
        #     if tap in tap_id_list:
        #         tap = self.taps[tap_id_list.index(tap)]
        #         local_coords = np.array(self.polygon.global_to_local(global_coords=[tap.coords])[0])
        #         tap_data = self.interpolate_extrapolate_data_birep([dim_1,dim_2],data,local_coords)
        #         tap.tap_cp = tap_data


    def interpolate_extrapolate_data_birep(self, dims: tuple, data: np.ndarray, output_points:np.ndarray ) -> np.ndarray:
        
        dims = np.array(dims)

        if len(data.shape) == 1:
            bisp_factors = bisplrep(dims[0],dims[1], data, s=0.01)
            new_data = bisplev(output_points[0],output_points[1],bisp_factors)

        else:
            new_data = []
            for th in range(data.shape[-1]):
                temp_data = data[:,th]

                bisp_factors = bisplrep(dims[0],dims[1], temp_data, s = 0.1)
                # bisp_factors = bisplrep(dims[0],dims[1], temp_data)
                temp_results = [bisplev(point[0],point[1],bisp_factors) for point in output_points]
                new_data.append(temp_results)

            new_data = np.array(new_data)
        return new_data

    def interpolate_extrapolate_data_rbf(self, dims: tuple, data: np.ndarray, output_points:np.ndarray ) -> np.ndarray:

        dims = np.array(dims).T
        new_data = RBFInterpolator(dims, data, kernel = 'gaussian', epsilon = 1)(output_points).T

        return new_data




    def point_data_to_grid(self, points: tuple, data:np.ndarray, output_dims:tuple, extrapolate:bool = False, num_x:int = 50, num_y:int = 50) -> tuple:

        x_min, x_max, y_min, y_max = output_dims
        new_x = np.linspace(x_min, x_max, int(num_x))
        new_y = np.linspace(y_min, y_max, int(num_y))

        grid_x, grid_y = np.meshgrid(new_x,new_y)

        inner_data = griddata(points,data, (grid_x, grid_y), method = 'cubic')

        if extrapolate:            
            remove =  np.invert(np.isnan(inner_data))

            mod_grid_x = grid_x[remove]
            mod_grid_y = grid_y[remove]
            temp_data = inner_data[remove]
            
            temp_x = np.reshape(mod_grid_x,mod_grid_x.size)
            temp_y = np.reshape(mod_grid_y,mod_grid_y.size)
            temp_data = np.reshape(temp_data,temp_data.size)

            # output_points = list(itertools.product(new_x,new_y))

            # bisp_factors = bisplrep(temp_x, temp_y , temp_data,)
            # temp_data = np.array([bisplev(x,y,bisp_factors) for x,y in output_points])
            # temp_data = temp_data.reshape((len(new_x),len(new_y))).T

            temp_data = griddata((temp_x,temp_y),temp_data, (grid_x, grid_y), method = 'nearest')
            

            inner_data[np.isnan(inner_data)] = temp_data[np.isnan(inner_data)]

        return grid_x, grid_y, inner_data

     

    # def get_grid_data_func(self, dims:list, data:np.ndarray) -> list:

    #     points = np.vstack((dims))

    #     new_x = np.linspace(np.min(dims[0]),np.max(dims[0]), 50)
    #     new_y = np.linspace(np.min(dims[1]),np.max(dims[1]), 50)

    #     grid_x, grid_y = np.meshgrid(new_x,new_y)

    #     zz = griddata(points.T,data,(grid_x,grid_y), method = 'cubic')

    #     return (grid_x, grid_y, zz, new_x, new_y)
        
    
    # def interpolate_extrapolate_data(self, dims: list, data: np.ndarray, output_points:list, grid:bool = False ) -> tuple:
        
        
    #     new_points = np.array(output_points)
    #     dims = np.array(dims).T

    #     inner_data = griddata(dims,data,new_points,'cubic',fill_value = np.nan)
    #     x,y = new_points.T
    #     _, outside_data = self.interpolate_extrapolate_data_birep(dims.T,data,(x,y),True)
    #     outside_data = np.array(outside_data)
    #     print(np.isnan(inner_data))
    #     #inner_data[np.isnan(inner_data)] = outside_data[np.isnan(inner_data)]
    #     new_data = inner_data.copy()
    #     return (new_points, new_data)
    
    # def interpolate_extrapolate_data_birep(self, dims: list, data: np.ndarray, output_points:list, grid:bool = False ) -> tuple:
        
    #     xb, xe, yb, ye = self.bounding_dims

    #     if len(data.shape) == 1:
    #         #new_data_function = RectBivariateSpline(dims[0],dims[1],data, bbox = self.bounding_dims)
    #         new_data_function = bisplrep(dims[0],dims[1], data, xb=xb, xe = xe, yb = yb, ye = ye)
            
    #         new_data = [bisplev(x,y,new_data_function) for x,y in zip(output_points[0],output_points[1])]
    #     else:
    #         new_data = []
    #         for th in range(0,data.shape[1]):
    #             hold_data = data[:,th]
    #             new_data_function = bisplrep(dims[0],dims[1],hold_data, xb=xb, xe = xe, yb = yb, ye = ye)
    #             new_data.append(bisplev(output_points[0],output_points[1],new_data_function))
            
    #         new_data = np.array(new_data).T
            
    #     if not grid:
    #         new_data = new_data.flatten()

    #     return (output_points, new_data)

    
    # def estimate_missing_taps (self, missing_taps: list,) -> None:
    #     """
    #     Estimate the missing taps for the current surface
    #     """

    #     tap_id_list = self.tap_id_list
    #     if not any([ tap in tap_id_list for tap in missing_taps]):
    #         return None
    #     dim_1, dim_2, data = self.get_surface_data('time_history',missing_taps)
    #     for tap in missing_taps:
    #         if tap in tap_id_list:
    #             tap = self.taps[tap_id_list.index(tap)]
    #             local_coords = self.polygon.global_to_local(global_coords=[tap.coords])[0]
    #             _, tap_data = self.interpolate_extrapolate_data_birep([dim_1,dim_2],data,local_coords)
    #             tap.tap_cp = tap_data

    def contour_grid_points(self,num_x:int = 50, num_y:int = 50) -> list:

        surface_dims = self.bounding_dims
        x_point = np.linspace(surface_dims[0],surface_dims[1], num_x)
        y_point = np.linspace(surface_dims[2],surface_dims[3], num_y)

        points = list(itertools.product(x_point,y_point))

        return points, x_point, y_point
    


class BuildingFloor:
    '''
    Building Floor Class
    Description: Contains information about building floor and the taps used to estimate the force at the current floor.

    General Variables:

        floor_num = the current floor number 
        tributary_height = a list containing two values: the tributary height below the floor elevation and the tributary height above the floor elevation.  
        floor_elevation = the current floors elevation above the ground.
        tap_list = a list of pressure taps which are associated with the current floor. These taps will be used to estimate the force time histories of the current floor.

    '''

    
    def __init__(self, floor_num: int, tributary_height: list, floor_elev:float, origin:tuple) -> None:

        self.floor_num=floor_num
        self.tributary_height = tributary_height
        self.elevation = floor_elev
        self.origin = np.array(origin)
        self.tap_list =[]
        self.floor_tap_trib_area = {}
        
    
    def find_floor_taps(self, surfaces: list) -> None:

        for index in range(len(surfaces)):
            surface = surfaces[index]
            check = np.cross(surface.normal_vector, np.array((0,0,1)))
            # check if surface is horizontal
            if np.round(np.linalg.norm(check),5) == 0:
                continue
            else:
                surface_normal = surface.normal_vector/np.linalg.norm(surface.normal_vector)
                intersection_polygon = self.create_intersection_polygon(surface)

                tap_polygons = [tap.tributary_area for tap in surface.taps]
                floor_tap_polygons = [polygon.intersection(intersection_polygon) for polygon in tap_polygons]
  
                for tap_index, tap_polygon in enumerate(floor_tap_polygons):
                    tap_id = surface.taps[tap_index].tap_id
                    if tap_polygon.area > 0:
                        area = np.round(tap_polygon.area,5)
                        moment_arm = self.get_moment_arm(tap_polygon, surface)
                        self.floor_tap_trib_area[tap_id] = [area,surface_normal, moment_arm]
                        self.tap_list.append(surfaces[index].taps[tap_index])
                        
                
    def create_intersection_polygon(self, surface: BuildingSurface) -> PolyShape:
        
        lower_plane = (0,0,1,-(self.elevation - self.tributary_height[0])) 
        lower_plane = np.round(np.array(lower_plane),5)
        upper_plane = (0,0,1,-(self.elevation + self.tributary_height[1]))
        upper_plane = np.round(np.array(upper_plane),5)
        
        surface_plane = surface.polygon.plane_equation()
        upper_line = plane_intersection(upper_plane,surface_plane)
        lower_line = plane_intersection(lower_plane,surface_plane)
        intersection_points = surface.polygon.intersection([upper_line,lower_line])
        intersection_polygon = PolyShape(intersection_points)

        return intersection_polygon

    def get_moment_arm(self, area_polygon: PolyShape, building_surface:WLEPolygon) -> tuple:
        """
        Get the tributary area of the tap
        """
        surface_normal = building_surface.normal_vector/np.linalg.norm(building_surface.normal_vector)
        area_centroid = area_polygon.centroid
        area_centroid = np.array([[area_centroid.x, area_centroid.y]])
        area_centroid = building_surface.polygon.local_to_global(local_coords = area_centroid)
        tap_vector = area_centroid - self.origin
        moment_arm = moment_arm_scalar(tap_vector, surface_normal, np.array([0,0,1]))
        return moment_arm

    def forces_calculation(self, velocity: float, air_density: float, x_axis: tuple, y_axis: tuple, z_axis:tuple ) -> np.ndarray:
        
        force_x = 0
        force_y = 0
        force_z = 0

        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)
        z_axis = z_axis/np.linalg.norm(z_axis)
                              
        
        for tap in self.tap_list:
            area, normal_vector, _ = self.floor_tap_trib_area[tap.tap_id]
            force = tap.force_calculation(velocity, air_density, area)
            
            force_x += force*np.dot(normal_vector,x_axis)
            force_y += force*np.dot(normal_vector,y_axis)
            force_z += force*np.dot(normal_vector,z_axis)

        self.force_x = np.round(force_x,5)
        self.force_y = np.round(force_y,5)
        self.force_z = np.round(force_z,5)

        return self.force_x, self.force_y, self.force_z

    def moment_calculation(self,  velocity: float, air_density: float, x_axis: tuple, y_axis: tuple, z_axis:tuple ) -> np.ndarray:
        moment_x = 0
        moment_y = 0
        moment_z = 0

        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)
        z_axis = z_axis/np.linalg.norm(z_axis)
                              
        
        for tap in self.tap_list:

            area, normal_vector,moment_arm = self.floor_tap_trib_area[tap.tap_id]
            force = tap.force_calculation(velocity, air_density, area)

            moment_y += force*np.dot(normal_vector,x_axis) * self.elevation
            moment_x += force*np.dot(normal_vector,y_axis) * self.elevation
            moment_z += force* - moment_arm

        self.moment_x = np.round(moment_x,5)
        self.moment_y = np.round(moment_y,5)
        self.moment_z = np.round(moment_z,5)

        return self.moment_x, self.moment_y, self.moment_z



