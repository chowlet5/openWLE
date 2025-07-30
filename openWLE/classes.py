import numpy as np
import pandas as pd

class pressure_tap:
    
    '''
    Pressure Tap Class
    Description: Contains infomation for each tap using in the wind tunnel model.

    General Variables:

        tap_id = numerical value or name of the tap (typically the one using the model). Example: 701, 1616
        tap_index = the current taps location/index within the main cp array obtained from the wind tunnel.
        home_wall = the name of the wall that the current tap is located on. Example: 'north', 'south'
        x = the x coordinate of the pressure tap. The center of the wall is consider zero which means x can be negative. The coordinates of full scale values.
        z = the z coordinate of the pressure tap/ the height in which the tap is located. The bottom of the wall is considered zero. The coordinates of full scale values.
        z_index = the index/location of the current tap's height within the height matrix.
        surface_width = the width dimension of the current home surface.
        surface_height = the height dimension of the current home wall.
        tap_cp = the cp time history for the tap.

    Additional Variables


        left = the tap id of the pressure tap located to the left of the current tap. If there is no tap to the left, the variable is set as "wall_left".
        right = the tap id of the pressure tap located to the right of the current tap. If there is no tap to the right, the variable is set as "wall_right".
        above = the tap id of the pressure tap located above of the current tap. If there is no tap above, the variable is set as "top".
        below = the tap id of the pressure tap located below of the current tap. If there is no tap below, the variable is set as "ground".

        width = the width dimension of the tributary area of the tap.
        height = the height dimension of the tributary area of the tap. This value only considered the pressure taps and is different than the floor height of the structure
        r = moment arm of the pressure tap from the center of the wall. Measured from the center of the wall to the center of the tap. 
            The value can be either negative (clockwise rotation/negative moment) or positive (counterclockwise rotation/ positive moment).

        

    Class Methods:

    __init__ (self, tap_id, tap_index, x, z, z_index = None, home_wall = None, surface_width = None, surface_height=None)
        Import the information required to create the pressure_tap instances. Required Information: tap_id, tap_index, x, z. The other values are set to None by default and can be set later on.

    surrounding_taps(self,tap_array,x_coor,z_coor)
        Copies the tap_id of the surrounding taps (above, left, below, right).The copied ids are placed in the corresponding variable (i.e. the tap id to the left is placed in the variable left.) 
            tap_array is an array of the tap_ids. x_coor is a list of the different x coordinates used for the current surface. z_coor is a list of the different z coordinates used on the current surface.

    tap_trib_dim_nc(self,pressure_taps)
        Determines the dimensions used to calculate the tributary area (also r). This method is used when the origin is bottom left corner of the surface. To use this method,
            the x and z coordinates provided in the __init__ method must be calcuated using the bottom left corner of the surface. pressure_taps is a dictionary of pressure_tap instances.
    tap_trib_dim_nc(self,pressure_taps)
        Determines the dimensions used to calculate the tributary area (also r). This method is used when the origin is the geometric center of the build. 
            pressure_taps is a dictionary of pressure_tap instances.
    trib_area(self,floor_height=None)
        Calcuates and returns the tributary area of the current tap. If a floor height is provide, the area will be calculated using the tributry width and the floor height.
            If a floor height is not provide, the area will be calculated using the tributry width and height.
    find_tap_cp (self, cp_data)
        Finds and stores the cp time history of the current tap. cp_data is the full array contains all the cp time histories from the wind tunnel.
            Returns the cp time history.
    cp_fix (self,pressure_taps)
        This method should be called current pressure tap has been identified as missing. If the tap is located inbetween two taps (along the x axis/ same height), the cp is estimated by taking an average between the two adjacent taps.
            If the tap is along one of the edges (if the left or right variable is set as "wall_left" or "wall_right" respectively), the cp is estimated using the adjacent tap.  
            pressure_taps is a dictionary of pressure taps instances. Returns the estimated cp time history.
    cp_fix_vertical(self,pressure_taps)
        This method should be called current pressure tap has been identified as missing and the adjacent taps are also missing (Typically used if there is a ring of bad taps/ broken scanner). 
            The cp is estimated using an average of cp values from the tap above and below the current tap.  
            pressure_taps is a dictionary of pressure taps instances. Returns the estimated cp time history.
    '''

    def __init__(self, tap_id, tap_index, x, z, z_index = None, home_wall = None, surface_width = None, surface_height=None):

        self.tap_id=tap_id                        
        self.tap_index= int(tap_index)
        self.home_wall = home_wall                   
        self.x = x
        self.z = z
        self.z_index = z_index
        self.surface_width = surface_width
        self.surface_height = surface_height
        self.tap_cp = None

    def surrounding_taps(self,tap_array,x_coor,z_coor):
        
        '''
        Description/ Methodology 

        This method uses a new methodology to determine the surrounding taps for a given tap. It will continune to check its surroundings until if finds a usable tap or the wall. This is an
        improvement over the pervious design of this function. 

        left = tap to the left of the current tap.
        below = tap below the current tap.
        right = tap to the right of the current tap.
        above = tap above the current tap.

        These values are later used to calculate the tributary area dimensions of the tap.

        if x_index is zero, that means that the current tap is the left most tap within the ring and therefore, no taps are located to the left of the current tap.
        if x_index is equal to the last index of the list, that means that the current tap is the right most tap along the ring and therefore, no taps are located to the right of the current tap.
        if x_index is inbetween the first and last index, there must be a tap to the left and right of the current tap.

        if z_index is zero, that means that the current tap is the bottom most tap within the column and therefore, no taps are located below of the current tap.
        if z_index is equal to the last index of the list, that means that the current tap is the top most tap within the column and therefore, no taps are located above of the current tap.
        if z_index is inbetween the first and last index, there must be a tap above and below of the current tap.

        '''
        import numpy as np
        import math

        x_index = x_coor.index(self.x)
        z_index = z_coor.index(self.z)

        
        left_missing = True
        right_missing = True
        #Check left tap
        for offset in np.arange(x_coor.index(self.x),0,-1):
            if not math.isnan(float(tap_array[z_index,offset-1])):
                self.left=tap_array[z_index,offset-1]
                left_missing = False
                break
        #Check right tap
        for offset in np.arange(x_coor.index(self.x),len(x_coor)-1,1):
            if not math.isnan(float(tap_array[z_index,offset+1])):
                self.right=tap_array[z_index,offset+1]
                right_missing = False
                break

        if left_missing:
            self.left = 'wall_left'
        if right_missing:
            self.right = 'wall_right'

        top_missing = True
        below_missing = True
        #Check top tap
        for offset in np.arange(z_coor.index(self.z),0,-1):
            if not math.isnan(float(tap_array[offset-1,x_index])):
                self.above=tap_array[offset-1,x_index]
                top_missing = False
                break
        #Check bottom tap
        for offset in np.arange(z_coor.index(self.z),len(z_coor)-1,1):
            if not math.isnan(float(tap_array[offset+1,x_index])):
                self.below=tap_array[offset+1,x_index]
                below_missing = False
                break

        if top_missing:
            self.above = 'top'
        if below_missing:
            self.below = 'ground'


    def tap_trib_dim_nc(self,pressure_taps):
        
        '''
        Description/ Methodology 

        This method is used to calculate the tributary area dimensions (width and height) for the current tap using the surrounding taps. Using the dictionary of pressure tap instance,
        the method calculates the different between current taps and the surrounding taps using their x and z values. The distance between the current tap and the surrounding is divided by two.
        This action is performed on their side of the current tap and summed to determine the dimension. The width is calculated using the left and right variable (using their x values) 
        and the height is calcuated using the above and below variables (using their z values). If a tap is along a surface edge ( variable contains either wall_right,wall_left, top or ground),
        the x or z values is assumed to be the wall width/surface_height or zero.

        if left = wall_left, the x value of the "tap" on the left = 0
        if right = wall_right, the x value of the "tap" on the left = surface_width
        if above = top, the z value of the "tap" above = surface_height
        if below = ground, the z value of the "tap" below = 0

        This method assumes the origin is located at the bottom left corner of the surface. See Pressure_Tap_NC.xlsx as an example of the input file.      

        The moment arm (r - for torsion) calculated about the origin the left edge of the surface.

        Note - this method is not recommended. The tap_trib_dim method should be used, especially for the torsion calculation.
        '''

        if self.left=='wall_left':
            x_neg_1 = 0 # No taps to the left, therefore the "tap"/x value is zero.
            x_plus_1 =pressure_taps[self.right].x # the value on the right is the x value of the tap on the right side.
            self.width = (float(x_plus_1)-float(self.x)/2)+(float(self.x)-float(x_neg_1)) # the width of the pressure tap tributary area is half the distance between the current tap and the tap on the right and the full distance between the wall edge and the current tap.
            
            self.r = self.x #the moment arm about the origin
            
        elif self.right=='wall_right':
            if self.left == 'wall_left':
                x_neg_1 =self.surface_width
            x_neg_1 = pressure_taps[self.left].x # the value on the left is the x value of the tap on the left side.
            x_plus_1 =self.surface_width # No taps to the right, therefore the "tap"/ x value is the wall width.
            self.width = (float(x_plus_1)-float(self.x))+(float(self.x)-float(x_neg_1)/2) # the width of the pressure tap tributary area is half the distance between the current tap and the tap on the left and the full distance between the wall edge and the current tap.
            
            self.r = self.x #the moment arm about the origin
        else:
            x_neg_1 = pressure_taps[self.left].x # the value on the left is the x value of the tap on the left side.
            x_plus_1 =pressure_taps[self.right].x # the value on the right is the x value of the tap on the right side.
            self.width = (float(x_plus_1)-float(self.x)/2)+(float(self.x)-float(x_neg_1)/2)# the width of the pressure tap tributary area is half the distance between the current tap and the left and right taps.
            
            self.r = self.x #the moment arm about the origin

        if self.below=='ground':
            z_neg_1 = 0 # No taps below, therefore the "tap"/z value is zero.
            z_plus_1 =pressure_taps[self.above].z # the value above is the z value of the tap above.
            self.height =  (float(z_plus_1)-float(self.z)/2)+(float(self.z)-float(z_neg_1)) # the height of the pressure tap tributary area is half the distance between the current tap and the tap above and the full distance between the bottom edge and the current tap.
            
        elif self.above=='top':
            z_neg_1 = pressure_taps[self.below].z  # the value below is the z value of the tap below.
            z_plus_1 =self.surface_height # No taps above, therefore the "tap"/z value is the wall height.
            self.height = (float(z_plus_1)-float(self.z))+(float(self.z)-float(z_neg_1)/2) # the height of the pressure tap tributary area is half the distance between the current tap and the tap below and the full distance between the top edge and the current tap.
            
        else:
            z_neg_1 = pressure_taps[self.below].z # the value below is the z value of the tap below.
            z_plus_1 =pressure_taps[self.above].z # the value above is the z value of the tap above.
            self.height = (float(z_plus_1)-float(self.z)/2)+(float(self.z)-float(z_neg_1)/2)  # the height of the pressure tap tributary area is half the distance between the current tap and the tap below and above.



    def tap_trib_dim(self,pressure_taps):

        '''
        Description/ Methodology 

        This method is used to calculate the tributary area dimensions (width and height) for the current tap using the surrounding taps. Using the dictionary of pressure tap instance,
        the method calculates the different between current taps and the surrounding taps using their x and z values. The distance between the current tap and the surrounding is divided by two.
        This action is performed on their side of the current tap and summed to determine the dimension. The width is calculated using the left and right variable (using their x values) 
        and the height is calcuated using the above and below variables (using their z values). If a tap is along a surface edge ( variable contains either wall_right,wall_left, top or ground),
        the x or z values is assumed to be the +/- wall width/2 or surface_height or zero.

        This method assumes the origin is located at the bottom center of the surface. See Pressure_Tap.xlsx as an example of the input file.   
        The left is direction is considered the negative side and the right side of the origin is the positive half.

        if left = wall_left, the x value of the "tap" on the left = - surface_width/2
        if right = wall_right, the x value of the "tap" on the left = surface_width/2
        if above = top, the z value of the "tap" above = surface_height
        if below = ground, the z value of the "tap" below = 0

        This method assumes the origin is located at the bottom center of the surface. See Pressure_Tap.xlsx as an example of the input file.      

        The moment arm (r - for torsion) calculated about the center of the surface. r can be both negative (negative moment clockwise) or positive (positive moment counterclockwise)

       
        '''
        if self.left=='wall_left':
            x_neg_1 = -1*self.surface_width/2 # No taps to the left, therefore the "tap"/x value is half the building width on the negative size.
            if self.right == 'wall_right':
                x_plus_1 = self.surface_width/2
            else:
                x_plus_1 =pressure_taps[self.right].x # the value on the right is the x value of the tap on the right side.
            self.width = abs((float(x_plus_1)-float(self.x)))/2+abs((float(x_neg_1)-float(self.x))) # the width of the pressure tap tributary area is half the distance between the current tap and the tap on the right and the full distance between the wall edge and the current tap.
            
        elif self.right=='wall_right':
            if self.right == 'wall_left':
                x_neg_1 = -1*self.surface_width/2
            else:
                x_neg_1 = pressure_taps[self.left].x # the value on the left is the x value of the tap on the left side.
            x_plus_1 =self.surface_width/2 # No taps to the right, therefore the "tap"/ x value is the wall width.
            self.width = abs((float(x_plus_1)-float(self.x)))+abs((float(x_neg_1)-float(self.x)))/2 # the width of the pressure tap tributary area is half the distance between the current tap and the tap on the left and the full distance between the wall edge and the current tap.
            
        else:
            x_neg_1 = pressure_taps[self.left].x
            x_plus_1 =pressure_taps[self.right].x
            self.width = abs((float(x_plus_1)-float(self.x)))/2+abs((float(x_neg_1)-float(self.x)))/2
        
        
        self.r = self.x #the moment arm about the center of the surface.

        if self.below=='ground':
            z_neg_1 = 0 # No taps below, therefore the "tap"/z value is zero.
            if self.above =='top':
                z_plus_1 = self.surface_height
            else:
                z_plus_1 =pressure_taps[self.above].z # the value above is the z value of the tap above.
            self.height =  (float(z_plus_1)-float(self.z)/2)+(float(self.z)-float(z_neg_1)) # the height of the pressure tap tributary area is half the distance between the current tap and the tap above and the full distance between the bottom edge and the current tap.
            
        elif self.above=='top':
           
            if self.below == 'ground':
                z_neg_1 = 0
            else:
                z_neg_1 = pressure_taps[self.below].z # the value below is the z value of the tap below.     
            z_plus_1 =self.surface_height # No taps above, therefore the "tap"/z value is the wall height.
            self.height = (float(z_plus_1)-float(self.z))+(float(self.z)-float(z_neg_1)/2) # the height of the pressure tap tributary area is half the distance between the current tap and the tap below and the full distance between the top edge and the current tap.
            
        else:
            z_neg_1 = pressure_taps[self.below].z # the value below is the z value of the tap below.
            z_plus_1 =pressure_taps[self.above].z # the value above is the z value of the tap above.
            self.height = (float(z_plus_1)-float(self.z)/2)+(float(self.z)-float(z_neg_1)/2) # the height of the pressure tap tributary area is half the distance between the current tap and the tap below and above.
            


    def trib_area(self,floor_height=None):
        '''
        Description/ Methodology 

        This method is used to calculate the tributary area area of the current tap and returns the value. If the loading is begin applied to floor diaphragm, provided tributary height of the diaphragm
        will calculate the area using the given height and the tap's tributary width. If the floor height is not provide, the tributary area returned is uses the tap's height and width.

        The floor height dimension should uses the same units as the x and z values provide.

        This method returns the area value.       
        '''
        if floor_height: 
            area = floor_height*self.width
        else:
            area = self.height*self.width

        return area
    def find_tap_cp (self, cp_data):
        '''
        Description/ Methodology 

        This method is uses the taps index value to extract the cp time history from the cp_data array. The cp time history is returned from this method.
        '''
        
        tap_cp = cp_data[:,self.tap_index]

        return tap_cp


    def cp_fix (self,pressure_taps):
        
        '''
        Description/ Methodology 

        This method is uses the surrounding taps to estimate the current taps cp time history. The fixing method currently only uses the left/right tap to estimate the cp values.
        This method should be used if the current taps has been identified as missing. If a tap is considered an edge tap ( the left/right variable contains "left_wall:/"right_wall"),
        the cp value is assumed to be the same as the value of the available tap opposite of the edge. If the tap has surrounding taps on either side, the cp value is estimated as an average 
        of the tap on the left and right. If a surrounding tap is also considered missing (i.e. two taps missing side by side), the method will used the next tap. 
        
        Example, if the left tap is missing, the cp will be average by the right and the second left tap:


        X = missing tap
        C = current tap 
        S = surrounding tap
        O = other taps

        O O O O O S C X S O O

        Since the left tap was missing, the method skipped it and used the tap to its left.

        This method does not check if the second tap is missing. This functionality will be added in the future.

        Once completed, the method returns the estimated cp time history.
        
        Notes:

        The skipping method is crude and needs refinement before implementing. The user should understand their data in order ensure this method will working properly for them.
        '''


        if self.left =='wall_left' or self.right =='wall_right': # checks if either the taps to the left or right exist (currently, no taps are expected to have no taps on either side) 
            if self.left == 'wall_left': # the tap has no tap to the left of it, therefore the cp time history is estimated using the tap to the right.
                if pressure_taps[self.right].missing_tap: #checks if the tap to the right is missing, if true, the next tap to the right is used to estimate the time history.
                    self.tap_cp = pressure_taps[pressure_taps[self.right].right].tap_cp 
                else:
                    self.tap_cp = pressure_taps[self.right].tap_cp # cp time history is assigned the TH from the right tap
            else: # the tap has no tap to the right of it, therefore the cp time history is estimated using the tap to the left.
                if pressure_taps[self.left].missing_tap: #checks if the tap to the left is missing, if true, the next tap to the left is used to estimate the time history.
                    self.tap_cp = pressure_taps[pressure_taps[self.left].left].tap_cp
                else:
                    self.tap_cp = pressure_taps[self.left].tap_cp # cp time history is assigned the TH from the left tap
                
        else: # both taps on either size exist
            if pressure_taps[self.left].missing_tap: #checks if the tap to the left is missing, if true, the next tap to the left is used to estimate the time history.
                left_tap = pressure_taps[pressure_taps[self.left].left].tap_cp
            else:
                left_tap = pressure_taps[self.left].tap_cp 
            if pressure_taps[self.right].missing_tap: #checks if the tap to the right is missing, if true, the next tap to the right is used to estimate the time history.
                right_tap = pressure_taps[pressure_taps[self.right].right].tap_cp
            else:
                right_tap = pressure_taps[self.right].tap_cp


            self.tap_cp = (left_tap + right_tap)/2 # cp time history is an average between the left and right taps.

        return self.tap_cp
    
    def cp_fix_vertical (self, pressure_taps):
        '''
        Description/ Methodology 

        This method estimates the taps cp time history using the tap above and below. The output is the estimated cp time history. This method is implemented when ring of taps fails 
        (example: broken scanner). This method does not check if the tap is missing, therefore use with caution.


        The cp fixing methods provided are crude and needs refinement before implementing. The user should understand their data in order ensure this method will working properly for them.
        More fixing method/options will be introduced at a later date
        
        '''
        self.tap_cp = (pressure_taps[pressure_taps[self.tap_id].below].tap_cp+ pressure_taps[pressure_taps[self.tap_id].above].tap_cp)/2

        return self.tap_cp


class building_floor:

    '''
     
    Building Floor Class
    Description: Contains information about building floor and the taps used to estimate the force at the current floor.

    General Variables:

        floor_num = the current floor number 
        floor_height = the tributary height of the current floor (typically the floor height)
        floor_elevation = the current floors elevation above the ground.
        tap_list = a list of pressure taps which are associated with the current floor. These taps will be used to estimate the force time histories of the current floor.

    Additional Variables


        left = the tap id of the pressure tap located to the left of the current tap. If there is no tap to the left, the variable is set as "wall_left".
        right = the tap id of the pressure tap located to the right of the current tap. If there is no tap to the right, the variable is set as "wall_right".
        above = the tap id of the pressure tap located above of the current tap. If there is no tap above, the variable is set as "top".
        below = the tap id of the pressure tap located below of the current tap. If there is no tap below, the variable is set as "ground".

        width = the width dimension of the tributary area of the tap.
        height = the height dimension of the tributary area of the tap. This value only considered the pressure taps and is different than the floor height of the structure
        r = moment arm of the pressure tap from the center of the wall. Measured from the center of the wall to the center of the tap. 
            The value can be either negative (clockwise rotation/negative moment) or positive (counterclockwise rotation/ positive moment).

        tap_cp = the cp time history for the tap.


    Class Methods:

    __init__ (self, floor_num, floor_height,floor_elev)
        Import the information required to create the pressure_tap instances. Required Information: floor_num, floor_height, floor_elev.

    floor_taps(self, pressure_tap_dict, tap_rows)
        This method is used to create a list of taps associated with the current floor. A ring of taps is chosen based on its proximity to the floor. This ring of taps is used to estimate the 
        force time history. pressure_tap_dict is a dictionary of the pressure taps and tap_rows is a list of the different elevations associated with the taps.

    forces(self, angle,air_density, velocity)
        This method calculates the force time histories for the current floor. The forces are discretized in force_x, force_y and moment_z.

        force_x is assumed to be parallel with wind tunnel
        force_y is perpendicular to the wind tunnel
        moment_z is about the axis used to calcualte r. Positive moment = counter clockwise and Negative moment= clockwise.

        The method requires three variables:
        1) angle  = the angle of attack of the wind
        2) air_density used to calculate the wind loading
        3) velocity used to calculate the wind loading

        The method return the three force time histories.
    forces_simple(self,air_density, velocity)
        This method calculates the force time histories for the current floor. The forces are discretized in force_x, force_y and moment_z. 

        force_x is perpendicular to the front wall surface. The front wall windward side of the building at test angle zero.
        force_y parallel to the front wall surface.
        moment_z is about the axis used to calcualte r. Positive moment = counter clockwise and Negative moment= clockwise.

        The method requires the following variables:
        1) air_density used to calculate the wind loading
        2) velocity used to calculate the wind loading

        The method return the three force time histories.


    '''
    
    def __init__(self, floor_num, floor_height,floor_elev):

        self.floor_num=floor_num+1
        self.floor_height = floor_height
        
        self.floor_elevation = floor_elev

        self.tap_list =[]
        
        
    
    def floor_taps(self, pressure_tap_dict, tap_rows):
        '''
        Description/ Methodology 

        This method is called when a tap list need to be generated for the current floor. The method assumes that the vertical taps are arranged in ring along the height. Each floor is assigned
        a ring of taps which is used to calculate the forces. The ring is assigned based on vertical proximity to the floor.  
        
        pressure_tap_dict is a dictionary of pressure tap instance which are used to assign the tap instances to the floor.
        tap_rows is a array/list of different ring elevations.

        '''


        import numpy as np
        
        if not isinstance(tap_rows,np.ndarray): # if tap_rows is not a numpy array, it will be converted the variable into a numpy array.
            tap_rows = np.array(tap_rows)
        
        height_difference = tap_rows - self.floor_elevation # calculate the elevation difference between the different rings of taps and the current floor.
        
        min_index=np.where(abs(height_difference)==abs(height_difference).min()) # find the ring index closest to the current floor elevation

        if len(min_index) > 1: # if there are two rings in equal proximity of the current floor, it will use the larger index value (which is the higher location)
            min_index = max(min_index)
        else:
            min_index = min_index[0][0] # min_index is a multi dimensional array
        
        for _, tap in pressure_tap_dict.items(): # finds the taps which are located with at the ring which has been assigned to the floor and creates a list of taps
            
            if tap.z_index == min_index:

                self.tap_list.append(tap)         
        
        
    def forces(self, angle,air_density, velocity):
        '''
        Description/ Methodology 

        This method is used to calculate the force time history of the current floor. The method calls the pressure tap method "trib_area" to calculate the tributary area using the floor_height
        of the current floor. This method uses the wind tunnel to create the x-axis and y-axis. The x-axis is assumed to the parallel with the wind tunnel and the y-axis is perpendicular to the
        to the wind.

        angle is the testing angle (angle of attack of the wind)
        air_density is the air density used to calculate the wind load
        velocity is the wind velocity used to calculate the wind load
        
        '''
        import numpy as np
        import math
        
        force_x = []
        force_y = []
        moment_z = []

        for tap in self.tap_list:

            if force_x==[]: # If the force_x list is empty, assign the calculated time history to force_x
         
                force_x=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),round(math.sin(math.radians(float(tap.wall_angle-angle))),4)))
                
            else:   # If the force_x has a force time history, add the calculated time history to force_x
                force_x+=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),round(math.sin(math.radians(float(tap.wall_angle-angle))),4)))
                

            if force_y==[]: # If the force_y list is empty, assign the calculated time history to force_y
                force_y=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),math.cos(math.radians(float(tap.wall_angle-angle)))))
            else:   # If the force_y has a force time history, add the calculated time history to force_y
                force_y+=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),math.cos(math.radians(float(tap.wall_angle-angle)))))
            if moment_z==[]: # If the moment_z list is empty, assign the calculated time history to moment_z   
                moment_z=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),tap.r))
            else:   # If the moment_z has a force time history, add the calculated time history to moment_z
                moment_z+=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),tap.r))
        
        force_x = np.multiply(force_x, (air_density* 0.5* velocity**2))
        
        force_y = np.multiply(force_y, (air_density* 0.5* velocity**2))
        moment_z = np.multiply(moment_z, (air_density* 0.5* velocity**2))

        return force_x, force_y, moment_z

    def forces_simple(self,air_density, velocity):
        '''
        Description/ Methodology 

        This method is used to calculate the force time history of the current floor. The method calls the pressure tap method "trib_area" to calculate the tributary area using the floor_height
        of the current floor. This method assumes that x axis is prependicular to the north facing wall (the windward wall add wind angle of attack = 0). 

        air_density is the air density used to calculate the wind load
        velocity is the wind velocity used to calculate the wind load
        
        '''

        import numpy as np
        import math
        
        
        force_x = []
        area = []
        force_y = []
        moment_z = []
        

        for tap in self.tap_list:

            if force_x==[]:
            
                force_x=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),round(math.sin(math.radians(float(tap.wall_angle))),4)))
                
            else:
                force_x+=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),round(math.sin(math.radians(float(tap.wall_angle))),4)))
                

            if force_y==[]:
                force_y=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),math.cos(math.radians(float(tap.wall_angle)))))
            else:
                force_y+=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),math.cos(math.radians(float(tap.wall_angle)))))
            if moment_z==[]:
                moment_z=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),tap.r))
            else:
                moment_z+=(np.multiply(np.multiply(tap.tap_cp,tap.trib_area(self.floor_height)),tap.r))
            area.append(tap.trib_area(self.floor_height))
        force_x = np.multiply(force_x, (air_density* 0.5* velocity**2))
        
        force_y = np.multiply(force_y, (air_density* 0.5* velocity**2))
        moment_z = np.multiply(moment_z, (air_density* 0.5* velocity**2))

        return force_x, force_y, moment_z, area



class wind_tunnel_test:
    '''
     
    wind tunnel test Class
    Description: Contains information about wind tunnel test setup.

    General Variables:

        ###########################################################################################
        VARIABLES ASSOCIATED WITH THE FULL SCALE BUILDING
        building_story = the number of building stories within the test building.
        building_surfaces= the names of the different surfaces of the structure.
        building_walls_angle = the wind angle of attack where the surface is facing the windward. The location of the value corresponds to wall name provided in building_surfaces
        building_width = the width of the corresponding wall provided in building_surfaces.
        building_floor_height = the single storey height of a floor.
        building_dim = the x,y,z dimensions of the building. (Assuming a simple rectangular prism)
        ###########################################################################################

        ###########################################################################################
        VARIABLES ASSOCIATED WITH THE WIND TUNNEL TEST
        tunnel_starting_tap = the id of the first connected to the scanner (typically 101)
        tunnel_missing_taps = list of missing taps ids which need to be estimated
        tunnel_tap_map = a dictionary which has the building wall names as the keys and a list of tap ids associated with the wall.
        tunnel_tap_layout_filepath = the full path to the xslx file which contains the tap layout information: See example.xlsx on the formatting of the file.
        tunnel_gradient_v = the gradient height velocity at full scale.
        tunnel_gradient_h = the gradient height used as a reference during the wind tunnel test.
        self.angle = the testing angle for the current setup
        self.name = the file name of the current setup
        self.alpha = the power law alpha value for the exposure used during the current test setup.
        self.exposure = the exposure used during the current test setup.
        ###########################################################################################

        ###########################################################################################
        VARIABLES ASSOCIATED WITH THE LOAD CALCULATION
        self.loading_air_density  = config_file['test_config'][2]['loading']['air_density']
        self.loading_velocity = config_file['test_config'][2]['loading']['velocity']
        

    Additional Variables


        config_file = The full filepath for the config_file used to populate the wind_tunnel_test class. The config file should be .yml file type. See example config_file. 
        right = the tap id of the pressure tap located to the right of the current tap. If there is no tap to the right, the variable is set as "wall_right".
        above = the tap id of the pressure tap located above of the current tap. If there is no tap above, the variable is set as "top".
        below = the tap id of the pressure tap located below of the current tap. If there is no tap below, the variable is set as "ground".

        width = the width dimension of the tributary area of the tap.
        height = the height dimension of the tributary area of the tap. This value only considered the pressure taps and is different than the floor height of the structure
        r = moment arm of the pressure tap from the center of the wall. Measured from the center of the wall to the center of the tap. 
            The value can be either negative (clockwise rotation/negative moment) or positive (counterclockwise rotation/ positive moment).

        tap_cp = the cp time history for the tap.


    Class Methods:

    __init__ (self, config_file)
        Opens the config_file which contains all the information about the wind tunnel experimental setup, the test building and the varaible used for the loading calculations. 
        The config_file is a .yml file. Once imported, the __init__ method creates a single list of all the taps ids used during the test and determines their index value within the list of cp time histories.
        The method also creates a list of the different floor elevations and a dictionary of building_floor class instances. For the dictionary, the floor number is used as the keys. 
        Following this, the tap_layout file is imported and the a dictionary for each pressure tap is create, the key is the tap id and the value is a pressure_tap class instance.

    cp_extraction(self, file_pssr,file_pssd)
        This method is used to extract the cp time histories from the pssr and pssd files. The data provided by the wind tunnel is referenced using the gradient height velocity, therefore a conversion factor is required
        to reference the cp values to the velocity at the top of the building. The method calculates the conversion factor and applies it to all the cp values. After
        applying the conversion, any tap which are considered to be missing are estimated using the cp_fixed method within the pressure_tap class. After adjusting the
        missing tap cps, any pressure taps associated with a broken scanner have their cp time histories estimated using the cp_fix_vertical method from the pressure_tap class.      
        After correcting the cp time histories, the cp_data is separated based on the tap_map provided ( depending on the surface the tap is located on).

        file_pssr = wind tunnel binary file used to extract the cp data. The file type is pssr.
        file_pssd = wind tunnel binary file used to extract the cp data. The file type is pssd.

    taps_per_floor(self)
        For each of the building floor class instance within the floor dictionary, the taps associated with the floor are assigned to the building floor instance using the 
        floor_taps methods, which is part of the building floor class.

    wind_loading(self)
        This method calculates the force time histories for each floor within the building. The force is calculated using the forces_simple method from the building
        floor class. The forces are discretized in force_x, force_y and moment_z. The forces along with the tap tributary area are stored in respective variables.

        force_x is perpendicular to the front wall surface. The front wall windward side of the building at test angle zero.
        force_y parallel to the front wall surface.
        moment_z is about the axis used to calcualte r. Positive moment = counter clockwise and Negative moment= clockwise.

        The method requires the following variables:
        1) air_density used to calculate the wind loading
        2) velocity used to calculate the wind loading

        The method return the three force time histories.

    print_contour_simple(self,folder)
        This method creates and save contour plots of each surface within the building_surfaces variable. Currently, the method only plots the mean value.

    '''

    def __init__(self, config_file):
        import numpy as np
        import pandas as pd
        from AutoCAD_Layout_2D import autoCAD_ref_cardir
        
        #import from config file
        
        
        data_input_method  = config_file['test_config'][0]['building_config']['data_input']
        
        if data_input_method=='AutoCAD':

            self.building_surfaces= config_file['test_config'][0]['building_config']['walls']
            self.building_walls_angle = config_file['test_config'][0]['building_config']['wall_angles']
            self.building_dim = config_file['test_config'][0]['building_config']['building_dim']
            
            self.building_story = config_file['test_config'][0]['building_config']['stories']
            self.building_story = config_file['test_config'][0]['building_config']['floor_height']
            
            self.tunnel_starting_tap = int(config_file['test_config'][1]['tunnel_model']['starting_tap'])
            self.tunnel_missing_taps = config_file['test_config'][1]['tunnel_model']['missing_taps']
            input_file = config_file['test_config'][1]['tunnel_model']['autoCAD_file']
            roof = config_file['test_config'][0]['building_config']['roof_name']
            self.broken_scanners_id = config_file['test_config'][1]['tunnel_model']['broken_scanner_ids']
            self.tunnel_gradient_v = config_file['test_config'][1]['tunnel_model']['gradient_v']
            self.tunnel_gradient_h = config_file['test_config'][1]['tunnel_model']['gradient_h']
            self.loading_air_density  = config_file['test_config'][2]['loading']['air_density']
            self.loading_velocity = config_file['test_config'][2]['loading']['velocity']
            self.tunnel_tap_map,self.tap_xyz = autoCAD_ref_cardir(input_file,'North',roof)
            '''
            if 'stories' in config_file['test_config'][0]['building_config']:
                self.building_story = config_file['test_config'][0]['building_config']['stories']
            else:
                self.building_story = None
            if 'floor_height' in config_file['test_config'][0]['building_config']:
                    self.building_story = config_file['test_config'][0]['building_config']['floor_height']
            else:
                self.building_floor_height = None 
            '''
            self.PLW_layout = config_file['test_config'][0]['PLW']['PLW_layout_file']
            self.PLW_ref_velocity = config_file['test_config'][0]['PLW']['Ref_Velocity']



        else:
            
            self.building_surfaces= config_file['test_config'][0]['building_config']['surfaces']
            self.building_walls_angle = config_file['test_config'][0]['building_config']['wall_angles']
            self.building_dim = config_file['test_config'][0]['building_config']['building_dim']

            if 'stories' in config_file['test_config'][0]['building_config']:
                self.building_story = config_file['test_config'][0]['building_config']['stories']
            else:
                self.building_story = None
            if 'floor_height' in config_file['test_config'][0]['building_config']:
                self.building_floor_height = config_file['test_config'][0]['building_config']['floor_height']
            else:
                self.building_floor_height = None 
            
    
            self.tunnel_starting_tap = int(config_file['test_config'][1]['tunnel_model']['starting_tap'])
            self.tunnel_missing_taps = config_file['test_config'][1]['tunnel_model']['missing_taps']
            self.broken_scanners_id = config_file['test_config'][1]['tunnel_model']['broken_scanner_ids']
            #self.tunnel_tap_map = config_file['test_config'][1]['tunnel_model']['tap_map']
            self.tunnel_tap_layout_filepath = config_file['test_config'][1]['tunnel_model']['tap_layout_file']
            self.tunnel_gradient_v = config_file['test_config'][1]['tunnel_model']['gradient_v']
            self.tunnel_gradient_h = config_file['test_config'][1]['tunnel_model']['gradient_h']
            self.loading_air_density  = config_file['test_config'][2]['loading']['air_density']
            self.loading_velocity = config_file['test_config'][2]['loading']['velocity']
            self.tap_xyz = None
            self.PLW_layout = config_file['test_config'][0]['PLW']['PLW_layout_file']
            self.PLW_ref_velocity = config_file['test_config'][0]['PLW']['Ref_Velocity']





        self.angle = None
        self.name = None
        self.alpha = None
        self.exposure = None

        #Create list of pressure taps used during the test 
    def tap_list_gen (self):    
        import numpy as np
        import pandas as pd

        self.pressure_tap_list=[]

        for index, sheet in enumerate(self.building_surfaces):
            
            df = pd.read_excel(r'{}'.format(self.tunnel_tap_layout_filepath),sheet_name= sheet ,header=0,index_col=0)
            tap_id_list=df.copy().values.flatten().tolist()
            tap_id_list = [x for x in tap_id_list if str(x) != 'nan']

            
            self.pressure_tap_list.extend(tap_id_list)
        
        self.pressure_tap_list.sort()
        

        self.pressure_tap_index =[i for i in np.arange((self.pressure_tap_list[0]-self.tunnel_starting_tap),(len(self.pressure_tap_list)+(self.pressure_tap_list[0]-self.tunnel_starting_tap)))]
        
        
    def floor_list_gen(self):
        #Create list of building floors within the structure
        import numpy as np
        import pandas as pd
        self.floors_elev = [ f for f in np.arange(self.building_floor_height,self.building_floor_height*self.building_story,self.building_floor_height)]
        
        self.floors= []
        for index,floor in enumerate(self.floors_elev):
            if index == len(self.floors_elev)-1:
                self.floors.append(building_floor(index,self.building_floor_height/2,floor))
            else:
                self.floors.append(building_floor(index,self.building_floor_height,floor))
        
    def surface_list_gen(self):
        self.surface_dim = {}
        for sur in self.building_surfaces:
            if sur.lower() == 'roof':
                self.surface_dim[sur] = [self.building_dim[0],self.building_dim[1]]
            elif sur.lower() in ['north','south']:
                self.surface_dim[sur] = [self.building_dim[0],self.building_dim[2]]
            else:
                self.surface_dim[sur] = [self.building_dim[1],self.building_dim[2]]
    def tap_layout_excel(self):
        # Import tap layout data from the excel file and create the dictionary
        import numpy as np
        import pandas as pd
        self.tap_class_dict={}
        self.columns= []
        self.rows= []
        self.tap_layout = []
        for index, sheet in enumerate(self.building_surfaces):
            
            df = pd.read_excel(r'{}'.format(self.tunnel_tap_layout_filepath),sheet_name= sheet ,header=0,index_col=0)
            self.tap_layout.append(df)
            self.columns.append(df.columns.tolist())   
            
            self.rows.append(df.index.tolist()) 
            
            tap_id_list=df.copy().values.flatten().tolist()
            tap_id_list = [x for x in tap_id_list if str(x) != 'nan']
            df_array=np.array(df)
            
                
            for tap_id in tap_id_list:

                missing_tap = False
                if tap_id in self.tunnel_missing_taps:
                    missing_tap=True
                
                axis_z,axis_x=np.where(df_array==tap_id)
                
                
                x = self.columns[index][axis_x[0]]
                z = self.rows[index][axis_z[0]]
                
                tap_instance= pressure_tap(tap_id,self.pressure_tap_index[self.pressure_tap_list.index(tap_id)],x,z,axis_z[0],sheet,surface_width=self.surface_dim[sheet.lower()][0],surface_height=self.surface_dim[sheet.lower()][1])
                if sheet.lower() =='roof':
                    pass
                else:
                    tap_instance.wall_angle = self.building_walls_angle[index]
                tap_instance.surrounding_taps(df_array,self.columns[index],self.rows[index])
                tap_instance.missing_tap = missing_tap
                self.tap_class_dict[tap_id] = tap_instance
        for _, value in self.tap_class_dict.items():
            value.tap_trib_dim(self.tap_class_dict)



    def cp_extraction(self, file_pssr,file_pssd):
        from readPSSfile import readPSSfile
        import numpy as np

        #create list of broken scanners
        self.broken_scanner = []
        for scanner_num in self.broken_scanners_id:
            for i in np.arange(1,17):
                self.broken_scanner.append(int(scanner_num)*100+i)
        self.velocity_conversion_factor = ((self.tunnel_gradient_h/self.building_dim[2])**self.alpha)**2

        [cp_data,analog,header]=readPSSfile(file_pssr,file_pssd)
        self.cp_data = cp_data 
        max_building_scanner = (int(str(max(self.pressure_tap_list))[:-2])*16)


        self.building_cp_data = cp_data[:,:max_building_scanner]
        self.PLW_cp_TH = cp_data[:,-len(self.pressure_tap_list):]
        
        self.building_cp_data = self.building_cp_data *(self.velocity_conversion_factor)

        self.building_cp_data = self.building_cp_data
        self.analog = analog
        self.header = header      
        for _,tap in self.tap_class_dict.items():
            tap.tap_cp = tap.find_tap_cp(self.building_cp_data)
    
    def cp_estimation(self):
        
        for tap_id in self.pressure_tap_list:
            
            if self.tap_class_dict[tap_id].missing_tap:
                
                self.tap_class_dict[tap_id].cp_data = self.tap_class_dict[tap_id].cp_fix(self.tap_class_dict)
        
        for tap_id in self.broken_scanner:
            self.tap_class_dict[tap_id].cp_data = self.tap_class_dict[tap_id].cp_fix_vertical(self.tap_class_dict)
    
    def surface_filter(self):
        self.filtered_cp_data={key:[] for key in self.building_surfaces}
        for tap_id in self.pressure_tap_list:
            
    
            
            self.filtered_cp_data[self.tap_class_dict[tap_id].home_wall].append(self.tap_class_dict[tap_id].tap_cp)     
        
    def PWL(self):

        self.PWL_data = PLW(self.PLW_layout)
        self.PWL_data.PLW_cp = self.PLW_cp_TH
        self.PWL_data.ref_vel = self.PLW_ref_velocity
        self.PWL_data.air_density = self.loading_air_density

    
    
    
    
    
    def taps_per_floor(self):

        # Import taps class into the floor class
        for floor in self.floors:
            floor.floor_taps(self.tap_class_dict,self.rows[0])


    def wind_loading(self):

        self.force_x = []
        self.force_y = []
        self.moment_z = []
        self.area = []
        for values in self.floors:

            force_x,force_y,moment_z, area= values.forces_simple(self.loading_air_density,self.loading_velocity)
            
            self.force_x.append(force_x)
            self.force_y.append(force_y)
            self.moment_z.append(moment_z)
            self.area.append(area)

    def print_contour_simple(self,folder,print_taps):
    
        
        import matplotlib.pyplot as plt
        
        from matplotlib import cm
        import matplotlib
        import numpy as np
        from scipy.interpolate import interp2d
        from scipy.interpolate import Rbf
        from scipy.interpolate import griddata
        plt.close('all')
        fig, ax = plt.subplots(1,len(self.building_surfaces)-1,figsize=(20,15))
        plt.suptitle(self.name)
        mean_cp_total = []
        for name, taps in self.filtered_cp_data.items():
            if name =='roof':
                continue
            for tap in taps:
                mean_cp_total.append(np.mean(tap))

        cp_max = max(mean_cp_total)
        cp_min = min(mean_cp_total)

        
        for index,name in enumerate(self.building_surfaces):
            if name == 'roof':
                continue
            
            #fig, ax = plt.subplots()
            tap_layout_current = self.tap_layout[index]
            
            x = []
            y = []
            tap_list =tap_layout_current.copy().values.flatten().tolist()
            tap_list = [x for x in tap_list if str(x) != 'nan']
            for tap_id in tap_list :
                x.append(self.tap_class_dict[tap_id].x)
                y.append(self.tap_class_dict[tap_id].z)

            x= np.array(x)
            y= np.array(y)

            mean_cp = []
            for tap in self.filtered_cp_data[name]:
                mean_cp.append(np.mean(tap))

            

            data = np.array(mean_cp)
            #fake_cp = np.ones(data.shape)
            #data = fake_cp
            #np.savetxt('{}\{}_{}_mean_cp.csv'.format(folder,self.name,name),data)
            
            
            grid_data = interp2d(
                x,y,data, kind ='cubic',
                bounds_error = False, fill_value= 0)
            
            '''
            grid_data = Rbf(
                x,y,data)
            '''

            new_x = np.linspace((-self.surface_dim[name][0]/2),(self.surface_dim[name][0]/2),200)
            new_y = np.linspace(0,self.surface_dim[name][1],200)
            ''' 
            new_x,new_y = np.meshgrid(new_x,new_y)
            points = np.vstack((x,y)).T
            
            zz = griddata(points,data,(new_x,new_y), method = 'cubic')

            '''

            zz= grid_data(new_x,new_y)




            matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
            

            
            CS = ax[index].contour(new_x,new_y,zz,5,colors='k')
            (ax[index].contourf(new_x,new_y,zz,5,cmap=cm.jet,vmin = cp_min, vmax = cp_max))
            ax[index].clabel(CS, CS.levels, inline=True,fontsize = 10)

            ax[index].set_title('{} wall'.format(name.capitalize()))
            
            if index !=0:
                ax[index].axes.get_yaxis().set_ticks([])
            
            SS = ax[index].scatter(x,y,1,c='k')
            
            if print_taps: 
                for x, y in zip(x,y):
                    text = tap_layout_current.loc[y,x]
                    ax[index].text(x,y+10,text, rotation = 'vertical', fontweight = 'bold')
            
            #SS = ax.scatter(x,y,1,c='k')
        
        
        
        #CB = fig.colorbar(CS, shrink=0.8, extend='both')
        plt.savefig('{}\{}_contours.png'.format(folder,self.name))
        #plt.show()

    def save(self, folder):
        import h5py
        import numpy as np

        atrr_list = {
            'Test Date' : 'December 11, 2018',
            'Tunnel': 'BLWT 2 (High Speed Test Section)',
            'Test Scale': '1:400',
            'Test Exposure' :  self.exposure,
            'Gradient Height': self.tunnel_gradient_h,
            'Building Height': self.building_dim[2],
            'Building Dimensions': self.building_dim[:-1],

            'Wind Angle of Attack': self.angle
        }

        with h5py.File('{}\{}.h5'.format(folder,self.name),'w') as f:
            dset = f.create_dataset('General',dtype = "f")
            for key, value in atrr_list.items():
                dset.attrs[key]= value


            f.create_dataset('cp_data_all',data=self.cp_data)
            

            for name,data in zip(self.building_surfaces,self.filtered_cp_data):
                f.create_dataset('cp_data_{}'.format(name),data=np.vstack(self.filtered_cp_data[data]))
            f.create_dataset('missing_taps',data=np.array(self.tunnel_missing_taps))
            #f.create_dataset('broken_scanner', data=np.array(self.broken_scanner))
            '''
            f.create_dataset('force_x',data=np.array(self.force_x))
            f.create_dataset('force_y',data=np.array(self.force_y))
            f.create_dataset('moment_z',data=np.array(self.moment_z))
            f.create_dataset('floor_elevation',data=self.floors_elev)
            f.create_dataset('area',data=np.array(self.area))
            '''

