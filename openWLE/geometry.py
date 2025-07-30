import numpy as np
from shapely.geometry import Polygon, LineString, MultiPoint, GeometryCollection
from shapely import voronoi_polygons
from shapely.ops import voronoi_diagram
from shapely.strtree import STRtree

from openWLE.funcs import plot_tap_check



def sort_points_clockwise(points: list, origin: np.ndarray = None) -> np.ndarray: 

    points_array = np.array(points)
    if origin is None:
        origin = np.mean(points, axis=0)
    else:
        origin = np.array(origin)

    angles = np.arctan2(points[:,1] - origin[1], points[:,0] - origin[0])
    sorted_taps = list(points_array[np.argsort(-angles)])
    angles = angles[np.argsort(-angles)]
    # angle_diff = np.rad2deg(angles[-1] - angles[0])
    # if angle_diff < 0:
    #     # If the angle difference is negative, reverse the order
    #     sorted_taps = sorted_taps[::-1]
    arc = (angles[0] - angles[-1]) % (2 * np.pi)
    
    # print(f'Arc 1: {a1}, Arc 2: {a2}')
    if arc > np.pi:
        sorted_taps = sorted_taps[::-1]

    return np.array(sorted_taps)

def vector_rotation(vector, angle:float, axis:str):

    angle = np.radians(angle)
    if axis == 'x':
        rotation_matrix = np.array([[1,0,0],
                                     [0, np.cos(angle), -np.sin(angle)],
                                     [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                     [0, 1, 0],
                                     [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                     [np.sin(angle), np.cos(angle), 0],
                                     [0, 0, 1]])
    else:
        raise ValueError('Axis must be x, y or z')

    rotate_vector =  np.dot(rotation_matrix, vector)

    return np.round(rotate_vector,5)

def vector_angle(vector1, vector2):
    return np.degrees(np.arccos(np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))))

def point_2_point_dist(point1, point2):
    return np.linalg.norm(point1 - point2)

def point_2_point_vector(point1, point2):
    return point2 - point1

def local_to_global(local_origin, local_x_axis, local_y_axis, local_coords):
    return [(local_origin + local_x_axis * x + local_y_axis * y) for x,y in local_coords]


def plane_intersection(plane1, plane2):
    n1 = np.array(plane1[:3])
    n2 = np.array(plane2[:3])
    aXb_vec = np.cross(n1, n2)

    if np.linalg.norm(aXb_vec) == 0:
        raise ValueError('Planes are parallel')

    A = np.array([n1, n2, aXb_vec])
    d = np.array([-plane1[3], -plane2[3], 0.]).reshape(3,1)


    if not np.linalg.det(A) == 0:
        p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]


def flatten_vector(vector,plane):

    n = plane
    m = vector
    m_perp = m - np.dot(m,n)*n
    r =  m_perp
    return r

def moment_arm_scalar(vector, normal_vector,plane):

    normal_vector = flatten_vector(normal_vector,plane)
    
    vector = flatten_vector(vector,plane)

    parallel_vector = np.dot(vector,normal_vector)*normal_vector
    
    moment_arm = vector - parallel_vector
    check = np.cross(moment_arm, normal_vector) 
    if np.round(np.linalg.norm(check),5) == 0:
        direction = 1
    elif np.all(check/np.linalg.norm(check) == [0,0,1]):
        direction = -1 
    else:
        direction = 1
            
    return np.round(np.linalg.norm(moment_arm),5) * direction






class WLEPolygon:

    def __init__(self, points: np.ndarray,origin=None, tangent_vector_1 = None, tangent_vector_2 = None, name = None):

        if origin is not None:
            self.origin = np.array(origin)
            self.origin = np.round(origin,5)  # round to 5 decimal places to prevent floating point errors
        else:
            self.origin = None
        self.points = np.round(points,5) # round to 5 decimal places to prevent floating point errors


        self.tangent_vector_1 = np.array(tangent_vector_1)
        self.tangent_vector_2 = np.array(tangent_vector_2)
        self.name = name
        self.polygon = None

    @property
    def width(self):
        return self.polygon.bounds[2] - self.polygon.bounds[0]
    
    @property
    def height(self):
        return self.polygon.bounds[3] - self.polygon.bounds[1]
    
    @property
    def bounding_dims(self):

        return self.polygon.bounds

    

    def local_coordinate_parameters(self):
        
        if self.origin is None:
            self.local_origin = self.points[0]
        else:
            self.local_origin = self.origin
        
        if self.tangent_vector_1 is None:
            self.tangent_vector_1 = self.points[1] - self.local_origin
            self.local_x_axis = self.tangent_vector_1/np.linalg.norm(self.tangent_vector_1)
        else:
            self.local_x_axis = self.tangent_vector_1/np.linalg.norm(self.tangent_vector_1)
        
        if self.tangent_vector_2 is None:
            self.tangent_vector_2 = self.points[2] - self.points[0]
            self.local_y_axis = self.tangent_vector_2/np.linalg.norm(self.tangent_vector_2)
        else:
            self.local_y_axis = self.tangent_vector_2/np.linalg.norm(self.tangent_vector_2)

        self.normal_vector = np.cross(self.tangent_vector_1, self.tangent_vector_2)/np.linalg.norm(np.cross(self.tangent_vector_1, self.tangent_vector_2))
        for p in self.points[3:]:
             
            if np.dot( p - self.local_origin, self.normal_vector) != 0:
                raise ValueError('Points are not coplanar')
                
    
        return self.normal_vector, self.local_origin, self.local_x_axis, self.local_y_axis

    def global_to_local(self, global_coords = None,local_origin = None, local_x_axis = None, local_y_axis =None ):
        
        if local_origin is None:
            local_origin = self.local_origin
        if local_x_axis is None:
            local_x_axis = self.local_x_axis
        if local_y_axis is None:
            local_y_axis = self.local_y_axis
        if global_coords is None:
            global_coords = self.points
        return [(np.dot((point - local_origin), local_x_axis), np.dot((point - local_origin), local_y_axis)) for point in global_coords]
    
    def local_to_global(self, local_coords = None, local_origin = None, local_x_axis = None, local_y_axis =None ,):
            
        if local_origin is None:
            local_origin = self.local_origin
        if local_x_axis is None:
            local_x_axis = self.local_x_axis
        if local_y_axis is None:
            local_y_axis = self.local_y_axis
        if local_coords is None:
            local_coords = self.points
        return [(local_origin + local_x_axis * x + local_y_axis * y) for x,y in local_coords]


    def shapely_polygon(self):
        
        self.polygon = Polygon(self.global_to_local())

        return self.polygon
        
    def plane_equation(self):

        d = -(self.normal_vector[0]*self.local_origin[0] + self.normal_vector[1]*self.local_origin[1] + self.normal_vector[2]*self.local_origin[2])
        return (self.normal_vector[0], self.normal_vector[1], self.normal_vector[2], d)
    
    def intersection(self, lines: list):

        if self.polygon is None:
            self.shapely_polygon()
        intersection_points = []
        for line in lines:
            if len(line) != 2:
                raise ValueError('Line must have 2 points')
            else:
                line_2D = self.global_to_local(global_coords = line)
                line_2D = self.extend_line_endpoints(line_2D,self.width*1.2)
                line_2D = LineString(line_2D)
                intersection = self.polygon.intersection(line_2D)

                if not np.isnan(intersection.bounds).any():
                    intersection_points.append(intersection.bounds[:2])
                    intersection_points.append(intersection.bounds[2:])
                
        
        intersection_points = self.coordinate_sort_2D(intersection_points)    

        return intersection_points
    
    
    def extend_line_endpoints(self, line: list, length: float = 10):
        p0 = np.array(line[0])
        p1 = np.array(line[1])

        scale = length/np.linalg.norm(p1 - p0)

        p0_new = p0 - (p1 - p0)*(scale*0.5)
        p1_new = p1 + (p1 - p0)*(scale*0.5)

        return [p0_new,p1_new]

    def coordinate_sort_2D(self, points: list):

        points = np.array(points)
        if points.shape[1] != 2:
            raise ValueError('Points must be 2D')
        
        center_x, center_y = np.mean(points, axis=0)
        points_adjusted = points - np.array([center_x, center_y])
        angles = np.arctan2(points_adjusted[:,1], points_adjusted[:,0])
        sort_order = np.argsort(angles)
        return points[sort_order]
        
    
    def voronoi_polygons(self, points):
        
        if self.polygon is None:
            self.shapely_polygon()

        points = self.global_to_local(global_coords = points)
        
        multi_point = MultiPoint(points)
        
        voronoi_diagram = voronoi_polygons(multi_point, extend_to=self.polygon)

        voronoi_diagram = [voronoi_diagram.intersection(self.polygon) for voronoi_diagram in voronoi_diagram.geoms]

        voronoi_diagram_list = []
        for point in multi_point.geoms:
            for poly in voronoi_diagram:
                if poly.contains(point):
                    voronoi_diagram_list.append(poly)
    


        return voronoi_diagram_list

                



        

        
        

