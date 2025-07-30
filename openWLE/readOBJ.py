import trimesh 
import numpy as np
from openWLE.geometry import WLEPolygon




def surface_bounds(mesh, facet):
    faces = mesh.faces[facet]
   
    face_bound = np.round(np.array(mesh.vertices[faces]),4)
    face_bound = np.vstack(face_bound)
    face_bound = np.unique(face_bound, axis=0)
    return face_bound

def surface_normal(mesh, index):

    normal = np.round(np.array(mesh.facets_normal[index]), 4)
    return normal

def surface_origin(mesh,index):

    origin = np.round(np.array(mesh.facets_origin[index]), 4)
    return origin

def read_geometry_file(input_file:str) -> dict:

    mesh = trimesh.load_mesh(input_file)
    
    if not mesh.is_watertight:
        raise ValueError('Mesh is not watertight')
    
    north_face = np.array([-1, 0, 0])

    
    for index,facet in enumerate(mesh.facets):
        surface_bound = surface_bounds(mesh, facet)

        if not all(surface_bound[:,2] <= 0):
            sur_normal = surface_normal(mesh, index)

            
            print(index)
            print(surface_bound)
            print(surface_normal(mesh, index))
            print(surface_origin(mesh, index))

            if all(np.cross(north_face, sur_normal) == np.array([0,0,0])) and (np.dot(north_face, sur_normal)>0):
                print('North_face')
        
            print('\n')
        
            
    
        



if __name__ == "__main__":

    stl = "test/Test_Config/CAARC.obj"
    read_geometry_file(stl)