import trimesh
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from skimage.measure import marching_cubes
import open3d as o3d
import time

RADIUS = 1.0

class ODFOBJ():
    def __init__(self, radius=RADIUS):
        self.radius = radius
        self.obj_mesh = None
        self.odf = {}
    
    def load_mesh(self, obj_path, normalize=True, vis=False):
        # load mesh and normalize it
        obj_mesh = trimesh.load(obj_path)
        if normalize:
            # deepsdf normalization
            mesh_vertices = obj_mesh.vertices
            mesh_faces = obj_mesh.faces
            center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
            max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
            max_dist = max_dist * 1.03
            mesh_vertices = (mesh_vertices - center) / max_dist
            obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        if vis:
            obj_mesh.show()
        self.obj_mesh = obj_mesh
    
    def construct_odf(self, grid_resolution=32, num_lines=16, vis=False):
        # start points and ray directions
        data = self.get_camera_pos_ray_dir(grid_resolution, num_lines, vis)
    
        # pyembree is faster than trimesh.ray.intersects_location
        #obj_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(obj_mesh)
        #intersect_loc, ray_idx, _ = obj_intersector.intersects_location(ray_origins=data[:,:3],
        #                                                                ray_directions=data[:,3:], 
        #                                                                multiple_hits=False)
    
        assert (self.obj_mesh!=None)
        intersect_loc, ray_idx, _ = self.obj_mesh.ray.intersects_location(ray_origins=data[:,:3],
                                                                          ray_directions=data[:,3:],                                                                        multiple_hits=False)

        # intersect or not
        intersect = np.zeros(len(data), dtype=bool)
        intersect[ray_idx] = True
        # inside or not
        #inside = obj_intersector.contains_points(points=data[:,:3])
        inside = self.obj_mesh.contains(points=data[:,:3])
        # depth
        depth = np.ones((len(data)))*100
        depth[ray_idx] = np.linalg.norm(intersect_loc - data[ray_idx,:3], axis=1)
        # flip the direction and depth if it is inside
        depth[inside] *= -1
        data[inside, 3:] *= -1
        
        
        self.odf['depth'] = depth.reshape(-1, num_lines**2).T
        self.odf['data'] = np.transpose(data.reshape(-1, num_lines**2, 6), (1, 0, 2))
        #self.odf['num_rays'] = num_lines**2

        if vis:
            #mask = np.ones(len(data), dtype=bool)
            mask = np.logical_and(intersect, np.logical_not(inside))
            #mask = np.logical_and(intersect, inside)
            start = data[mask][:num_lines**2, :3]
            surface = start+data[mask][:num_lines**2, 3:]*depth[mask][:num_lines**2][:,np.newaxis]
            segments = np.stack((start, surface), axis=1)
            lines = trimesh.load_path(segments)
            scene = trimesh.Scene([self.obj_mesh, lines])
            scene.show()

    def get_camera_pos_ray_dir(self, resolution=32, num_lines=16, vis=False):
        side = np.linspace(start=-self.radius, stop=self.radius, num=resolution)
        xv, yv, zv = np.meshgrid(side, side, side)
        # transpose x and y so we can recover the grid by reshape(res, res, res)
        camera_pos = np.transpose(np.stack([xv, yv, zv], axis=-1), (1, 0, 2, 3)).reshape(-1, 3)
        rays = self.get_ray_dir(num_lines, vis)
        data = np.hstack((np.tile(camera_pos, (1, len(rays))).reshape(-1, 3), 
                          np.tile(rays, (len(camera_pos), 1))))
        return data    

    def get_ray_dir(self, num_line=16, vis=False):
        tris, quads, sphere_mesh = self.uv_sphere(radius=1, count=[num_line, num_line])

        # get ray directions through the centroid of the triangle
        top_bottom_rays = np.squeeze(self.triangle_centroid(tris), axis=1)
    
        # ray directions for quads
        # step1: get centroids of the four triangles
        quads_centroids = self.triangle_centroid(quads)
        # step2: get centroids of the quad by the intersection of two lines
        middle_rays = self.line_interaction(quads_centroids[:,:,:])

        rays = np.vstack((top_bottom_rays, middle_rays))

        rays = rays/np.linalg.norm(rays, axis=-1)[:, np.newaxis]
        
        if vis:
            segments = np.stack((np.zeros(rays.shape), rays*1.2), axis=1)
            lines = trimesh.load_path(segments)
            scene = trimesh.Scene([sphere_mesh, lines])
            scene.show()

        return rays

    def line_interaction(self, points):

        coeffs = np.stack((points[:,1,:] - points[:,0,:], points[:,2,:] - points[:,3,:]), -1)
        constants = points[:,2,:]-points[:,0,:]
        solutions = torch.linalg.lstsq(torch.tensor(coeffs), torch.tensor(constants)).solution
        solutions = solutions.numpy()
        intersections = points[:,0,:]+(points[:,1,:] - points[:,0,:])*solutions[:, 0:1]
        #intersections = points[:,2,:]+(points[:,3,:] - points[:,2,:])*solutions[:,1:2]
        return intersections

    def triangle_centroid(self, triangles):
        centroids = np.mean(triangles, axis=2)
        return centroids

    # uv sphere modified from trimesh github
    #https://github.com/mikedh/trimesh/blob/main/trimesh/creation.py
    def uv_sphere(self, radius=1.0, count=[32, 32]):

        count = np.array(count, dtype=np.int64)+1

        # generate vertices on a sphere using spherical coordinates
        theta = np.linspace(0, np.pi, count[0])
        phi = np.linspace(0, np.pi * 2, count[1])[:-1]
        spherical = np.dstack((np.tile(phi, (len(theta), 1)).T,
                               np.tile(theta, (len(phi), 1)))).reshape((-1, 2))
        vertices = trimesh.util.spherical_to_vector(spherical) * radius

        # generate faces by creating a bunch of pie wedges
        c = len(theta)
        # a quad face as four triangles
        triangles = np.array([[c, 0, 1],
                              [c + 1, c, 1],
                              [c + 1, c, 0],
                              [0, 1, c + 1]])

        # increment both triangles in each quad face by the same offset
        incrementor = np.tile(np.arange(c - 1), (4, 1)).T.reshape((-1, 1))
        # create the faces for a single pie wedge of the sphere
        strip = np.tile(triangles, (c - 1, 1))
        strip += incrementor

        # tile pie wedges into a sphere
        faces = np.vstack([strip + (i * c) for i in range(len(phi))])

        # poles are repeated in every strip, so a mask to merge them
        mask = np.arange(len(vertices))
        # the top pole are all the same vertex
        mask[0::c] = 0
        # the bottom pole are all the same vertex
        mask[c - 1::c] = c - 1

        # faces masked to remove the duplicated pole vertices
        # and mod to wrap to fill in the last pie wedge
        faces = mask[np.mod(faces, len(vertices))]

        mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces, 
                                    process=False,
                                    metadata={'shape': 'sphere',
                                              'radius': radius})

        faces = faces.reshape(-1, 4, 3)
        # mask quad
        mask_top = np.zeros(faces.shape[0], dtype=bool)
        mask_bottom = np.zeros(faces.shape[0], dtype=bool)
        mask_top[::c-1] = True
        mask_bottom[c-2::c-1] = True
        # keep a valid triangle
        top_bottom = np.vstack((faces[mask_top][:,1,:], faces[mask_bottom][:,2,:]))

        mask = np.logical_or(mask_top, mask_bottom)
        middle = faces[np.logical_not(mask)]

        top_bottom_coord = vertices[top_bottom]
        middle_coord = vertices[middle]
        return np.expand_dims(top_bottom_coord, 1), middle_coord, mesh
    
    def vis_coord_ray(self):
        assert ('data' in self.odf)
        #data = self.odf['data'][:,:5,:].reshape(-1, 6)
        data = self.odf['data'].reshape(-1, 6)
        segments = np.stack((data[:,:3], 
                             data[:,:3]+data[:,3:]*0.01), axis=1)
        lines = trimesh.load_path(segments)
        scene = trimesh.Scene([lines])
        scene.show()

    def odf_marching_cube(self, vote=True, vis=True):
        assert('depth' in self.odf)
        resolution = round(self.odf['depth'].shape[1]**(1./3))
        spacing = 2.*self.radius/(resolution-1)
        
        if vote:
            implicit_values = np.copy(self.odf['depth'])
            implicit_values[implicit_values>0] = 1.0
            implicit_values[implicit_values<0] = -1.0
            implicit_values = np.mean(implicit_values, axis=0)
        else:
            implicit_values = np.copy(self.odf['depth'][0,:])

        implicit_values = implicit_values.reshape(resolution, resolution, resolution)

        verts, faces, normals, _ = marching_cubes(implicit_values, 
                                                  level=0.0, 
                                                  spacing=(spacing, spacing, spacing))
        verts = verts - 1.
        odf_obj_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        if vis:
            odf_obj_mesh.show()
        #print(len(odf_obj_mesh.vertices))
        return odf_obj_mesh

    def odf_pointcloud(self, mode='all', vis=True):
        assert('depth' in self.odf)
        resolution = round(self.odf['depth'].shape[1]**(1./3))
        
        if mode=='internal':
            mask = np.logical_and(self.odf['depth']<0., self.odf['depth']>-2.)
        elif mode=='external':
            mask = np.logical_and(self.odf['depth']>=0, self.odf['depth']<2.)
        else:
            mask = np.logical_and(self.odf['depth']<2., self.odf['depth']>-2.)
        data = self.odf['data'][mask]
        depth = self.odf['depth'][mask]
        surf_pts = data[:,:3]+data[:,3:]*depth[:,np.newaxis]
        surf_pts = surf_pts.reshape(-1, 3)
        
        pcd = o3d.geometry.PointCloud()
        #pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 255], (len(surf_pts), 1)))
        pcd.points = o3d.utility.Vector3dVector(surf_pts)
        pcd.normals = o3d.utility.Vector3dVector(-data[:,3:])
        if vis:
            o3d.visualization.draw_geometries([pcd])
        return pcd

    def odf_sdf(self, vis=True):
        assert('depth' in self.odf)
        resolution = round(self.odf['depth'].shape[1]**(1./3))
        
        odf_depth = np.copy(self.odf['depth'])
        # check the point is inside or outside
        outside = np.sum(self.odf['depth']>=0, axis=0)>=(self.odf['depth'].shape[0]//2)
        inside = np.logical_not(outside)
        # inside but pos and outside but neg
        outside_neg = np.logical_and(self.odf['depth']<0, outside[np.newaxis,:])
        inside_pos = np.logical_and(self.odf['depth']>=0, inside[np.newaxis,:])
        odf_depth[outside_neg] = 100.
        odf_depth[inside_pos] = -100.

        sdf = np.zeros(odf_depth.shape[1])
        sdf[outside] = np.min(odf_depth[:, outside], axis=0)
        sdf[inside] = np.max(odf_depth[:, inside], axis=0)       
        sdf = sdf.reshape(resolution, resolution, resolution)
        
        if vis:
            maxi = np.max((np.abs(np.min(sdf)), np.abs(np.max(sdf)), 1e-1))
            mini = -maxi
            fig = plt.figure()
            sdf_cmap = self.get_depth_cmap(mini, maxi)
            frames = [[plt.imshow(sdf[:,i,:], vmin=mini, vmax=maxi, cmap=sdf_cmap),
                       plt.text(0.5, 1.100, 'Y: {}'.format(i)),
                     ] for i in range(resolution)]
            plt.colorbar()
            plt.clim(mini, maxi)
            ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat_delay=0)
            #ani.save(os.path.join(model_dir, '{}.mp4'.format(NeuralODF.Config.Args.expt_name)))
            plt.show()


    def get_depth_cmap(self, vmin, vmax):
        #middle is the 0-1 value that 0.0 maps to after normalization
        middle = (0.-vmin)/(vmax-vmin)
        cdict = {
            'red': [[0.0, 217./255., 217./255.],
                [middle, 232./255, 173./255.],
                [1.0, 2./255., 2./255.]],
            'green': [[0.0, 114./255., 114./255.],
                [middle, 196./255., 196./255.],
                [1.0, 10./255., 10./255.]],
            'blue': [[0.0, 4./255., 4./255.],
                [middle, 158./255., 247./255.],
                [1.0, 94./255., 94./255.]],
        }

        depth_cmap = colors.LinearSegmentedColormap('depth_cmap', segmentdata=cdict, N=256)

        return depth_cmap

    def odf_marching_cube_refine(self, vote=True, vis=True):
        assert('depth' in self.odf)
        resolution = round(self.odf['depth'].shape[1]**(1./3))
        spacing = 2.*self.radius/(resolution-1)
        
        if vote:
            implicit_values = np.copy(self.odf['depth'])
            implicit_values[implicit_values>0] = 1.0
            implicit_values[implicit_values<0] = -1.0
            implicit_values = np.mean(implicit_values, axis=0)
        else:
            implicit_values = np.copy(self.odf['depth'][0,:])

        implicit_values = implicit_values.reshape(resolution, resolution, resolution)

        verts, faces, normals, _ = marching_cubes(implicit_values, 
                                                  level=0.0, 
                                                  spacing=(spacing, spacing, spacing))
        verts -= 1.
        obj_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        div_obj_mesh = obj_mesh.subdivide().subdivide()
        verts = div_obj_mesh.vertices
        faces = div_obj_mesh.faces
        normals = div_obj_mesh.vertex_normals
        # refine the mesh
        rays_dir = -normals/np.linalg.norm(normals, axis=-1)[:, np.newaxis]
        data = np.hstack((verts, rays_dir)).reshape(-1, 6)

        assert (self.obj_mesh!=None)
        intersect_loc, ray_idx, _ = self.obj_mesh.ray.intersects_location(ray_origins=data[:,:3],
                                                                          ray_directions=data[:,3:],                                                                          multiple_hits=False)
        # intersect or not
        intersect = np.zeros(len(data), dtype=bool)
        intersect[ray_idx] = True
        # inside or not
        inside = self.obj_mesh.contains(points=data[:,:3])
        # depth
        depth = np.ones((len(data)))*100
        depth[ray_idx] = np.linalg.norm(intersect_loc - data[ray_idx,:3], axis=1)

         
        mask = np.logical_and(depth<=(self.radius*2/resolution), intersect)
        

        # flip the direction and depth if it is inside
        depth[inside] *= -1
        data[inside, 3:] *= -1
    
        verts[mask, :] = (data[:,:3]+data[:,3:]*depth[:,np.newaxis])[mask, :]
        
        odf_obj_mesh = trimesh.Trimesh(vertices=verts+[1.0, 0., 0.], faces=faces)
        if vis:
            odf_obj_mesh.show()
        #segments = np.stack((data[mask, :3], verts[mask, :]), axis=1)
        #lines = trimesh.load_path(segments)
        #obj_mesh.vertices += [-2., 0, 0]
        #scene = trimesh.Scene([odf_obj_mesh, self.obj_mesh, obj_mesh])
        #scene.show()

    def odf_mesh_ball_pivoting(self, radius=[0.03, 0.05, 0.1, 0.2], vis=True):
        assert('depth' in self.odf)
        pcd = self.odf_pointcloud(mode='internal', vis=False)
        # open3d can estimate normals for pointcloud
        #pcd.estimate_normals()      
        downpcd = pcd.voxel_down_sample(voxel_size=0.03)
        downpcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=10))      
        #downpcd.compute_convex_hull()
        #downpcd.estimate_normals()
        #downpcd.orient_normals_consistent_tangent_plane(10)
        
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                downpcd, o3d.cpu.pybind.utility.DoubleVector(radius))
        """
        rec_mesh = rec_mesh.remove_degenerate_triangles()
        rec_mesh = rec_mesh.remove_non_manifold_edges()
        rec_mesh = rec_mesh.remove_duplicated_vertices()
        rec_mesh = rec_mesh.remove_duplicated_triangles()
        rec_mesh = rec_mesh.remove_unreferenced_vertices()
        """
        o3d.visualization.draw_geometries([rec_mesh, downpcd], point_show_normal=True)
        
    def odf_mesh_poisson(self, depth=7, vis=True):
        assert('depth' in self.odf)
        pcd = self.odf_pointcloud(mode='all', vis=False)
        # open3d can estimate normals for pointcloud
        #pcd.estimate_normals()      
        #downpcd = pcd.voxel_down_sample(voxel_size=0.03)
        #downpcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=10))      
        #downpcd.compute_convex_hull()
        #downpcd.estimate_normals()
        #downpcd.orient_normals_consistent_tangent_plane(10)
        
        rec_mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        rec_mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
        rec_mesh.compute_vertex_normals()
        if vis:
            o3d.visualization.draw_geometries([rec_mesh])
        #print(rec_mesh.is_watertight())

def main():
    start = time.time()
    # get object path
    obj = 'armadillo.obj'
    #obj = 'bunny_watertight.obj'
    #obj = 'torus.obj'
    root = os.path.join(os.sep, 'home', 'johnnylu', 'Documents')
    data_path = os.path.join(root, 'data', 'common-3d-test-models', 'data/')
    
    odf = ODFOBJ()
    odf.load_mesh(obj_path = os.path.join(root, data_path, obj), normalize=True, vis = False)
    odf.construct_odf(grid_resolution=32, num_lines=16, vis=False)
    end = time.time()
    print("ODF construction time: {}".format(end-start))
    start = time.time()
    odf.odf_marching_cube(vote=True, vis=False)
    end = time.time()
    print("Marching cube time: {}".format(end-start))
    start = time.time()
    odf.odf_pointcloud(mode='all', vis=False)
    end = time.time()
    print("Pointcloud time: {}".format(end-start))
    start = time.time()   
    odf.odf_sdf(vis=False)
    end = time.time()
    print("SDF time: {}".format(end-start))
    start = time.time()   
    odf.odf_marching_cube_refine(vote=True, vis=False)
    end = time.time()
    print("Refine mesh time: {}".format(end-start))
    #start = time.time()   
    #odf.odf_mesh_ball_pivoting()
    #end = time.time()
    #print("Refine mesh time: {}".format(end-start))
    start = time.time()      
    odf.odf_mesh_poisson(depth=6, vis=True)
    end = time.time()
    print("Poisson time: {}".format(end-start))
    #odf.vis_coord_ray()

if __name__=="__main__":
    main()
