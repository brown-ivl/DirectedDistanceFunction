import numpy as np
import argparse
import trimesh
import open3d as o3d
import os
import glob

dataset_dir = "F:\\ivl-data\\common-3d-test-models\\rendered-data-5d"


class ODFCamera():
    '''
    This class represents a camera and allows view to be rendered either by rasterizing a mesh or by querying a learned network.
        cam_center        - the coordinates of the camera center (x,y,z)
        direction         - a vector defining the direction that the camera is pointing, relative to the camera center
        up                - the up vector of the camera
        focal_length      - the focal length of the camera
        sensor_size       - the dimensions of the sensor (u,v)
        sensor_resolution - The number of pixels on each edge of the sensor (u,v)
    '''

    def __init__(self, center=[1.,1.,1.], direction=[-1.,-1.,-1.], up=[0.,1.,0.], focal_length=1.0, sensor_size=[1.,1.], sensor_resolution=[256,256], verbose=True):
        super().__init__()
        self.verbose = verbose
        self.center = np.array(center)
        assert(np.linalg.norm(direction) != 0.)
        assert(np.linalg.norm(up) != 0.)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.up = np.array(up) / np.linalg.norm(up)
        self.right = np.cross(self.direction, self.up)
        assert(np.linalg.norm(self.right) != 0.)
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.sensor_resolution = sensor_resolution

    def change_resolution(self, resolution):
        self.sensor_resolution = resolution

    def generate_rays(self):
        '''
        Returns a list of rays ( [start point, end point] ), where each ray intersects one pixel. The start point of each ray is the camera center.
        Rays are returned top to bottom, left to right.
        '''
        u_steps = np.linspace(-self.sensor_size[0], self.sensor_size[0], num=self.sensor_resolution[0])
        v_steps = np.linspace(-self.sensor_size[1], self.sensor_size[1], num=self.sensor_resolution[1])
        us, vs = np.meshgrid(u_steps, v_steps)
        us = us.flatten()
        vs = vs.flatten()
        rays = [[np.array(self.center), np.array(self.center + self.focal_length * self.direction + us[i]*self.up) + vs[i]*self.right] for i in range(us.shape[0])]
        return rays


def make_line_set(verts, lines, colors=None):
    '''
    Returns an open3d line set given vertices, line indices, and optional color
    '''
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(verts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if colors is not None:
        line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def make_point_cloud(points, colors=None):
    '''
    Returns an open3d point cloud given a list of points and optional colors
    '''
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    return point_cloud

def make_mesh(verts, faces, color=None):
    '''
    Returns an open3d triangle mesh given vertices, mesh indices, and optional color
    '''
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if color is not None:
        if len(color.shape) == 1:
            mesh.paint_uniform_color(color)
        else:
            mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    return mesh

def load_object(obj_name, data_path):
    obj_file = os.path.join(data_path, obj_name)

    obj_mesh = trimesh.load(obj_file)
    # obj_mesh.show()

    ## deepsdf normalization
    mesh_vertices = obj_mesh.vertices
    mesh_faces = obj_mesh.faces
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
    max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
    max_dist = max_dist * 1.03
    mesh_vertices = (mesh_vertices - center) / max_dist
    obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
    return mesh_vertices, mesh_faces, obj_mesh

def get_odf_cam_elements(center, look, up):
    odf_cam = ODFCamera(center=center, direction=look, up=up, focal_length=1.0, sensor_size=[1.,1.], sensor_resolution=[256,256], verbose=False)
    odf_rays = odf_cam.generate_rays()

    ray_ends = [ray[1] for ray in odf_rays]
    verts = [center] + ray_ends
    edges = [(0, x) for x in range(1, len(verts))]
    return make_line_set(verts, edges, colors=[[0.2,0.2,1.0]]*len(verts))


def get_graph_elements_for_image(img_data, show_odf_cam=False):
    cam_pose_ray_length = 0.1

    cam_center = img_data["viewpoint"]
    look = img_data["camera_viewpoint_pose"][0:3,2]
    look = look / np.linalg.norm(look) * cam_pose_ray_length
    up = img_data["camera_viewpoint_pose"][0:3,1]
    up = up / np.linalg.norm(up) * cam_pose_ray_length
    right = img_data["camera_viewpoint_pose"][0:3,0]
    right = right / np.linalg.norm(right) * cam_pose_ray_length

    vertices = [cam_center, cam_center+look, cam_center+right, cam_center+up]
    edges = [[0,1], [0,2], [0,3]]
    colors = [[0.,1.,0.], [1.,1.,0.], [0.,0.,1.]]
    intersection_points = []

    unproj_pts = img_data['unprojected_normalized_pts']
    unproj_pts = np.reshape(unproj_pts, (256,256,3))
    int_mask = np.reshape(img_data["invalid_depth_mask"], (256,256))
    intersecting_color = [1.,0.5,0.5]
    nonintersecting_color = [0.8,0.8,0.8]
    stride= 1 if show_odf_cam else 8
    curr_vert_index = 4
    for u in range(0,unproj_pts.shape[0], stride):
        for v in range(0, unproj_pts.shape[1], stride):
            vertices.append(unproj_pts[u,v])
            edges.append([0,curr_vert_index])
            if int_mask[u,v]:
                colors.append(nonintersecting_color)
            else:
                colors.append(intersecting_color)
                intersection_points.append(unproj_pts[u,v])
            curr_vert_index += 1

    graph_elements = [make_line_set(vertices, edges, colors=colors)]
    if len(intersection_points) > 0:
        graph_elements.append(make_point_cloud(intersection_points, colors=[[1.,0.,0.]]*len(intersection_points)))
    
    if show_odf_cam:
        graph_elements.append(get_odf_cam_elements(cam_center, -look, up))
    

    # print(f"cam_center: {cam_center.shape}")
    # print(f"look: {look.shape}")
    # print(f"up: {up.shape}")
    # print(f"right: {right.shape}")
    # print(f"unproj pts: {img_data['unprojected_normalized_pts'].shape}")
    # print("=="*20)

    return graph_elements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the camera position and object intersections of each depth image")
    parser.add_argument("--data-dir", "-d", type=str, default="F:\\ivl-data\\common-3d-test-models", help="Data directory")
    parser.add_argument("--object", "-o", type=str, default="armadillo", help="Object to view")
    parser.add_argument("--skip-nonintersecting", action="store_true", help="Skip depth images that don't intersect the object")
    parser.add_argument("--interior", action="store_true", help="Only show depth images from the object interior")
    parser.add_argument("--show-odf-cam", action="store_true", help="Show the ODF Camera rays with the same camera parameters")
    args = parser.parse_args()

    #3D data and rendered data
    mesh_dir = "data"
    rendered_dir = "rendered-data-5d"

    mesh_vertices, _, obj_mesh = load_object(args.object + ".obj", os.path.join(args.data_dir, mesh_dir))
    
    wireframe = make_line_set(obj_mesh.vertices, obj_mesh.edges, colors=[[0.,0.,0.]]*obj_mesh.edges.shape[0])
    mesh = make_mesh(obj_mesh.vertices, obj_mesh.faces)
    print(os.path.join(args.data_dir, rendered_dir, args.object, '*.npy'))
    data_files = glob.glob(os.path.join(args.data_dir, rendered_dir, args.object, "depth", "train", '*.npy'))
    data_files.sort()

    for df in data_files:
        depth_data = np.load(df, allow_pickle=True).item()

        if (not args.skip_nonintersecting or np.min(depth_data["invalid_depth_mask"]) < 1.) and (not args.interior or np.min(depth_data["depth_map"]) < 0.):
            graph_elements = get_graph_elements_for_image(depth_data, show_odf_cam=args.show_odf_cam)
            o3d.visualization.draw_geometries([mesh] + graph_elements)




    # o3d.visualization.draw_geometries(to_show)

