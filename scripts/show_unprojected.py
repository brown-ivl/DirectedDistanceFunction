import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import trimesh

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

dataset_dir = "F:\\ivl-data\\common-3d-test-models\\rendered-data-5d"
mesh_dir = "F:\\ivl-data\\common-3d-test-models\\data"
dataset = "armadillo"

_, _, mesh = load_object(dataset+".obj", mesh_dir)

files_path = os.path.join(dataset_dir, dataset, "depth", "train")

data_files = glob.glob(os.path.join(files_path, '*.npy'))
data_files.sort()

loaded_depths = []
for f in data_files:
    depth_data = np.load(f, allow_pickle=True).item()
    loaded_depths.append(depth_data)

#setup 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#after the files are loaded, read and display the unprojected points
n_plotted_inside = 0
n_plotted_outside = 0
img_max = 20
for iter, data in enumerate(loaded_depths):
    print(np.min(data["depth_map"]),  np.max(data["depth_map"]))
    # plot cam center

    cam_color = "purple" if np.min(data["depth_map"]) < 0. else "green"
    if np.min(data["depth_map"]) < 0.:
        assert(mesh.contains([data["viewpoint"]]))
    else:
        assert(not mesh.contains([data["viewpoint"]]))
    ax.scatter(data["viewpoint"][0], data["viewpoint"][1], data["viewpoint"][2], color=cam_color)
    if n_plotted_outside == img_max and n_plotted_inside == img_max:
        break
    if np.min(data["invalid_depth_mask"]) > 0:
        continue
    color = "tab:red"
    if np.min(data["depth_map"]) < 0.:
        if n_plotted_inside >= img_max:
            continue
        else:
            n_plotted_inside += 1
    else:
        color = "tab:blue"
        if n_plotted_outside >= img_max:
            continue
        else:
            n_plotted_outside += 1
    unproj_points = []
    for i in range(len(data["unprojected_normalized_pts"])):
        if not data['invalid_depth_mask'][i]:
            # print(data["unprojected_normalized_pts"][i])
            # print(unproj_points)
            if np.random.random() < 0.001:
                unproj_points.append(data["unprojected_normalized_pts"][i])
    unproj_points = np.array(unproj_points)
    if unproj_points.shape[0] > 0:
        pass
        # ax.scatter(unproj_points[:,0], unproj_points[:,1], unproj_points[:,2], color=color)

plt.show()