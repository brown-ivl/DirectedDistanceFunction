import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-class", type=str, help="Name of class to visualize")
    parser.add_argument("--data-dir", default="/gpfs/data/ssrinath/neural-odf/data/shapenetv2/", type=str, help="Location of dataset")
    args = parser.parse_args()

    #TODO
    # 1) Find the right class directory
    subdirs = os.listdir(args.data_dir)

    class_dir = None
    for subdir in subdirs:
        if args.obj_class in subdir:
            class_dir = subdir
    if class_dir is None:
        print(f"Unable to find directory for '{args.obj_class}' in '{args.data_dir}'")
    else:
        # 2) Randomly load object instances
        class_dir = os.path.join(args.data_dir, class_dir)
        instances = os.listdir(class_dir)
        random.shuffle(instances)
        instances = instances[:6]

        # 3) Find sufficiently covered view from each instance
        depth_maps = []
        for instance in instances:
            train_dir = os.path.join(class_dir, instance, "depth", "train")
            views = os.listdir(train_dir)
            for view in views:
                npy_path = os.path.join(train_dir, view)
                data = np.load(npy_path, allow_pickle=True).item()
                frac_invalid = np.mean(data["invalid_depth_mask"])
                if frac_invalid < 0.75 and frac_invalid > 0.25:
                    depth_maps.append(data["depth_map"])
                    break
        
                # print(np.mean(data["invalid_depth_mask"]))
                # dict_keys(['depth_map', 'camera_viewpoint_pose', 'camera_projection_matrix', 'viewpoint', 'unprojected_normalized_pts', 'invalid_depth_mask'])
        
        # 4) Show each instance on a plot
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        for i in range(len(axes)):
            axes[i].set_title(instances[i])
            axes[i].imshow(depth_maps[i])
        plt.show()