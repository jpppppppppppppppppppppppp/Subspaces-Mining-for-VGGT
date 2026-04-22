"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import hashlib

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

from mathutils import Vector, Matrix
import numpy as np

import bpy
from mathutils import Vector
import pickle
import pdb
import random
import cv2
import concurrent.futures

def stable_uid_to_env_light(object_uid: int) -> float:
    uid_str = str(object_uid).encode('utf-8')
    hash_digest = hashlib.md5(uid_str).hexdigest()
    hash_int = int(hash_digest[:8], 16)  # use first 8 hex digits
    normalized = hash_int / 0xFFFFFFFF  # scale to [0, 1)
    return 0.55 + normalized * (0.75 - 0.55)

def zigzag_trajectory(num_cameras, azimuth_range=(0, 360), elevation_range=(-45, 45), cycles=2):
    """
    Generate a zigzag trajectory for cameras in spherical coordinates.

    Parameters:
        num_cameras (int): Number of cameras in the trajectory.
        azimuth_range (tuple): The range of azimuth angles (in degrees) as (min, max).
        elevation_range (tuple): The range of elevation angles (in degrees) as (min, max).
        cycles (int): Number of zigzag cycles (back and forth) over the elevation range.

    Returns:
        azimuths (list): List of azimuth angles in degrees.
        elevations (list): List of elevation angles in degrees.
    """
    # Generate azimuth values uniformly distributed over the azimuth range
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], num_cameras, endpoint=True)

    # Generate elevation values for a zigzag pattern
    elevations = []
    for i in range(cycles):
        # Create one upward and one downward segment of the zigzag
        elevations.extend(np.linspace(elevation_range[0], elevation_range[1], num_cameras // (2 * cycles)))
        elevations.extend(np.linspace(elevation_range[1], elevation_range[0], num_cameras // (2 * cycles)))

    # Adjust elevation array to match the exact number of cameras
    elevations = np.tile(elevations, int(np.ceil(num_cameras / len(elevations))))[:num_cameras]

    return list(azimuths), list(elevations)

def spiral_trajectory(num_cameras, azimuth_range=(0, 360), elevation_range=(-45, 45)):
    """
    Generate a spiral trajectory for cameras in spherical coordinates.

    Parameters:
        num_cameras (int): Number of cameras in the trajectory.
        azimuth_range (tuple): The range of azimuth angles (in degrees) as (min, max).
        elevation_range (tuple): The range of elevation angles (in degrees) as (min, max).

    Returns:
        azimuths (list): List of azimuth angles in degrees.
        elevations (list): List of elevation angles in degrees.
    """
    # Generate azimuth values uniformly distributed over the specified range
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], num_cameras, endpoint=True)
    
    # Generate elevation values uniformly distributed between elevation_range
    elevations = np.linspace(elevation_range[0], elevation_range[1], num_cameras)

    return list(azimuths), list(elevations)

def equatorial_trajectory(num_cameras, azimuth_range=(0, 360), elevation=0):
    """
    Generate a equatorial trajectory for cameras in spherical coordinates.

    Parameters:
        num_cameras (int): Number of cameras in the trajectory.
        azimuth_range (tuple): The range of azimuth angles (in degrees) as (min, max).
        elevation (float): The fixed elevation angle (in degrees).

    Returns:
        azimuths (list): List of azimuth angles in degrees.
        elevations (list): List of fixed elevation angles in degrees.
    """
    # Generate azimuth values uniformly distributed in the specified range
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], num_cameras, endpoint=True)

    # All elevations are the same as the fixed elevation
    elevations = [elevation] * num_cameras

    return list(azimuths), elevations

def sinusoidal_trajectory(num_cameras, azimuth_range=(0, 360), elevation_range=(-45, 45)):
    """
    Generate a sinusoidal trajectory for camera positions in spherical coordinates.

    Parameters:
        num_cameras (int): Number of cameras in the trajectory.
        azimuth_range (tuple): The range of azimuth angles (in degrees) as (min, max).
        elevation_range (tuple): The range of elevation angles (in degrees) as (min, max).

    Returns:
        azimuths (list): List of azimuth angles in degrees.
        elevations (list): List of elevation angles in degrees.
    """
    # Azimuth angles are uniformly distributed in the specified range
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], num_cameras)

    # Elevation follows a sinusoidal pattern based on azimuth
    elevation_amplitude = (elevation_range[1] - elevation_range[0]) / 2
    elevation_offset = (elevation_range[1] + elevation_range[0]) / 2
    elevations = elevation_amplitude * np.sin(np.linspace(0, 2 * np.pi, num_cameras)) + elevation_offset

    return list(azimuths), list(elevations)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--camera_type", type=str, default='even')
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--elevation", type=float, default=30)
parser.add_argument("--elevation_start", type=float, default=-10)
parser.add_argument("--elevation_end", type=float, default=40)
parser.add_argument("--device", type=str, default='CUDA')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

bpy.context.scene.render.use_persistent_data = True

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
# render.resolution_x = 576
# render.resolution_y = 576
render.resolution_x = 1024
render.resolution_y = 1024
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 64
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

scene.view_layers["ViewLayer"].use_pass_z = True

# Detect devices
bpy.context.preferences.addons["cycles"].preferences.get_devices()

# Use OptiX
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
bpy.context.scene.cycles.device = "GPU"
bpy.context.preferences.addons["cycles"].preferences.get_devices()

for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1  # Using all devices, include GPU and CPU

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #

def set_camera_location(cam_pt):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = cam_pt # sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    return camera

def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)),np.float32)
    return K


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        # if obj.type not in {"CAMERA", "LIGHT"}:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=object_path)
        obj = bpy.context.active_object
        if obj.type == 'MESH':
            mat = bpy.data.materials.new(name="VertexColorMaterial")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            mesh = obj.data
            vcol_name = "Color"
            mat = bpy.data.materials.new(name="VC_Mat")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            for n in nodes:
                nodes.remove(n)
            out_node = nodes.new(type="ShaderNodeOutputMaterial")
            out_node.location = (300, 0)
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
            attr = nodes.new(type="ShaderNodeAttribute")
            attr.location = (-300, 0)
            attr.attribute_name = vcol_name  # 对应 mesh.attributes["Color"]
            links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)

            # center the object
            bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            min_corner = Vector([min(bbox, key=lambda v: v[i])[i] for i in range(3)])
            max_corner = Vector([max(bbox, key=lambda v: v[i])[i] for i in range(3)])
            center = (min_corner + max_corner) / 2
            translation = -center
            obj.location += translation

            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
            rotation_matrix = Matrix.Rotation(math.radians(90), 4, 'Z')
            obj.rotation_euler = rotation_matrix.to_euler()
        # bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
    return RT

def normalize_scene(scale, uid):
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        if obj.name == "Camera":
            continue
        obj.scale = obj.scale * scale
        # pdb.set_trace()
        if int(uid) >= 526:
            obj.rotation_euler = (0, 0, 0)
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    
    # if int(uid) >= 526:
    #     Rot = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    #     offset = Vector((Rot @ np.array(offset).reshape(3,1)).reshape(3,))
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")

def setup_camera_rendering(rgba_path, depth_path):
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    for node in nodes:
        nodes.remove(node)

    render_layers = nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (0, 0)

    file_output_rgba = nodes.new(type='CompositorNodeOutputFile')
    file_output_rgba.location = (400, 0)
    file_output_rgba.base_path = ""
    file_output_rgba.file_slots[0].path = rgba_path
    file_output_rgba.format.file_format = 'OPEN_EXR'
    file_output_rgba.format.color_mode = 'RGBA'
    links.new(render_layers.outputs["Image"], file_output_rgba.inputs[0])

    file_output_depth = nodes.new(type='CompositorNodeOutputFile')
    file_output_depth.location = (400, -200)
    file_output_depth.base_path = ""
    file_output_depth.file_slots[0].path = depth_path
    file_output_depth.format.file_format = 'OPEN_EXR'
    file_output_depth.format.color_mode = 'RGB' #'BW'
    file_output_depth.format.use_zbuffer = True
    links.new(render_layers.outputs["Depth"], file_output_depth.inputs[0])

def read_one_image(fpath):
    im = cv2.imread(fpath, -1)
    im, alpha = im[:, :, :3], im[:, :, 3]
    valid_pixels = im[alpha > 0.95]
    minval, maxval = np.percentile(valid_pixels, [1, 99])
    return (im, alpha, minval, maxval, fpath)

def read_images_parallel(fpaths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads = [executor.submit(read_one_image, f) for f in fpaths]
        return [t.result() for t in threads]

def write_one_image(im_data, use_white_bg=True):
    im, alpha, minval, maxval, fpath = im_data

    valid_mask = alpha > 1e-3
    if np.any(valid_mask):
        valid_pixels = im[valid_mask] / alpha[valid_mask][:, None]
        valid_pixels = (valid_pixels - minval) / (maxval - minval)
        im[valid_mask] = valid_pixels
    im = np.clip(im, 0.0, 1.0)
    # blender by default uses black background; we use white background
    if use_white_bg:
        im = im * alpha[:, :, None] + np.ones_like(im) * (1.0 - alpha[:, :, None])
    im = np.power(im, 1.0 / 2.2)

    im = (im * 255.0).clip(0.0, 255.0).astype(np.uint8)
    alpha = (alpha * 255.0).clip(0.0, 255.0).astype(np.uint8)

    im = np.concatenate([im, alpha[:, :, None]], axis=2)
    cv2.imwrite(fpath, im)

def write_images_parallel(im_datas):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads = [executor.submit(write_one_image, im_data) for im_data in im_datas]
        return [t.result() for t in threads]

def tonemap_folder(rendering_dir, keep_exr=False):
    exr_fpaths = [
        os.path.join(rendering_dir, f)
        for f in os.listdir(rendering_dir)
        if f.endswith("_rgba.exr")
    ]
    im_datas = read_images_parallel(exr_fpaths)
    mean_minval = np.mean([d[2] for d in im_datas])
    mean_maxval = np.mean([d[3] for d in im_datas])
    print(f"Minval: {mean_minval}, maxval: {mean_maxval}")
    with open(os.path.join(rendering_dir, "minmax.txt"), "w") as f:
        f.write(f"{mean_minval} {mean_maxval}")

    png_fpaths = [f.replace("_rgba.exr", "_rgba.png") for f in exr_fpaths]
    for idx in range(len(png_fpaths)):
        im_datas[idx] = im_datas[idx][:2] + (
            mean_minval,
            mean_maxval,
            png_fpaths[idx],
        )
    write_images_parallel(im_datas)

    if not keep_exr:
        for f in exr_fpaths:
            os.remove(f)

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def get_camera_params(camera_object):
    c2w = np.array(listify_matrix(camera_object.matrix_world))
    resolution_x = render.resolution_x
    resolution_y = render.resolution_y
    cx = resolution_x / 2.0
    cy = resolution_y / 2.0
    fx = cx / (camera_object.data.sensor_width / 2.0 / camera_object.data.lens)
    fy = fx
    w2c = np.linalg.inv(c2w)
    w2c = np.diag([1.0, -1.0, -1.0, 1.0]) @ w2c
    cam_dict = {
        "w": resolution_x,
        "h": resolution_y,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "w2c": w2c.tolist(),
    }
    return cam_dict

def save_images(object_file: str) -> None:
    object_uid = os.path.basename(object_file).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()
    # load the object
    load_object(object_file)
    subject_id = object_file.split('/')[-1].split('.')[0]
    # scale = np.load(f'/home/xiyichen/scratch.lin-lab/smplx/{subject_id}/smplx_param.pkl', allow_pickle=True)['scale']
    # normalize_scene(scale, int(subject_id))

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    # env_light = random.uniform(0.55, 0.75)
    env_light = stable_uid_to_env_light(int(subject_id[:4]))
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 1.0

    distances = np.asarray([1.1 for _ in range(args.num_images)])

    # azimuths = (np.arange(args.num_images)/args.num_images*np.pi*2).astype(np.float32)
    # elevations = np.deg2rad(np.asarray([args.elevation] * args.num_images).astype(np.float32))
    # elevations = np.deg2rad(np.asarray([0] * args.num_images).astype(np.float32))
    num_cameras = args.num_images
    azimuth_range = (0, 360)
    elevation_range = (-15, 15)

    # stats = np.load(f'/fs/gamma-datasets/MannequinChallenge/training_examples/{subject_id}/meta.pkl', allow_pickle=True)
    # trajectory_type = random.choice(['sin', 'equ', 'spiral'])
    trajectory_type = 'equ'
    if trajectory_type == 'sin':
        azimuths, elevations = sinusoidal_trajectory(num_cameras, azimuth_range, elevation_range)
    elif trajectory_type == 'equ':
        azimuths, elevations = equatorial_trajectory(num_cameras, azimuth_range, 0)
    elif trajectory_type == 'spiral':
        azimuths, elevations = spiral_trajectory(num_cameras, azimuth_range, elevation_range)
    azimuths = np.deg2rad(np.array(azimuths))
    elevations = np.deg2rad(np.array(elevations))
    # azimuths = np.array(stats[1])
    # elevations = np.array(stats[2])

    cam_pts = az_el_to_points(azimuths, elevations) * distances[:,None]
    (Path(args.output_dir) / object_uid).mkdir(exist_ok=True, parents=True)
    opencv_cameras = {"frames": []}
    for i in range(args.num_images - 1):
        # set camera
        camera = set_camera_location(cam_pts[i])
        render_rgba_path = os.path.join(args.output_dir, object_uid, "renderings", f"{i:03d}_rgba.exr")
        render_depth_path = os.path.join(args.output_dir, object_uid, "renderings", f"{i:03d}_depth.exr")
        setup_camera_rendering(render_rgba_path, render_depth_path)
        bpy.ops.render.render(write_still=True)
        actual_rgba_path, actual_depth_path = render_rgba_path.replace('.exr', '.exr0001.exr'), render_depth_path.replace('.exr', '.exr0001.exr')
        os.system(
            f"mv {actual_rgba_path} {render_rgba_path}; mv {actual_depth_path} {render_depth_path}"
        )
        cam_dict = get_camera_params(camera)
        cam_dict["file_path"] = os.path.relpath(render_rgba_path, args.output_dir)
        cam_dict["blender_camera_location"] = cam_pts[i].tolist()
        opencv_cameras["frames"].append(cam_dict)

    tonemap_folder(os.path.join(args.output_dir, object_uid, "renderings"), keep_exr=False)
    camera_fpath = os.path.join(args.output_dir, object_uid, "opencv_cameras.json")
    with open(camera_fpath, "w") as f:
        json.dump(opencv_cameras, f, indent=4)

if __name__ == "__main__":
    save_images(args.object_path)
