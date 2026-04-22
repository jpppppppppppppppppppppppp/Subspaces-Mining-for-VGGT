import sys
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import bpy
import pickle
import random
import numpy as np
import json
import cv2
import concurrent.futures
import time
import os
from rich import print
import tempfile
import copy
from multiprocessing import cpu_count
import tempfile
import uuid
import augment_shape
import copy
from mathutils import Euler, Vector, Matrix
import math
from scipy.interpolate import CubicSpline
import mathutils
from scipy.interpolate import make_interp_spline

# scene parameters
SCENE_SCALE_MAX = 1.0                               # for normalizing the scene into [-scale, scale] box
CAMERA_SAMPLE_DIST_THRE = 0.15 #0.15                      # the smallest distance from the sampled cameras to any 3D cubes
POINT_TRANSFORMATION = np.array([[1.0, 0, 0, 0],    # switch y-axis and z-axis, and flip the new y-axis (original z-axis)
                                 [0, 0, 1.0, 0],
                                 [0, -1.0, 0, 0],
                                 [0, 0, 0, 1.0]])
# camera parameters
AZIMUTH_RANGE = [-180, 180]      # yaw
ELEVATION_RANGE = [-10, 10]      # picth
ROLL_RANGE = [-10, 10]           # roll
FOV_RANGE = [45, 70]
NUM_BDRY_VIEWS = 75
NUM_CNTR_VIEWS = 25
#NEAR_PLANE = 0.01
CAMERA_POS_EPS_RANGE = [0.012, 0.03]
# material parameters
PROB_MODIFY_METERIAL = 0.5
PROB_MODIFY_MATERIAL_SLOT = 0.4
PROB_MODIFY_METERIAL_SPECULAR_SCENE = 0.2
ROUGHNESS_RANGE = [0.001, 0.2]
METALLIC_RANGE = [0.001, 1.0]
ROUGHNESS_RANGE_SPECULAR = [0.0, 0.05]
METALLIC_RANGE_SPECULAR = [0.6, 1.0]
# lighting-window parameters
PROB_ADDITION_LIGHT = 1.0
MAX_NUM_ADDITION_LIGHT = 3
PROB_ENV_MAP = 0.0
LIGHT_STR_UNIFORM = 1.0
LIGHT_STR_ADDITION_LIGHT = [0.1, 0.3]
USE_SOLIDIFIER = False
SOLIDIFY_THICKNESS=0.003
# additional geometry
PROB_WINDOW_GLASS = 0.5
PROB_WINDOW_FENCE = 0.5
PROB_AA_GLASS = 0.8
STICK_LIGHT_STRENGTH1 = [0.2, 2.0]
STICK_LIGHT_STRENGTH2 = [5.0, 8.0]
PROB_STICK_LIGHT = 0.7
PROB_STICK_LIGHT_SLOT = 0.2
PROB_LIGHT_BULB = 1.0
LIGHT_BULB_STRENGTH1 = [0.2, 2.0]
LIGHT_BULB_STRENGTH2 = [5.0, 8.0]
LIGHT_BULB_SIZE = [0.05, 0.03]
MAX_NUM_LIGHT_BULBS = 3
GLASS_IOR_RANGE = [1.4, 1.6]
GLASS_ROUGHNESS_RANGE = [0.001, 0.1]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="../renderings")
parser.add_argument("--ibl_path", type=str, default="")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--proj_names", type=str, default="megasynth")
parser.add_argument("--save_norm_glb", action="store_true", help="Save normalized glb")
parser.add_argument("--only_use_cpu", action="store_true", help="Use CPU rendering")
parser.add_argument(
    "--keep_exr", action="store_true", help="Keep EXR files after rendering"
)
parser.add_argument(
    "--no_tonemap", action="store_true", help="Do not tonemap the images"
)
parser.add_argument("--local_cache_dir", type=str, default="../local_cache")
parser.add_argument("--augment_probability", type=float, default=0.0)
parser.add_argument("--wireframe_probability", type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--np_seed', type=int, default=19260817)
parser.add_argument('--radius_min', type=float, default=1.5)
parser.add_argument('--radius_max', type=float, default=2.8)
parser.add_argument('--scan', action='store_true', default=False)
parser.add_argument('--micro', action='store_true', default=False)
parser.add_argument('--render_times', type=int, default=1)
parser.add_argument('--render_views', type=int, default=15)
parser.add_argument('--control_points_num', type=int, default=7)
parser.add_argument('--repeat', action='store_true', default=False)
args = parser.parse_args()

raw_args = copy.deepcopy(args)


# Set up temp dir
print(f"old temp_dir: {tempfile.gettempdir()}")
temp_dir = os.path.join(args.local_cache_dir, "tmp", str(uuid.uuid4()))
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir
print(f"new temp_dir: {tempfile.gettempdir()}")

# Detect devices
bpy.context.preferences.addons["cycles"].preferences.get_devices()

# Use OptiX
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
bpy.context.scene.cycles.device = "GPU"
bpy.context.preferences.addons["cycles"].preferences.get_devices()

for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1  # Using all devices, include GPU and CPU


if args.only_use_cpu:
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "NONE"
    bpy.context.scene.cycles.device = 'CPU'

    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        if d["type"] == "GPU":
            d["use"] = 0  # disable GPU

print(
    f"bpy.context.preferences.addons['cycles'].preferences.compute_device_type: {bpy.context.preferences.addons['cycles'].preferences.compute_device_type}"
)

# Speed up rendering of the same scene
bpy.context.scene.render.use_persistent_data = True

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "OPEN_EXR"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 518
render.resolution_y = 518
render.resolution_percentage = 100

scene.cycles.samples = 64
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 1.5
scene.cycles.use_denoising = True
scene.render.film_transparent = True
# scene.render.film_transparent = False

# depth setting
scene.view_layers["ViewLayer"].use_pass_z = True


def get_random_color():
    r = random.random()
    return r


def get_random_color_array():
    r = random.random() * 1.0
    g = random.random() * 1.0
    b = random.random() * 1.0
    # max_value = max(r, g, b)
    # if max_value > 0:
    #     r /= max_value
    #     g /= max_value
    #     b /= max_value
    return (r, g, b)


def add_area_light(name, location, size_x, size_y, flip=False, strength=LIGHT_STR_UNIFORM, color=(1.0, 1.0, 1.0), light_type="AREA"):
    # Create a new light datablock
    light_data = bpy.data.lights.new(name=name, type=light_type)

    # Create a new object with this light datablock
    light_object = bpy.data.objects.new(name=name, object_data=light_data)

    # Link light object to the collection
    bpy.context.collection.objects.link(light_object)

    # Set light location, strength and color
    light_object.location = location
    light_data.energy = strength
    light_data.color = color #get_random_color() #color

    light_data.shape = 'RECTANGLE'
    light_data.size = SCENE_SCALE_MAX * 3
    light_data.size_y = SCENE_SCALE_MAX * 3

    if flip:
        yaw, pitch, roll = 0, np.pi, 0
        euler = np.array([pitch, roll, yaw])
        euler_blender = Euler((euler[0], euler[1], euler[2]), 'XYZ')
        light_object.rotation_euler = euler_blender

    return light_object


def add_sun_light(name, location, strength, target=(0, 0, 0), light_type='SUN'):
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_data.energy = strength
    light_data.color = get_random_color_array()
    light_data.use_shadow = True
    direction = Vector(target) - Vector(location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    light_object.rotation_euler = rot_quat.to_euler()
    print('Added sunlight at', location)
    return light_object


def add_ambient_light(strength=1.0, color=(1.0, 1.0, 1.0, 1.0)):
    # Add ambient lighting
    bpy.context.scene.cycles.use_fast_gi = True

    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("AmbientWorld")

    bpy.context.scene.world.use_nodes = True
    world_node_tree = bpy.context.scene.world.node_tree

    if "Background" not in world_node_tree.nodes:
        background_node = world_node_tree.nodes.new(type="ShaderNodeBackground")
    else:
        background_node = world_node_tree.nodes["Background"]

    background_node.inputs[0].default_value = color  # Set color
    background_node.inputs[1].default_value = strength  # Set strength

    if "World Output" not in world_node_tree.nodes:
        world_output_node = world_node_tree.nodes.new(type="ShaderNodeOutputWorld")
    else:
        world_output_node = world_node_tree.nodes["World Output"]

    world_node_tree.links.new(background_node.outputs[0], world_output_node.inputs[0])


def add_uniform_lighting():
    # Add uniform lighting
    bpy.context.scene.world = bpy.data.worlds.new("UniformWorld")
    bpy.context.scene.world.use_nodes = True
    shader = bpy.context.scene.world.node_tree.nodes["Background"]
    shader.inputs[0].default_value = (1, 1, 1, 1)  # RGB + Alpha
    shader.inputs[1].default_value = 1.0  # Strength


def add_envmap_lighting(filepath, render_env_map=False, strength=1.0):
    # Create new world material
    world = bpy.data.worlds.get("EnvmapWorld")
    if world is None:
        world = bpy.data.worlds.new("EnvmapWorld")

    world.use_nodes = True

    # Get the node tree
    node_tree = world.node_tree
    node_tree.nodes.clear()

    background_node = node_tree.nodes.new(type='ShaderNodeBackground')
    env_texture_node = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    output_node = node_tree.nodes.new(type='ShaderNodeOutputWorld')
    env_texture_node.image = bpy.data.images.load(filepath)

    node_tree.links.new(env_texture_node.outputs['Color'], background_node.inputs['Color'])
    node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

    background_node.inputs['Strength'].default_value = strength

    bpy.context.scene.world = world

    # set envirnment map visible
    bpy.context.scene.render.film_transparent = not render_env_map
    bpy.context.scene.world.cycles_visibility.camera = True


def reset_scene():
    """Resets the scene to a clean state."""
    # delete everything: object, camera, light
    for obj in bpy.data.objects:
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


def delete_all_lights():
    # Select all light objects in the scene
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)


def load_object(object_path):
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def update_bounding_box(obj):
    obj.update_from_editmode()  # Ensures the bounding box is recalculated
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    mesh.update()
    obj_eval.to_mesh_clear()


def transform_scene():
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            mesh = obj.data
            vertices = np.array([v.co[:] for v in mesh.vertices])
            vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
            vertices_transformed = vertices_homo @ POINT_TRANSFORMATION.T[:, :3]
            for i, vert in enumerate(mesh.vertices):
                vert.co = vertices_transformed[i]
            mesh.calc_normals()
            mesh.update()
            update_bounding_box(obj)
            bpy.context.view_layer.update()


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (np.inf,) * 3
    bbox_max = (-np.inf,) * 3
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
    return np.array(bbox_min), np.array(bbox_max)


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    print("Scene scale from loaded ply file:", bbox_min, bbox_max)
    target_scale = 2 * SCENE_SCALE_MAX
    scale_factor = target_scale / np.max(bbox_max - bbox_min)
    print("Scale the loaded scene to [{}, {}] with a scaling factor {}".format(-SCENE_SCALE_MAX, SCENE_SCALE_MAX, scale_factor))
    offset = -(bbox_min + bbox_max) / 2
    offset = Vector((offset[0], offset[1], offset[2]))
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
        obj.matrix_world.translation *= scale_factor
        obj.scale *= scale_factor
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    return bbox_min, bbox_max, scale_factor


def normalize_scale_info(scale_info, scale_factor):
    neg_x, neg_y, neg_z = scale_info[0]
    pos_x, pos_y, pos_z = scale_info[1]
    print('Scene scale read from object.info: [{}, {}]'.format(scale_info[0], scale_info[1]))
    new_scale_info = np.array([[neg_x, neg_y, neg_z],
                               [pos_x, pos_y, pos_z]])
    new_scale_info *= scale_factor
    print('Normalized scene scale read from object.info: [{}, {}]'.format(new_scale_info[0], new_scale_info[1]))
    return new_scale_info


def normalize_scene_info(scale_info, scene_info, bbox_min, bbox_max, scale_factor_prev):
    cur_scale = (bbox_max - bbox_min) * scale_factor_prev     # should be the same as scale_info, normalized with SCENE_SCALE_MAX
    offset = (bbox_min + bbox_max) / 2                        # should be 0
    origin_scale = np.array(scene_info['scene_size'])         # no-centered scale (in normalize_scene_info) , but should be centered in practice
    scale_factor = cur_scale / origin_scale

    def transform_cubes(cubes):
        new_cubes = []
        for cube in cubes:
            position, size, cube_type = cube
            position = [position[0], position[1], position[2]]
            size = [size[0], size[1], size[2]]
            new_position = np.array(position) * scale_factor - offset
            new_size = np.array(size) * scale_factor
            new_cube = [new_position.tolist(), new_size.tolist(), cube_type]
            new_cubes.append(new_cube)
        return new_cubes

    def transform_position(positions):
        new_positions = []
        for position in positions:
            new_position = np.array(position) * scale_factor - offset
            new_positions.append(new_position)
        return new_positions

    cubes = scene_info['cubes']
    new_cubes = transform_cubes(cubes)

    new_cubes_frame = []
    if scene_info['scene_has_frames']:
        cubes_frame = scene_info['cubes_frame']
        new_cubes_frame = transform_cubes(cubes_frame)

    new_cubes_window = []
    new_positions_light = []
    new_window_additional_planer = []
    if scene_info['scene_has_window']:
        cubes_window = scene_info['cubes_window']
        new_cubes_window = transform_cubes(cubes_window)
        positions_light = scene_info['positions_light']
        new_positions_light = transform_position(positions_light)
        window_additional_planer = scene_info['window_additional_planer']
        new_window_additional_planer = transform_cubes(window_additional_planer)

    new_cubes_aa = []
    if scene_info['scene_has_aa']:
        cubes_aa = scene_info['cubes_aa']
        new_cubes_aa = transform_cubes(cubes_aa)

    new_cubes_thin = []
    if scene_info['scene_has_sticks']:
        cubes_thin = scene_info['cubes_thin']
        new_cubes_thin = transform_cubes(cubes_thin)

    new_scene_info = {
        'scene_size': np.array([bbox_min.tolist(), bbox_max.tolist()]) * scale_factor_prev,
        'cubes': new_cubes,
        'scene_has_frames': scene_info['scene_has_frames'],
        'cubes_frame': new_cubes_frame,
        'scene_has_window': scene_info['scene_has_window'],
        'cubes_window': new_cubes_window,
        'positions_light': new_positions_light,
        'window_additional_planer': new_window_additional_planer,
        'scene_has_aa': scene_info['scene_has_aa'],
        'cubes_aa': new_cubes_aa,
        'scene_has_sticks': scene_info['scene_has_sticks'],
        'cubes_thin': new_cubes_thin
    }
    return new_scene_info


def parse_scene_cube_info(bbox_min, bbox_max, scale_factor):
    '''
    bbox_min and bbox_max are min and max values of 3 axes before normalization
    scale_factor is the factor to normalize the scene into [-1,1] box
    '''
    base_path = os.path.dirname(args.object_path)
    scene_info_path = os.path.join(base_path, 'info_layout.json')
    scale_info_path = os.path.join(base_path, 'object.info')

    with open(scene_info_path, 'r') as f:
        scene_info = json.load(f)
    scale_info = np.loadtxt(scale_info_path)

    scale_info = normalize_scale_info(scale_info, scale_factor)
    scene_info = normalize_scene_info(scene_info, scene_info, bbox_min, bbox_max, scale_factor)
    return scale_info, scene_info


def add_camera(constrained=True, target_location=(0, 0, 0)):
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    camera_object = bpy.context.object
    scene.camera = camera_object  # make this the current camera
    camera_object.location = (0, 1.2, 0)    # a dummy location

    hfov = np.random.uniform(FOV_RANGE[0], FOV_RANGE[1]) #50
    camera_object.data.sensor_width = 32
    camera_object.data.lens = camera_object.data.sensor_width / (
        2 * np.tan(np.deg2rad(hfov / 2))
    )
    #camera_object.data.clip_start = NEAR_PLANE

    if constrained:
        cam_constraint = camera_object.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"

        # create an empty object to track; look at the specified target location
        empty = bpy.data.objects.new("Empty", None)
        empty.location = target_location
        scene.collection.objects.link(empty)
        cam_constraint.target = empty

    return camera_object


def is_outside_cubes(point, cubes, threshold):
    for center, scale, _ in cubes:
        center = np.array(center)
        scale = np.array(scale)
        half_scale = scale / 2.0
        min_corner = center - half_scale
        max_corner = center + half_scale

        if np.all((min_corner - threshold) <= point) and np.all(point <= (max_corner + threshold)):
            return False
    return True


def sample_point_scene_boundary(bbox_min, bbox_max, boundary_sample_type='all_faces'):
    eps = random.uniform(CAMERA_POS_EPS_RANGE[0], CAMERA_POS_EPS_RANGE[1])
    new_bbox_min = [it+eps for it in bbox_min]
    new_bbox_max = [it-eps for it in bbox_max]

    # Define the regions
    # all_regions = [
    #     (new_bbox_min, np.array([new_bbox_min[0], new_bbox_max[1], new_bbox_max[2]])),  # Region 1: x in min range
    #     (new_bbox_min, np.array([new_bbox_max[0], new_bbox_min[1], new_bbox_max[2]])),  # Region 2: y in min range
    #     (new_bbox_min, np.array([new_bbox_max[0], new_bbox_max[1], new_bbox_min[2]])),  # Region 3: z in min range
    #     (np.array([new_bbox_max[0], new_bbox_min[1], new_bbox_min[2]]), new_bbox_max),  # Region 4: x in max range
    #     (np.array([new_bbox_min[0], new_bbox_max[1], new_bbox_min[2]]), new_bbox_max),  # Region 5: y in max range
    #     (np.array([new_bbox_min[0], new_bbox_min[1], new_bbox_max[2]]), new_bbox_max),  # Region 6: z in max range
    #     # (np.array([new_bbox_min[0], new_bbox_max[1], new_bbox_max[2]]), new_bbox_max),  # Region 7: x in min range, y in max range
    #     # (np.array([new_bbox_max[0], new_bbox_min[1], new_bbox_max[2]]), new_bbox_max)   # Region 8: x in max range, y in min range
    # ]
    all_regions = [
        (new_bbox_min, np.array([new_bbox_min[0], new_bbox_max[1], new_bbox_max[2]])),  # Region 1: x in min range
        (new_bbox_min, np.array([new_bbox_max[0], new_bbox_min[1], new_bbox_max[2]])),  # Region 2: y in min range
        (new_bbox_min, np.array([new_bbox_max[0], new_bbox_max[1], new_bbox_min[2]])),  # Region 3: z in min range
        (np.array([new_bbox_max[0], new_bbox_min[1], new_bbox_min[2]]), new_bbox_max),  # Region 4: x in max range
        (np.array([new_bbox_min[0], new_bbox_max[1], new_bbox_min[2]]), new_bbox_max),  # Region 5: y in max range
        (np.array([new_bbox_min[0], new_bbox_min[1], new_bbox_max[2]*0.7]), [new_bbox_max[0], new_bbox_max[1], new_bbox_max[2]*0.7]),  # Region 6: z in max range
        # (np.array([new_bbox_min[0], new_bbox_max[1], new_bbox_max[2]]), new_bbox_max),  # Region 7: x in min range, y in max range
        # (np.array([new_bbox_max[0], new_bbox_min[1], new_bbox_max[2]]), new_bbox_max)   # Region 8: x in max range, y in min range
    ]

    if boundary_sample_type == 'all_faces':
        regions = all_regions
    elif boundary_sample_type == '3_faces':
        regions = [all_regions[2], all_regions[0], all_regions[3]] + [random.choice([all_regions[1], all_regions[4]])]
        if random.uniform(0,1) < 0.5:
            regions += [all_regions[-1]]
    elif boundary_sample_type == '2_faces':
        regions = [all_regions[2]] + [random.choice([all_regions[1], all_regions[4]])] + [random.choice([all_regions[0], all_regions[3]])]
        if random.uniform(0,1) < 0.5:
            regions += [all_regions[-1]]
    else:
        raise NotImplementedError

    # Randomly select a region
    region_min, region_max = regions[np.random.randint(0, len(regions))]

    # Sample a point within the selected region
    point = np.random.uniform(region_min, region_max)
    return point


def hanwen_sample_cam_loc(scene_info, type_camera='center', threshold=CAMERA_SAMPLE_DIST_THRE, num_samples=12, max_trials=3000):
    '''
    We have different sampling strategies
    Type 'center': sample cameras at the center of room (size is half of the room), and give random pose within pre-defined ranges
    '''
    assert type_camera in ['center', 'boundary']
    bbox_min, bbox_max = scene_info['scene_size']
    cubes = scene_info['cubes'] + scene_info['cubes_frame'] + scene_info['cubes_aa'] + scene_info['cubes_thin']
    sampled_points = []
    boundary_sample_type = random.choice(['all_faces', '3_faces', '2_faces'])

    trials = 0
    while len(sampled_points) < num_samples:
        trials += 1
        if trials > max_trials:
            threshold *= 0.95
            print(f'Current camera location sampling threshold {threshold}')
            trials = 0

        if type_camera == 'center':
            point = np.random.uniform(bbox_min / 4, bbox_max / 4)
        elif type_camera == 'boundary':
            point = sample_point_scene_boundary(bbox_min, bbox_max, boundary_sample_type)
        else:
            raise NotImplementedError

        if is_outside_cubes(point, cubes, threshold):
            sampled_points.append(point)
    print('Camera sampled at:')
    print(sampled_points)
    return np.array(sampled_points)


def hanwen_sample_cam_rot():
    # default camera with Euler((0.0, 0.0, 0.0), 'XYZ') looks downwards at the ground
    yaw = np.random.uniform(AZIMUTH_RANGE[0], AZIMUTH_RANGE[1])  # azimuth, in degree
    pitch = 90.0 + np.random.uniform(ELEVATION_RANGE[0], ELEVATION_RANGE[1])
    roll = np.random.uniform(ROLL_RANGE[0], ROLL_RANGE[1])
    yaw, pitch, roll = math.radians(yaw), math.radians(pitch), math.radians(roll)
    euler = np.array([pitch, roll, yaw])
    euler_blender = Euler((euler[0], euler[1], euler[2]), 'XYZ')
    return euler_blender


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
    with open(os.path.join(rendering_dir, "../minmax.txt"), "w") as f:
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


def scene_add_frames(scene_info):
    cubes_frame = scene_info['cubes_frame']
    for cube in cubes_frame:
        prim_type = random.choice(['sphere', 'cube', 'torus'])      # 'cylinder', 'cone',
        location, scale, cube_type = cube
        size = np.mean(scale) #random.uniform(0.3, 0.6)
        subdivide_type = 'SIMPLE'
        subdivide_level = 0
        thickness = random.uniform(size/30, size/20)
        if prim_type == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location, segments=8, ring_count=8)
        elif prim_type == 'cube':
            subdivide_level = random.choice([0, 1, 2])
            thickness *= random.uniform(1.0, 3.0 - subdivide_level)
            bpy.ops.mesh.primitive_cube_add(size=2 * size, location=location)  # Cube size is edge length
        elif prim_type == 'torus':
            bpy.ops.mesh.primitive_torus_add(location=location, major_radius=size, minor_radius=size * 0.3, major_segments=8, minor_segments=4)
        else:
            raise ValueError(f"Unsupported primitive type: {prim_type}")

        # The newly added object becomes the active object in the scene.
        obj = bpy.context.active_object
        obj.location = (location[0], location[1], location[2])
        obj.scale = (scale[0], scale[1], scale[2])
        cutter = obj.name
        mat_name = augment_shape.get_a_random_material_name()
        augment_shape.link_material_to_object_material_slot(cutter, mat_name)

        # get modifier for sub-deviding primitive
        if subdivide_level > 0:
            subdiv_mod_name, _ = augment_shape.add_subdivision_modifier(cutter, type=subdivide_type, levels=subdivide_level, render=subdivide_level, overwrite=True)
        else:
            subdiv_mod_name = None

        # get modifier for getting wireframe
        wf_mod_name, _ = augment_shape.add_wireframe_modifier(cutter, thickness=thickness, overwrite=True)

        # apply modifier
        if subdiv_mod_name:
            print(f"Added wireframe modifier: {subdiv_mod_name}")
            augment_shape.apply_modifier(cutter, subdiv_mod_name)
        if wf_mod_name:
            print(f"Added wireframe modifier: {wf_mod_name}")
            augment_shape.apply_modifier(cutter, wf_mod_name)


def scene_add_window(obj_name, scene_info, modify_mat_flag):
    # create hole on the wall using boolean operation
    for window in scene_info['cubes_window']:
        location, scale, _ = window
        min_idx = np.argmin(scale)
        scale[min_idx] *= 10
        # bpy.ops.mesh.primitive_plane_add(scale=scale, location=location)
        bpy.ops.mesh.primitive_cube_add(scale=[it/2.0 for it in scale], location=location)
        cutter_obj = bpy.context.active_object

        cutter = cutter_obj.name
        mat_name = augment_shape.get_a_random_material_name()
        augment_shape.link_material_to_object_material_slot(cutter, mat_name)
        bool_mod_name = augment_shape.add_boolean_modifier_to_target(obj_name, cutter, operation='DIFFERENCE', solver='FAST')
        if bool_mod_name:
            print(f"Added Boolean modifier for creating window: {bool_mod_name}")
            augment_shape.apply_modifier(obj_name, bool_mod_name)
            augment_shape.remove_object_and_data(cutter)
            # solid_mod_name = augment_shape.add_solidify_modifier(obj_name, SOLIDIFY_THICKNESS)
            # if solid_mod_name and USE_SOLIDIFIER:
            #     augment_shape.apply_modifier(obj_name, solid_mod_name)

    # create glass on the wall
    for window in scene_info['cubes_window']:
        if random.uniform(0,1) < PROB_WINDOW_GLASS:
            location, scale, _ = window
            min_idx = np.argmin(scale)
            scale[min_idx] *= 3
            bpy.ops.mesh.primitive_cube_add(scale=[it/2.0 for it in scale], location=location)
            glass_obj = bpy.context.active_object
            glass_material = bpy.data.materials.new(name="GlassMaterial")
            glass_material.use_nodes = True  # Enable nodes
            glass_material.node_tree.nodes.clear()
            bsdf = glass_material.node_tree.nodes.new(type='ShaderNodeBsdfGlass')
            bsdf.inputs['IOR'].default_value = random.uniform(GLASS_IOR_RANGE[0], GLASS_IOR_RANGE[1])  # Index of Refraction for glass
            bsdf.inputs['Roughness'].default_value = random.uniform(GLASS_ROUGHNESS_RANGE[0], GLASS_ROUGHNESS_RANGE[1])  # Adjust roughness for frosted glass effect
            material_output = glass_material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
            glass_material.node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])
            if glass_obj.data.materials:
                glass_obj.data.materials[0] = glass_material
            else:
                glass_obj.data.materials.append(glass_material)


    # create fence on window
    for planer in scene_info['window_additional_planer']:
        if random.uniform(0,1) < PROB_WINDOW_FENCE:
            location, scale, _ = planer
            min_idx = np.argmin(scale)
            thickness = random.uniform(np.median(scale)/40, np.median(scale)/30)
            scale[min_idx] = thickness / 2 #0.005
            bpy.ops.mesh.primitive_cube_add(scale=[it/2 for it in scale], location=location)
            obj = bpy.context.active_object
            cutter = obj.name
            mat_name = augment_shape.get_a_random_material_name()
            augment_shape.link_material_to_object_material_slot(cutter, mat_name)
            subdivide_type = 'SIMPLE'
            subdivide_level, thickness_factor = random.choice([(0,3), (1,3), (2,3), (3,4), (4,5)])
            thickness *= random.uniform(1.0, thickness_factor - subdivide_level)
            if subdivide_level == 4:
                thickness *= 0.5
            if subdivide_level > 0:
                subdiv_mod_name, _ = augment_shape.add_subdivision_modifier(cutter, type=subdivide_type, levels=subdivide_level, render=subdivide_level, overwrite=True)
            else:
                subdiv_mod_name = None
            wf_mod_name, _ = augment_shape.add_wireframe_modifier(cutter, thickness=thickness, overwrite=True)
            # apply modifier
            if subdiv_mod_name:
                print(f"Added window fence modifier: {subdiv_mod_name}")
                augment_shape.apply_modifier(cutter, subdiv_mod_name)
            if wf_mod_name:
                print(f"Added window fence modifier: {wf_mod_name}")
                augment_shape.apply_modifier(cutter, wf_mod_name)

    # create lighting
    if random.uniform(0, 1) < PROB_ADDITION_LIGHT:
        for light_location in scene_info['positions_light']:
            light_strength = random.uniform(LIGHT_STR_ADDITION_LIGHT[0], LIGHT_STR_ADDITION_LIGHT[1])
            print('Add additional lighting of strength {} at {}'.format(light_strength, light_location))
            add_sun_light('sunlight', light_location, light_strength)


def scene_add_aa(obj_name, scene_info):
    # create axis-aligned geometry
    for window in scene_info['cubes_aa']:
        location, scale, cube_type = window
        bpy.ops.mesh.primitive_cube_add(scale=[it/2.0 for it in scale], location=location)
        obj = bpy.context.active_object
        name = obj.name
        if random.uniform(0,1) < PROB_AA_GLASS and cube_type == 'aaB_thin':
            glass_material = bpy.data.materials.new(name="GlassMaterial")
            glass_material.use_nodes = True  # Enable nodes
            glass_material.node_tree.nodes.clear()
            bsdf = glass_material.node_tree.nodes.new(type='ShaderNodeBsdfGlass')
            bsdf.inputs['IOR'].default_value = random.uniform(GLASS_IOR_RANGE[0], GLASS_IOR_RANGE[1])  # Index of Refraction for glass
            bsdf.inputs['Roughness'].default_value = random.uniform(GLASS_ROUGHNESS_RANGE[0], GLASS_ROUGHNESS_RANGE[1])  # Adjust roughness for frosted glass effect
            material_output = glass_material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
            glass_material.node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])
            if obj.data.materials:
                obj.data.materials[0] = glass_material
            else:
                obj.data.materials.append(glass_material)
        else:
            mat_name = augment_shape.get_a_random_material_name()
            augment_shape.link_material_to_object_material_slot(name, mat_name)


def scene_add_sticks(obj_name, scene_info):
    stick_as_light = random.uniform(0,1) < PROB_STICK_LIGHT
    light_strength = STICK_LIGHT_STRENGTH1 if random.uniform(0,1) < 0.8 else STICK_LIGHT_STRENGTH2
    cubes_thin = scene_info['cubes_thin']
    for cube in cubes_thin:
        #print(cube)
        location, scale, cube_type = cube
        bpy.ops.mesh.primitive_cube_add(scale=[it/2.0 for it in scale], location=location)
        obj = bpy.context.active_object
        if random.uniform(0,1) < PROB_STICK_LIGHT_SLOT and stick_as_light:
            print('no stick lighting')
            # color = get_random_color()
            # subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            # subsurf.levels = 3
            # subsurf.render_levels = 3
            # bpy.ops.object.modifier_apply(modifier="Subdivision")
            # bpy.ops.object.shade_smooth()
            # emission_material = bpy.data.materials.new(name="Bulb_Emission")
            # emission_material.use_nodes = True
            # nodes = emission_material.node_tree.nodes
            # emission_node = nodes.new(type='ShaderNodeEmission')
            # emission_node.inputs['Strength'].default_value = random.uniform(light_strength[0], light_strength[1])
            # emission_node.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
            # material_output = nodes.get('Material Output')
            # links = emission_material.node_tree.links
            # links.new(emission_node.outputs['Emission'], material_output.inputs['Surface'])
            # obj.data.materials.append(emission_material)
        else:
            name = obj.name
            mat_name = augment_shape.get_a_random_material_name()
            augment_shape.link_material_to_object_material_slot(name, mat_name)


def scene_add_light_bulb(obj_name, scene_info, no_light_bulb=False):
    bbox_min, bbox_max = scene_info['scene_size']
    num_light_bulbs = random.randint(1, MAX_NUM_LIGHT_BULBS)
    light_strength = LIGHT_BULB_STRENGTH1 if random.uniform(0,1) < 0.85 else LIGHT_BULB_STRENGTH2
    for i in range(num_light_bulbs):
        location = [random.uniform(val_min, val_max) for (val_min, val_max) in zip(bbox_min.tolist(), bbox_max.tolist())]
        # add light object
        scale = np.random.uniform(LIGHT_BULB_SIZE[0], LIGHT_BULB_SIZE[1], 3).tolist()
        color = get_random_color_array()
        bpy.ops.mesh.primitive_cube_add(scale=[it/2.0 for it in scale], location=location)
        obj = bpy.context.active_object
        subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf.levels = 3
        subsurf.render_levels = 3
        bpy.ops.object.modifier_apply(modifier="Subdivision")
        bpy.ops.object.shade_smooth()
        emission_material = bpy.data.materials.new(name="Bulb_Emission")
        emission_material.use_nodes = True
        nodes = emission_material.node_tree.nodes
        emission_node = nodes.new(type='ShaderNodeEmission')
        emission_node.inputs['Strength'].default_value = 1.0
        emission_node.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
        material_output = nodes.get('Material Output')
        links = emission_material.node_tree.links
        links.new(emission_node.outputs['Emission'], material_output.inputs['Surface'])
        obj.data.materials.append(emission_material)
        # add light
        # bpy.ops.mesh.primitive_cube_add(scale=[it/1.95 for it in scale], location=location)
        # obj = bpy.context.active_object
        light_data = bpy.data.lights.new(name=f"Light_bulb_{i}", type='POINT')
        light_data.energy = random.uniform(light_strength[0], light_strength[1])
        light_data.color = color
        light_object = bpy.data.objects.new(name=f"Light_bulb_{i}_Object", object_data=light_data)
        light_object.location = (location[0], location[1], location[2])
        bpy.context.collection.objects.link(light_object)


def adjust_metallic_and_roughness(obj):
    if random.uniform(0,1) < PROB_MODIFY_METERIAL_SPECULAR_SCENE:
        metallic_range = METALLIC_RANGE_SPECULAR
        roughness_range = ROUGHNESS_RANGE_SPECULAR
    else:
        metallic_range = METALLIC_RANGE
        roughness_range = ROUGHNESS_RANGE

    for material in obj.data.materials:
        if random.uniform(0,1) > PROB_MODIFY_MATERIAL_SLOT:
            continue
        metallic_factor = random.uniform(metallic_range[0], metallic_range[1])
        roughness_value = random.uniform(roughness_range[0], roughness_range[1])
        if material.use_nodes:
            node_tree = material.node_tree
            # Find the Principled BSDF node
            principled_bsdf = None
            for node in node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    principled_bsdf = node
            if principled_bsdf:
                material.node_tree.nodes["Math"].inputs[
                    1
                ].default_value = metallic_factor
                # Disconnect the roughness channel and set roughness
                for link in node_tree.links:
                    if (
                        link.to_node == principled_bsdf
                        and link.to_socket.name == "Roughness"
                    ):
                        node_tree.links.remove(link)
                        print("Disconnected roughness input")
                principled_bsdf.inputs["Roughness"].default_value = roughness_value
                print("Set roughness to constant", roughness_value)


def load_object_return_name(object_path: str) -> str:
    # Capture current objects in the scene to find the newly added object(s)
    before_import = set(obj.name for obj in bpy.context.scene.objects)

    # Import object based on its file type
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}. Only .glb and .fbx files are supported.")

    # Determine the names of the newly imported object(s)
    after_import = set(obj.name for obj in bpy.context.scene.objects)
    new_objects = after_import - before_import

    # Handle the case where multiple objects are imported
    if not new_objects:
        raise RuntimeError("No new objects were added to the scene.")
    elif len(new_objects) > 1:
        print("Multiple objects imported. Returning the name of the first new object.")

    modify_mat_flag = False
    if random.uniform(0, 1) < PROB_MODIFY_METERIAL:
        modify_mat_flag = True
        print('Modifying material (roughness and metallic) of the scene')
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                adjust_metallic_and_roughness(obj)

    return next(iter(new_objects)), modify_mat_flag  # Returns the name of one of the new objects


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


def yujiang_sample_cam_rot():
    # default camera with Euler((0.0, 0.0, 0.0), 'XYZ') looks downwards at the ground
    yaw = np.random.uniform(AZIMUTH_RANGE[0], AZIMUTH_RANGE[1])  # azimuth, in degree
    pitch = 90.0 + np.random.uniform(ELEVATION_RANGE[0], ELEVATION_RANGE[1])
    roll = np.random.uniform(ROLL_RANGE[0], ROLL_RANGE[1])
    yaw, pitch, roll = math.radians(yaw), math.radians(pitch), math.radians(roll)
    euler = np.array([pitch, roll, yaw])
    euler_blender = Euler((euler[0], euler[1], euler[2]), 'XYZ')
    return euler_blender


def SLERP_between_two_rot(rot1, rot2, t):
    if not isinstance(rot1, Euler) or not isinstance(rot2, Euler):
        raise TypeError("Both rot1 and rot2 must be Blender Euler objects.")

    return rot1.to_quaternion().slerp(rot2.to_quaternion(), t).to_euler('XYZ')


def yujiang_sample_cam_loc(scene_info, type_camera='center', threshold=CAMERA_SAMPLE_DIST_THRE,
                           num_samples=12, num_control_points=5,
                           max_trials=3000, max_step_length=0.03, max_rotation_degree_limit=1.0
                           ):
    '''
    We have different sampling strategies
    Type 'center': sample cameras at the center of room (size is half of the room), and give random pose within pre-defined ranges
    '''
    assert type_camera in ['center', 'boundary']
    bbox_min, bbox_max = scene_info['scene_size']
    cubes = scene_info['cubes'] + scene_info['cubes_frame'] + scene_info['cubes_aa'] + scene_info['cubes_thin']
    control_points = []
    control_rotations = []
    # boundary_sample_type = random.choice(['all_faces', '3_faces', '2_faces'])

    trials = 0
    while len(control_points) < num_control_points:
        trials += 1
        if trials > max_trials:
            threshold *= 0.95
            print(f'Current camera location sampling threshold {threshold}')
            trials = 0

        if type_camera == 'center':
            if len(control_points) == 0:
                point = np.random.uniform(bbox_min / 4, bbox_max / 4)
                random_sample_flag = np.random.uniform(0, 1) < 0.5
                if random_sample_flag:
                    rotation_euler = yujiang_sample_cam_rot()
                else:
                    prob = []
                    for cube in scene_info['cubes']:
                        center, scale, _ = cube
                        dist = np.linalg.norm(center - point)
                        prob.append(np.linalg.norm(scale) / (dist + 1e-6))
                    prob = np.array(prob)
                    prob /= np.sum(prob)
                    idx = np.random.choice(len(scene_info['cubes']), p=prob)
                    center, _, _ = scene_info['cubes'][idx]
                    y_axis = point - center
                    y_axis /= np.linalg.norm(y_axis)
                    y_axis = mathutils.Vector(y_axis)
                    rotation_euler = y_axis.to_track_quat('Z', 'Y').to_euler('XYZ')
            else:
                step_length = np.random.rand() * max_step_length
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                dx = step_length * np.sin(phi) * np.cos(theta)
                dy = step_length * np.sin(phi) * np.sin(theta)
                dz = step_length * np.cos(phi)
                point = control_points[-1] + np.array([dx, dy, dz])
                yaw = np.random.uniform(-max_rotation_degree_limit, max_rotation_degree_limit)
                pitch = np.random.uniform(-max_rotation_degree_limit, max_rotation_degree_limit)
                roll = np.random.uniform(-max_rotation_degree_limit, max_rotation_degree_limit)
                yaw, pitch, roll = math.radians(yaw), math.radians(pitch), math.radians(roll)
                euler = np.array([pitch, roll, yaw])
                euler_blender = Euler((euler[0], euler[1], euler[2]), 'XYZ')
                rotation_euler = (euler_blender.to_matrix() @ control_rotations[-1].to_matrix()).to_euler('XYZ')
        # elif type_camera == 'boundary':
        #     point = sample_point_scene_boundary(bbox_min, bbox_max, boundary_sample_type)
        else:
            raise NotImplementedError

        if is_outside_cubes(point, cubes, threshold):
            control_points.append(point)
            control_rotations.append(rotation_euler)

    sampled_points = []
    sampled_rotations = []
    control_point_time = np.linspace(0, 1, num_control_points)
    spline = CubicSpline(control_point_time, np.array(control_points), bc_type='natural')
    for t in np.linspace(0, 1, num_samples):
        sampled_points.append(spline(t))
        begin_idx = int(t // (1/(num_control_points-1)))
        end_idx = min(begin_idx + 1, num_control_points - 1)
        t_local = t * (num_control_points - 1) - begin_idx
        sampled_rot = SLERP_between_two_rot(control_rotations[begin_idx], control_rotations[end_idx], t_local)
        sampled_rotations.append(sampled_rot)

    return np.array(sampled_points), sampled_rotations


def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #


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


def yujiang_sample_cam_loc_rot_scan(scene_info, num_samples=12, scale=1.0):
    '''
    We have different sampling strategies
    Type 'center': sample cameras at the center of room (size is half of the room), and give random pose within pre-defined ranges
    '''
    assert len(scene_info['cubes']) == 1
    bbox_min, bbox_max = scene_info['scene_size']
    cubes = scene_info['cubes'] + scene_info['cubes_frame'] + scene_info['cubes_aa'] + scene_info['cubes_thin']
    sampled_points = []
    boundary_sample_type = random.choice(['all_faces', '3_faces', '2_faces'])
    distance = np.linalg.norm(scene_info['cubes'][0][1]) * 5 * scale
    azimuths, elevations = equatorial_trajectory(num_samples + 1)
    azimuths = np.deg2rad(np.array(azimuths))[:num_samples]
    elevations = np.deg2rad(np.array(elevations))[:num_samples]
    distances = np.asarray([distance for _ in range(num_samples)])
    cam_pts = az_el_to_points(azimuths, elevations) * distances[:,None]
    return cam_pts


def yujiang_sample_control_points_on_sphere(num_samples=2, distances=[1,2]):
    while True:
        theta = np.asin(np.random.rand(num_samples-1)*2-1)
        phi = np.random.rand(num_samples-1) * 2 * np.pi
        distances_sample = np.random.rand(num_samples-1)
        theta = np.concat([theta, theta[:1]])
        phi = np.concat([phi, phi[:1]])
        distances_sample = np.concat([distances_sample, distances_sample[:1]])
        cam_pts = az_el_to_points(phi, theta)
        angle_distance = np.acos(cam_pts[1:][:, None, :] @ cam_pts[:-1][:, :, None]).squeeze()
        if np.any(angle_distance > 2*np.pi/3):
            continue
        else:
            break
    return cam_pts * (distances_sample * (distances[1]-distances[0]) + distances[0])[:, None], angle_distance
    
    
def yujiang_slerp_between_control_points(control_pts, control_pts_interval, num_samples=10):
    control_pts_interval = np.concat([[0], np.cumsum(control_pts_interval)])
    spl = make_interp_spline(control_pts_interval, control_pts, k=2)
    sample_point_interval = np.linspace(control_pts_interval[0], control_pts_interval[-1], num_samples)
    sample_point = spl(sample_point_interval)
    return sample_point
    
    
def render_images():
    try:
        """Saves rendered images of the object in the scene."""
        if os.path.exists(os.path.join(args.output_dir, "000", "opencv_cameras.json")):
            print(os.path.join(args.output_dir, "000", "opencv_cameras.json"), "exist!")
            return True
        os.system(f"rm -rf {args.output_dir}")
        print(f"rm -rf {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

        # reset scene, load from glb file
        reset_scene()
        obj_name, modify_mat_flag = load_object_return_name(args.object_path)

        # transform coordinate frame to the one used in create_scenes.py
        transform_scene()
        bbox_min, bbox_max, scale_factor = normalize_scene()    # normalized the scene
        scale_info, scene_info = parse_scene_cube_info(bbox_min, bbox_max, scale_factor)    # normalized

        # add primitives-based wireframes
        if scene_info['scene_has_frames']:
            scene_add_frames(scene_info)

        # add window and window wireframe, lighting of sun or env map, and window glass
        if scene_info['scene_has_window']:
            scene_add_window(obj_name, scene_info, modify_mat_flag)

        # add axis-aligned geometry
        if scene_info['scene_has_aa']:
            scene_add_aa(obj_name, scene_info)

        # add sticks and stick light
        if scene_info['scene_has_sticks']:
            scene_add_sticks(obj_name, scene_info)

        # add default area lighting
        # add_area_light(name='area_light_downwards', location=np.zeros(3), size_x=None, size_y=None)
        # add_area_light(name='area_light_upwards', location=np.zeros(3), size_x=None, size_y=None, flip=True)
        env_light = get_random_color()
        world_tree = bpy.context.scene.world.node_tree
        back_node = world_tree.nodes['Background']
        back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
        back_node.inputs['Strength'].default_value = 1e-3
        # add light bulbs
        # if random.uniform(0,1) < PROB_LIGHT_BULB:
        #     scene_add_light_bulb(obj_name, scene_info)
        # create lighting
        if random.uniform(0, 1) < PROB_ADDITION_LIGHT:
            positions_light = []
            for _ in range(random.randint(2, MAX_NUM_ADDITION_LIGHT)):
                ### near
                sampled_pos = [0 for _ in range(3)]
                if random.uniform(0, 1) < 1.0:
                    axis = random.choice([(0,1,2), (1,2,0), (2,0,1)])
                    face = random.choice([0, 1])
                    rand0 = random.uniform(0, 1)
                    rand1 = random.uniform(0, 1)
                    sampled_pos[axis[0]] = rand0 * bbox_min[axis[0]] + (1-rand0) * bbox_max[axis[0]]
                    sampled_pos[axis[1]] = rand1 * bbox_min[axis[1]] + (1-rand1) * bbox_max[axis[1]]
                    sampled_pos[axis[2]] = face  * bbox_min[axis[2]] + (1-face ) * bbox_max[axis[2]]
                ### far
                positions_light.append(sampled_pos)
            for light_location in positions_light:
                light_strength = random.uniform(LIGHT_STR_ADDITION_LIGHT[0], LIGHT_STR_ADDITION_LIGHT[1])
                print('Add additional lighting of strength {} at {}'.format(light_strength, light_location))
                add_sun_light('sunlight', light_location, light_strength)
            
        if args.save_norm_glb:
            bpy.ops.export_scene.gltf(
                filepath=os.path.join(args.output_dir, "norm_scene.glb"),
                export_format="GLB",
            )

        # rendering loop for different camera sampling settings
        proj_names = args.proj_names.split(",")
        
        if args.repeat:
            seed_everything(19260817)
        if args.micro:
            for render_idx in range(args.render_times):
                try:
                    proj_name = str(render_idx).zfill(3)
                    out_dir = os.path.join(args.output_dir, proj_name)
                    if os.path.isdir(out_dir):
                        os.system(f"rm -rf {out_dir}")

                    out_rendering_dir = os.path.join(out_dir, "renderings")
                    os.makedirs(out_rendering_dir, exist_ok=True)

                    # start rendering
                    opencv_cameras = {"frames": []}
                    cur_views = 0
                    # for camera_type in ["boundary", "center"]: #["boundary"]:#["boundary", "center"]:
                    for camera_type in ["center"]:
                        if camera_type == "boundary":
                            camera = add_camera(constrained=True)
                        elif camera_type == "center":
                            camera = add_camera(constrained=False)
                        else:
                            raise NotImplementedError

                        # sample camera location
                        # if proj_name == "megasynth":
                        #     print('Sampling camera locations')
                        #     if camera_type == "boundary":
                        #         cam_locations = hanwen_sample_cam_loc(scene_info, type_camera='boundary', num_samples=NUM_BDRY_VIEWS)
                        #     elif camera_type == "center":

                        #     else:
                        #         raise NotImplementedError
                        # else:
                        #     raise ValueError(f"Unknown projection name: {args.proj_names}")
                        if camera_type == "center":
                            cam_locations, cam_rotations = yujiang_sample_cam_loc(scene_info, type_camera='center', num_samples=args.render_views, num_control_points=args.control_points_num,)
                        else:
                            raise NotImplementedError

                        # for all sampled camera, do render, and save results
                        for idx in range(cam_locations.shape[0]):
                            camera.location = cam_locations[idx]
                            if camera_type == "center":
                                camera.rotation_euler = cam_rotations[idx]

                            rgba_path = os.path.join(out_rendering_dir, f"{(idx + cur_views):08d}_rgba.exr")
                            depth_path = os.path.join(out_rendering_dir, f"{(idx + cur_views):08d}_depth.exr")
                            setup_camera_rendering(rgba_path, depth_path)
                            bpy.ops.render.render(write_still=True)
                            actual_rgba_path, actual_depth_path = rgba_path.replace('.exr', '.exr0001.exr'), depth_path.replace('.exr', '.exr0001.exr')
                            print(f"renaming {actual_rgba_path} to {rgba_path}")
                            print(f"renaming {actual_depth_path} to {depth_path}")
                            os.system(
                                f"mv {actual_rgba_path} {rgba_path}; mv {actual_depth_path} {depth_path}"
                            )

                            cam_dict = get_camera_params(camera)
                            cam_dict["file_path"] = os.path.relpath(rgba_path, out_dir)
                            cam_dict["blender_camera_location"] = cam_locations[idx].tolist()
                            opencv_cameras["frames"].append(cam_dict)
                        cur_views += cam_locations.shape[0]

                    if not args.no_tonemap:
                        tonemap_folder(out_rendering_dir, keep_exr=args.keep_exr)
                        # change file_path to png
                        for frame in opencv_cameras["frames"]:
                            frame["file_path"] = frame["file_path"][:-4] + ".png"

                    camera_fpath = f"{out_dir}/opencv_cameras.json"
                    with open(camera_fpath, "w") as f:
                        json.dump(opencv_cameras, f, indent=4)
                except Exception as e:
                    if os.path.isdir(out_dir):
                        os.system(f"rm -rf {out_dir}")
                    print(f"Error during rendering: {e}")
                # remove temp_dir

                print(f"Removing {temp_dir}")
                os.system(f"rm -rf {temp_dir}")
        elif args.scan:
            for render_idx in range(args.render_times):
                proj_name = str(render_idx).zfill(3)
                out_dir = os.path.join(args.output_dir, proj_name)
                if os.path.isdir(out_dir):
                    os.system(f"rm -rf {out_dir}")

                out_rendering_dir = os.path.join(out_dir, "renderings")
                os.makedirs(out_rendering_dir, exist_ok=True)

                # start rendering
                opencv_cameras = {"frames": []}
                cur_views = 0
                x, y, z = np.random.randn(3) * 0.01
                camera = add_camera(constrained=True, target_location=(x, y, z))
                # sample camera location
                print('Sampling camera locations')
                # cam_locations = yujiang_sample_cam_loc_rot_scan(scene_info, num_samples=96, scale=1.0)
                distance = SCENE_SCALE_MAX * (3 ** 0.5) * 1.5
                control_points, control_points_dis = yujiang_sample_control_points_on_sphere(num_samples=args.control_points_num, distances=[distance*0.7, distance*1.0])
                cam_locations = yujiang_slerp_between_control_points(control_points, control_points_dis, num_samples=100)

                # for all sampled camera, do render, and save results
                for idx in range(cam_locations.shape[0]):
                    camera.location = cam_locations[idx]

                    rgba_path = os.path.join(out_rendering_dir, f"{(idx + cur_views):08d}_rgba.exr")
                    depth_path = os.path.join(out_rendering_dir, f"{(idx + cur_views):08d}_depth.exr")
                    setup_camera_rendering(rgba_path, depth_path)
                    bpy.ops.render.render(write_still=True)
                    actual_rgba_path, actual_depth_path = rgba_path.replace('.exr', '.exr0001.exr'), depth_path.replace('.exr', '.exr0001.exr')
                    print(f"renaming {actual_rgba_path} to {rgba_path}")
                    print(f"renaming {actual_depth_path} to {depth_path}")
                    os.system(
                        f"mv {actual_rgba_path} {rgba_path}; mv {actual_depth_path} {depth_path}"
                    )

                    cam_dict = get_camera_params(camera)
                    cam_dict["file_path"] = os.path.relpath(rgba_path, out_dir)
                    cam_dict["blender_camera_location"] = cam_locations[idx].tolist()
                    opencv_cameras["frames"].append(cam_dict)
                cur_views += cam_locations.shape[0]

                if not args.no_tonemap:
                    tonemap_folder(out_rendering_dir, keep_exr=args.keep_exr)
                    # change file_path to png
                    for frame in opencv_cameras["frames"]:
                        frame["file_path"] = frame["file_path"][:-4] + ".png"

                camera_fpath = f"{out_dir}/opencv_cameras.json"
                with open(camera_fpath, "w") as f:
                    json.dump(opencv_cameras, f, indent=4)

            # remove temp_dir
            print(f"Removing {temp_dir}")
            os.system(f"rm -rf {temp_dir}")
        else:
            for render_idx in range(args.render_times):
                proj_name = str(render_idx).zfill(3)
                out_dir = os.path.join(args.output_dir, proj_name)
                if os.path.isdir(out_dir):
                    os.system(f"rm -rf {out_dir}")

                out_rendering_dir = os.path.join(out_dir, "renderings")
                os.makedirs(out_rendering_dir, exist_ok=True)

                # start rendering
                opencv_cameras = {"frames": []}
                cur_views = 0
                for camera_type in ["boundary", "center"]: #["boundary"]:#["boundary", "center"]:
                    if camera_type == "boundary":
                        camera = add_camera(constrained=True)
                    elif camera_type == "center":
                        camera = add_camera(constrained=False)
                    else:
                        raise NotImplementedError

                    # sample camera location
                    print('Sampling camera locations')
                    if camera_type == "boundary":
                        cam_locations = hanwen_sample_cam_loc(scene_info, type_camera='boundary', num_samples=NUM_BDRY_VIEWS)
                    elif camera_type == "center":
                        cam_locations = hanwen_sample_cam_loc(scene_info, type_camera='center', num_samples=NUM_CNTR_VIEWS)
                    else:
                        raise NotImplementedError


                    # for all sampled camera, do render, and save results
                    for idx in range(cam_locations.shape[0]):
                        camera.location = cam_locations[idx]
                        if camera_type == "center":
                            rotation_euler = hanwen_sample_cam_rot()
                            camera.rotation_euler = rotation_euler

                        rgba_path = os.path.join(out_rendering_dir, f"{(idx + cur_views):08d}_rgba.exr")
                        depth_path = os.path.join(out_rendering_dir, f"{(idx + cur_views):08d}_depth.exr")
                        setup_camera_rendering(rgba_path, depth_path)
                        bpy.ops.render.render(write_still=True)
                        actual_rgba_path, actual_depth_path = rgba_path.replace('.exr', '.exr0001.exr'), depth_path.replace('.exr', '.exr0001.exr')
                        print(f"renaming {actual_rgba_path} to {rgba_path}")
                        print(f"renaming {actual_depth_path} to {depth_path}")
                        os.system(
                            f"mv {actual_rgba_path} {rgba_path}; mv {actual_depth_path} {depth_path}"
                        )

                        cam_dict = get_camera_params(camera)
                        cam_dict["file_path"] = os.path.relpath(rgba_path, out_dir)
                        cam_dict["blender_camera_location"] = cam_locations[idx].tolist()
                        opencv_cameras["frames"].append(cam_dict)
                    cur_views += cam_locations.shape[0]

                if not args.no_tonemap:
                    tonemap_folder(out_rendering_dir, keep_exr=args.keep_exr)
                    # change file_path to png
                    for frame in opencv_cameras["frames"]:
                        frame["file_path"] = frame["file_path"][:-4] + ".png"

                camera_fpath = f"{out_dir}/opencv_cameras.json"
                with open(camera_fpath, "w") as f:
                    json.dump(opencv_cameras, f, indent=4)

            # remove temp_dir
            print(f"Removing {temp_dir}")
            os.system(f"rm -rf {temp_dir}")
    
    except Exception as e:
        # remove temp_dir
        print(f"Removing {temp_dir}")
        os.system(f"rm -rf {temp_dir}")

        print(f"Removing {args.local_cache_dir}")
        os.system(f"rm -rf {args.local_cache_dir}")
        raise e


def seed_everything(random_seed, np_seed):
    np.random.seed(np_seed)
    random.seed(random_seed)
    print(f'Seed: {random_seed}, {np_seed}')


if __name__ == "__main__":
    start_time = time.time()
    seed_everything(args.seed, args.np_seed)
    render_images()
    print(f'TIME - megasynth_rgbd.py: rendering time: {time.time() - start_time:.2f}s')
