import sys
import numpy as np
import pickle
from pathlib import Path
import argparse
import sys
import cv2
import os
from glob import glob
import random
import shutil
import time
import uuid
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datasets import load_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from common import *
from create_shapes_planes import *
from convert_obj_to_glb import convert_file


SCENE_SCALE_MAX = 1.0            # for normalizing the scene into [-scale, scale] box
CAMERA_SAMPLE_DIST_THRE = 0.1    # 0.15, the smallest distance from the sampled cameras to any 3D cubes
PROB_THINB_WALL = 1.0
PROB_THINB_SPACE = 0.5
PROB_FRAME = 0.8
PROB_FRAME_ADD_GEOM = 0.5
PROB_WINDOW = 0.8
PROB_AA = 0.7
WALL_OPTIONS = {'xy_front': ['x', 'y'], 'xy_back': ['x', 'y'],
                'xz_left': ['x', 'z'], 'xz_right': ['x', 'z'],
                'yz_top': ['y', 'z'], 'yz_bottom': ['y', 'z']}

# possibility of different number of objects in each box
CUBE_TO_SHAPE = {
    'lB': {"sub_obj_nums": [4, 5, 6, 7, 8, 9], "sub_obj_num_poss": [5, 7, 10, 7, 5]},   
    'sB': {"sub_obj_nums": [2, 3, 4, 5], "sub_obj_num_poss": [2, 3, 2, 1]},
    'rB': {"sub_obj_nums": [2, 3, 4, 5], "sub_obj_num_poss": [2, 3, 2, 1]},
    'wB': {"sub_obj_nums": [2, 3, 4, 5], "sub_obj_num_poss": [2, 3, 2, 1]},
    'fB': {"sub_obj_nums": [1, 2, 3], "sub_obj_num_poss": [2, 1, 1]},
}
CUBE_TO_EDGE_COLOR = {
    'lB': 'r', 
    'sB': 'g', 
    'rB': 'b', 
    'wB': 'orange',
    'thinB_wall': 'c',
    'thinB_space': 'm',
    'fB': 'lime',
    'window': 'gray',
    'aaB_thin': 'y',
    'aaB_thick': 'y',
}


def sample_point_scene_boundary(bbox_min, bbox_max):
    # Calculate the center and half size of the box
    center = (bbox_min + bbox_max) / 2
    half_size = (bbox_max - bbox_min) / 4

    # Define the regions outside the half box
    regions = [
        (bbox_min, np.array([bbox_min[0], bbox_max[1], bbox_max[2]])),  # Region 1: x in min range
        (bbox_min, np.array([bbox_max[0], bbox_min[1], bbox_max[2]])),  # Region 2: y in min range
        (bbox_min, np.array([bbox_max[0], bbox_max[1], bbox_min[2]])),  # Region 3: z in min range
        (np.array([bbox_max[0], bbox_min[1], bbox_min[2]]), bbox_max),  # Region 4: x in max range
        (np.array([bbox_min[0], bbox_max[1], bbox_min[2]]), bbox_max),  # Region 5: y in max range
        (np.array([bbox_min[0], bbox_min[1], bbox_max[2]]), bbox_max),  # Region 6: z in max range
        (np.array([bbox_min[0], bbox_max[1], bbox_max[2]]), bbox_max),  # Region 7: x in min range, y in max range
        (np.array([bbox_max[0], bbox_min[1], bbox_max[2]]), bbox_max)   # Region 8: x in max range, y in min range
    ]

    # Randomly select a region
    region_min, region_max = regions[np.random.randint(0, len(regions))]

    # Sample a point within the selected region
    point = np.random.uniform(region_min, region_max)
    return point


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


# helper functions
def draw_cube(ax, position, scale, color=None, cube_type='lB'):
    edge_color = CUBE_TO_EDGE_COLOR[cube_type]

    x = [position[0] - scale[0] / 2, position[0] + scale[0] / 2]
    y = [position[1] - scale[1] / 2, position[1] + scale[1] / 2]
    z = [position[2] - scale[2] / 2, position[2] + scale[2] / 2]

    vertices = [
        [x[0], y[0], z[0]],
        [x[1], y[0], z[0]],
        [x[1], y[1], z[0]],
        [x[0], y[1], z[0]],
        [x[0], y[0], z[1]],
        [x[1], y[0], z[1]],
        [x[1], y[1], z[1]],
        [x[0], y[1], z[1]]
    ]
    faces = [
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [7, 6, 2, 3]],
        [vertices[j] for j in [0, 3, 7, 4]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]]
    ]
    if not isinstance(color, np.ndarray):
        color = np.random.rand(3,)
    ax.add_collection3d(Poly3DCollection(faces, color=edge_color, linewidths=1, edgecolors=edge_color, alpha=.25))


def intersects(pos1, scale1, pos2, scale2):
    # Check for intersection between two cubes
    return (
        abs(pos1[0] - pos2[0]) < (scale1[0] + scale2[0]) / 2 and
        abs(pos1[1] - pos2[1]) < (scale1[1] + scale2[1]) / 2 and
        abs(pos1[2] - pos2[2]) < (scale1[2] + scale2[2]) / 2
    )


def normalize_possibility(sub_obj_num_poss):
    sub_obj_bound = np.reshape(sub_obj_num_poss, -1).astype(float)
    sub_obj_bound = sub_obj_bound / np.sum(sub_obj_bound)
    sub_obj_bound = np.cumsum(sub_obj_bound)  # normalized, cumulative sum of subObjPoss (possibility)
    if sub_obj_bound[-1] != 1.0:
        sub_obj_bound[-1] = 1.0  # setting 0.999... to 1.0
    return sub_obj_bound


def sample_sub_object_num(type_cube, shapeIds=[0]):
    sub_obj_nums, sub_obj_num_poss = CUBE_TO_SHAPE[type_cube]['sub_obj_nums'], CUBE_TO_SHAPE[type_cube]['sub_obj_num_poss']
    sub_obj_bound = normalize_possibility(sub_obj_num_poss)
    counts = np.zeros(len(sub_obj_nums))

    chooses = np.random.uniform(0, 1.0, len(shapeIds))
    for ii, i in enumerate(shapeIds):  # for each MultiShape (only one in practice)
        choose = chooses[ii]
        sub_obj_num = sub_obj_nums[-1]
        for iO in range(len(sub_obj_bound)):
            if choose < sub_obj_bound[iO]:  # randomly choose a sub obj
                sub_obj_num = sub_obj_nums[iO]
                counts[iO] += 1
                break
    return sub_obj_num


class Scene(Shape):
    def __init__(self, 
                # scene scale info
                scene_size_range=(17.0, 30.0), 
                scene_height_range=(10.0, 15.0), 
                wall_thickness=0.01,           # actually 2*wall_thickness
                # pre-defined scene box/object scale
                largeBox_size_range=(4,8,4,8),  # in format [min_wd, max_wd, min_h, max_h]
                smallBox_size_range=(2,4,2,6),  # in format [min_size_2D, max_size_2D, on_ground_min_h, on_ground_max_h], two types of small boxes
                roofBox_size_range=(2,5,3,4),   # in format [min_size, max_size, thin_rB_thickness, thick_rB_thickness], two types of roof boxes
                wallBox_size_range=(2,5,2,6),   # in format [min_size, max_size, thin_wB_thickness, thick_wB_thickness], two types of on-wall boxes
                thin_structure_size_range = (0.1, 0.6, 0.8, 1.8), # in format [min_size_space, max_size_space]
                frameBox_size_range=(3,6,3,6),  # in format [min_wd, max_wd, min_h, max_h]
                aaBox_size_range=(2,5,0.2,1),   # in format [min_size, max_size, thin_wB_thickness, thick_wB_thickness], two types of aa boxes
                # predefined scene box/object number
                max_num_large_cubes=5, 
                max_num_small_cubes=8,
                max_num_roof_cubes=4, 
                max_num_wall_cubes=6,
                max_num_frame_cubes=3,
                max_num_wall_thin_cubes=16,
                max_num_space_thin_cubes=6,
                max_num_aa_cubes=2,
                fixed_aspect_ratio=True):
        super(Scene, self).__init__()

        # scene layout parameters
        self.scene_size_range = scene_size_range
        self.scene_height_range = scene_height_range
        self.wall_thickness = wall_thickness

        # 3D cubes size parameters
        self.largeBox_size_range = largeBox_size_range
        self.smallBox_size_range = smallBox_size_range
        self.roofBox_size_range = roofBox_size_range
        self.wallBox_size_range = wallBox_size_range
        self.thin_structure_size_range = thin_structure_size_range  # stick-like thin structure
        self.frameBox_size_range = frameBox_size_range              # wireframe thin structure
        self.aaBox_size_range = aaBox_size_range                    # axis-aligned geometry

        # 3D cubes number parameters
        self.max_num_large_cubes = max_num_large_cubes
        self.max_num_small_cubes = max_num_small_cubes
        self.max_num_roof_cubes = max_num_roof_cubes
        self.max_num_wall_cubes = max_num_wall_cubes
        self.max_num_frame_cubes = max_num_frame_cubes
        self.max_num_wall_thin_cubes = max_num_wall_thin_cubes      # for each wall
        self.max_num_space_thin_cubes = max_num_space_thin_cubes
        self.max_num_aa_cubes = max_num_aa_cubes

        # MultiShape rescale setting
        self.fixed_aspect_ratio = fixed_aspect_ratio

        # scene property
        self.cubes = []         # type ['lB' (large Box), 'sB' (small Box), 'rB' (roof Box), 'wB' (wall Box)]
        self.cubes_frame = []   # type ['lB_frame', 'sB_frame']
        self.cubes_thin = []    # type ['thinB_wall', 'thinB_space']
        self.cubes_aa = []
        self.walls = []         # type ['lB' (large Box), 'sB' (small Box), 'rB' (roof Box), 'wB' (wall Box), 'W' (Wall)]
        self.thin_geoemtry = []
        self.used_wall = None

        # generate scene with heuristic
        self.scene_valid_flag = False
        while self.scene_valid_flag == False:
            self._generate_scene_size()
            self._generate_cubes()
            self._generate_cubes_window()       # no usage
            self._generate_cubes_thin()         # no usage
            self._validate_scene_layout()
        # self._generate_thin_structure()     # add stick-like thin structure geometry
        self._generate_walls()
        print(f"The scene has: \n"
               f"{self.num_lB}/{self.max_num_large_cubes} largeB, \n"
                f"{self.num_sB}/{self.max_num_small_cubes} smallB, \n"
                f"{self.num_rB}/{self.max_num_roof_cubes} roofB, \n"
                f"{self.num_wB}/{self.max_num_wall_cubes} wallB, \n"
                f"{self.num_geom_frame}/{self.max_num_frame_cubes} wireframeB, \n"
                f"{self.num_thinB_wall}/{self.max_num_wall_thin_cubes} stickB-wall, \n"
                f"{self.num_thinB_space}/{self.max_num_space_thin_cubes} stickB-space, \n"
                f"{self.num_aa_geom}/{self.max_num_aa_cubes} axis-alignB")


    def _reset_scene(self):
        self.scene_size = None
        self.cubes = []
        self.cubes_frame = []
        self.cubes_window = []
        self.cubes_thin = []
        self.cubes_aa = []
        self.walls = []
        self.used_wall = None
        self.thin_geometry = []

        
    def _generate_scene_size(self):
        # xy = np.random.uniform(self.scene_size_range[0], self.scene_size_range[1], 2)
        xy = np.array([20.0, 20.0])
        # z = np.random.uniform(self.scene_height_range[0], self.scene_height_range[1], 1)
        z = np.array([12.0])
        xyz = np.concatenate((xy, z), axis=0)   # [3,]
        self.scene_size = xyz   # [2D width, 2D depth, 3D height]
        print('Generate scene size', xyz)

    
    def _generate_cubes(self):
        """Procedual generation of 3D cubes, each cube represent one MultiShape"""
        fl_w, fl_d, fl_h = self.scene_size

        num_lB = random.randint(self.max_num_large_cubes // 2, self.max_num_large_cubes)
        num_sB = random.randint(self.max_num_small_cubes // 2, self.max_num_small_cubes)
        num_rB = random.randint(self.max_num_roof_cubes // 2, self.max_num_roof_cubes)
        num_wB = random.randint(self.max_num_wall_cubes // 2, self.max_num_wall_cubes)

        # # generate cubes for wireframes
        # self.scene_has_frames = random.random() < PROB_FRAME
        self.scene_has_frames = False
        # if self.scene_has_frames:
        #     num_geom_frame = random.randint(1, self.max_num_frame_cubes)
        #     if num_geom_frame == 1:
        #         num_sB = max(num_sB - 1, 1)
        #     elif num_geom_frame == 2:
        #         num_lB = max(num_lB - 1, 1)
        #     elif num_geom_frame == 3 or num_geom_frame == 4:
        #         num_lB = max(num_lB - 1, 1)
        #         num_sB = max(num_sB - 1, 1)
        #     else:
        #         raise NotImplementedError
        #     self.cubes_frame = [self.generate_frame_cube(cube_type='fB') for _ in range(num_geom_frame)]
        # else:
        #     num_geom_frame = 0
        num_geom_frame = 0

        # generate cubes for axis-aligned geometry
        self.scene_has_aa = False
        if self.scene_has_aa:
            num_aa_geom = random.randint(1, self.max_num_aa_cubes)
            self.cubes_aa = [self.generate_aa_cube() for _ in range(num_aa_geom)]
        else:
            num_aa_geom = 0

        # generate cubes for multiObj
        # self.cubes = [self.generate_large_cube() for _ in range(num_lB)]
        # small_cubes = [self.generate_small_cube() for _ in range(num_sB)]
        # self.cubes.extend(small_cubes)
        # self.cubes.extend(self.generate_roof_cube() for _ in range(num_rB))
        min_size, max_size, thin_wB_thickness, thick_wB_thickness = self.wallBox_size_range
        thin_wB_thickness = 0.3
        # self.cubes.extend(self.generate_wall_cube() for _ in range(num_wB))
        self.cubes.extend([((thin_wB_thickness/2, self.scene_size[1]/2, self.scene_size[2]/2), (thin_wB_thickness, 3, 4), 'wB')])

        self.num_lB = 0
        self.num_sB = 0
        self.num_rB = 0
        self.num_wB = 1
        self.num_geom_frame = 0
        self.num_aa_geom = 0

        # move the scene to be centered
        cubes = self.cubes.copy()
        self.cubes = []
        x, y, z = self.scene_size
        for cube in cubes:
            position, scale, cube_type = cube
            new_position = (position[0] - x / 2, position[1] - y / 2, position[2] - z / 2)
            self.cubes.append((new_position, scale, cube_type))

        cubes = self.cubes_frame.copy()
        self.cubes_frame = []
        for cube in cubes:
            position, scale, cube_type = cube
            new_position = (position[0] - x / 2, position[1] - y / 2, position[2] - z / 2)
            self.cubes_frame.append((new_position, scale, cube_type))

        cubes = self.cubes_aa.copy()
        self.cubes_aa = []
        for cube in cubes:
            position, scale, cube_type = cube
            new_position = (position[0] - x / 2, position[1] - y / 2, position[2] - z / 2)
            self.cubes_aa.append((new_position, scale, cube_type))


    def _generate_cubes_window(self):
        x, y, z = self.scene_size
        scene_size_max = max(x, y, z)
        light_distance_range = (scene_size_max/4, scene_size_max/2)
        t = self.wall_thickness
        self.scene_has_window = False
        self.cubes_window = []
        self.positions_light = []
        self.additional_planer = []

        # if random.random() < PROB_WINDOW:
        #     self.scene_has_window = True
        #     wall = random.choice(['xy_front', 'xy_back', 'xz_left', 'xz_right'])
        #     self.used_wall = wall
        #     wall = wall.split('_')[-1]
        #     height = random.uniform(z/5, 3*z/5)
        #     light_distance = random.uniform(light_distance_range[0], light_distance_range[1])
        #     if wall in ['front', 'back']:
        #         width = random.uniform(x/5, 3*x/5)
        #         size = [width, 2*t, height]
        #         center = [random.uniform(size[0]/2, x - size[0]/2),
        #                   y - size[1]/2 if wall == 'front' else size[1]/2,
        #                   random.uniform(size[2]/2, z - size[2]/2)]
        #         position_light = [random.uniform(center[0]-size[0]/2, center[0]+size[0]/2),
        #                           center[1] + light_distance if wall == 'front' else center[1] - light_distance,
        #                           random.uniform(center[2]-size[2]/2, center[2]+size[2]/2)]
        #         size_planer = [width, random.uniform(self.thin_structure_size_range[0], self.thin_structure_size_range[1]), height]
        #         position_planer = center.copy()
        #     elif wall in ['left', 'right']:
        #         width = random.uniform(y/5, 3*y/5)
        #         size = [2*t, width, height]
        #         center = [x - size[0]/2 if wall == 'right' else size[0]/2,
        #                   random.uniform(size[1]/2, y - size[1]/2),
        #                   random.uniform(size[2]/2, z - size[2]/2)]
        #         position_light = [center[0] + light_distance if wall == 'right' else center[0] - light_distance,
        #                           random.uniform(center[1]-size[1]/2, center[1]+size[1]/2),
        #                           random.uniform(center[2]-size[2]/2, center[2]+size[2]/2)]
        #         size_planer = [random.uniform(self.thin_structure_size_range[0], self.thin_structure_size_range[1]), width, height]
        #         position_planer = center.copy()
        #     else:
        #         raise NotImplementedError
        #     cube_window = (center, size, 'window')
        #     planer = (position_planer, size_planer, 'planer')
        #     self.cubes_window.append(cube_window)
        #     self.positions_light.append(position_light)
        #     self.additional_planer.append(planer)
        
        cubes = self.cubes_window.copy()
        self.cubes_window = []
        x, y, z = self.scene_size
        for cube in cubes:
            position, scale, cube_type = cube
            new_position = (position[0] - x / 2, position[1] - y / 2, position[2] - z / 2)
            self.cubes_window.append((new_position, scale, cube_type))
        positions_light = self.positions_light.copy()
        self.positions_light = []
        for position in positions_light:
            new_position_light = (position[0] - x / 2, position[1] - y / 2, position[2] - z / 2)
            self.positions_light.append(new_position_light)
        additional_planer = self.additional_planer.copy()
        self.additional_planer = []
        for planer in additional_planer:
            position, scale, planer_type = planer
            new_position = (position[0] - x / 2, position[1] - y / 2, position[2] - z / 2)
            self.additional_planer.append((new_position, scale, planer_type))


    def _generate_cubes_thin(self):
        """Generation of thin cubes (stick-like) for (1) on walls (2) in space"""
        x, y, z = self.scene_size
        cubes = []

        # # generate on-wall cubes
        # if random.random() < PROB_THINB_WALL:
        #     num_thinB_wall = random.randint(self.max_num_wall_thin_cubes // 3, self.max_num_wall_thin_cubes)
        #     wall = np.random.choice(list(set(WALL_OPTIONS.keys()) - set([self.used_wall])))
        #     # align = np.random.choice(WALL_OPTIONS[wall])
        #     for _ in range(num_thinB_wall):
        #         if random.random() < 0.7:
        #             thin_cross_size = np.random.uniform(self.thin_structure_size_range[0], self.thin_structure_size_range[1], size=2)
        #         else:
        #             thin_cross_size = np.random.uniform(self.thin_structure_size_range[1], self.thin_structure_size_range[2], size=2)
        #         align = np.random.choice(WALL_OPTIONS[wall])
        #         if wall in ['xy_front', 'xy_back']:
        #             if align == 'x':
        #                 size = [np.random.uniform(x/4, x/2), thin_cross_size[0], thin_cross_size[1]]
        #             elif align == 'y':
        #                 size = [thin_cross_size[0], np.random.uniform(y/4, y/2), thin_cross_size[1]]
        #             center = [
        #                 np.random.uniform(size[0]/2, x - size[0]/2),
        #                 np.random.uniform(size[1]/2, y - size[1]/2),
        #                 z - size[2]/2 if wall == 'xy_front' else size[2]/2
        #             ]
        #         elif wall in ['xz_left', 'xz_right']:
        #             if align == 'x':
        #                 size = [np.random.uniform(x/4, x/2), thin_cross_size[0], thin_cross_size[1]]
        #             elif align == 'z':
        #                 size = [thin_cross_size[0], thin_cross_size[1], np.random.uniform(z/4, z/2)]
        #             center = [
        #                 np.random.uniform(size[0]/2, x - size[0]/2),
        #                 y - size[1]/2 if wall == 'xz_left' else size[1]/2,
        #                 np.random.uniform(size[2]/2, z - size[2]/2)
        #             ]
        #         elif wall in ['yz_top', 'yz_bottom']:
        #             if align == 'y':
        #                 size = [thin_cross_size[0], np.random.uniform(y/4, y/2), thin_cross_size[1]]
        #             elif align == 'z':
        #                 size = [thin_cross_size[0], thin_cross_size[1], np.random.uniform(z/4, z/2)]
        #             center = [
        #                 size[0]/2 if wall == 'yz_top' else x - size[0]/2,
        #                 np.random.uniform(size[1]/2, y - size[1]/2),
        #                 np.random.uniform(size[2]/2, z - size[2]/2)
        #             ]
        #         cubes.append((center, size, 'thinB_wall'))
        # else:
        #     num_thinB_wall = 0
        num_thinB_wall = 0

        # # generate in-space cubes
        # if random.random() < PROB_THINB_SPACE:
        #     num_thinB_space = random.randint(self.max_num_space_thin_cubes//3, self.max_num_space_thin_cubes)
        #     for _ in range(num_thinB_space):
        #         align = np.random.choice(['x', 'y', 'z'])
        #         thin_cross_size = np.random.uniform(self.thin_structure_size_range[0], self.thin_structure_size_range[1], size=2)
        #         if align == 'x':
        #             size = [np.random.uniform(x/4, x/2), thin_cross_size[0], thin_cross_size[1]]
        #             center = [np.random.uniform(size[0]/2, x-size[0]/2), np.random.uniform(thin_cross_size[0]/2, y-thin_cross_size[0]/2), np.random.uniform(thin_cross_size[1]/2, z-thin_cross_size[1]/2)]
        #         elif align == 'y':
        #             size = [thin_cross_size[0], np.random.uniform(y/4, y/2), thin_cross_size[1]]
        #             center = [np.random.uniform(thin_cross_size[0]/2, x-thin_cross_size[0]/2), np.random.uniform(size[1]/2, y-size[1]/2), np.random.uniform(thin_cross_size[1]/2, z-thin_cross_size[1]/2)]
        #         elif align == 'z':
        #             size = [thin_cross_size[0], thin_cross_size[1], np.random.uniform(z/4, z/2)]
        #             center = [np.random.uniform(thin_cross_size[0]/2, x-thin_cross_size[0]/2), np.random.uniform(thin_cross_size[1]/2, y-thin_cross_size[1]/2), np.random.uniform(size[2]/2, z-size[2]/2)]
        #         cubes.append((center, size, 'thinB_space'))
        # else:
        #     num_thinB_space = 0
        num_thinB_space = 0

        self.num_thinB_wall = num_thinB_wall
        self.num_thinB_space = num_thinB_space

        tmp = cubes.copy()
        cubes = []
        for cube in tmp:
            position, scale, cube_type = cube
            new_position = (position[0] - x / 2, position[1] - y / 2, position[2] - z / 2)
            cubes.append((new_position, scale, cube_type))
        self.cubes_thin = cubes.copy()


    def _validate_scene_layout(self):
        scene_scale = np.max(np.array(self.scene_size))
        scene_bbox_min, scene_bbox_max = -0.5 * self.scene_size, 0.5 * self.scene_size
        threshold = scene_scale / (2.0 * SCENE_SCALE_MAX) * CAMERA_SAMPLE_DIST_THRE
        camera_types = ['center', 'boundary']
        valid_flags = [False] * len(camera_types)

        for i, camera_type in enumerate(camera_types):
            trial = 5000
            target_num = 100
            valid_camera_location = []

            for _ in range(trial):
                if camera_type == 'center':
                    camera_location = np.random.uniform(scene_bbox_min/4.0, scene_bbox_max/4.0)
                elif camera_type == 'boundary':
                    camera_location = sample_point_scene_boundary(scene_bbox_min, scene_bbox_max)
                else:
                    raise NotImplementedError
                
                if is_outside_cubes(camera_location, self.cubes + self.cubes_frame + self.cubes_thin, threshold):
                    valid_camera_location.append(camera_location)
                
                if len(valid_camera_location) == target_num:
                    valid_flags[i] = True
                    break
        
        self.scene_valid_flag = True if np.all(valid_flags) else False
        if not self.scene_valid_flag:
            print('The generated scene is invalid for camera pose sampling, doing re-generation...')
            self._reset_scene()


    def _generate_walls(self):
        print('Generating box')
        x, y, z = self.scene_size
        t = self.wall_thickness

        self.walls.append((Cube(x/2, y/2, t), 'floor', np.array([0,0,-z/2 + t]).reshape(1,3)))
        self.walls.append((Cube(x/2, y/2, t), 'roof', np.array([0,0,z/2 - t]).reshape(1,3)))
        self.walls.append((Cube(x/2, t, z/2), 'wall_0', np.array([0,-y/2 + t,0]).reshape(1,3)))
        self.walls.append((Cube(x/2, t, z/2), 'wall_1', np.array([0,y/2 - t,0]).reshape(1,3)))
        self.walls.append((Cube(t, y/2, z/2), 'wall_2', np.array([-x/2 + t,0,0]).reshape(1,3)))
        self.walls.append((Cube(t, y/2, z/2), 'wall_3', np.array([x/2 - t,0,0]).reshape(1,3)))
        self.num_walls = len(self.walls)


    def _generate_thin_structure(self):
        for cube_idx, cube in enumerate(self.cubes_thin):
            position, scale, type_cube = cube
            if random.random() < 0.5:
                geometry = Cube(scale[0]/2, scale[1]/2, scale[2]/2)
            else:
                geometry = Cylinder(scale[0]/2, scale[1]/2, scale[2]/2)
            self.thin_geoemtry.append((geometry, type_cube, np.array(position).reshape(1,3)))


    def _add_wall_to_multiShape(self, matIdx_start=0):
        tmp = self.walls.copy()     # [floor, roof, wall, wall, wall, wall]
        self.walls = MultiShape_hanwen(None, candShapes=None, smoothPossibility=None)
        for wall_idx, (wall, wall_type, wall_trans) in enumerate(tmp):
            if wall_type == 'floor':
                wall.genShape(matName="mat_shape%d"%(matIdx_start+0))
            elif wall_type == 'roof':
                wall.genShape(matName="mat_shape%d"%(matIdx_start+1))
            elif 'wall' in wall_type:
                wall.genShape(matName="mat_shape%d"%(matIdx_start+2))
            else:
                raise NotImplementedError
            wall.translate(wall_trans)
            self.walls.addShape(wall)
        self.add_multiShape_geometry(self.walls)

    
    def _add_thin_geometry_to_multiShape(self, matIdx_start=0):
        tmp = self.thin_geoemtry.copy()
        self.thin_geoemtry = MultiShape_hanwen(None, candShapes=None, smoothPossibility=None)
        for geometry_idx, (geom, geom_type, geom_trans) in enumerate(tmp):
            geom.genShape(matName="mat_shape%d"%(matIdx_start+geometry_idx))
            # if geom_type == 'thinB_wall':
            #     geom.genShape(matName="mat_shape%d"%(matIdx_start+0))
            # elif geom_type == 'thinB_space':
            #     geom.genShape(matName="mat_shape%d"%(matIdx_start+1))
            geom.translate(geom_trans)
            self.thin_geoemtry.addShape(geom)
        self.add_multiShape_geometry(self.thin_geoemtry)

    
    def _generate_objects(self, outFolder, uuid_str='', candShapes=[0,1,2], smooth_probability=1.0, no_hf=False, 
                          bPermuteMat=True, bScaleMesh=True, bMaxDimRange=[0.3, 0.5], repeat_times=1, texture_idx=0):
        """
        Generate Random Objects, workflow as:
            1. For each box, generate a random multiShape (geometry only) using ms.genShape
            2. Normalize the multiShape into the current box
            3. Update the properties of the current scene
            4. After all geometry generated, generate material and save the results
        """
        # initialize properties
        super(Scene, self).__init__()

        if not os.path.isdir(outFolder):
            os.makedirs(outFolder)

        # origin shape parameters
        all_shape_parameters = {'uuid_str': uuid_str}

        cur_matIdx_start = 0

        # multiShape generation for each cube
        for cube_idx, cube in enumerate(self.cubes + self.cubes_frame):
            position, scale, type_cube = cube
            if type_cube == 'fB':
                if random.uniform(0,1) < PROB_FRAME_ADD_GEOM:
                    scale = [it/random.uniform(1.5,2.5) for it in scale]
                    position = [it+random.uniform(-0.7, 0.7)*it2 for (it, it2) in zip(position, scale)]
                else:
                    continue

            # sub_obj_num = sample_sub_object_num(type_cube)
            sub_obj_num = 1
            print(f'cube idx {cube_idx} ({type_cube}) has {sub_obj_num} object premitives')
            
            shape_parameters = {
                'cube_center': position,
                'cube_scale': scale,
                'cube_type': type_cube,
            }
            shape_parameters['sub_obj_num'] = sub_obj_num
            shape_parameters['sub_objs'] = [{} for _ in range(sub_obj_num)]

            ms = MultiShape_hanwen(sub_obj_num, candShapes=candShapes, smoothPossibility=smooth_probability)
            sub_objs_vals = list(ms.genShape(no_hf=no_hf, matIdx_start=cur_matIdx_start))
            if bPermuteMat:
                ms.permuteMatIds()
            cur_matIdx_start += sub_obj_num

            ms.normalize_shape(cube_center=position, cube_scale=scale)
            self.add_multiShape_geometry(ms)
            print(f'cur scene has {len(self.points)} points, current multiShape has {(len(ms.points))} points')

            for i_key, key in enumerate(['primitive_id', 'axis_vals', 'translation', 'translation1', 'rotation', 'rotation1', 'height_fields']):
                for iS in range(sub_obj_num):
                    shape_parameters['sub_objs'][iS][key] = sub_objs_vals[i_key][iS].tolist() if isinstance(sub_objs_vals[i_key][iS], np.ndarray) else sub_objs_vals[i_key][iS]
            all_shape_parameters[f'cube_{cube_idx}'] = shape_parameters

        self._add_wall_to_multiShape(cur_matIdx_start)

        output_path_list = []
        all_shape_parameters_list = []
        return_uuid_list = []
        
        for idx in range(repeat_times):
            new_uuid = str(texture_idx).zfill(2) + '_' + str(idx).zfill(2)
            subFolder = Path(outFolder) / new_uuid
            subFolder.mkdir(parents=True, exist_ok=True)
            output_path = subFolder / 'object.obj'
            subFolder = str(subFolder.resolve())
            return_uuid = new_uuid

            # save .obj file and sample texture
            max_dim, material_ids = self.genObj(subFolder + "/object.obj", outFolder, bMat=True, bComputeNormal=True, bScaleMesh=bScaleMesh, bMaxDimRange=bMaxDimRange, texture_path=f"imgs/{str(texture_idx).zfill(2)}")
            all_shape_parameters['max_dim'] = max_dim
            all_shape_parameters['all_material'] = material_ids
            self.genMatList(subFolder + "/object.txt")
            self.genInfo(subFolder + "/object.info")

            output_path_list.append(output_path)
            all_shape_parameters_list.append(all_shape_parameters)
            return_uuid_list.append(return_uuid)

        return output_path_list, all_shape_parameters_list, return_uuid_list

    def add_multiShape_geometry(self, ms):
        curPN = len(self.points)
        curUN = len(self.uvs)
        curFN = len(self.faces)
        if curPN == 0:
            self.points = np.copy(ms.points)
            self.uvs = np.copy(ms.uvs)
            self.faces = np.copy(ms.faces)
            self.facesUV = np.copy(ms.facesUV)
        else:
            self.points = np.row_stack([self.points, ms.points])
            self.uvs = np.row_stack([self.uvs, ms.uvs])
            self.faces = np.row_stack([self.faces, ms.faces+curPN])
            self.facesUV = np.row_stack([self.facesUV, ms.facesUV+curUN])
        self.matNames += ms.matNames

        ms_matStartId = ms.matStartId
        ms_matStartId = np.array([it+curFN for it in ms_matStartId]).astype(int)
        self.matStartId = np.concatenate([self.matStartId, ms_matStartId], axis=0).astype(int)


    def generate_frame_cube(self, cube_type='fB'):
        """Generate a wireframe cube on the ground with more variation."""
        """ self.frameBox_size_range = [min_wd, max_wd, min_h, max_h], default=[3,6,3,6]"""
        min_wd, max_wd, min_h, max_h = self.frameBox_size_range
        
        scale = (random.uniform(min_wd, max_wd), random.uniform(min_wd, max_wd), random.uniform(min_h, max_h))
        position = (
            random.uniform(scale[0] / 2, self.scene_size[0] - scale[0] / 2),
            random.uniform(scale[1] / 2, self.scene_size[1] - scale[1] / 2),
            random.uniform(scale[2] / 2, self.scene_size[2] - scale[2] / 2),
        )
        return (position, scale, cube_type)

    
    def generate_aa_cube(self):
        """Generate a cube where the geometry is axis-aligned"""
        """ self.aaBox_size_range = [min_size, max_size, thin_rB_thickness, thick_rB_thickness], default=[1,3,1,2]"""
        min_size, max_size, thin_aaB_thickness, thick_aaB_thickness = self.aaBox_size_range
        if random.random() < 0.5:
            # thin aa Box
            type = 'aaB_thin'
            scale = (random.uniform(min_size, max_size), random.uniform(min_size, max_size), random.uniform(0.01, thin_aaB_thickness))
        else:
            # thick aa Box
            type = 'aaB_thick'
            scale = (random.uniform(min_size, max_size), random.uniform(min_size, max_size), random.uniform(0.5, thick_aaB_thickness))
        position = (
            random.uniform(scale[0] / 2, self.scene_size[0] - scale[0] / 2),
            random.uniform(scale[1] / 2, self.scene_size[1] - scale[1] / 2),
            random.uniform(scale[2] / 2, self.scene_size[2] - scale[2] / 2),
        )
        return (position, scale, type)
            
    
    def generate_large_cube(self, cube_type='lB'):
        """Generate a large cube on the ground with more variation."""
        """ self.largeBox_size_range = [min_wd, max_wd, min_h, max_h], default=[4,8,2,4]"""
        min_wd, max_wd, min_h, max_h = self.largeBox_size_range
        while True:
            scale = (random.uniform(min_wd, max_wd), random.uniform(min_wd, max_wd), random.uniform(min_h, max_h))
            position = (
                random.uniform(scale[0] / 2, self.scene_size[0] - scale[0] / 2),
                random.uniform(scale[1] / 2, self.scene_size[1] - scale[1] / 2),
                scale[2] / 2
            )
            if not any(intersects(position, scale, pos, scl) for pos, scl, _ in (self.cubes + self.cubes_frame + self.cubes_aa)):
                return (position, scale, cube_type)
            
    
    def generate_small_cube(self, cube_type='sB'):
        """Generate a small cube. Either on the ground or on top of large cubes"""
        """ self.smallBox_size_range = [min_size_2D, max_size_2D, on_ground_min_h, on_ground_max_h], default=[1,3,1,4]"""
        min_size_2D, max_size_2D, on_ground_min_h, on_ground_max_h = self.smallBox_size_range

        while True:
            if random.random() < 0.5:
                # on-ground small cube
                scale = (random.uniform(min_size_2D, max_size_2D), random.uniform(min_size_2D, max_size_2D), random.uniform(on_ground_min_h, on_ground_max_h))
                position = (
                    random.uniform(scale[0] / 2, self.scene_size[0] - scale[0] / 2),
                    random.uniform(scale[1] / 2, self.scene_size[1] - scale[1] / 2),
                    scale[2] / 2
                )
            else:
                # on-top-of-lB cube
                scale = (random.uniform(min_size_2D, max_size_2D), random.uniform(min_size_2D, max_size_2D), random.uniform(min_size_2D, max_size_2D))
                base_cube = random.choice(self.cubes)
                bx, by, bz = base_cube[0]
                bwidth, bdepth, bheight = base_cube[1]
                position = (
                    bx + random.uniform(-bwidth / 2 + scale[0] / 2, bwidth / 2 - scale[0] / 2),
                    by + random.uniform(-bdepth / 2 + scale[1] / 2, bdepth / 2 - scale[1] / 2),
                    bz + bheight / 2 + scale[2] / 2
                )
            if not any(intersects(position, scale, pos, scl) for pos, scl, _ in (self.cubes + self.cubes_frame)):
                return (position, scale, cube_type)


    def generate_roof_cube(self):
        """Generate a cube hanging from the roof."""
        """ self.roofBox_size_range = [min_size, max_size, thin_rB_thickness, thick_rB_thickness], default=[1,3,1,2]"""
        min_size, max_size, thin_rB_thickness, thick_rB_thickness = self.roofBox_size_range
        thin_rB_thickness = max(0.55, thin_rB_thickness)
        while True:
            if random.random() < 0.5:
                # thin roof Box
                scale = (random.uniform(min_size, max_size), random.uniform(min_size, max_size), random.uniform(0.5, thin_rB_thickness))
            else:
                # thick roof Box
                scale = (random.uniform(min_size, max_size), random.uniform(min_size, max_size), random.uniform(max(1.0, thick_rB_thickness-2), thick_rB_thickness))
            position = (
                random.uniform(scale[0] / 2, self.scene_size[0] - scale[0] / 2),
                random.uniform(scale[1] / 2, self.scene_size[1] - scale[1] / 2),
                self.scene_size[2] - scale[2] / 2
            )
            if not any(intersects(position, scale, pos, scl) for pos, scl, _ in (self.cubes + self.cubes_frame)):
                return (position, scale, 'rB')


    def generate_wall_cube(self):
        """Generate a cube on the wall."""
        """ self.wallBox_size_range = [min_size, max_size, thin_wB_thickness, thick_wB_thickness], default=[2,4,1,3]"""
        min_size, max_size, thin_wB_thickness, thick_wB_thickness = self.wallBox_size_range
        thick_wB_thickness = max(0.55, thin_wB_thickness)
        while True:
            if random.random() < 0.5:  # Wall along y axis
                if random.random() < 0.5:
                    # thin wall Box
                    scale = (random.uniform(min_size, max_size), random.uniform(0.5, thin_wB_thickness), random.uniform(min_size, max_size))
                else:
                    # thick wall Box
                    scale = (random.uniform(min_size, max_size), random.uniform(min(max(1.0, thick_wB_thickness-2), thick_wB_thickness), thick_wB_thickness), random.uniform(min_size, max_size))
                position = (
                    random.uniform(scale[0] / 2, self.scene_size[0] - scale[0] / 2),
                    scale[1] / 2 if random.random() < 0.5 else self.scene_size[1] - scale[1] / 2,
                    random.uniform(scale[2] / 2, self.scene_size[2] - scale[2] / 2)
                )
            else:  # Wall along x axis
                if random.random() < 0.5: # thin
                    scale = (random.uniform(0.5, thin_wB_thickness), random.uniform(min_size, max_size), random.uniform(min_size, max_size))
                else:   # thick
                    scale = (random.uniform(min(max(1.0, thick_wB_thickness-2), thick_wB_thickness), thick_wB_thickness), random.uniform(min_size, max_size), random.uniform(min_size, max_size))
                position = (
                    scale[0] / 2 if random.random() < 0.5 else self.scene_size[0] - scale[0] / 2,
                    random.uniform(scale[1] / 2, self.scene_size[1] - scale[1] / 2),
                    random.uniform(scale[2] / 2, self.scene_size[2] - scale[2] / 2)
                )
            if not any(intersects(position, scale, pos, scl) for pos, scl, _ in (self.cubes + self.cubes_frame)):
                return (position, scale, 'wB')


    def visualize_cubes(self, save_dir):
        views = {
            'default_view': {'elev': 30, 'azim': 45},  # Default 3D view
            'front_view': {'elev': 0, 'azim': 90},
            'side_view': {'elev': 0, 'azim': 0},
            'top_view': {'elev': 90, 'azim': 0}
        }
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
        all_cubes = self.cubes + self.cubes_thin + self.cubes_frame + self.cubes_window + self.cubes_aa
        num_cubes = len(all_cubes)
        colors = np.random.rand(num_cubes, 3)
        
        for ax, (view_name, view_params) in zip(axs, views.items()):
            for cube_idx, cube in enumerate(all_cubes):
                position, scale, cube_type = cube
                color = colors[cube_idx]
                draw_cube(ax, position, scale, color, cube_type)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-self.scene_size[0] / 2, self.scene_size[0] / 2)
            ax.set_ylim(-self.scene_size[1] / 2, self.scene_size[1] / 2)
            ax.set_zlim(-self.scene_size[2] / 2, self.scene_size[2] / 2)
            
            # Set the view
            ax.view_init(elev=view_params['elev'], azim=view_params['azim'])
            ax.set_title(view_name.replace('_', ' ').title())

        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'vis_cubes.png'))
        print('save fig to', os.path.join(save_dir, 'vis_cubes.png'))


    def save_info_layout(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, 'info_layout.json')
        save_info = {
            'scene_size': self.scene_size.tolist(),
            'cubes': self.cubes,
            'scene_has_sticks': len(self.cubes_thin) > 0,
            'cubes_thin': self.cubes_thin,
            'scene_has_frames': self.scene_has_frames,
            'cubes_frame': self.cubes_frame,
            'scene_has_window': self.scene_has_window,
            'cubes_window': self.cubes_window,
            'positions_light': self.positions_light,
            'window_additional_planer': self.additional_planer,
            'scene_has_aa': self.scene_has_aa,
            'cubes_aa': self.cubes_aa,
        }
        with open(file_name, 'w') as f:
            json.dump(save_info, f, indent=4)
        print('save layout to', os.path.join(save_dir, 'info_layout.json'))


class MultiShape_hanwen(Shape):
    """
    0: ellipsoid
    1: cube
    2: cylinder

    """
    def __init__(self,
                 numShape = 6, smoothPossibility = 0.1, axisRange = (0.25, 2.0), heightRangeRate = (0, 0.3),
                 translateRangeRate = (0, 0.5), rotateRange = (0, 180), candShapes=[0,1,2]):
        super(MultiShape_hanwen, self).__init__()
        self.numShape = numShape
        self.smoothPossibility = smoothPossibility
        self.axisRange = axisRange
        self.heightRangeRate = heightRangeRate
        self.translateRangeRate = translateRangeRate
        self.rotateRange = rotateRange
        self.candShapes = candShapes
        
    def genShape(self, no_hf=False, matIdx_start=0):
        """ For each shape, randomly sample parameters (axis, height field, rotation, translation) and create the shape. """
        super(MultiShape_hanwen, self).__init__()
        primitive_ids = []
        axis_vals_s = []
        translations = []
        translation1s = []
        rotations = []
        rotation1s = []
        height_fields_s = []
        subShapes = []
        for iS in range(self.numShape):
            rp = [1]
            # axisVals = np.random.uniform(self.axisRange[0], self.axisRange[1], 3)
            axisVals = np.array([1.0, 1.0, 1.0])
            # hfs = []
            # minA = axisVals.min()*2.0
            # maxA = axisVals.max()*2.0
            # maxH = np.random.uniform(self.heightRangeRate[0]*minA, self.heightRangeRate[1]*minA*2, 6)   # 2x HF augmentation
            # translation = np.random.uniform(self.translateRangeRate[0]*maxA, self.translateRangeRate[1]*maxA, 3)
            translation = np.array([0.0, 0.0, 0.0])
            # translation1 = np.random.uniform(self.translateRangeRate[0] * maxA, self.translateRangeRate[1] * maxA, 3)
            translation1 = np.array([0.0, 0.0, 0.0])
            # rotation = np.random.uniform(self.rotateRange[0], self.rotateRange[1], 3)
            rotation = np.array([-90.0, 0.0, 0.0])
            # rotation1 = np.random.uniform(self.rotateRange[0], self.rotateRange[1], 3)
            rotation1 = np.array([0.0, 0.0, 0.0])
            # for ih in range(6):
            #     smoothR = np.random.uniform(0,1,1)[0]
            #     if smoothR <= self.smoothPossibility or maxH[ih] == 0:
            #         hf = np.zeros((36,36))
            #     else:
            #         hfg = HeightFieldCreator(maxHeight=(-maxH[ih], maxH[ih]))
            #         hf = hfg.genHeightField()
            #     hfs.append(hf)
            # hfs = np.reshape(hfs, (6,) + hf.shape)
            #print(hfs)
            # if no_hf:
            hfs = np.zeros((6, 36, 36))

            # if rp[0] == 0:
                # subShape = Ellipsoid(axisVals[0], axisVals[1], axisVals[2])
            # elif rp[0] == 1:
            subShape = Cube(axisVals[0], axisVals[1], axisVals[2])
            # elif rp[0] == 2:
            #     subShape = Cylinder(axisVals[0], axisVals[1], axisVals[2])

            subShape.genShape(matName="mat_shape%d"%(iS+matIdx_start))
            # subShape.applyHeightField(hfs)


            subShape.rotate((1, 0, 0), rotation[0])
            subShape.rotate((0, 1, 0), rotation[1])
            subShape.rotate((0, 0, 1), rotation[2])
            subShape.translate(translation)

            if iS != 0:
                self.rotate((1, 0, 0), rotation1[0])
                self.rotate((0, 1, 0), rotation1[1])
                self.rotate((0, 0, 1), rotation1[2])
                self.translate(translation1)

            self.addShape(subShape)
            primitive_ids.append(rp[0])
            axis_vals_s.append(axisVals)
            translations.append(translation)
            translation1s.append(translation1)
            rotations.append(rotation)
            rotation1s.append(rotation1)
            height_fields_s.append(hfs)

        self.reCenter()
        return primitive_ids, axis_vals_s, translations, translation1s, rotations, rotation1s, height_fields_s


    def normalize_shape(self, cube_center, cube_scale):
        self.reCenter()
        cur_scale = self.points.max(axis=0) - self.points.min(axis=0)   # [3,]
        self.points = self.points / cur_scale.reshape((1,3)) * np.array(cube_scale).reshape((1,3))
        self.points += np.array(cube_center).reshape((1,3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create scene shapes")
    parser.add_argument('--project_dir', default='./generated_scenes', help='project directory, where output train_shapes/ and brdf/ are located')
    parser.add_argument('--num_scenes', default=1, type=int, help='number of shapes to create')
    parser.add_argument('--dont_convert_to_glb', default=False, action='store_true', help='converts the generated objs to glbs')
    parser.add_argument('--uuid_str', default='', type=str, help='uuid to use for the shape (only used if single_shape is True)')
    parser.add_argument('--seed', default=1, type=int, help='seed for random number generation')
    parser.add_argument('--sub_obj_num_poss', type=str, default='5,5,5,4,4,3,2,1,1', help='comma separated list of possibilities for number of sub objects')
    parser.add_argument('--no_hf', default=False, action='store_true', help='do not use height field')
    parser.add_argument('--smooth_probability', default=0.1, type=float, help='possibility of smoothing the height field')
    parser.add_argument('--repeat', default=1, type=int, help='number of times to repeat the generation of the same scene')
    args = parser.parse_args()
    seed_everything(args.seed)

    project_dir = args.project_dir
    out_dir = f'{project_dir}/scenes'
    num_scenes = args.num_scenes
    dataset = 'rgb2x'
    uuid_str = ''
    dont_convert_to_glb = False
    no_hf = args.no_hf
    bScaleMesh = True
    bPermuteMat = False
    smooth_probability = args.smooth_probability

    # mat_path = get_matsynth_material(out_dir)
    for i in range(num_scenes):
        scene = Scene()
        new_uuid = i

        output_path_list, all_shape_parameters_list, new_uuid_list = scene._generate_objects(outFolder=out_dir,
                                                                            #   mat_path=mat_path,
                                                                            uuid_str=uuid_str, 
                                                                            candShapes=[0,1,2], 
                                                                            smooth_probability=smooth_probability, 
                                                                            no_hf=no_hf, 
                                                                            bPermuteMat=bPermuteMat, 
                                                                            bScaleMesh=bScaleMesh, 
                                                                            bMaxDimRange=[0.3, 0.45],
                                                                            repeat_times=args.repeat,
                                                                            texture_idx=i
                                                                            )
        print(output_path_list)
        for output_path, all_shape_parameters, uuid_str in zip(output_path_list, all_shape_parameters_list, new_uuid_list):
            scene.visualize_cubes(str(output_path).replace('object.obj', ''))
            scene.save_info_layout(str(output_path).replace('object.obj', ''))

            if dont_convert_to_glb:
                pass
            else:
                json_output_fn = str(output_path).replace('object.obj', f'{uuid_str.split("/")[-1]}_original_parameters.json')
                with open(json_output_fn, 'w') as f:
                    json.dump(all_shape_parameters, f, indent=4, cls=NpEncoder)
                print(f'Saved {json_output_fn}')
                convert_file(output_path)