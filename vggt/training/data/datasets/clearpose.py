# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
from scipy.io import loadmat

from data.dataset_util import *
from data.base_dataset import BaseDataset

class ClearPose(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CLEARPOSE_DIR: str = None,
        min_num_images: int = 10,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the MegaDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            MEGA_DIR (str): Directory path to MEGA data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If MEGA_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if CLEARPOSE_DIR is None:
            raise ValueError("CLEARPOSE_DIR must be specified.")

        category = ["set1", "set2", "set3", "set4", "set5", "set6", "set7", "set8", "set9"]

        if self.debug:
            category = category[:1]
        if split == "train":
            category = category[:7]
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = []

        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"CLEARPOSE_DIR is {CLEARPOSE_DIR}")

        self.CLEARPOSE_DIR = CLEARPOSE_DIR

        total_frame_num = 0

        for c in category:
            seq_names = sorted(os.listdir(osp.join(CLEARPOSE_DIR, c)))
            if split == "train":
                seq_names = seq_names[:-1]
            elif split == "test":
                if int(c[-1]) < 8:
                    seq_names = seq_names[-1:]
            for seq_name in seq_names:
                anno_file_path = osp.join(CLEARPOSE_DIR, c, seq_name, "metadata.mat")
                if not osp.exists(anno_file_path):
                    print("no anno file", c, seq_name)
                    continue
                seq_data = loadmat(anno_file_path)
                if len(seq_data) < min_num_images:
                    print("less", c, seq_name)
                    continue
                image_list = sorted([int(f.split("-color")[0]) for f in os.listdir(osp.join(CLEARPOSE_DIR, c, seq_name)) if f.endswith("-color.png")])
                for idx in image_list:
                    assert f"{idx:06d}" in seq_data, f"{c}:{seq_name} {idx:06d} not in {anno_file_path}"
                # if seq_name in self.invalid_sequence:
                #     continue
                total_frame_num += len(seq_data)
                self.data_store[c+":"+seq_name] = (image_list, seq_data)

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ClearPose Data size: {self.sequence_list_len}")
        logging.info(f"{status}: ClearPose Data total frames size: {self.total_frame_num}")
        logging.info(f"{status}: ClearPose Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        idx_list, metadata = self.data_store[seq_name]
        c, seq_name = seq_name.split(":")

        if ids is None:
            ids = np.random.choice(
                len(idx_list), img_per_seq, replace=self.allow_duplicate_img
            )

        annos = [metadata[f"{idx_list[i]:06d}"] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for anno, i in zip(annos, ids):
            i = idx_list[i]
            image_path = osp.join(self.CLEARPOSE_DIR, c, seq_name, f"{i:06d}-color.png")
            image = read_image_cv2(image_path)

            if self.load_depth:
                depth_path = image_path.replace("color", "depth_true")
                depth_map: np.ndarray   = np.array(Image.open(depth_path))
                depth_map = depth_map / anno['factor_depth'].item()
                # mvs_mask_path = image_path.replace(
                #     "/images", "/depth_masks"
                # ).replace(".jpg", ".png")
                # mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                # depth_map[~mvs_mask] = 0

                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            extri_opencv = anno['rotation_translation_matrix'][0,0].astype(np.float32)
            extri_opencv = np.concatenate([extri_opencv, np.array([[0,0,0,1]], dtype=np.float32)], axis=0)
            extri_opencv = np.linalg.inv(extri_opencv)[:3,:]
            intri_opencv = anno['intrinsic_matrix'][0,0].astype(np.float32)

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "clearpose"

        batch = {
            "seq_name": set_name + "_" + c + ":" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch
