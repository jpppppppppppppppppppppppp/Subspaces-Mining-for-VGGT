import os.path as osp
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import torch
import cv2
import torchvision.transforms as tvf

from typing import Optional, Union, List
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.vggt.utils.geometry import unproject_depth_map_to_point_map
from datasets.utils.cropping import resize_image, resize_image_depth_and_intrinsic
import json

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()

def read_depth(path: str, scale_adjustment=1.0) -> np.ndarray:
    """
    Reads a depth map from disk in either .exr or .png format. The .exr is loaded using OpenCV
    with the environment variable OPENCV_IO_ENABLE_OPENEXR=1. The .png is assumed to be a 16-bit
    PNG (converted from half float).

    Args:
        path (str):
            File path to the depth image. Must end with .exr or .png.
        scale_adjustment (float):
            A multiplier for adjusting the loaded depth values (default=1.0).

    Returns:
        np.ndarray:
            A float32 array (H, W) containing the loaded depth. Zeros or non-finite values
            may indicate invalid regions.

    Raises:
        ValueError:
            If the file extension is not supported.
    """
    if path.lower().endswith(".exr"):
        # Ensure OPENCV_IO_ENABLE_OPENEXR is set to "1"
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    else:
        raise ValueError(f'unsupported depth file name "{path}"')

    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0

    return d

class Mega(Dataset):
    def __init__(
        self,
        MEGA_DIR: str,
        split: str = "test",
        load_img_size: int = 518,
        cache_file: str = "data/dataset_cache/mega_mv_recon_cache.npy",
    ):

        self.MEGA_DIR = MEGA_DIR
        if MEGA_DIR == None:
            raise NotImplementedError
        print(f"MEGA_DIR is {MEGA_DIR}")
        train_split = 9 / 10
        category = sorted(os.listdir(MEGA_DIR))
        self.split = split
        assert split == 'test', "Only test set preprocessed."
        if self.split == 'train':
            seq_numbers = category[:int(len(category) * train_split)]
        elif self.split == 'valid':
            raise NotImplementedError
        elif self.split == 'test':
            seq_numbers = category[:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'valid' or 'test'.")

        if osp.exists(cache_file):
            print(f"[MEGA] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(list(self.metadata.keys()))
        else:
            print(f"[MEGA] Cache file not found, loading from {MEGA_DIR}")
            sequence_lists = {seq : os.listdir(osp.join(MEGA_DIR, seq)) for seq in seq_numbers}
            sequence_lists = [seq+'|'+seq_list for seq in seq_numbers for seq_list in sequence_lists[seq]]
            self.sequence_list = sequence_lists

            self.metadata = {}
            for seq in tqdm(self.sequence_list):
                cat, seq_name = seq.split('|')
                rgb_root = osp.join(MEGA_DIR, cat, seq_name, 'renderings')
                all_imgs = sorted([d for d in os.listdir(rgb_root) if d.endswith('.png')])
                all_img_numbers = [int(imgname.split('_')[0]) for imgname in all_imgs]
                if all_img_numbers[0] != 0 or all_img_numbers[-1] + 1 != len(all_img_numbers):
                    raise ValueError(f"Image number not regular, with first image {all_imgs[0]} and last image {all_imgs[-1]} but number of images {len(all_imgs)}")

                self.metadata[seq] = len(all_imgs)

            np.save(cache_file, self.metadata)

        self.load_img_size = load_img_size
        print(f"[MEGA] Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return self.metadata[sequence_name]

    def __getitem__(self, idx_N):
        """Fetch item by index and a dynamic variable n_per_seq."""

        # Different from most pytorch datasets,
        # here we not only get index, but also a dynamic variable n_per_seq
        # supported by DynamicBatchSampler

        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(
            self,
            index: Optional[int] = None,
            sequence_name: Optional[str] = None,
            ids: Union[List[int], np.ndarray, None] = None,
        ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name: str = self.sequence_list[index]
        seq_len: int = self.metadata[sequence_name]

        if ids is None:
            ids = np.arange(seq_len).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()

        cat, seq_name = sequence_name.split('|')

        image_path = osp.join(self.MEGA_DIR, cat, seq_name, "renderings")
        depth_path = osp.join(self.MEGA_DIR, cat, seq_name, "renderings")
        cam_path = osp.join(self.MEGA_DIR, cat, seq_name, "opencv_cameras.json")
        cam_meta = json.load(open(cam_path, 'r'))
        image_paths: list      = [""] * len(ids)
        images: list           = [0]  * len(ids)
        depths: list           = [0]  * len(ids)
        extrinsics: np.ndarray = np.zeros((len(ids), 3, 4))
        intrinsics: np.ndarray = np.zeros((len(ids), 3, 3))

        for id_index, id in enumerate(ids):
            impath = osp.join(image_path, f"{id:08d}_rgba.png")
            depthpath = osp.join(depth_path, f"{id:08d}_depth.exr")
            maskpath = osp.join(depth_path, f"{id:08d}_mask.npy")

            rgb_image: Image.Image = Image.open(impath).convert('RGB')
            depthmap: np.ndarray   = read_depth(depthpath)

            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)
            if os.path.exists(maskpath):
                mask: np.ndarray       = np.load(maskpath).astype(bool)
                depthmap = depthmap * mask

            cam_cur = cam_meta["frames"][id]
            fx = cam_cur['fx']
            fy = cam_cur['fy']
            cx = cam_cur['cx']
            cy = cam_cur['cy']
            intrinsic = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0, 0, 1]], dtype=np.float32,)
            extrinsic = np.array(cam_cur["w2c"])[:4, :4]

            rgb_image, depthmap, intrinsic = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsic,
                output_width=self.load_img_size, # finally width = 518, height = 388
            )

            image_paths[id_index] = impath
            images[id_index]      = to_tensor(rgb_image)
            depths[id_index]      = depthmap
            intrinsics[id_index]  = intrinsic
            extrinsics[id_index]  = extrinsic[:3, :]

        depths = np.array(depths)  # (S, H, W)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics
        )

        # def save_ply(points, colors, filename):
        #     import open3d as o3d
        #     if torch.is_tensor(points):
        #         points_visual = points.reshape(-1, 3).cpu().numpy()
        #     else:
        #         points_visual = points.reshape(-1, 3)
        #     if torch.is_tensor(colors):
        #         points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
        #     else:
        #         points_visual_rgb = colors.reshape(-1, 3)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
        #     pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
        #     o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
        # save_ply(
        #     pointclouds,
        #     torch.stack(images, dim=0).permute(0,2,3,1),
        #     "debug.ply"
        # )
        # exit(0)

        batch = {"seq_id": sequence_name, "seq_len": seq_len, "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths  # list of str
        batch['images']      = torch.stack(images, dim=0)
        batch['pointclouds'] = pointclouds  # in numpy
        batch['valid_mask']  = depths > 1e-4
        batch['depths']     = torch.from_numpy(depths).float()
        # batch["extrs"] = extrinsics
        # batch["intrs"] = intrinsics
        # batch["w"] = metadata["w"]
        # batch["h"] = metadata["h"]

        return batch

