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
from scipy.io import loadmat

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()

def load_16big_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

def read_depth(path: str, scale_adjustment=1.0) -> np.ndarray:
    if path.lower().endswith(".exr"):
        # Ensure OPENCV_IO_ENABLE_OPENEXR is set to "1"
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = load_16big_png_depth(path)
    else:
        raise ValueError(f'unsupported depth file name "{path}"')

    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0

    return d

class ClearPose(Dataset):
    def __init__(
        self,
        ClearPose_DIR: str,
        split: str = "test",
        load_img_size: int = 518,
        cache_file: str = "data/dataset_cache/clearpose_mv_recon_cache.npy",
    ):

        self.ClearPose_DIR = ClearPose_DIR
        if ClearPose_DIR == None:
            raise NotImplementedError
        print(f"ClearPose_DIR is {ClearPose_DIR}")
        category = ["set1", "set2", "set3", "set4", "set5", "set6", "set7", "set8", "set9"]
        self.split = split
        assert split == 'test', "Only test set preprocessed."
        if self.split == 'train':
            seq_numbers = category[:7]
        elif self.split == 'test':
            seq_numbers = category
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'test'.")

        if osp.exists(cache_file):
            print(f"[ClearPose] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(list(self.metadata.keys()))
        else:
            print(f"[ClearPose] Cache file not found, loading from {ClearPose_DIR}")
            sequence_lists = {seq : sorted(os.listdir(osp.join(ClearPose_DIR, seq))) for seq in seq_numbers}
            sequence_lists = [seq+'|'+seq_list for seq in seq_numbers for seq_list in sequence_lists[seq]]
            self.sequence_list = sequence_lists

            self.metadata = {}
            for seq in tqdm(self.sequence_list):
                cat, seq_name = seq.split('|')
                rgb_root = osp.join(ClearPose_DIR, cat, seq_name)
                all_imgs = sorted([d for d in os.listdir(rgb_root) if d.endswith('-color.png')])
                all_img_numbers = [int(imgname.split('-')[0]) for imgname in all_imgs]
                self.metadata[seq] = len(all_imgs)

            np.save(cache_file, self.metadata)

        self.load_img_size = load_img_size
        print(f"[ClearPose] Data size: {len(self)}")

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
        image_path = osp.join(self.ClearPose_DIR, cat, seq_name)
        depth_path = osp.join(self.ClearPose_DIR, cat, seq_name)
        meta_path = osp.join(self.ClearPose_DIR, cat, seq_name, "metadata.mat")
        meta_data = loadmat(meta_path)
        image_paths: list      = [""] * len(ids)
        images: list           = [0]  * len(ids)
        depths: list           = [0]  * len(ids)
        extrinsics: np.ndarray = np.zeros((len(ids), 3, 4))
        intrinsics: np.ndarray = np.zeros((len(ids), 3, 3))

        for id_index, id in enumerate(ids):
            impath = osp.join(image_path, f"{id:04d}00-color.png")
            depthpath = osp.join(depth_path, f"{id:04d}00-depth_true.png")
            mask_path = osp.join(image_path, f"{id:04d}00-label.png")
            rgb_image: Image.Image = Image.open(impath).convert('RGB')
            mask = np.array(Image.open(mask_path))
            mask = mask != 0
            depthmap: np.ndarray   = np.array(Image.open(depthpath))
            meta = meta_data[f"{id:04d}00"]
            depthmap = depthmap / meta['factor_depth'].item()
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)
            # depthmap = depthmap * mask

            intrinsic = meta['intrinsic_matrix'][0,0].astype(np.float32)
            extrinsic = meta['rotation_translation_matrix'][0,0].astype(np.float32)
            extrinsic = np.concatenate([extrinsic, np.array([[0,0,0,1]], dtype=np.float32)], axis=0)
            extrinsic = np.linalg.inv(extrinsic)[:3,:]
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
            extrinsics[id_index]  = extrinsic

        depths = np.array(depths)  # (S, H, W)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics
        )

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
        #     pointclouds[batch['valid_mask']],
        #     torch.stack(images, dim=0).permute(0,2,3,1)[batch['valid_mask']],
        #     "debug.ply"
        # )
        # exit(0)
        
        return batch

