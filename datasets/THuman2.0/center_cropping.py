import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from PIL import Image
import cv2
import json

def read_rgba(file_path):
    return np.asarray(Image.open(file_path))

def rgba2amask(rgba_array, threshold=192):
    return rgba_array[:,:,3] > threshold

def read_depth(path: str, scale_adjustment=1.0) -> np.ndarray:
    if path.lower().endswith(".exr"):
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    else:
        raise ValueError(f'unsupported depth file name "{path}"')
    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d

def recenter(rgba_img, depth_img, mask, camera_param, output_size=518, border_ratio=0.2):
    return_int = False
    if rgba_img.dtype == np.uint8:
        rgba_img = rgba_img.astype(np.float32) / 255
        return_int = True
    
    H, W, C = rgba_img.shape
    size = max(H, W)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result_rgb = np.ones((output_size, output_size, C), dtype=np.float32)
    else:
        result_rgb = np.ones((output_size, output_size, C), dtype=np.float32)
        result_rgb[..., 3] = 0.0
    result_depth = np.zeros((output_size, output_size, 1), dtype=np.float32)

    fx, fy, cx, cy = (camera_param["fx"],
                      camera_param["fy"],
                      camera_param["cx"],
                      camera_param["cy"])

    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    x_cx = int(max(cx-x_min, x_max-cx))
    x_min, x_max = int(cx-x_cx), int(cx+x_cx)
    y_min, y_max = coords[1].min(), coords[1].max()
    y_cy = int(max(cy-y_min, y_max-cy))
    y_min, y_max = int(cy-y_cy), int(cy+y_cy)

    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(output_size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (output_size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (output_size - w2) // 2
    y2_max = y2_min + w2
    result_rgb[x2_min:x2_max, y2_min:y2_max] = cv2.resize(rgba_img[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_NEAREST)
    result_depth[x2_min:x2_max, y2_min:y2_max, 0] = cv2.resize(depth_img[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_NEAREST)

    if return_int:
        result_rgb = (result_rgb * 255).astype(np.uint8)

    cx = output_size // 2
    cy = output_size // 2
    fx = int(fx * scale)
    fy = int(fy * scale)

    return result_rgb, result_depth, {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": output_size, "h": output_size}

def process_rank(rank, size, folders, chunk_size):
    folders_chunk = folders[rank * chunk_size : (rank + 1) * chunk_size]
    if rank == size - 1:
        folders_chunk += folders[size * chunk_size:]
    for folder in folders_chunk:
        folder_path = os.path.join(base_dir, folder, "renderings")
        rgba_imgs = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
        depth_imgs = sorted([f for f in os.listdir(folder_path) if f.endswith(".exr")])
        camera_meta_file = os.path.join(base_dir, folder, "opencv_cameras.json")
        camera_meta = json.load(open(camera_meta_file, 'r'))["frames"]
        new_meta = {"frames": []}
        for i in range(len(rgba_imgs)):
            rgba_image = read_rgba(os.path.join(folder_path, rgba_imgs[i]))
            depth_image = read_depth(os.path.join(folder_path, depth_imgs[i]))
            mask = rgba2amask(rgba_image)
            rgba_image, depth_image, camera_meta_new = recenter(rgba_image, depth_image, mask, camera_meta[i], output_size=518, border_ratio=0.1)

            os.makedirs(os.path.join(save_dir, folder, "renderings"), exist_ok=True)
            image = Image.fromarray(rgba_image, 'RGBA')
            image.save(os.path.join(save_dir, folder, "renderings", rgba_imgs[i]))
            cv2.imwrite(os.path.join(save_dir, folder, "renderings", depth_imgs[i]), depth_image.astype(np.float32))
            camera_meta_new["w2c"] = camera_meta[i]["w2c"]
            new_meta["frames"].append(camera_meta_new)
        json.dump(new_meta, open(os.path.join(save_dir, folder, "opencv_cameras.json"), 'w'), indent=4)

    
if __name__ == "__main__":
    from multiprocessing import Pool
    size = 8
    base_dir = "/datasets/THUman/test"
    save_dir = "/datasets/THUman/save"
    folders = sorted(os.listdir(base_dir))
    chunk_size = len(folders) // size
    with Pool(processes=size) as pool:
        pool.starmap(process_rank, [(rank, size, folders, chunk_size) for rank in range(size)])
