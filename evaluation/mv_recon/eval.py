import os
import json
import torch
import numpy as np
import open3d as o3d
import os.path as osp
import hydra
import logging

from omegaconf import DictConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from mv_recon.utils import umeyama, accuracy, completion
# from utils.debug import setup_debug
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto
from utils.depth import depth_evaluation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)

    all_eval_models: DictConfig   = hydra_cfg.eval_models    # see configs/evaluation/mv_recon.yaml
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    logger = logging.getLogger("mv_recon-eval")

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0.1 look up model config from configs/model, decide the model name (to save)
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]

        # 0.2 load the model
        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        logger.info(f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from {model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}")
        
        # 0.3 route the correct infer function for the model
        infer_func_cfg = model_info.get(
            "infer_mv_pointclouds",
            DictConfig({
                '_target_': f'interfaces.{"pi3" if "pi3" in model_keyname else "vggt"}.infer_mv_pointclouds',
                '_partial_': True,
            })
        )
        infer_mv_pointclouds = hydra.utils.instantiate(infer_func_cfg)

        model_logger = logging.getLogger(f"mv_recon-eval-{model_keyname}")
        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
            dataset_info = all_data_info[dataset_name]
            dataset = hydra.utils.instantiate(dataset_info.cfg)

            # 1.2 ready for output directory & metrics
            output_root = osp.join(hydra_cfg.output_dir, model_keyname, dataset_name)
            os.makedirs(output_root, exist_ok=True)
            all_data_dict = {
                "model": model_keyname,
                "Acc-mean":  0.0,  "Acc-med":  0.0,
                "Comp-mean": 0.0,  "Comp-med": 0.0,
                "NC-mean":   0.0,  "NC-med":   0.0,
                "NC1-mean":  0.0,  "NC1-med":  0.0,
                "NC2-mean":  0.0,  "NC2-med":  0.0,
            }

            # 1.3 load pre-sampled seq-id-map
            model_logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating Multi-View Pointcloud Reconstruction on dataset {dataset_name}...")
            sample_config: DictConfig = dataset_info.sampling
            model_logger.info(f"Sampling strategy: {sample_config.strategy}")
            with open(dataset_info.seq_id_map, "r") as f:
                seq_id_map: dict = json.load(f)

            model_logger.info(f"Evaluating {dataset_name} with {model_keyname}...")
            if osp.exists(osp.join(output_root, "_all_samples.csv")):
                os.remove(osp.join(output_root, "_all_samples.csv"))  # remove old csv file
            for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
                # 2. load data, choose specific ids of a sequence
                data = dataset.get_data(sequence_name=seq_name, ids=ids)
                filelist: list         = data['image_paths']  # [str] * N
                images: torch.Tensor   = data['images']       # (N, 3, H, W)
                gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
                valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

                # 3. real inference, predicted pointcloud aligned to ground truth (data_h, data_w)
                data_h, data_w         = images.shape[-2:]
                pred_pts, pred_depth   = infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
                assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

                # 4. save input images
                seq_name = seq_name.replace("/", "-")
                # save_image_grid_auto(images, osp.join(output_root, f"{seq_name}.png"))
                colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

                # 5. coarse align
                c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
                pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

                # 6. filter invalid points
                pred_pts = pred_pts[valid_mask].reshape(-1, 3)
                gt_pts = gt_pts[valid_mask].reshape(-1, 3)

                # 7. save predicted & ground truth point clouds
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pred_pts)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                # o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-pred.ply"), pcd)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
                pcd_gt.colors = o3d.utility.Vector3dVector(colors)
                # o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt.ply"), pcd_gt)

                # 8. ICP align refinement
                if "DTU" in dataset_name:
                    threshold = 100
                else:
                    threshold = 0.1

                trans_init = np.eye(4)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                transformation = reg_p2p.transformation
                pcd = pcd.transform(transformation)
                
                # 9. estimate normals
                pcd.estimate_normals()
                pcd_gt.estimate_normals()
                pred_normal = np.asarray(pcd.normals)
                gt_normal = np.asarray(pcd_gt.normals)

                # o3d.io.write_point_cloud(
                #     os.path.join(
                #         output_root, f"{seq_name.replace('|', '_')}-mask-icp.ply"
                #     ),
                #     pcd,
                # )

                # 10. compute metrics
                acc, acc_med, nc1, nc1_med, acc_distance = accuracy(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                comp, comp_med, nc2, nc2_med, comp_distance = completion(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                model_logger.info(
                    f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                )

                

                # save_images = images.permute(0, 2, 3, 1).cpu().numpy()
                # save_dir = os.path.join("save_video", dataset_name, model_keyname, seq_name.replace("|", "_"))
                # cur_valid_point_num = 0 
                # for i in range(len(save_images)):
                #     save_np = np.zeros((data_h, 2 * data_w, 3), dtype=np.uint8)
                #     save_image = (save_images[i] * 255.0).astype(np.uint8)
                #     cur_distance = np.zeros_like(save_image[..., 0], dtype=np.float32)
                #     cur_distance[valid_mask[i]] = acc_distance[cur_valid_point_num:cur_valid_point_num+valid_mask[i].sum()]
                #     cur_distance[cur_distance == 0] = np.min(cur_distance[cur_distance>0])
                #     cur_valid_point_num += valid_mask[i].sum()
                #     norm = Normalize(vmin=np.min(acc_distance), vmax=np.max(acc_distance))
                #     acc_image = (cm.get_cmap("coolwarm")(norm(cur_distance))[:, :, :3] * 255.0).astype(np.uint8)
                #     acc_image[~valid_mask[i]] = 0.
                #     save_np[:, :data_w, :] = save_image
                #     save_np[:, data_w:, :] = acc_image
                #     os.makedirs(save_dir, exist_ok=True)
                #     Image.fromarray(save_np).save(os.path.join(save_dir, f"{i:03d}.png"))
                # images_2_video = "ffmpeg -framerate 2 -i {input_path}/%3d.png -c:v libx264 -pix_fmt yuv420p {save_path}/{file_name}.mp4"
                # os.system(images_2_video.format(input_path=save_dir, save_path=save_dir, file_name="acc"))
                # rm_cmd = f"rm -rf {save_dir}/*.png"
                # os.system(rm_cmd)

                # 11. save metrics to csv
                write_csv(osp.join(output_root, f"_all_samples.csv"), {
                    "seq":       seq_name,
                    "Acc-mean":  acc,
                    "Acc-med":   acc_med,
                    "Comp-mean": comp,
                    "Comp-med":  comp_med,
                    "NC1-mean":  nc1,
                    "NC1-med":   nc1_med,
                    "NC2-mean":  nc2,
                    "NC2-med":   nc2_med,
                })
                all_data_dict["Acc-mean"]  += acc
                all_data_dict["Acc-med"]   += acc_med
                all_data_dict["Comp-mean"] += comp
                all_data_dict["Comp-med"]  += comp_med
                all_data_dict["NC-mean"]   += (nc1 + nc2) / 2
                all_data_dict["NC-med"]    += (nc1_med + nc2_med) / 2
                all_data_dict["NC1-mean"]  += nc1
                all_data_dict["NC1-med"]   += nc1_med
                all_data_dict["NC2-mean"]  += nc2
                all_data_dict["NC2-med"]   += nc2_med

                # release cuda memory
                torch.cuda.empty_cache()

            num_samples = len(dataset)
            metric_dict = {
                metric: value / num_samples
                for metric, value in all_data_dict.items()
                if metric != "model"
            }

            statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
            if getattr(hydra_cfg, "save_suffix", None) is not None:
                statistics_file += f"-{hydra_cfg.save_suffix}"
            statistics_file += ".csv"
            write_csv(statistics_file, {"model": model_keyname, **metric_dict})
        
        del model
        torch.cuda.empty_cache()
        model_logger.info(f"Finished evaluating {model_keyname} on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()