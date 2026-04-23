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
from mesh_recon.utils import umeyama, accuracy, completion
# from utils.debug import setup_debug
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto
from utils.depth import depth_evaluation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import pytorch3d
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
import open3d.cuda as o3c
from evaluate_3d_reconstruction import run_evaluation
import shutil

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)
    all_eval_models: DictConfig   = hydra_cfg.eval_models    # see configs/evaluation/mv_recon.yaml
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    logger = logging.getLogger("mesh_recon-eval")

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
            "infer_mesh_pointclouds",
            DictConfig({
                '_target_': f'interfaces.vggt.infer_mesh_pointclouds',
                '_partial_': True,
            })
        )
        infer_mesh_pointclouds = hydra.utils.instantiate(infer_func_cfg)

        model_logger = logging.getLogger(f"mesh_recon-eval-{model_keyname}")
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
                "precision":  0.0,
                "recall": 0.0,
                "f-score": 0.0,
                "precision-mean": 0.0,
                "recall-mean": 0.0,
            }

            # 1.3 load pre-sampled seq-id-map
            model_logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating Mesh Reconstruction on dataset {dataset_name}...")
            sample_config: DictConfig = dataset_info.sampling
            model_logger.info(f"Sampling strategy: {sample_config.strategy}")
            with open(dataset_info.seq_id_map, "r") as f:
                seq_id_map: dict = json.load(f)

            model_logger.info(f"Evaluating {dataset_name} with {model_keyname}...")
            if osp.exists(osp.join(output_root, "_all_samples.csv")):
                os.remove(osp.join(output_root, "_all_samples.csv"))  # remove old csv file
            for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
                output_sample_root = osp.join(output_root, seq_name.replace("/", "-"))
                # 2. load data, choose specific ids of a sequence
                data = dataset.get_data(sequence_name=seq_name, ids=ids)
                filelist: list         = data['image_paths']  # [str] * N
                images: torch.Tensor   = data['images']       # (N, 3, H, W)
                gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
                valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)
                gt_mesh_path: str      = data['mesh_path']    # str

                # 3. real inference, predicted pointcloud aligned to ground truth (data_h, data_w)
                data_h, data_w         = images.shape[-2:]
                pred_pts               = infer_mesh_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
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
                # o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-test.ply"), pcd)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
                pcd_gt.colors = o3d.utility.Vector3dVector(colors)
                # # o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt.ply"), pcd_gt)

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

                pcd.estimate_normals()
                pcd.orient_normals_consistent_tangent_plane(100)
                
                pcd_gt.estimate_normals()
                pcd_gt.orient_normals_consistent_tangent_plane(100)

                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
                    gt_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_gt, depth=9)

                # densities = np.asarray(densities)
                # vertices_to_remove = densities < np.quantile(densities, 0.001)
                # mesh.remove_vertices_by_mask(vertices_to_remove)

                # ### OLD
                # gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
                # mesh = mesh.scale(scale=1.0 / max(mesh.get_minimal_oriented_bounding_box().extent), center=mesh.get_center())
                # center = (np.asarray(mesh.vertices).max(axis=0) + np.asarray(mesh.vertices).min(axis=0)) / 2
                # mesh = mesh.translate(-center)
                # gt_mesh = gt_mesh.scale(scale=1.0 / max(gt_mesh.get_minimal_oriented_bounding_box().extent), center=gt_mesh.get_center())
                # center = (np.asarray(gt_mesh.vertices).max(axis=0) + np.asarray(gt_mesh.vertices).min(axis=0)) / 2
                # gt_mesh = gt_mesh.translate(-center)

                # v1 = torch.from_numpy(np.asarray(mesh.vertices)).float().to(hydra_cfg.device)
                # f1 = torch.from_numpy(np.asarray(mesh.triangles)).long().to(hydra_cfg.device)
                # mesh_pred = pytorch3d.structures.Meshes(verts=[v1], faces=[f1])
                # v2 = torch.from_numpy(np.asarray(gt_mesh.vertices)).float().to(hydra_cfg.device)
                # f2 = torch.from_numpy(np.asarray(gt_mesh.triangles)).long().to(hydra_cfg.device)
                # mesh_gt = pytorch3d.structures.Meshes(verts=[v2], faces=[f2])

                # num_points = 1000000
                # sampled_points_pred = sample_points_from_meshes(mesh_pred, num_points)[0].cpu().numpy()
                # sampled_points_gt   = sample_points_from_meshes(mesh_gt, num_points)[0].cpu().numpy()

                # # o3d.io.write_triangle_mesh(osp.join(output_root, f"{seq_name}-pred.obj"), mesh)
                # # o3d.io.write_triangle_mesh(osp.join(output_root, f"{seq_name}-gt.obj"), gt_mesh)

                # pcd_sample_pred = o3d.geometry.PointCloud()
                # pcd_sample_pred.points = o3d.utility.Vector3dVector(sampled_points_pred)
                # pcd_sample_gt = o3d.geometry.PointCloud()
                # pcd_sample_gt.points = o3d.utility.Vector3dVector(sampled_points_gt)

                # # o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-pred-sample.ply"), pcd_sample_pred)
                # # o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt-sample.ply"), pcd_sample_gt)
                ### NEW
                os.makedirs(output_sample_root, exist_ok=True)
                o3d.io.write_triangle_mesh(osp.join(output_sample_root, f"{seq_name}-pred.obj"), mesh)
                o3d.io.write_triangle_mesh(osp.join(output_sample_root, f"{seq_name}-gt.obj"), gt_mesh)
                pred_mesh_path = osp.join(output_sample_root, f"{seq_name}-pred.obj")
                ret = run_evaluation(pred_ply=f"{seq_name}-pred.obj", scene=seq_name, path_to_pred_ply=output_sample_root, icp_align=False, full_path_to_gt_ply=osp.join(output_sample_root, f"{seq_name}-gt.obj"))
                
                # 10. compute metrics
                # acc, acc_med = accuracy(
                #     sampled_points_gt, sampled_points_pred
                # )
                # comp, comp_med = completion(
                #     sampled_points_gt, sampled_points_pred
                # )
                precision, recall, fscore, precision_mean, recall_mean = ret["precision"], ret["recall"], ret["f-score"], ret["mean precision"], ret["mean recall"]
                model_logger.info(
                    f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, Precision: {precision}, Recall: {recall} - F-score: {fscore}, Precision_mean: {precision_mean}, Recall_mean: {recall_mean}"
                )

                # 11. save metrics to csv
                write_csv(osp.join(output_root, f"_all_samples.csv"), {
                    "seq":       seq_name,
                    # "Acc-top-5%":   acc_avg,
                    "precision":  precision,
                    "recall":     recall,
                    "f-score":    fscore,
                    "precision-mean":  precision_mean,
                    "recall-mean":     recall_mean,
                })
                # all_data_dict["Acc-top-5%"] += acc_avg
                all_data_dict["precision"]  += precision
                all_data_dict["recall"]   += recall
                all_data_dict["f-score"] += fscore
                all_data_dict["precision-mean"]  += precision_mean
                all_data_dict["recall-mean"]     += recall_mean
                # release cuda memory
                torch.cuda.empty_cache()
                # shutil.rmtree(output_sample_root)

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
    set_default_arg("evaluation", "mesh_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
