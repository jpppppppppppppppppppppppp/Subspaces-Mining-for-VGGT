import trimesh
import numpy as np
import os

dataset_dir = "data"
dataset_source_list = sorted(os.listdir(dataset_dir))

for subject_id in dataset_source_list:
    mesh = trimesh.load_mesh(f"data/{subject_id}/{subject_id}.ply")
    os.makedirs(f"data_reordered/{subject_id}", exist_ok=True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors)
    mesh.export(f"data_reordered/{subject_id}/{subject_id}.ply")
