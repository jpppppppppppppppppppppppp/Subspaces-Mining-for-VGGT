import subprocess
import argparse
import os
from multiprocessing import Pool
import argparse

def process_rank(rank, size, subject_ids, chunk_size, dir_path, output_path, seed, camera_rand):
    # Calculate the subject chunk assigned to this rank
    subject_ids_chunk = subject_ids[rank * chunk_size : (rank + 1) * chunk_size]
    print(subject_ids_chunk)

    # Ensure the last process gets any remaining subjects
    if rank == size - 1:
        subject_ids_chunk += subject_ids[size * chunk_size:]
    # seed = seed + rank
    for scene_name in subject_ids_chunk:
        object_path = os.path.join(dir_path, "scenes", scene_name, "scenes.glb")
        output_dir = os.path.join(dir_path, output_path, scene_name)
        # seed = seed + size

        cmds = ['python', 'render_object_rgbd.py',
                '--object_path', f"{object_path}",
                '--output_dir', f"{output_dir}",
                '--seed', f"{seed}",
                '--render_times', "1",
                '--scan', '--random_camera_sampling' if camera_rand else '',
                ]
        subprocess.run(cmds)

def main(dir_path, output_path, seed, camera_rand):
    size = 5 # 8
    subject_ids = sorted(os.listdir(os.path.join(dir_path, "scenes")))
    chunk_size = len(subject_ids) // size

    # Create a pool of workers to run the ranks in parallel
    with Pool(processes=size) as pool:
        pool.starmap(process_rank, [(rank, size, subject_ids, chunk_size, dir_path, output_path, seed, camera_rand) for rank in range(size)])
    # process_rank(0, size, subject_ids, chunk_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create scene shapes")
    parser.add_argument('--dir_path', default='./test/')
    parser.add_argument('--output_path', default='renderings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--camera_rand', action='store_true')
    args = parser.parse_args()
    main(args.dir_path, args.output_path, args.seed, args.camera_rand)
