import subprocess
import argparse
import os
from multiprocessing import Pool

def process_rank(rank, size, subject_ids, chunk_size):
    # Calculate the subject chunk assigned to this rank
    subject_ids_chunk = subject_ids[rank * chunk_size : (rank + 1) * chunk_size]
    # Ensure the last process gets any remaining subjects
    if rank == size - 1:
        subject_ids_chunk += subject_ids[size * chunk_size:]
    print(subject_ids_chunk)

    len_to_generate = len(subject_ids_chunk)
    cmds = ['python', 'create_object_texture.py',
            '--project_dir', f"{args.project_dir}",
            '--start_id', f"{subject_ids_chunk[0]}",
            '--end_id', f"{subject_ids_chunk[0] + len_to_generate}",
            '--seed', f"{args.seed + rank}",
            '--group_idx', f"{args.num_project}",
            ]
    subprocess.run(cmds)

def main(args):
    size = 20 # 20
    subject_ids = list(range(0, args.num_scenes))
    chunk_size = len(subject_ids) // size

    # Create a pool of workers to run the ranks in parallel
    with Pool(processes=size) as pool:
        pool.starmap(process_rank, [(rank, size, subject_ids, chunk_size) for rank in range(size)])
    # process_rank(0, size, subject_ids, chunk_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create scene shapes")
    parser.add_argument('--project_dir', default='./generated_scenes', help='project directory, where output train_shapes/ and brdf/ are located')
    parser.add_argument('--num_project', default=1, type=int, help='idx of the project')
    parser.add_argument('--num_scenes', default=1, type=int, help='number of shapes to create')
    parser.add_argument('--dont_convert_to_glb', default=False, action='store_true', help='converts the generated objs to glbs')
    parser.add_argument('--uuid_str', default='', type=str, help='uuid to use for the shape (only used if single_shape is True)')
    parser.add_argument('--seed', default=42, type=int, help='seed for random number generation')
    parser.add_argument('--sub_obj_num_poss', type=str, default='5,5,5,4,4,3,2,1,1', help='comma separated list of possibilities for number of sub objects')
    parser.add_argument('--no_hf', default=False, action='store_true', help='do not use height field')
    parser.add_argument('--smooth_probability', default=0.1, type=float, help='possibility of smoothing the height field')
    parser.add_argument('--repeat', default=1, type=int, help='number of times to repeat the generation of the same scene')
    args = parser.parse_args()
    main(args)