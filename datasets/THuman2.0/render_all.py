import subprocess
import argparse
import os
from multiprocessing import Pool

def process_rank(rank, size, subject_ids, chunk_size):
    # Calculate the subject chunk assigned to this rank
    subject_ids_chunk = subject_ids[rank * chunk_size : (rank + 1) * chunk_size]
    print(subject_ids_chunk)

    # Ensure the last process gets any remaining subjects
    if rank == size - 1:
        subject_ids_chunk += subject_ids[size * chunk_size:]

    # Iterate over subject IDs assigned to this process
    for subject_id in subject_ids_chunk:
        # Construct the command to execute
        cmds = ['python', 'blender_script.py', '--',
                '--object_path', f'/datasets/THUman/model/{subject_id}/{subject_id}.obj',
                '--output_dir', 'test/',
                '--num_images', '97']
        subprocess.run(cmds)

def main(args):
    size = 8
    subject_ids = list(range(args.start_idx, args.end_idx))
    subject_ids = [str(subject_id).zfill(4) for subject_id in subject_ids]
    chunk_size = len(subject_ids) // size

    # Create a pool of workers to run the ranks in parallel
    with Pool(processes=size) as pool:
        pool.starmap(process_rank, [(rank, size, subject_ids, chunk_size) for rank in range(size)])
    # process_rank(0, size, subject_ids, chunk_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=0)
    args = parser.parse_args()
    main(args)