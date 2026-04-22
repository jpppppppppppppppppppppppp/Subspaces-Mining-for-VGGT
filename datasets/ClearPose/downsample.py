import os

src_dir = 'clearpose_dataset'
dst_dir = 'clearpose_downsample_100'
sets = [s for s in os.listdir(src_dir) if 'set' in s]
for s in sets:
    for scene in os.listdir(src_dir + '/' + s):
        print(f'{s}/{scene}')
        os.system(f'mkdir -p {dst_dir}/{s}/{scene}')
        os.system(f'cp {src_dir}/{s}/{scene}/*00-* {dst_dir}/{s}/{scene}')
        os.system(f'cp {src_dir}/{s}/{scene}/metadata.mat {dst_dir}/{s}/{scene}')