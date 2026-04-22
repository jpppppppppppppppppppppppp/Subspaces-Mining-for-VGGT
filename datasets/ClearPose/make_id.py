import os
import json

output = {}

src_dir = 'clearpose_downsample_100'
sets = [s for s in sorted(os.listdir(src_dir)) if 'set' in s]
for s in sets:
    scenes = sorted(os.listdir(os.path.join(src_dir, s)))
    if int(s[-1]) < 8:
        scenes = scenes[-1:]
    for scene in scenes:
        rgb_root = os.path.join(src_dir, s, scene)
        all_imgs = sorted([d for d in os.listdir(rgb_root) if d.endswith('-color.png')])
        m = len(all_imgs)
        n = 12
        indices = [int((i/(n-1))*(m-1)) for i in range(n)] if n>1 else [m//2]
        print(f'{s}/{scene}', len(all_imgs), indices)
        output[f"{s}|{scene}"] = indices
with open('clearpose_mv-recon_seq-id-map.json', 'w') as f:
    json.dump(output, f, indent=4)