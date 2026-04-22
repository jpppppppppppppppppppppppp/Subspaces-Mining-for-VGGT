import json
import os
import random

root_dir = "plane_eval/rendering"
categories = sorted(os.listdir(root_dir))

# train_split = 9 / 10
# categories = categories[int(len(categories) * train_split):]

output_meta = {}
select_num = [i for i in range(0, 42, 6)]
for cat in categories:
    seq_names = os.listdir(os.path.join(root_dir, cat))
    # seq_idx = random.choices(range(len(seq_names)), k=...)
    # seq_names = [seq_names[i] for i in seq_idx]
    for seq in seq_names:
        print(cat+'|'+seq)
        output_meta[cat+'|'+seq] = select_num

with open("{save_name}_mv-recon_seq-id-map.json".format(save_name="PLANE"), "w") as f:
    json.dump(output_meta, f)
