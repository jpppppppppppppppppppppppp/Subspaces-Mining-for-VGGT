import os
import json
ids = [i for i in range(0, 96, 96//8)]
seq_names = sorted(os.listdir('./renderings'))
data = {}
for seq in seq_names[-100:]:
    data[seq] = ids
json.dump(data, open('THuman_mv-recon_seq-id-map.json', 'w'), indent=4)