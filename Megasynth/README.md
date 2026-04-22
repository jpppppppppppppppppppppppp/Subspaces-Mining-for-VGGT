<div align="center">

# MegaSynth: Scaling Up 3D Scene Renstruction with Synthesized Data

</div>



## Installation
```
sudo apt install libxi6 libsm6 libxext6
pip install datasets opencv-python Pillow rich bpy==3.6.0 numpy scipy matplotlib mathutils==3.3.0
# install blender, then
export PATH=path/to/blender/:$PATH  # needs blender binary in addition to bpy to run
```

## Generate Scenes

The provided [`render.sh`](./render.sh) script generates several types of synthetic data under a single root directory:

- plane-based face anti-spoofing scenes
- texture-variation scenes
- geometry-variation scenes
- camera-variation renders
- lighting-variation renders

Before running the script, edit the configuration variables at the top of `render.sh`:

- `TARGET_PATH`: root directory for generated scenes and render outputs
- `NUM_SCENES`: number of scenes to generate
- `SEED`: global random seed used by the generation scripts

Then run:

```
. ./render.sh
```

The script performs the following stages:

1. Generates plane-based scene GLBs with `create_planes.py`.
2. Renders each generated plane scene with `render_plane_rgbd.py`.
3. Creates texture-variation data with `create_object_texture_para.py`.
4. Creates geometry-variation data with `create_object_geometry_para.py`.
5. Renders the texture- and geometry-variation scenes with `render_scenes_rgbd_para.py`.
6. Renders a camera-variation version with `render_scenes_rgbd_para.py --camera_rand`.
7. Renders a lighting-variation version with `render_object_rgbd_light_para.py`.

Notes:

- `--camera_rand` enables randomized camera targets and a wider control-point distance range during camera-path sampling.
- In the lighting-variation stage, `seed` controls lighting randomness, while `np_seed` controls camera sampling and can be fixed independently.
- You need to comment out unused commands in `render.sh`.

## BibTex
If you find this code useful, please consider citing:
```
@article{jiang2024megasynth,
  title={MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data},
  author={Jiang, Hanwen and Xu, Zexiang and Xie, Desai and Chen, Ziwen and Jin, Haian and Luan, Fujun and Shu, Zhixin and Zhang, Kai and Bi, Sai and Sun, Xin and Gu, Jiuxiang and Huang, Qixing and Pavlakos, Georgios and Tan, Hao},
  booktitle={arXiv preprint arXiv:2412.14166},
  year={2024},
}
```
