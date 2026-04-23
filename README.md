## Synthetic Dataset Generation

All dataset synthesis code, scripts, and documentation are located in the [`Megasynth/`](./Megasynth/) directory.
For details, see [`Megasynth/README.md`](./Megasynth/README.md).

## Real-application Datasets

- The scripts contain dataset paths and output paths that should be adjusted to match your local installation before running them.

### THuman2.0

The preprocessing and rendering scripts for [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset) are located in [`datasets/THuman2.0/`](./datasets/THuman2.0/).

To render all models, run:

```bash
cd datasets/THuman2.0
python render_all.py --start_idx 0 --end_idx 2444
```

After rendering, you may `optionally` run center cropping:

```bash
python center_cropping.py
```

### 2K2K

The preprocessing and rendering scripts for [2K2K](https://github.com/SangHunHan92/2K2K) are located in [`datasets/2K2K/`](./datasets/2K2K/).

Before rendering, run [`datasets/2K2K/reoder_ply_head.py`](./datasets/2K2K/reoder_ply_head.py) so Blender can recognize the mesh textures.

```bash
cd datasets/2K2K
python reoder_ply_head.py
python render_all.py

# Optionally
python center_cropping.py
```

### ClearPose

For [ClearPose](https://github.com/opipari/ClearPose), we do not render the dataset. We only downsample it by keeping 1 frame out of every 100 frames.


