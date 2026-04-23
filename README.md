<div align="center">

# Mining Attribute Subspaces for Efficient Fine-tuning of 3D Foundation Models (CVPR2026)

</div>

<div align="center">
    <a href="https://arxiv.org/abs/2604.10095"><strong>Paper</strong></a>
</div>

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

## Experiments

Our experiments are organized into two parts:

- [`vggt/`](./vggt/): finetuning code for VGGT with our modified PEFT implementation.
- [`evaluation/`](./evaluation/): evaluation code for downstream geometry tasks.

### Finetuning

The finetuning pipeline is implemented in [`vggt/`](./vggt/). It includes the installation steps, training
configuration, and the PEFT modifications required by our method.

For details, see [`vggt/README.md`](./vggt/README.md).

### Evaluation

The evaluation pipeline is implemented in [`evaluation/`](./evaluation/), which is adapted from
[`recons_eval`](https://github.com/ZhouTimeMachine/recons_eval). It provides evaluation code for:

- video depth estimation
- multi-view reconstruction

For task-specific usage, see [`evaluation/README.md`](./evaluation/README.md).

## TODO
- [ ] Release our extracted subspaces.
- [ ] Release spectrum analysis code for subspace mining.


## BibTex
If you find this code useful, please consider citing:
```
@article{jiang2026mining,
  title={Mining Attribute Subspaces for Efficient Fine-tuning of 3D Foundation Models},
  author={Jiang, Yu and Jiang, Hanwen and Abdelkader, Ahmed and Chu, Wen-Sheng and Feng, Brandon Y and Wang, Zhangyang and Huang, Qixing},
  journal={arXiv preprint arXiv:2604.10095},
  year={2026}
}

@article{jiang2024megasynth,
  title={MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data},
  author={Jiang, Hanwen and Xu, Zexiang and Xie, Desai and Chen, Ziwen and Jin, Haian and Luan, Fujun and Shu, Zhixin and Zhang, Kai and Bi, Sai and Sun, Xin and Gu, Jiuxiang and Huang, Qixing and Pavlakos, Georgios and Tan, Hao},
  booktitle={arXiv preprint arXiv:2412.14166},
  year={2024},
}

@inproceedings{wang2025vggt,
  title={Vggt: Visual geometry grounded transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5294--5306},
  year={2025}
}

@article{wang2025pi,
  title={$$\backslash$pi\^{} 3$: Permutation-Equivariant Visual Geometry Learning},
  author={Wang, Yifan and Zhou, Jianjun and Zhu, Haoyi and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Chen, Junyi and Pang, Jiangmiao and Shen, Chunhua and He, Tong},
  journal={arXiv preprint arXiv:2507.13347},
  year={2025}
}
```