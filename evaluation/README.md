This repo is modified from [recons_eval](https://github.com/ZhouTimeMachine/recons_eval).

## 1. Video Depth Estimation

configs in `configs/evaluation/videodepth.yaml`, see [videodepth/README.md](videodepth/README.md) for more details.

```bash
python videodepth/infer.py
# torchrun --nnodes=1 --nproc_per_node=8 videodepth/infer_mp.py  # accelerate with multi gpus
python videodepth/eval.py
```

## 2. Multi-view Reconstruction (Point Map Estimation)

See [mv_recon/README.md](mv_recon/README.md) for more details.

```bash
# python mv_recon/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python mv_recon/eval.py
# torchrun --nnodes=1 --nproc_per_node=8 mv_recon/eval_mp.py  # accelerate with multi gpus
```

## Acknowledgement

Our work mainly builds upon:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
- [Spann3R](https://github.com/HengyiWang/spann3r)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [MoGe](https://github.com/microsoft/MoGe)
- [VGGT](https://github.com/facebookresearch/vggt)

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{wang2025pi3,
      title={$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning},
      author={Yifan Wang and Jianjun Zhou and Haoyi Zhu and Wenzheng Chang and Yang Zhou and Zizun Li and Junyi Chen and Jiangmiao Pang and Chunhua Shen and Tong He},
      year={2025},
      eprint={2507.13347},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13347},
}
```

## License

This project is licensed under CC BY-NC-SA 4.0 License. See the LICENSE file and https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.