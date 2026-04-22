# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
print("Using Python version:", sys.version)

import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model, PeftModel

def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    parser.add_argument(
        "--lora", 
        action="store_true",
        help="Enable LoRA training"
    )
    parser.add_argument(
        "--adalora", 
        action="store_true",
        help="Enable AdaLoRA training"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=123,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=16,
        help="Rank of the LoRA adaptors"
    )
    parser.add_argument(
        "--svd_lora", 
        action="store_true",
        help="Apply svd LoRA subspace"
    )
    parser.add_argument(
        "--saved_lora_rank", 
        type=int, 
        default=256,
        help="Rank of the svd LoRA subspace"
    )
    parser.add_argument(
        "--sparse_lora", 
        action="store_true",
        help="Apply extra sparse LoRA parameters"
    )
    parser.add_argument(
        "--p_svd", 
        type=str, 
        default="qkv",
        help="Pattern of svd LoRA"
    )
    parser.add_argument(
        "--p_sparse", 
        type=str, 
        default="qkv",
        help="Pattern of sparse LoRA"
    )
    parser.add_argument(
        "--init_lora_weights", 
        type=str, 
        default="True",
        help="Weights init function"
    )
    parser.add_argument(
        "--use_predefined_space", 
        type=str, 
        default="",
        help="Predefined space file path"
    )
    parser.add_argument(
        "--use_mlp", 
        action="store_true",
        help="Apply MLP"
    )
    args = parser.parse_args()
    if args.init_lora_weights == "True":
        args.init_lora_weights = True

    lora_config = None
    if args.lora:
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank*2,
            lora_dropout=0.0,
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none",
            apply_svd_lora=args.svd_lora,
            saved_svd_rank=args.saved_lora_rank,
            apply_sparse_lora=args.sparse_lora,
            pattern_svd_lora=args.p_svd,
            pattern_sparse_lora=args.p_sparse,
            init_lora_weights=args.init_lora_weights,
            use_predefined_space=args.use_predefined_space,
            use_mlp=args.use_mlp,
        )
    if args.adalora:
        lora_config = AdaLoraConfig(
            peft_type="ADALORA",
            target_r=args.rank,
            init_r=int(args.rank*1.5),
            lora_alpha=args.rank*2,
            lora_dropout=0.0,
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none",
            total_step=100,  # This should be set according to your training schedule
            tinit=10,
            tfinal=80,
            deltaT=10,
        )

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    trainer = Trainer(is_lora=(args.lora, args.adalora), lora_config=lora_config, seed=args.seed, **cfg)
    trainer.run()


if __name__ == "__main__":
    main()


