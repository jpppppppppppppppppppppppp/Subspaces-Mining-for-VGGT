import torch
import torch.nn as nn

from typing import Optional, Dict

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from peft import PeftModel, LoraConfig, TaskType, PeftConfig
import os

class Pi3(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            predefined_space: str = "",
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.pi3.models.pi3 import Pi3 as Pi3Model
            if pretrained_model_name_or_path.startswith("yyfz233"):
                self.model = Pi3Model.from_pretrained(pretrained_model_name_or_path)
            elif pretrained_model_name_or_path.startswith("/") and pretrained_model_name_or_path.endswith(".pt"):
                self.model = Pi3Model()
                checkpoint = torch.load(pretrained_model_name_or_path)
                if "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                _, _ = self.model.load_state_dict(checkpoint, strict=False)
            elif pretrained_model_name_or_path.startswith("/") and "lora" in pretrained_model_name_or_path:
                _URL = "yyfz233/Pi3"
                self.model = Pi3Model.from_pretrained(_URL)
                modules = [(n, m) for n, m in self.model.named_modules()]
                lora_modules_name = []
                for n, m in modules:
                    if "encoder" in n:
                        continue
                    if "conf_decoder" in n:
                        continue
                    if "conf_head" in n:
                        continue
                    if isinstance(m, torch.nn.modules.linear.Linear) or isinstance(m, torch.nn.modules.conv.Conv2d):
                        lora_modules_name.append(n)
                pattern = "block"
                pattern_svd_lora = ""
                if "qkv" in pretrained_model_name_or_path:
                    pattern = "qkv"
                elif "attn" in pretrained_model_name_or_path:
                    pattern = "attn"
                elif "block" in pretrained_model_name_or_path:
                    pattern = "block"

                rank = 16
                if "_lora_" in pretrained_model_name_or_path and predefined_space == "":
                    rank_list = [i for i in pretrained_model_name_or_path.split("_lora_")[-1].split("_") if i.isdigit()]
                    if len(rank_list) > 0:
                        rank = int(rank_list[0])
                init_lora_weights = True
                if "pissa" in pretrained_model_name_or_path:
                    init_lora_weights = "pissa"
                if predefined_space != "" and os.path.exists(predefined_space):
                    pass
                else:
                    predefined_space = ""
                lora_config = LoraConfig(
                    r=rank,
                    lora_alpha=rank*2,
                    lora_dropout=0.0,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    bias="none",
                    target_modules=lora_modules_name,
                    apply_svd_lora=pattern_svd_lora != "",
                    pattern_svd_lora=pattern_svd_lora,
                    init_lora_weights=init_lora_weights,
                    use_predefined_space=predefined_space,
                )
                # print(pretrained_model_name_or_path, lora_config)
                self.model = PeftModel.from_pretrained(self.model, config=lora_config, model_id=pretrained_model_name_or_path)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, imgs: torch.Tensor):
        return self.model(imgs=imgs)


class VGGT(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            predefined_space: str = "",
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.vggt.models.vggt import VGGT as VGGTModel
            if pretrained_model_name_or_path.startswith("facebook"):
                self.model = VGGTModel.from_pretrained(pretrained_model_name_or_path)
        
            elif pretrained_model_name_or_path.startswith("/") and pretrained_model_name_or_path.endswith(".pt"):
                self.model = VGGTModel()
                checkpoint = torch.load(pretrained_model_name_or_path, map_location="cpu")
                if "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                _, _ = self.model.load_state_dict(checkpoint, strict=False)
            elif pretrained_model_name_or_path.startswith("/") and "adalora" in pretrained_model_name_or_path:
                _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
                self.model = VGGTModel()
                self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
                self.model = PeftModel.from_pretrained(self.model, pretrained_model_name_or_path)
                self.model.print_trainable_parameters()
            elif pretrained_model_name_or_path.startswith("/") and "lora" in pretrained_model_name_or_path:
                _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
                self.model = VGGTModel()
                self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
                modules = [(n, m) for n, m in self.model.named_modules()]
                lora_modules_name = []
                for n, m in modules:
                    if "aggregator.patch_embed" in n:
                        continue
                    if "track_head" in n:
                        continue
                    if "point_head" in n:
                        continue
                    if isinstance(m, torch.nn.modules.linear.Linear) or isinstance(m, torch.nn.modules.conv.Conv2d):
                        lora_modules_name.append(n)
                pattern = "block"
                pattern_svd_lora = ""
                if "qkv" in pretrained_model_name_or_path:
                    pattern = "qkv"
                elif "attn" in pretrained_model_name_or_path:
                    pattern = "attn"
                elif "block" in pretrained_model_name_or_path:
                    pattern = "block"

                rank = 16
                if "_lora_" in pretrained_model_name_or_path:
                    rank_list = [i for i in pretrained_model_name_or_path.split("_lora_")[-1].split("_") if i.isdigit()]
                    if len(rank_list) > 0:
                        rank = int(rank_list[0])
                init_lora_weights = True
                if "pissa" in pretrained_model_name_or_path:
                    init_lora_weights = "pissa"
                if predefined_space != "" and os.path.exists(predefined_space):
                    pass
                else:
                    predefined_space = ""
                lora_config = LoraConfig(
                    r=rank,
                    lora_alpha=rank*2,
                    lora_dropout=0.0,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    bias="none",
                    target_modules=lora_modules_name,
                    apply_svd_lora=pattern_svd_lora != "",
                    pattern_svd_lora=pattern_svd_lora,
                    init_lora_weights=init_lora_weights,
                    use_predefined_space=predefined_space,
                )
                # print(pretrained_model_name_or_path, lora_config)
                self.model = PeftModel.from_pretrained(self.model, config=lora_config, model_id=pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        return self.model(images=images, query_points=query_points)


class MoGe(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            ori_model: bool = True,
        ):
        super().__init__()

        if ori_model and pretrained_model_name_or_path is not None:
            from models.moge.model.v1 import MoGeModel
            self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        return self.model(image, num_tokens)


# class AetherV1(nn.Module):
#     def __init__(
#             self,
#             pretrained_model_name_or_path: Optional[str] = None,
#             ori_model: bool = True,
#         ):
#         super().__init__()

#         if ori_model and pretrained_model_name_or_path is not None:
#             from models.aether.v1 import AetherV1Model
#             self.model = AetherV1Model.from_pretrained(pretrained_model_name_or_path)
#         else:
#             raise NotImplementedError
#     def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
#         return self.model(images, query_points)

