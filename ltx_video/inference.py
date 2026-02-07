#!/usr/bin/env python3
"""
改进的推理脚本 - 支持三分支独立加载
从独立的root/trans/pose检查点加载模型
"""

import os
import random
import copy
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging
from typing import Optional, List, Union, Dict, Tuple
import yaml
import pickle

import imageio
import json
import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file
from PIL import Image
import torchvision.transforms.functional as TVF
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
)
from huggingface_hub import hf_hub_download
from dataclasses import dataclass, field

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
    Decoder
)
from ltx_video.models.autoencoders.motion_decoder import (
    ElasticMotionDecoder,
    ElasticMotionOnlyDecoder,
    create_elastic_motion_decoder_config
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    ConditioningItem,
    LTXVideoPipeline,
    LTXMultiScalePipeline,
    vae_decode_motion,
    MotionVAEOutput,
    save_motion_params,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
import ltx_video.pipelines.crf_compressor as crf_compressor

logger = logging.get_logger("LTX-Video")

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['DIFFUSERS_OFFLINE'] = '1'


def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return total_memory
    return 0


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DummyEncoder(nn.Module):
    """虚拟的encoder，用于兼容需要访问encoder属性的代码"""

    def __init__(self, out_channels=128):
        super().__init__()
        self.down_blocks = []
        self.out_channels = out_channels
        self.config = {
            "out_channels": out_channels,
            "_class_name": "DummyEncoder",
            "norm_num_groups": 32,
            "double_z": False,
            "sample_size": 64,
        }

    def __len__(self):
        return 0

    def __getattr__(self, name):
        if name == 'down_blocks':
            return self.down_blocks
        elif name == 'out_channels':
            return self.out_channels
        elif name == 'config':
            return self.config
        raise AttributeError(f"'DummyEncoder' object has no attribute '{name}'")

    def forward(self, x):
        raise NotImplementedError("DummyEncoder只用于占位，不支持前向传播")

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def train(self, mode=True):
        return self


class TriBranchMotionDecoderOnly(nn.Module):
    """
    三分支独立的运动解码器封装
    分别加载root/trans/pose三个分支的检查点
    """

    def __init__(
            self,
            root_checkpoint_path: str,
            trans_checkpoint_path: str,
            pose_checkpoint_path: str,
            device: str = "cuda",
            latent_channels: int = 128,
            motion_channels_per_person: int = 69,
            temporal_downscale_factor: int = 8,
            spatial_downscale_factor: int = 1,
            causal: bool = True,
            timestep_conditioning: bool = True,
    ):
        super().__init__()

        self.is_motion_vae = True
        self.motion_channels_per_person = motion_channels_per_person
        self.latent_channels = latent_channels
        self.temporal_downscale_factor = temporal_downscale_factor
        self.spatial_downscale_factor = spatial_downscale_factor
        self.scaling_factor = nn.Parameter(torch.tensor(1.0))
        self.dtype = torch.float32  # 初始设为float32
        self.device = device

        print("=" * 60)
        print("🎯 加载三分支独立训练的运动解码器")
        print("=" * 60)

        # 加载三个分支的解码器
        self.root_decoder = self._load_single_branch_decoder(
            checkpoint_path=root_checkpoint_path,
            branch_name="root",
            motion_channels=3,  # root分支输出3维
            device=device
        )

        self.trans_decoder = self._load_single_branch_decoder(
            checkpoint_path=trans_checkpoint_path,
            branch_name="trans",
            motion_channels=3,  # trans分支输出3维
            device=device
        )

        self.pose_decoder = self._load_single_branch_decoder(
            checkpoint_path=pose_checkpoint_path,
            branch_name="pose",
            motion_channels=63,  # pose分支输出63维
            device=device
        )

        # 虚拟encoder用于兼容
        self.encoder = DummyEncoder(out_channels=latent_channels)

        print("=" * 60)
        print("✅ 三分支模型加载完成")
        print("=" * 60)

        # 移动到设备并转换为float16以匹配管道
        self.to(device)
        self.eval()
        
        # 转换为float16
        self._convert_to_float16()

    def _convert_to_float16(self):
        """将解码器转换为float16以匹配管道精度"""
        print("🔧 将解码器转换为float16以匹配管道精度...")
        
        def convert_module(module):
            for param in module.parameters():
                if param.dtype == torch.float32:
                    param.data = param.data.to(torch.float16)
            for buffer in module.buffers():
                if buffer.dtype == torch.float32:
                    buffer.data = buffer.data.to(torch.float16)
        
        convert_module(self.root_decoder)
        convert_module(self.trans_decoder)
        convert_module(self.pose_decoder)
        
        # 更新dtype
        self.dtype = torch.float16
        print("✅ 解码器已转换为float16")

    def _load_single_branch_decoder(
            self,
            checkpoint_path: str,
            branch_name: str,
            motion_channels: int,
            device: str
    ) -> ElasticMotionOnlyDecoder:
        """加载单个分支的解码器"""
        print(f"\n📥 加载{branch_name}分支检查点: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{branch_name}分支检查点不存在: {checkpoint_path}")

        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print(f"  ✓ {branch_name}检查点加载成功")

            # 🔥 根据训练代码，权重在'decoder_state_dict'中
            # 但经过提取工具处理后，权重在'branch_state_dict'中
            checkpoint_keys = list(checkpoint.keys())
            print(f"  📋 检查点键数量: {len(checkpoint_keys)}")
            print(f"  📋 检查点键: {checkpoint_keys}")

            # 优先使用提取后的分支权重
            state_dict = None
            if 'branch_state_dict' in checkpoint:
                state_dict = checkpoint['branch_state_dict']
                print(f"  ✅ 从'branch_state_dict'加载权重")
            elif 'decoder_state_dict' in checkpoint:
                state_dict = checkpoint['decoder_state_dict']
                print(f"  ✅ 从'decoder_state_dict'加载权重")
            else:
                # 如果没有找到标准键，检查点可能就是state_dict本身
                print(f"  🔍 检查点可能就是state_dict本身")
                state_dict = checkpoint

            # 从检查点获取深度信息
            current_depth = checkpoint.get('current_depth', 2)
            max_depth = checkpoint.get('max_depth', 20)

            print(f"  📊 {branch_name}配置: 当前深度={current_depth}, 最大深度={max_depth}")

            # 创建单个分支的解码器
            decoder = ElasticMotionOnlyDecoder(
                dims=3,
                in_channels=self.latent_channels,
                motion_channels_per_person=motion_channels,
                base_channels=128,
                norm_layer="group_norm",
                causal=self.causal,
                timestep_conditioning=self.timestep_conditioning,
                spatial_padding_mode="zeros",
                dropout_rate=0.1,
                use_layer_norm=False,
                use_stochastic_depth=True,
                stochastic_depth_rate=0.1,
                max_res_layers=max_depth,
                initial_res_layers=current_depth,
                use_elastic_depth=True,
            )

            # 加载权重
            print(f"  🔧 加载权重到{branch_name}解码器...")
            decoder.load_state_dict(state_dict, strict=True)
            print(f"  ✅ {branch_name}分支权重加载成功")

            # 统计参数
            total_params = sum(p.numel() for p in decoder.parameters())
            print(f"  📊 {branch_name}分支参数量: {total_params:,}")

            decoder.to(device)
            decoder.eval()

            return decoder

        except Exception as e:
            print(f"  ❌ {branch_name}分支加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    @property
    def causal(self):
        """获取因果性配置"""
        return True  # 根据你的训练参数

    @property
    def timestep_conditioning(self):
        """获取时间步条件配置"""
        return True  # 根据你的训练参数

    def decode(self, latents, target_shape, timestep=None, return_dict=True):
        """
        三分支独立解码
        输入: latents [batch, latent_channels, T_compressed, 1, n_persons]
        输出: motion [batch, 69, target_frames, 1, n_persons]
        """
        batch_size, channels, T_compressed, H, W = latents.shape
        n_persons = W
        target_frames = target_shape[2]

        # 验证输入
        assert H == 1, f"高度维度应为1，当前为{H}"
        assert n_persons > 0, f"宽度维度（人数）应大于0，当前为{W}"
        assert channels == self.latent_channels, \
            f"输入通道数{channels} != 预期通道数{self.latent_channels}"

        # 🔥 确保latents与解码器在同一数据类型上
        if latents.dtype != self.dtype:
            print(f"⚠️  转换latents类型: {latents.dtype} -> {self.dtype}")
            latents = latents.to(self.dtype)

        # 🔥 修复：正确处理timestep参数
        if timestep is not None:
            # 如果timestep是浮点数或整数，转换为张量
            if isinstance(timestep, (int, float)):
                print(f"🔧 转换timestep类型: {type(timestep)} -> tensor")
                timestep = torch.tensor([timestep], device=latents.device, dtype=latents.dtype)
            # 确保timestep有正确的形状 [batch_size]
            if isinstance(timestep, torch.Tensor):
                if timestep.dim() == 0:
                    timestep = timestep.unsqueeze(0)
                # 确保timestep广播到batch_size
                if timestep.shape[0] != batch_size:
                    if timestep.shape[0] == 1:
                        timestep = timestep.expand(batch_size)
                    else:
                        raise ValueError(f"timestep形状{timestep.shape}与batch_size{batch_size}不匹配")
            
            print(f"✅ timestep形状: {timestep.shape}, dtype: {timestep.dtype}")
        else:
            # 如果timestep是None，创建一个随机timestep（训练时就是这样的）
            print(f"⚠️  timestep为None，创建随机timestep")
            timestep = torch.rand(batch_size, device=latents.device, dtype=latents.dtype)
            print(f"✅ 创建随机timestep: {timestep.shape}, dtype: {timestep.dtype}")

        # 三个分支分别解码
        target_shape_branch = (batch_size, 3, target_frames, 1, n_persons)

        # trans解码
        trans_output = self.trans_decoder(
            latents,
            target_shape=target_shape_branch,
            timestep=timestep
        )

        # root解码
        root_output = self.root_decoder(
            latents,
            target_shape=target_shape_branch,
            timestep=timestep
        )

        # pose解码
        pose_target_shape = (batch_size, 63, target_frames, 1, n_persons)
        pose_output = self.pose_decoder(
            latents,
            target_shape=pose_target_shape,
            timestep=timestep
        )

        # 拼接三个部分
        motion = torch.cat([trans_output, root_output, pose_output], dim=1)

        # 验证输出形状
        target_shape_full = (batch_size, 69, target_frames, 1, n_persons)
        assert motion.shape == target_shape_full, \
            f"输出形状{motion.shape} != 目标形状{target_shape_full}"

        if return_dict:
            return {
                "motion_params": motion,
                "latents": latents,
                "target_frames": target_frames,
                "num_persons": n_persons,
                "trans_output": trans_output,
                "root_output": root_output,
                "pose_output": pose_output,
                "timestep": timestep,
            }
        else:
            return motion

    def split_by_person(self, motion_output: torch.FloatTensor) -> List[torch.FloatTensor]:
        """
        将运动输出按人分割
        """
        from einops import rearrange

        if isinstance(motion_output, dict):
            motion_params = motion_output["motion_params"]
        else:
            motion_params = motion_output

        batch_size, channels, T, H, n_persons = motion_params.shape

        persons_motion = []
        for i in range(n_persons):
            person_motion = motion_params[:, :, :, :, i:i + 1]
            person_motion = rearrange(person_motion, 'b c t 1 1 -> b t c')
            persons_motion.append(person_motion)

        return persons_motion

    def to(self, *args, **kwargs):
        """移动到设备"""
        self.root_decoder = self.root_decoder.to(*args, **kwargs)
        self.trans_decoder = self.trans_decoder.to(*args, **kwargs)
        self.pose_decoder = self.pose_decoder.to(*args, **kwargs)
        return self

    def eval(self):
        """设置为评估模式"""
        self.root_decoder.eval()
        self.trans_decoder.eval()
        self.pose_decoder.eval()
        return self

    def train(self, mode=True):
        """设置为训练模式"""
        self.root_decoder.train(mode)
        self.trans_decoder.train(mode)
        self.pose_decoder.train(mode)
        return self


def load_trained_motion_decoder(
        root_checkpoint_path: str,
        trans_checkpoint_path: str,
        pose_checkpoint_path: str,
        device: str = "cuda",
        latent_channels: int = 128,
        motion_channels_per_person: int = 69,
):
    """
    加载三分支独立训练的运动解码器
    """
    print("=" * 60)
    print("📥 加载三分支独立训练的运动解码器")
    print("=" * 60)

    # 解析路径
    def resolve_path(path):
        if not os.path.isabs(path):
            current_dir = Path(__file__).parent
            return (current_dir / path).resolve()
        return Path(path)

    root_path = resolve_path(root_checkpoint_path)
    trans_path = resolve_path(trans_checkpoint_path)
    pose_path = resolve_path(pose_checkpoint_path)

    print(f"  root分支路径: {root_path}")
    print(f"  trans分支路径: {trans_path}")
    print(f"  pose分支路径: {pose_path}")

    # 创建三分支解码器
    motion_vae = TriBranchMotionDecoderOnly(
        root_checkpoint_path=str(root_path),
        trans_checkpoint_path=str(trans_path),
        pose_checkpoint_path=str(pose_path),
        device=device,
        latent_channels=latent_channels,
        motion_channels_per_person=motion_channels_per_person,
        temporal_downscale_factor=8,
        spatial_downscale_factor=1,
        causal=True,  # 你的训练参数 --causal
        timestep_conditioning=True  # 你的训练参数 --use_timestep
    )

    print(f"✅ 三分支MotionDecoder加载成功")
    print(f"  设备: {device}")

    # 统计参数
    root_params = sum(p.numel() for p in motion_vae.root_decoder.parameters())
    trans_params = sum(p.numel() for p in motion_vae.trans_decoder.parameters())
    pose_params = sum(p.numel() for p in motion_vae.pose_decoder.parameters())
    total_params = root_params + trans_params + pose_params

    print(f"  参数统计:")
    print(f"    root分支: {root_params:,}")
    print(f"    trans分支: {trans_params:,}")
    print(f"    pose分支: {pose_params:,}")
    print(f"    总参数: {total_params:,}")

    return motion_vae


def load_image_to_tensor_with_resize_and_crop(
        image_input: Union[str, Image.Image],
        target_height: int = 512,
        target_width: int = 768,
        just_crop: bool = False,
) -> torch.Tensor:
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be either a file path or a PIL Image object")

    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    if not just_crop:
        image = image.resize((target_width, target_height))

    frame_tensor = TVF.to_tensor(image)
    frame_tensor = TVF.gaussian_blur(frame_tensor, kernel_size=3, sigma=1.0)
    frame_tensor_hwc = frame_tensor.permute(1, 2, 0)
    frame_tensor_hwc = crf_compressor.compress(frame_tensor_hwc)
    frame_tensor = frame_tensor_hwc.permute(2, 0, 1) * 255.0
    frame_tensor = (frame_tensor / 127.5) - 1.0
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def calculate_padding(
        source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    words = clean_text.split()

    result = []
    current_length = 0

    for word in words:
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)


def get_unique_filename(
        base: str,
        ext: str,
        prompt: str,
        seed: int,
        resolution: tuple[int, int, int],
        dir: Path,
        endswith=None,
        index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )


def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def check_local_path(path: str, model_name: str) -> str:
    if not path:
        raise FileNotFoundError(f"{model_name} 路径未配置")

    if not os.path.isabs(path):
        current_dir = Path(__file__).parent
        abs_path = (current_dir / path).resolve()
    else:
        abs_path = Path(path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"{model_name} 本地文件不存在: {abs_path}")

    if not (os.path.isfile(abs_path) or os.path.isdir(abs_path)):
        raise FileNotFoundError(f"{model_name} 路径既不是文件也不是目录: {abs_path}")

    print(f"✓ {model_name} 找到本地文件: {abs_path}")
    return str(abs_path)


def check_model_directory_structure(path: str, required_files: list = None):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"模型目录不存在: {path}")

    if path.is_file():
        print(f"模型路径是文件: {path}")
        return

    print(f"检查模型目录结构: {path}")

    common_files = ["config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    if required_files:
        common_files.extend(required_files)

    found_files = []
    for file in common_files:
        file_path = path / file
        if file_path.exists():
            found_files.append(file)
            print(f"  ✓ 找到: {file}")
        else:
            print(f"  ✗ 未找到: {file}")

    if not found_files:
        print(f"警告: 在目录 {path} 中未找到任何模型文件")

    subdirs = [d for d in path.iterdir() if d.is_dir()]
    for subdir in subdirs:
        print(f"  📁 子目录: {subdir.name}")


def create_transformer(ckpt_path: str, precision: str) -> Transformer3DModel:
    ckpt_path = check_local_path(ckpt_path, "Transformer主模型")

    if precision == "float8_e4m3fn":
        try:
            from q8_kernels.integration.patch_transformer import (
                patch_diffusers_transformer as patch_transformer_for_q8_kernels,
            )

            transformer = Transformer3DModel.from_pretrained(
                ckpt_path, local_files_only=True, dtype=torch.float8_e4m3fn
            )
            patch_transformer_for_q8_kernels(transformer)
            return transformer
        except ImportError:
            raise ValueError(
                "Q8-Kernels not found. To use FP8 checkpoint, please install Q8 kernels from https://github.com/Lightricks/LTXVideo-Q8-Kernels"
            )
    elif precision == "bfloat16":
        return Transformer3DModel.from_pretrained(ckpt_path, local_files_only=True).to(torch.bfloat16)
    else:
        return Transformer3DModel.from_pretrained(ckpt_path, local_files_only=True)


def load_transformers_model_with_fallback(model_path: str, model_class, **kwargs):
    model_path = check_local_path(model_path, f"{model_class.__name__}模型")

    try:
        print(f"尝试标准方式加载模型: {model_path}")
        model = model_class.from_pretrained(model_path, local_files_only=True, **kwargs)
        print(f"✓ 模型加载成功: {model_path}")
        return model
    except Exception as e:
        print(f"标准方式加载失败: {e}")
        print(f"尝试备选方式加载模型...")

        try:
            model = model_class.from_pretrained(model_path, **kwargs)
            print(f"✓ 模型加载成功 (备选方式): {model_path}")
            return model
        except Exception as e2:
            print(f"备选方式加载失败: {e2}")

            model_dir = Path(model_path)
            if model_dir.is_dir():
                try:
                    config_path = model_dir / "config.json"
                    if config_path.exists():
                        print(f"尝试从配置文件加载: {config_path}")
                        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)

                        model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                        if model_files:
                            print(f"找到模型权重文件: {model_files[0]}")
                            model = model_class.from_pretrained(
                                str(model_dir),
                                config=config,
                                local_files_only=True,
                                **kwargs
                            )
                            print(f"✓ 模型加载成功 (手动方式): {model_path}")
                            return model
                except Exception as e3:
                    print(f"手动方式加载失败: {e3}")

            raise FileNotFoundError(f"无法加载模型 {model_path}: {e2}")


def load_processor_with_fallback(processor_path: str, processor_class, **kwargs):
    processor_path = check_local_path(processor_path, f"{processor_class.__name__}处理器")

    try:
        print(f"尝试加载处理器: {processor_path}")
        processor = processor_class.from_pretrained(processor_path, local_files_only=True, **kwargs)
        print(f"✓ 处理器加载成功: {processor_path}")
        return processor
    except Exception as e:
        print(f"处理器加载失败: {e}")
        print(f"尝试备选方式加载处理器...")

        try:
            processor = processor_class.from_pretrained(processor_path, **kwargs)
            print(f"✓ 处理器加载成功 (备选方式): {processor_path}")
            return processor
        except Exception as e2:
            print(f"备选方式加载失败: {e2}")

            processor_dir = Path(processor_path)
            if processor_dir.is_dir():
                try:
                    config_path = processor_dir / "processor_config.json"
                    if not config_path.exists():
                        config_path = processor_dir / "config.json"

                    if config_path.exists():
                        print(f"尝试从配置文件加载处理器: {config_path}")
                        processor = processor_class.from_pretrained(str(processor_dir), **kwargs)
                        print(f"✓ 处理器加载成功 (手动方式): {processor_path}")
                        return processor
                except Exception as e3:
                    print(f"手动方式加载失败: {e3}")

            raise FileNotFoundError(f"无法加载处理器 {processor_path}: {e2}")


def create_ltx_video_pipeline(
        ckpt_path: str,
        precision: str,
        text_encoder_model_name_or_path: str,
        sampler: Optional[str] = None,
        device: Optional[str] = None,
        enhance_prompt: bool = False,
        prompt_enhancer_image_caption_model_name_or_path: Optional[str] = None,
        prompt_enhancer_llm_model_name_or_path: Optional[str] = None,
        motion_mode: bool = False,
        motion_channels_per_person: int = 69,
        # 三分支检查点路径
        root_checkpoint_path: str = None,
        trans_checkpoint_path: str = None,
        pose_checkpoint_path: str = None,
) -> LTXVideoPipeline:
    ckpt_path = check_local_path(ckpt_path, "主模型checkpoint")
    text_encoder_model_name_or_path = check_local_path(text_encoder_model_name_or_path, "文本编码器模型")

    print("检查模型目录结构:")
    check_model_directory_structure(text_encoder_model_name_or_path)

    # 完全移除提示增强相关的检查和加载逻辑
    print("提示增强功能已完全关闭")

    with safe_open(ckpt_path, framework="pt") as f:
        metadata = f.metadata()
        config_str = metadata.get("config")
        configs = json.loads(config_str)
        allowed_inference_steps = configs.get("allowed_inference_steps", None)

    # 根据运动模式选择VAE创建方式
    if motion_mode:
        print("\n" + "=" * 50)
        print("🔄 运动模式：使用三分支独立训练的运动解码器")
        print("=" * 50)

        # 验证三个检查点路径
        if not all([root_checkpoint_path, trans_checkpoint_path, pose_checkpoint_path]):
            raise ValueError("运动模式需要提供root、trans、pose三个分支的检查点路径")

        print(f"  root分支检查点: {root_checkpoint_path}")
        print(f"  trans分支检查点: {trans_checkpoint_path}")
        print(f"  pose分支检查点: {pose_checkpoint_path}")

        # 使用三分支独立训练的解码器
        vae = load_trained_motion_decoder(
            root_checkpoint_path=root_checkpoint_path,
            trans_checkpoint_path=trans_checkpoint_path,
            pose_checkpoint_path=pose_checkpoint_path,
            device=device,
            latent_channels=128,
            motion_channels_per_person=motion_channels_per_person,
        )
        print("=" * 50)
        
        # 🔥 注意：解码器已转换为float16，不需要再转换为bfloat16
        print("🔧 运动解码器已转换为float16，跳过bfloat16转换")
        
    else:
        # 标准模式：从原始checkpoint加载标准VAE
        print("\n📥 加载标准VAE...")
        vae = CausalVideoAutoencoder.from_pretrained(ckpt_path, local_files_only=True)
        print("✅ 标准VAE加载完成")
        vae = vae.to(torch.bfloat16)

    # 其他组件正常加载
    transformer = create_transformer(ckpt_path, precision)

    # 加载scheduler
    if sampler == "from_checkpoint" or not sampler:
        try:
            scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)
        except TypeError as e:
            print(f"警告: RectifiedFlowScheduler.from_pretrained() 失败: {e}")
            print("创建默认scheduler")
            scheduler = RectifiedFlowScheduler()
    else:
        scheduler = RectifiedFlowScheduler(
            sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
        )

    text_encoder = load_transformers_model_with_fallback(
        text_encoder_model_name_or_path,
        T5EncoderModel,
        subfolder="text_encoder"
    )

    patchifier = SymmetricPatchifier(patch_size=1)

    tokenizer = load_transformers_model_with_fallback(
        text_encoder_model_name_or_path,
        T5Tokenizer,
        subfolder="tokenizer"
    )

    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    # 完全不加载提示增强相关模型
    prompt_enhancer_image_caption_model = None
    prompt_enhancer_image_caption_processor = None
    prompt_enhancer_llm_model = None
    prompt_enhancer_llm_tokenizer = None

    # 转换为bfloat16（运动解码器已为float16，不需要转换）
    if not motion_mode:
        vae = vae.to(torch.bfloat16)
    text_encoder = text_encoder.to(torch.bfloat16)

    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
        "prompt_enhancer_image_caption_model": prompt_enhancer_image_caption_model,
        "prompt_enhancer_image_caption_processor": prompt_enhancer_image_caption_processor,
        "prompt_enhancer_llm_model": prompt_enhancer_llm_model,
        "prompt_enhancer_llm_tokenizer": prompt_enhancer_llm_tokenizer,
        "allowed_inference_steps": allowed_inference_steps,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)

    return pipeline


def create_latent_upsampler(latent_upsampler_model_path: str, device: str):
    if not latent_upsampler_model_path:
        raise ValueError("潜在上采样器模型路径未提供")

    latent_upsampler_model_path = check_local_path(latent_upsampler_model_path, "潜在上采样器模型")
    try:
        latent_upsampler = LatentUpsampler.from_pretrained(latent_upsampler_model_path, local_files_only=True)
    except TypeError:
        print(f"警告: LatentUpsampler.from_pretrained() 可能不支持 local_files_only 参数")
        print(f"尝试从本地文件直接加载: {latent_upsampler_model_path}")
        latent_upsampler = LatentUpsampler.from_pretrained(latent_upsampler_model_path)

    latent_upsampler.to(device)
    latent_upsampler.eval()
    return latent_upsampler


def load_pipeline_config(pipeline_config: str):
    current_file = Path(__file__)

    path = None
    if os.path.isfile(current_file.parent / pipeline_config):
        path = current_file.parent / pipeline_config
    elif os.path.isfile(pipeline_config):
        path = pipeline_config
    else:
        raise ValueError(f"Pipeline config file {pipeline_config} does not exist")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    print("=== 加载的管道配置 ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("======================")

    return config


@dataclass
class InferenceConfig:
    prompt: str = field(metadata={"help": "Prompt for the generation"})

    output_path: str = field(
        default_factory=lambda: Path(
            f"outputs/{datetime.today().strftime('%Y-%m-%d')}"
        ),
        metadata={"help": "Path to the folder to save the output video"},
    )

    pipeline_config: str = field(
        default="configs/ltxv-2b-0.9.8-distilled.yaml",
        metadata={"help": "Path to the pipeline config file"},
    )
    seed: int = field(
        default=171198, metadata={"help": "Random seed for the inference"}
    )
    height: int = field(
        default=704, metadata={"help": "Height of the output video frames"}
    )
    width: int = field(
        default=1216, metadata={"help": "Width of the output video frames"}
    )
    num_frames: int = field(
        default=121,
        metadata={"help": "Number of frames to generate in the output video"},
    )
    frame_rate: int = field(
        default=30, metadata={"help": "Frame rate for the output video"},
    )
    offload_to_cpu: bool = field(
        default=False, metadata={"help": "Offloading unnecessary computations to CPU."}
    )
    negative_prompt: str = field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        metadata={"help": "Negative prompt for undesired features"},
    )

    motion_mode: bool = field(
        default=True,
        metadata={"help": "是否启用运动推理模式，生成69维运动参数"},
    )

    motion_channels_per_person: int = field(
        default=69,
        metadata={"help": "每人运动参数通道数，默认为69"},
    )

    # 三分支检查点路径
    root_checkpoint_path: str = field(
        default="/hy-tmp/elastic_root_models/latest_root_checkpoint.pt",
        metadata={"help": "root分支检查点路径"},
    )

    trans_checkpoint_path: str = field(
        default="/hy-tmp/elastic_trans_models/latest_trans_checkpoint.pt",
        metadata={"help": "trans分支检查点路径"},
    )

    pose_checkpoint_path: str = field(
        default="/hy-tmp/elastic_pose_models/latest_pose_checkpoint.pt",
        metadata={"help": "pose分支检查点路径"},
    )

    motion_target_frames: Optional[int] = field(
        default=None,
        metadata={"help": "目标帧数（运动模式必需）。如果不指定，则自动计算"},
    )

    save_motion_params_path: Optional[str] = field(
        default=None,
        metadata={"help": "保存运动参数到指定路径（可选）"},
    )

    enable_second_stage: bool = field(
        default=False,
        metadata={"help": "是否启用第二阶段高分辨率优化，默认关闭"},
    )

    save_first_stage_video: bool = field(
        default=True,
        metadata={"help": "是否保存第一阶段低分辨率视频，默认开启"},
    )

    first_stage_filename: str = field(
        default="first_stage_low_res_video.mp4",
        metadata={"help": "第一阶段视频的文件名"},
    )

    input_media_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the input video (or image) to be modified using the video-to-video pipeline"
        },
    )

    image_cond_noise_scale: float = field(
        default=0.15,
        metadata={"help": "Amount of noise to add to the conditioned image"},
    )
    conditioning_media_paths: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of paths to conditioning media (images or videos). Each path will be used as a conditioning item."
        },
    )
    conditioning_strengths: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "List of conditioning strengths (between 0 and 1) for each conditioning item. Must match the number of conditioning items."
        },
    )
    conditioning_start_frames: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "List of frame indices where each conditioning item should be applied. Must match the number of conditioning items."
        },
    )

    output_format: str = field(
        default="pkl",
        metadata={"help": "输出格式: pkl, npy, pt, 默认pkl"}
    )


def decode_latents_to_motion_simple(
        latents: torch.FloatTensor,
        vae,
        target_frames: int,
        motion_channels_per_person: int = 69
) -> torch.FloatTensor:
    """简化的latents解码为运动参数"""
    batch_size, channels, T_compressed, H, W = latents.shape

    print(f"\n🔍 解码参数:")
    print(f"  - latents形状: {latents.shape}")
    print(f"  - 目标帧数: {target_frames}")

    try:
        # 🔥 创建随机timestep（与训练时一致）
        timestep = torch.rand(batch_size, device=latents.device, dtype=latents.dtype)
        print(f"  - 创建timestep: {timestep.shape}, dtype: {timestep.dtype}")

        # 使用vae的decode方法
        if hasattr(vae, 'decode'):
            motion = vae.decode(
                latents=latents,
                target_shape=(batch_size, motion_channels_per_person, target_frames, 1, W),
                timestep=timestep,  # 🔥 提供timestep参数
                return_dict=False
            )
        else:
            # 尝试直接调用
            motion = vae(
                latents=latents,
                target_frames=target_frames,
                timestep=timestep,  # 🔥 提供timestep参数
                return_dict=False
            )

        print(f"✅ VAE解码成功")
        return motion

    except Exception as e:
        print(f"❌ VAE解码失败: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"VAE解码失败: {e}")


def decode_69d_to_smpl_params(motion_69d):
    """将69维运动参数解码为SMPL格式"""
    if isinstance(motion_69d, torch.Tensor):
        motion_cpu = motion_69d.cpu().float().detach()
        motion_np = motion_cpu.numpy()
    else:
        motion_np = motion_69d

    if motion_np.shape[1] != 69:
        raise ValueError(f"期望69维，但得到{motion_np.shape[1]}维")

    # 分割为三部分
    trans = motion_np[:, :3]  # 平移 (3维)
    root_orient = motion_np[:, 3:6]  # 根方向 (3维)
    pose_body = motion_np[:, 6:69]  # 身体姿势 (63维)

    return {
        'trans': trans,
        'root_orient': root_orient,
        'pose_body': pose_body,
        'gender': 'neutral'
    }


def infer(config: InferenceConfig):
    print("=" * 70)
    print("🚀 开始推理 - 使用三分支独立训练的运动解码器")
    print("=" * 70)

    print(f"📋 配置信息:")
    print(f"  运动模式: {'开启' if config.motion_mode else '关闭'}")
    if config.motion_mode:
        print(f"  每人运动通道数: {config.motion_channels_per_person}")
        print(f"  目标帧数: {config.motion_target_frames or '自动计算'}")
        print(f"  root检查点: {config.root_checkpoint_path}")
        print(f"  trans检查点: {config.trans_checkpoint_path}")
        print(f"  pose检查点: {config.pose_checkpoint_path}")
        print(f"  保存路径: {config.save_motion_params_path or '不保存单独文件'}")

    print(f"  第二阶段: {'开启' if config.enable_second_stage else '关闭'}")
    print(f"  输出格式: {config.output_format}")
    print(f"  提示: {config.prompt[:50]}...")
    print(f"  分辨率: {config.height}x{config.width}x{config.num_frames}")
    print("=" * 70)

    if config.output_path:
        output_dir = Path(config.output_path)
    else:
        output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")

    pipeline_config = load_pipeline_config(config.pipeline_config)

    ltxv_model_name_or_path = pipeline_config["checkpoint_path"]

    ltxv_model_path = check_local_path(ltxv_model_name_or_path, "LTX-Video主模型")
    print(f"主模型路径: {ltxv_model_path}")

    spatial_upscaler_model_name_or_path = None
    spatial_upscaler_model_path = None

    if config.enable_second_stage and not config.motion_mode:
        spatial_upscaler_model_name_or_path = pipeline_config.get(
            "spatial_upscaler_model_path"
        )
        if spatial_upscaler_model_name_or_path:
            spatial_upscaler_model_path = check_local_path(
                spatial_upscaler_model_name_or_path,
                "空间上采样器模型"
            )
            print(f"上采样器路径: {spatial_upscaler_model_path}")
        else:
            raise ValueError(
                "启用第二阶段需要配置空间上采样器模型路径 (spatial_upscaler_model_path)，但未在配置文件中找到"
            )
    elif config.enable_second_stage and config.motion_mode:
        print("注意: 运动模式不支持第二阶段，将忽略启用第二阶段的设置")
    elif not config.enable_second_stage:
        print("第二阶段已禁用，跳过空间上采样器加载")

    conditioning_media_paths = config.conditioning_media_paths
    conditioning_strengths = config.conditioning_strengths
    conditioning_start_frames = config.conditioning_start_frames

    if conditioning_media_paths:
        if not conditioning_strengths:
            conditioning_strengths = [1.0] * len(conditioning_media_paths)
        if not conditioning_start_frames:
            raise ValueError(
                "If `conditioning_media_paths` is provided, "
                "`conditioning_start_frames` must also be provided"
            )
        if len(conditioning_media_paths) != len(conditioning_strengths) or len(
                conditioning_media_paths
        ) != len(conditioning_start_frames):
            raise ValueError(
                "`conditioning_media_paths`, `conditioning_strengths`, "
                "and `conditioning_start_frames` must have the same length"
            )
        if any(s < 0 or s > 1 for s in conditioning_strengths):
            raise ValueError("All conditioning strengths must be between 0 and 1")
        if any(f < 0 or f >= config.num_frames for f in conditioning_start_frames):
            raise ValueError(
                f"All conditioning start frames must be between 0 and {config.num_frames - 1}"
            )

    seed_everething(config.seed)
    if config.offload_to_cpu and not torch.cuda.is_available():
        logger.warning(
            "offload_to_cpu is set to True, but offloading will not occur since the model is already running on CPU."
        )
        offload_to_cpu = False
    else:
        offload_to_cpu = config.offload_to_cpu and get_total_gpu_memory() < 30

    output_dir = (
        Path(config.output_path)
        if config.output_path
        else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    height_padded = ((config.height - 1) // 32 + 1) * 32
    width_padded = ((config.width - 1) // 32 + 1) * 32
    num_frames_padded = ((config.num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(
        config.height, config.width, height_padded, width_padded
    )

    logger.warning(
        f"调整后的维度: {height_padded}x{width_padded}x{num_frames_padded}"
    )
    print(f"📏 维度调整:")
    print(f"  原始: {config.height}x{config.width}x{config.num_frames}")
    print(f"  调整后: {height_padded}x{width_padded}x{num_frames_padded}")

    device = get_device()
    print(f"使用设备: {device}")

    prompt_enhancement_words_threshold = pipeline_config[
        "prompt_enhancement_words_threshold"
    ]

    prompt_word_count = len(config.prompt.split())
    enhance_prompt = (
            prompt_enhancement_words_threshold > 0
            and prompt_word_count < prompt_enhancement_words_threshold
    )

    if prompt_enhancement_words_threshold > 0 and not enhance_prompt:
        logger.info(
            f"Prompt has {prompt_word_count} words, which exceeds the threshold of {prompt_enhancement_words_threshold}. Prompt enhancement disabled."
        )

    # 强制关闭提示增强
    enhance_prompt = False
    print("提示增强功能已强制关闭")

    precision = pipeline_config["precision"]
    text_encoder_model_name_or_path = pipeline_config["text_encoder_model_name_or_path"]
    sampler = pipeline_config.get("sampler", None)
    prompt_enhancer_image_caption_model_name_or_path = None
    prompt_enhancer_llm_model_name_or_path = None

    print("\n" + "=" * 50)
    print("检查所有模型路径")
    print("=" * 50)
    text_encoder_model_name_or_path = check_local_path(text_encoder_model_name_or_path, "文本编码器模型")
    print("提示增强功能已关闭，跳过相关模型检查")
    print("=" * 50)
    print("所有模型路径检查完成")
    print("=" * 50 + "\n")

    print("开始创建管道...")
    pipeline = create_ltx_video_pipeline(
        ckpt_path=ltxv_model_path,
        precision=precision,
        text_encoder_model_name_or_path=text_encoder_model_name_or_path,
        sampler=sampler,
        device=device,
        enhance_prompt=enhance_prompt,
        prompt_enhancer_image_caption_model_name_or_path=None,
        prompt_enhancer_llm_model_name_or_path=None,
        motion_mode=config.motion_mode,
        motion_channels_per_person=config.motion_channels_per_person,
        # 三分支检查点路径
        root_checkpoint_path=config.root_checkpoint_path,
        trans_checkpoint_path=config.trans_checkpoint_path,
        pose_checkpoint_path=config.pose_checkpoint_path,
    )
    print("✅ 管道创建完成")

    pipeline_type = pipeline_config.get("pipeline_type", None)
    is_multi_scale = pipeline_type == "multi-scale"

    if config.enable_second_stage and is_multi_scale and not config.motion_mode:
        if not spatial_upscaler_model_path:
            raise ValueError(
                "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
            )
        print("创建潜在上采样器...")
        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, pipeline.device
        )
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)
        print("多尺度管道创建完成")
    elif config.enable_second_stage and not is_multi_scale and not config.motion_mode:
        print("注意: 启用第二阶段但配置文件中 pipeline_type 不是 'multi-scale'，将使用单尺度管道")
    elif config.motion_mode:
        print("运动模式启用，使用单尺度管道（运动模式不支持多尺度）")
    else:
        print("第二阶段已禁用，使用单尺度管道")

    media_item = None
    if config.input_media_path:
        media_item = load_media_file(
            media_path=config.input_media_path,
            height=config.height,
            width=config.width,
            max_frames=num_frames_padded,
            padding=padding,
        )

    conditioning_items = (
        prepare_conditioning(
            conditioning_media_paths=conditioning_media_paths,
            conditioning_strengths=conditioning_strengths,
            conditioning_start_frames=conditioning_start_frames,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            padding=padding,
            pipeline=pipeline,
        )
        if conditioning_media_paths
        else None
    )

    stg_mode = pipeline_config.get("stg_mode", "attention_values")
    del pipeline_config["stg_mode"]
    if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

    sample = {
        "prompt": config.prompt,
        "prompt_attention_mask": None,
        "negative_prompt": config.negative_prompt,
        "negative_prompt_attention_mask": None,
    }

    generator = torch.Generator(device=device).manual_seed(config.seed)

    single_scale_params = pipeline_config.get("first_pass", {}).copy()

    base_params = {}
    for key, value in pipeline_config.items():
        if key not in ["first_pass", "second_pass", "downscale_factor", "pipeline_type"]:
            base_params[key] = value

    if single_scale_params:
        base_params.update(single_scale_params)

    required_params = ["timesteps", "guidance_scale", "stg_scale", "rescaling_scale"]
    for param in required_params:
        if param not in base_params:
            if param == "timesteps":
                with safe_open(ltxv_model_path, framework="pt") as f:
                    metadata = f.metadata()
                    config_str = metadata.get("config")
                    configs = json.loads(config_str)
                    allowed_inference_steps = configs.get("allowed_inference_steps", None)
                    if allowed_inference_steps:
                        base_params[param] = allowed_inference_steps
                    else:
                        base_params[param] = [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219]
            elif param == "guidance_scale":
                base_params[param] = 1
            elif param == "stg_scale":
                base_params[param] = 0
            elif param == "rescaling_scale":
                base_params[param] = 1

    if "timesteps" in base_params:
        base_params["num_inference_steps"] = len(base_params["timesteps"])

    print(f"\n⚙️ 生成参数:")
    print(f"  - timesteps: {base_params.get('timesteps', '未设置')}")
    print(f"  - guidance_scale: {base_params.get('guidance_scale', '未设置')}")
    print(f"  - stg_scale: {base_params.get('stg_scale', '未设置')}")
    print(f"  - 推理步数: {base_params.get('num_inference_steps', '未设置')}")

    print("\n" + "=" * 70)
    print("开始生成...")
    print("=" * 70)

    if config.motion_mode:
        print("\n" + "=" * 70)
        print("🚀 启动运动推理模式生成69维运动参数")
        print("=" * 70)

        motion_target_frames = config.motion_target_frames
        if motion_target_frames is None:
            motion_target_frames = num_frames_padded
            print(f"📏 目标帧数设置为输入帧数: {motion_target_frames}")

        # 使用pipeline的运动模式
        try:
            motion_output = pipeline.motion_inference(
                height=height_padded,
                width=width_padded,
                num_frames=num_frames_padded,
                frame_rate=config.frame_rate,
                **base_params,
                skip_layer_strategy=skip_layer_strategy,
                generator=generator,
                callback_on_step_end=None,
                **sample,
                media_items=media_item,
                conditioning_items=conditioning_items,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=config.image_cond_noise_scale,
                mixed_precision=False,  # 🔥 禁用混合精度以避免数据类型不匹配
                offload_to_cpu=offload_to_cpu,
                device=device,
                enhance_prompt=enhance_prompt,
                motion_channels_per_person=config.motion_channels_per_person,
                motion_target_frames=motion_target_frames,
                save_motion_params_path=config.save_motion_params_path,
            )

            print(f"\n✅ 运动推理完成")
            print(f"  - 运动参数形状: {motion_output.motion_params.shape}")

            # 保存运动参数（如果指定了路径）
            if config.save_motion_params_path and hasattr(motion_output, 'motion_params'):
                save_motion_params(
                    motion_output=motion_output,
                    filepath=config.save_motion_params_path,
                    format="pt"
                )
                print(f"✅ 运动参数已保存到: {config.save_motion_params_path}")

        except AttributeError:
            print("⚠️ pipeline没有motion_inference方法，使用手动解码方式")
            # 手动解码方式
            original_output_type = "latent"

            print(f"\n🎬 开始推理...")
            result = pipeline(
                **base_params,
                skip_layer_strategy=skip_layer_strategy,
                generator=generator,
                output_type=original_output_type,
                callback_on_step_end=None,
                height=height_padded,
                width=width_padded,
                num_frames=num_frames_padded,
                frame_rate=config.frame_rate,
                **sample,
                media_items=media_item,
                conditioning_items=conditioning_items,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=config.image_cond_noise_scale,
                mixed_precision=False,  # 🔥 禁用混合精度
                offload_to_cpu=offload_to_cpu,
                device=device,
                enhance_prompt=enhance_prompt,
            )

            if hasattr(result, 'images'):
                latents = result.images
            else:
                latents = result[0] if isinstance(result, tuple) else result

            print(f"\n✅ 推理完成")
            print(f"  - 生成的latents形状: {latents.shape}")

            print(f"🔧 解码latents为运动参数...")
            motion_params = decode_latents_to_motion_simple(
                latents=latents,
                vae=pipeline.vae,
                target_frames=motion_target_frames,
                motion_channels_per_person=config.motion_channels_per_person
            )

            print(f"✅ 解码完成")
            print(f"  - 解码后的运动参数形状: {motion_params.shape}")

            motion_output = MotionVAEOutput(
                motion_params=motion_params,
                latents=latents,
                metadata={
                    'prompt': config.prompt,
                    'seed': config.seed,
                    'target_frames': motion_target_frames,
                    'channels_per_person': config.motion_channels_per_person,
                    'num_persons': motion_params.shape[4],
                    'height': config.height,
                    'width': config.width,
                    'num_frames': config.num_frames,
                }
            )

        persons_motion = motion_output.split_by_person()
        print(f"👥 运动参数已按{len(persons_motion)}人分割")

        print(f"\n💾 保存运动参数...")
        for i in range(motion_output.motion_params.shape[0]):
            print(f"处理第{i + 1}个batch...")

            if len(persons_motion) > i:
                person1_motion_cpu = persons_motion[0][i].cpu() if i < persons_motion[0].shape[0] else \
                    persons_motion[0][0].cpu()
                person1_params = decode_69d_to_smpl_params(person1_motion_cpu)

                if len(persons_motion) > 1:
                    person2_motion_cpu = persons_motion[1][i].cpu() if i < persons_motion[1].shape[0] else \
                        persons_motion[1][0].cpu()
                    person2_params = decode_69d_to_smpl_params(person2_motion_cpu)
                else:
                    zeros_data = np.zeros_like(person1_motion_cpu.numpy())
                    person2_params = decode_69d_to_smpl_params(zeros_data)

                save_data = {
                    'person1': person1_params,
                    'person2': person2_params,
                    'mocap_framerate': float(config.frame_rate),
                    'frames': int(config.num_frames),

                    'metadata': {
                        'prompt': config.prompt,
                        'seed': config.seed,
                        'original_height': config.height,
                        'original_width': config.width,
                        'num_frames': config.num_frames,
                        'timestamp': datetime.now().isoformat(),
                        'generator': 'LTX-Video Motion Mode',
                        'motion_channels_per_person': config.motion_channels_per_person,
                        'motion_target_frames': motion_target_frames,
                    }
                }

                output_filename = get_unique_filename(
                    f"motion_{i}",
                    f".{config.output_format}",
                    prompt=config.prompt,
                    seed=config.seed,
                    resolution=(save_data['person1']['trans'].shape[0],
                                save_data['person1']['pose_body'].shape[1],
                                config.motion_channels_per_person),
                    dir=output_dir,
                )

                print(f"准备保存到: {output_filename}")

                if config.output_format.lower() == 'pkl':
                    with open(output_filename, 'wb') as f:
                        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"✅ 运动参数已保存为pkl文件: {output_filename}")

                elif config.output_format.lower() == 'npy':
                    np.save(output_filename, save_data)
                    print(f"✅ 运动参数已保存为npy文件: {output_filename}")

                elif config.output_format.lower() == 'pt':
                    torch.save(save_data, output_filename)
                    print(f"✅ 运动参数已保存为pt文件: {output_filename}")

                else:
                    raise ValueError(f"不支持的输出格式: {config.output_format}")

            else:
                print(f"⚠️ 警告: persons_motion长度不足，跳过第{i}个batch")

        print("\n" + "=" * 70)
        print("🎉 运动推理完成")
        print("=" * 70)
        print(f"运动推理模式完成")
        print(f"输出目录: {output_dir}")
        print("=" * 70)
        return

    else:
        print("\n" + "=" * 70)
        print("使用单尺度管道进行生成...")
        print("=" * 70)

        images = pipeline(
            **base_params,
            skip_layer_strategy=skip_layer_strategy,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=config.frame_rate,
            **sample,
            media_items=media_item,
            conditioning_items=conditioning_items,
            is_video=True,
            vae_per_channel_normalize=True,
            image_cond_noise_scale=config.image_cond_noise_scale,
            mixed_precision=(precision == "mixed_precision"),
            offload_to_cpu=offload_to_cpu,
            device=device,
            enhance_prompt=enhance_prompt,
        ).images

        print(f"\n✅ 视频生成完成")
        print(f"  - 生成的视频形状: {images.shape}")

        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom
        pad_right = -pad_right
        if pad_bottom == 0:
            pad_bottom = images.shape[3]
        if pad_right == 0:
            pad_right = images.shape[4]
        images = images[:, :, : config.num_frames, pad_top:pad_bottom, pad_left:pad_right]

        print(f"  - 裁剪后的视频形状: {images.shape}")

        print(f"\n💾 保存图像/视频数据...")
        for i in range(images.shape[0]):
            image_data = images[i].cpu()

            save_data = {
                'image_data': image_data,
                'prompt': config.prompt,
                'seed': config.seed,
                'original_height': config.height,
                'original_width': config.width,
                'num_frames': config.num_frames,
                'frame_rate': config.frame_rate,
                'negative_prompt': config.negative_prompt,
                'timestamp': datetime.now().isoformat(),
                'image_shape': image_data.shape,
                'image_dtype': str(image_data.dtype),
            }

            save_data['inference_params'] = {
                'timesteps': base_params.get('timesteps', []),
                'guidance_scale': base_params.get('guidance_scale', 1),
                'stg_scale': base_params.get('stg_scale', 0),
                'num_inference_steps': base_params.get('num_inference_steps', 0),
            }

            if config.enable_second_stage:
                output_filename = get_unique_filename(
                    f"video_output_stage2_{i}",
                    f".{config.output_format}",
                    prompt=config.prompt,
                    seed=config.seed,
                    resolution=(image_data.shape[2], image_data.shape[3], config.num_frames),
                    dir=output_dir,
                )
            else:
                output_filename = get_unique_filename(
                    f"video_output_stage1_{i}",
                    f".{config.output_format}",
                    prompt=config.prompt,
                    seed=config.seed,
                    resolution=(image_data.shape[2], image_data.shape[3], config.num_frames),
                    dir=output_dir,
                )

            if config.output_format.lower() == 'pkl':
                with open(output_filename, 'wb') as f:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"✅ 图像/视频数据已保存为pkl文件: {output_filename}")

            elif config.output_format.lower() == 'npy':
                np.save(output_filename, image_data.numpy())
                print(f"✅ 图像/视频数据已保存为npy文件: {output_filename}")

            elif config.output_format.lower() == 'pt':
                torch.save(save_data, output_filename)
                print(f"✅ 图像/视频数据已保存为pt文件: {output_filename}")

            else:
                raise ValueError(f"不支持的输出格式: {config.output_format}")

        print("\n" + "=" * 70)
        print("🎉 推理完成")
        print("=" * 70)
        print(f"总共生成 {images.shape[0]} 个图像/视频")
        print(f"输出目录: {output_dir}")
        print("=" * 70)


def prepare_conditioning(
        conditioning_media_paths: List[str],
        conditioning_strengths: List[float],
        conditioning_start_frames: List[int],
        height: int,
        width: int,
        num_frames: int,
        padding: tuple[int, int, int, int],
        pipeline: LTXVideoPipeline,
) -> Optional[List[ConditioningItem]]:
    conditioning_items = []
    for path, strength, start_frame in zip(
            conditioning_media_paths, conditioning_strengths, conditioning_start_frames
    ):
        num_input_frames = orig_num_input_frames = get_media_num_frames(path)
        if hasattr(pipeline, "trim_conditioning_sequence") and callable(
                getattr(pipeline, "trim_conditioning_sequence")
        ):
            num_input_frames = pipeline.trim_conditioning_sequence(
                start_frame, orig_num_input_frames, num_frames
            )
        if num_input_frames < orig_num_input_frames:
            logger.warning(
                f"Trimming conditioning video {path} from {orig_num_input_frames} to {num_input_frames} frames."
            )

        media_tensor = load_media_file(
            media_path=path,
            height=height,
            width=width,
            max_frames=num_input_frames,
            padding=padding,
            just_crop=True,
        )
        conditioning_items.append(ConditioningItem(media_tensor, start_frame, strength))
    return conditioning_items


def get_media_num_frames(media_path: str) -> int:
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]
    )
    num_frames = 1
    if is_video:
        reader = imageio.get_reader(media_path)
        num_frames = reader.count_frames()
        reader.close()
    return num_frames


def load_media_file(
        media_path: str,
        height: int,
        width: int,
        max_frames: int,
        padding: tuple[int, int, int, int],
        just_crop: bool = False,
) -> torch.Tensor:
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]
    )
    if is_video:
        reader = imageio.get_reader(media_path)
        num_input_frames = min(reader.count_frames(), max_frames)

        frames = []
        for i in range(num_input_frames):
            frame = Image.fromarray(reader.get_data(i))
            frame_tensor = load_image_to_tensor_with_resize_and_crop(
                frame, height, width, just_crop=just_crop
            )
            frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
            frames.append(frame_tensor)
        reader.close()

        media_tensor = torch.cat(frames, dim=2)
    else:
        media_tensor = load_image_to_tensor_with_resize_and_crop(
            media_path, height, width, just_crop=just_crop
        )
        media_tensor = torch.nn.functional.pad(media_tensor, padding)
    return media_tensor