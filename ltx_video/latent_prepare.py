import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging
from typing import Optional, List, Union
import yaml

import imageio
import json
import numpy as np
import torch
from safetensors import safe_open
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
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    ConditioningItem,
    LTXVideoPipeline,
    LTXMultiScalePipeline,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
import ltx_video.pipelines.crf_compressor as crf_compressor

logger = logging.get_logger("LTX-Video")

# 设置离线模式，禁止所有在线加载
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['DIFFUSERS_OFFLINE'] = '1'

# ============================================
# 解码前特征保存钩子功能
# ============================================

# 全局变量存储解码前特征
_decoder_input_features = []
_decoder_hook_handle = None


def register_decoder_hook(vae):
    """
    为VAE解码器注册钩子，捕获解码前的特征

    参数:
        vae: VAE模型实例

    返回:
        hook_handle: 钩子句柄，用于后续移除
    """
    global _decoder_input_features, _decoder_hook_handle

    # 清空之前保存的特征
    _decoder_input_features = []

    print("[钩子] 正在寻找解码器输入层...")

    # 寻找解码器的conv_in层
    if hasattr(vae, 'decoder'):
        decoder = vae.decoder

        # 方法1: 直接访问conv_in属性
        if hasattr(decoder, 'conv_in'):
            conv_in_layer = decoder.conv_in
            print(f"[钩子] 找到 decoder.conv_in: {conv_in_layer}")
        else:
            # 方法2: 查找第一个Conv3d层
            conv_in_layer = None
            for name, module in decoder.named_modules():
                if isinstance(module, torch.nn.Conv3d):
                    conv_in_layer = module
                    print(f"[钩子] 使用第一个Conv3d层: {name}")
                    break

        if conv_in_layer is not None:
            # 定义钩子函数
            def save_decoder_input_hook(module, input, output):
                """保存解码器的输入特征（只保存最后一次）"""
                global _decoder_input_features
                if input is not None and len(input) > 0:
                    # input[0] 是输入张量
                    features = input[0] if isinstance(input, tuple) else input
                    # 保存特征（不保存梯度）
                    features_detached = features.detach().cpu().clone()

                    # 清空之前的特征，只保留当前这一个
                    _decoder_input_features = [features_detached]

                    print(f"[钩子] 更新最后一次特征形状: {features.shape}")

            # 注册前向钩子
            _decoder_hook_handle = conv_in_layer.register_forward_hook(save_decoder_input_hook)
            print(f"[钩子] 成功注册钩子到: {conv_in_layer}")

            return _decoder_hook_handle
        else:
            print("[钩子] 警告: 未找到解码器的Conv3d层")
            return None
    else:
        print("[钩子] 警告: VAE没有decoder属性")
        return None


def clear_saved_features():
    """清空保存的特征"""
    global _decoder_input_features
    _decoder_input_features = []
    print("[钩子] 已清空保存的特征")


def get_saved_features():
    """获取保存的特征"""
    global _decoder_input_features
    return _decoder_input_features


def get_last_feature():
    """获取最后一个保存的特征"""
    global _decoder_input_features
    if _decoder_input_features:
        return _decoder_input_features[-1]
    return None


def remove_decoder_hook():
    """移除钩子"""
    global _decoder_hook_handle
    if _decoder_hook_handle is not None:
        _decoder_hook_handle.remove()
        _decoder_hook_handle = None
        print("[钩子] 已移除钩子")
    clear_saved_features()


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


def load_image_to_tensor_with_resize_and_crop(
        image_input: Union[str, Image.Image],
        target_height: int = 512,
        target_width: int = 768,
        just_crop: bool = False,
) -> torch.Tensor:
    """Load and process an image into a tensor.

    Args:
        image_input: Either a file path (str) or a PIL Image object
        target_height: Desired height of output tensor
        target_width: Desired width of output tensor
        just_crop: If True, only crop the image to the target size without resizing
    """
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

    frame_tensor = TVF.to_tensor(image)  # PIL -> tensor (C, H, W), [0,1]
    frame_tensor = TVF.gaussian_blur(frame_tensor, kernel_size=3, sigma=1.0)
    frame_tensor_hwc = frame_tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    frame_tensor_hwc = crf_compressor.compress(frame_tensor_hwc)
    frame_tensor = frame_tensor_hwc.permute(2, 0, 1) * 255.0  # (H, W, C) -> (C, H, W)
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def calculate_padding(
        source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:
    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)


# Generate output video name
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
    """检查本地路径是否存在，如果不存在则抛出详细错误"""
    if not path:
        raise FileNotFoundError(f"{model_name} 路径未配置")

    # 如果路径是相对路径，转换为绝对路径
    if not os.path.isabs(path):
        current_dir = Path(__file__).parent
        abs_path = (current_dir / path).resolve()
    else:
        abs_path = Path(path)

    # 检查路径是否存在
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"{model_name} 本地文件不存在: {abs_path}")

    # 检查是否是文件或目录
    if not (os.path.isfile(abs_path) or os.path.isdir(abs_path)):
        raise FileNotFoundError(f"{model_name} 路径既不是文件也不是目录: {abs_path}")

    print(f"✓ {model_name} 找到本地文件: {abs_path}")
    return str(abs_path)


def check_model_directory_structure(path: str, required_files: list = None):
    """检查模型目录结构是否完整"""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"模型目录不存在: {path}")

    if path.is_file():
        print(f"模型路径是文件: {path}")
        return

    print(f"检查模型目录结构: {path}")

    # 检查常见模型文件
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

    # 检查子目录
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    for subdir in subdirs:
        print(f"  📁 子目录: {subdir.name}")


def create_transformer(ckpt_path: str, precision: str) -> Transformer3DModel:
    # 检查本地路径
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
    """加载Transformers模型，如果标准方式失败则尝试其他方式"""
    model_path = check_local_path(model_path, f"{model_class.__name__}模型")

    try:
        # 首先尝试标准方式加载
        print(f"尝试标准方式加载模型: {model_path}")
        model = model_class.from_pretrained(model_path, local_files_only=True, **kwargs)
        print(f"✓ 模型加载成功: {model_path}")
        return model
    except Exception as e:
        print(f"标准方式加载失败: {e}")
        print(f"尝试备选方式加载模型...")

        # 尝试不使用 local_files_only
        try:
            model = model_class.from_pretrained(model_path, **kwargs)
            print(f"✓ 模型加载成功 (备选方式): {model_path}")
            return model
        except Exception as e2:
            print(f"备选方式加载失败: {e2}")

            # 检查是否是目录结构问题
            model_dir = Path(model_path)
            if model_dir.is_dir():
                # 尝试直接加载配置文件
                try:
                    config_path = model_dir / "config.json"
                    if config_path.exists():
                        print(f"尝试从配置文件加载: {config_path}")
                        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)

                        # 尝试加载模型权重
                        model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                        if model_files:
                            print(f"找到模型权重文件: {model_files[0]}")
                            # 这里需要根据具体模型类型处理
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
    """加载处理器，如果标准方式失败则尝试其他方式"""
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

            # 检查是否是目录结构问题
            processor_dir = Path(processor_path)
            if processor_dir.is_dir():
                # 尝试直接加载配置文件
                try:
                    config_path = processor_dir / "processor_config.json"
                    if not config_path.exists():
                        config_path = processor_dir / "config.json"

                    if config_path.exists():
                        print(f"尝试从配置文件加载处理器: {config_path}")
                        # 这里需要根据具体处理器类型处理
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
) -> LTXVideoPipeline:
    # 检查所有本地路径
    ckpt_path = check_local_path(ckpt_path, "主模型checkpoint")
    text_encoder_model_name_or_path = check_local_path(text_encoder_model_name_or_path, "文本编码器模型")

    print("检查模型目录结构:")
    check_model_directory_structure(text_encoder_model_name_or_path)

    # 提示增强功能已关闭，跳过相关模型检查
    print("提示增强功能已关闭，跳过相关模型检查")

    with safe_open(ckpt_path, framework="pt") as f:
        metadata = f.metadata()
        config_str = metadata.get("config")
        configs = json.loads(config_str)
        allowed_inference_steps = configs.get("allowed_inference_steps", None)

    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path, local_files_only=True)
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

    # 使用增强的加载函数
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

    vae = vae.to(torch.bfloat16)
    text_encoder = text_encoder.to(torch.bfloat16)

    # Use submodels for the pipeline
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

    # ============================================
    # 新增：注册解码器钩子
    # ============================================
    print("[钩子] 正在为VAE解码器注册特征保存钩子...")
    hook_handle = register_decoder_hook(pipeline.vae)
    if hook_handle:
        print("[钩子] 特征保存钩子注册成功")
    else:
        print("[钩子] 警告：特征保存钩子注册失败")
    # ============================================

    return pipeline


def create_latent_upsampler(latent_upsampler_model_path: str, device: str):
    """创建潜在上采样器，仅在启用第二阶段时调用"""
    if not latent_upscaler_model_path:
        raise ValueError("潜在上采样器模型路径未提供")

    # 检查本地路径
    latent_upscaler_model_path = check_local_path(latent_upscaler_model_path, "潜在上采样器模型")
    try:
        latent_upsampler = LatentUpsampler.from_pretrained(latent_upscaler_model_path, local_files_only=True)
    except TypeError:
        print(f"警告: LatentUpsampler.from_pretrained() 可能不支持 local_files_only 参数")
        print(f"尝试从本地文件直接加载: {latent_upscaler_model_path}")
        latent_upsampler = LatentUpsampler.from_pretrained(latent_upscaler_model_path)

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

    # 打印配置以帮助调试
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

    # Pipeline settings
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
        default=30, metadata={"help": "Frame rate for the output video"}
    )
    offload_to_cpu: bool = field(
        default=False, metadata={"help": "Offloading unnecessary computations to CPU."}
    )
    negative_prompt: str = field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        metadata={"help": "Negative prompt for undesired features"},
    )

    # 新增：第二阶段运行控制
    enable_second_stage: bool = field(
        default=False,
        metadata={"help": "是否启用第二阶段高分辨率优化，默认关闭"},
    )

    # 新增：第一阶段视频保存控制
    save_first_stage_video: bool = field(
        default=True,
        metadata={"help": "是否保存第一阶段低分辨率视频，默认开启"},
    )

    # 新增：第一阶段视频文件名
    first_stage_filename: str = field(
        default="first_stage_low_res_video.mp4",
        metadata={"help": "第一阶段视频的文件名"},
    )

    # Video-to-video arguments
    input_media_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the input video (or image) to be modified using the video-to-video pipeline"
        },
    )

    # Conditioning
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

    # 新增：特征保存配置
    features_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "特征保存目录，如果不指定则不保存特征"}
    )
    feature_filename: str = field(
        default="feature.pth",
        metadata={"help": "特征文件名"}
    )


def infer(config: InferenceConfig):
    print("=== 开始推理，强制本地加载模式 ===")
    print(f"第二阶段启用状态: {'开启' if config.enable_second_stage else '关闭'}")
    print(f"第一阶段视频保存: {'开启' if config.save_first_stage_video else '关闭'}")

    if config.output_path:
        output_dir = Path(config.output_path)
    else:
        output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")

    pipeline_config = load_pipeline_config(config.pipeline_config)

    ltxv_model_name_or_path = pipeline_config["checkpoint_path"]

    # 强制本地加载，不再尝试在线下载
    ltxv_model_path = check_local_path(ltxv_model_name_or_path, "LTX-Video主模型")
    print(f"主模型路径: {ltxv_model_path}")

    # 只在启用第二阶段时才检查和加载空间上采样器
    spatial_upscaler_model_name_or_path = None
    spatial_upscaler_model_path = None

    if config.enable_second_stage:
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
    else:
        print("第二阶段已禁用，跳过空间上采样器加载")
        # 如果配置文件中指定了上采样器但第二阶段被禁用，只打印警告
        if "spatial_upscaler_model_path" in pipeline_config:
            print(f"注意: 配置文件中存在上采样器路径但第二阶段被禁用: {pipeline_config['spatial_upscaler_model_path']}")

    conditioning_media_paths = config.conditioning_media_paths
    conditioning_strengths = config.conditioning_strengths
    conditioning_start_frames = config.conditioning_start_frames

    # Validate conditioning arguments
    if conditioning_media_paths:
        # Use default strengths of 1.0
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

    # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
    height_padded = ((config.height - 1) // 32 + 1) * 32
    width_padded = ((config.width - 1) // 32 + 1) * 32
    num_frames_padded = ((config.num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(
        config.height, config.width, height_padded, width_padded
    )

    logger.warning(
        f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
    )

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
    # 设置为None，避免检查不存在的路径
    prompt_enhancer_image_caption_model_name_or_path = None
    prompt_enhancer_llm_model_name_or_path = None

    # 检查所有模型路径
    print("=== 检查所有模型路径 ===")
    text_encoder_model_name_or_path = check_local_path(text_encoder_model_name_or_path, "文本编码器模型")
    print("提示增强功能已关闭，跳过相关模型检查")
    print("=== 所有模型路径检查完成 ===")

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
    )
    print("管道创建完成")

    # 判断是否为多尺度管道
    pipeline_type = pipeline_config.get("pipeline_type", None)
    is_multi_scale = pipeline_type == "multi-scale"

    # 只有在启用第二阶段且为多尺度管道时才创建LTXMultiScalePipeline
    if config.enable_second_stage and is_multi_scale:
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
    elif config.enable_second_stage and not is_multi_scale:
        print("注意: 启用第二阶段但配置文件中 pipeline_type 不是 'multi-scale'，将使用单尺度管道")
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

    # Prepare input for the pipeline
    sample = {
        "prompt": config.prompt,
        "prompt_attention_mask": None,
        "negative_prompt": config.negative_prompt,
        "negative_prompt_attention_mask": None,
    }

    generator = torch.Generator(device=device).manual_seed(config.seed)

    print("开始生成视频...")

    # 根据是否启用第二阶段选择不同的调用方式
    if config.enable_second_stage and is_multi_scale:
        print("使用多尺度管道进行两阶段生成...")

        # 准备第一阶段参数
        first_pass = pipeline_config.get("first_pass", {})

        # 准备第二阶段参数
        second_pass = pipeline_config.get("second_pass", {})

        # 获取下采样因子
        downscale_factor = pipeline_config.get("downscale_factor", 0.5)

        # 获取最终视频保存路径
        final_video_path = output_dir / f"final_video_{config.seed}.mp4"

        images = pipeline(
            downscale_factor=downscale_factor,
            first_pass=first_pass,
            second_pass=second_pass,
            save_first_stage_video=config.save_first_stage_video,
            final_video_path=str(final_video_path),
            first_stage_filename=config.first_stage_filename,
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
    else:
        print("使用单尺度管道进行生成...")

        # 单尺度生成 - 直接调用 LTXVideoPipeline
        # 如果是多尺度配置文件但第二阶段被禁用，直接使用 first_pass 的参数
        if is_multi_scale and not config.enable_second_stage:
            print("多尺度配置文件用于单尺度生成，使用 first_pass 参数")
            single_scale_params = pipeline_config.get("first_pass", {}).copy()
        else:
            # 单尺度配置文件，直接使用所有参数
            print("使用单尺度配置文件参数")
            single_scale_params = {}
            # 复制所有基本的管道参数
            for key, value in pipeline_config.items():
                if key not in ["first_pass", "second_pass", "downscale_factor", "pipeline_type"]:
                    single_scale_params[key] = value

        # 合并基础配置和阶段配置
        # 首先复制基础配置
        base_params = {}
        for key, value in pipeline_config.items():
            if key not in ["first_pass", "second_pass", "downscale_factor", "pipeline_type"]:
                base_params[key] = value

        # 将阶段参数合并到基础参数上（阶段参数优先）
        if single_scale_params:
            base_params.update(single_scale_params)

        # 确保必要的参数存在
        required_params = ["timesteps", "guidance_scale", "stg_scale", "rescaling_scale"]
        for param in required_params:
            if param not in base_params:
                # 设置合理的默认值
                if param == "timesteps":
                    # 使用模型允许的所有 timesteps
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

        # 设置 inference steps 为 timesteps 数量
        if "timesteps" in base_params:
            base_params["num_inference_steps"] = len(base_params["timesteps"])

        print(f"单尺度生成参数: timesteps={base_params.get('timesteps', '未设置')}")
        print(f"单尺度生成参数: guidance_scale={base_params.get('guidance_scale', '未设置')}")
        print(f"单尺度生成参数: stg_scale={base_params.get('stg_scale', '未设置')}")

        # 单尺度管道调用
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

    # ============================================
    # 新增：保存解码前特征
    # ============================================
    print("\n[钩子] 开始保存解码前特征...")
    saved_features = get_saved_features()

    if saved_features:
        print(f"[钩子] 捕获到 {len(saved_features)} 个特征")

        # 只保存最后一个特征
        if saved_features:
            last_feature = get_last_feature()
            if last_feature is not None:
                # 创建特征目录
                if config.features_output_dir:
                    features_dir = Path(config.features_output_dir)
                    features_dir.mkdir(parents=True, exist_ok=True)
                else:
                    print("[钩子] 警告: features_output_dir 未指定，不保存特征")

                if config.features_output_dir:
                    # 生成特征文件名
                    feature_filename = config.feature_filename
                    feature_path = features_dir / feature_filename

                    # 保存为PyTorch文件
                    torch.save({
                        'feature': last_feature,
                        'prompt': config.prompt,
                        'seed': config.seed,
                        'height': config.height,
                        'width': config.width,
                        'num_frames': config.num_frames,
                        'timestamp': datetime.now().isoformat(),
                        'description': '解码前的最后一次特征'
                    }, feature_path)

                    print(f"[钩子] 最后一次特征已保存: {feature_path}")
                    print(f"[钩子] 特征形状: {last_feature.shape}")
                    print(f"[钩子] 特征类型: {last_feature.dtype}")
    else:
        print("[钩子] 警告: 未捕获到任何特征")

    # 清理钩子
    remove_decoder_hook()
    # ============================================

    # Crop the padded images to the desired resolution and number of frames
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, : config.num_frames, pad_top:pad_bottom, pad_left:pad_right]

    for i in range(images.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = config.frame_rate
        height, width = video_np.shape[1:3]

        # 确定输出文件名
        if config.enable_second_stage:
            output_filename = get_unique_filename(
                f"video_output_stage2_{i}",
                ".mp4",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )
        else:
            output_filename = get_unique_filename(
                f"video_output_stage1_{i}",
                ".mp4",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )

        # In case a single image is generated
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            # Write video
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

        logger.warning(f"输出保存至: {output_filename}")

    print("=== 推理完成 ===")


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
    """Prepare conditioning items based on input media paths and their parameters.

    Args:
        conditioning_media_paths: List of paths to conditioning media (images or videos)
        conditioning_strengths: List of conditioning strengths for each media item
        conditioning_start_frames: List of frame indices where each item should be applied
        height: Height of the output frames
        width: Width of the output frames
        num_frames: Number of frames in the output video
        padding: Padding to apply to the frames
        pipeline: LTXVideoPipeline object used for condition video trimming

    Returns:
        A list of ConditioningItem objects.
    """
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

        # Read and preprocess the relevant frames from the video file.
        frames = []
        for i in range(num_input_frames):
            frame = Image.fromarray(reader.get_data(i))
            frame_tensor = load_image_to_tensor_with_resize_and_crop(
                frame, height, width, just_crop=just_crop
            )
            frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
            frames.append(frame_tensor)
        reader.close()

        # Stack frames along the temporal dimension
        media_tensor = torch.cat(frames, dim=2)
    else:  # Input image
        media_tensor = load_image_to_tensor_with_resize_and_crop(
            media_path, height, width, just_crop=just_crop
        )
        media_tensor = torch.nn.functional.pad(media_tensor, padding)
    return media_tensor