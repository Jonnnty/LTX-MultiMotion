from typing import Tuple, Optional, List
import torch
from diffusers import AutoencoderKL
from einops import rearrange
from torch import Tensor

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
    MotionCausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.video_autoencoder import (
    Downsample3D,
    VideoAutoencoder,
)

# 导入ElasticMotionDecoder和相关类
from ltx_video.models.autoencoders.motion_decoder import (
    ElasticMotionDecoder,
    ElasticMotionOnlyDecoder
)

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def is_elastic_motion_decoder(vae) -> bool:
    """
    判断是否为弹性运动解码器类型
    """
    # 检查是否为ElasticMotionDecoder
    if isinstance(vae, ElasticMotionDecoder):
        return True
    
    # 检查是否为ElasticMotionOnlyDecoder
    if isinstance(vae, ElasticMotionOnlyDecoder):
        return True
    
    # 检查是否为自定义的三分支解码器（来自inference.py）
    if hasattr(vae, 'root_decoder') and hasattr(vae, 'trans_decoder') and hasattr(vae, 'pose_decoder'):
        return True
    
    # 检查是否有is_motion_vae属性
    if hasattr(vae, 'is_motion_vae') and vae.is_motion_vae:
        return True
    
    return False


def vae_encode(
        media_items: Tensor,
        vae: AutoencoderKL,
        split_size: int = 1,
        vae_per_channel_normalize=False,
) -> Tensor:
    """
    Encodes media items (images or videos) into latent representations using a specified VAE model.
    The function supports processing batches of images or video frames and can handle the processing
    in smaller sub-batches if needed.

    Args:
        media_items (Tensor): A torch Tensor containing the media items to encode. The expected
            shape is (batch_size, channels, height, width) for images or (batch_size, channels,
            frames, height, width) for videos.
        vae (AutoencoderKL): An instance of the `AutoencoderKL` class from the `diffusers` library,
            pre-configured and loaded with the appropriate model weights.
        split_size (int, optional): The number of sub-batches to split the input batch into for encoding.
            If set to more than 1, the input media items are processed in smaller batches according to
            this value. Defaults to 1, which processes all items in a single batch.

    Returns:
        Tensor: A torch Tensor of the encoded latent representations. The shape of the tensor is adjusted
            to match the input shape, scaled by the model's configuration.

    Examples:
        >>> import torch
        >>> from diffusers import AutoencoderKL
        >>> vae = AutoencoderKL.from_pretrained('your-model-name')
        >>> images = torch.rand(10, 3, 8 256, 256)  # Example tensor with 10 videos of 8 frames.
        >>> latents = vae_encode(images, vae)
        >>> print(latents.shape)  # Output shape will depend on the model's latent configuration.

    Note:
        In case of a video, the function encodes the media item frame-by frame.
    """
    is_video_shaped = media_items.dim() == 5
    batch_size, channels = media_items.shape[0:2]

    if channels != 3:
        raise ValueError(f"Expects tensors with 3 channels, got {channels}.")

    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)
    
    if is_video_shaped and not isinstance(
            vae, (VideoAutoencoder, CausalVideoAutoencoder, MotionCausalVideoAutoencoder)
    ) and not is_elastic_motion:
        media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
        
    if split_size > 1:
        if len(media_items) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(media_items) // split_size
        latents = []
        if media_items.device.type == "xla":
            xm.mark_step()
        for image_batch in media_items.split(encode_bs):
            # 弹性运动解码器无法编码
            if is_elastic_motion:
                raise ValueError("ElasticMotionDecoder cannot encode, it's decoder-only")
            latents.append(vae.encode(image_batch).latent_dist.sample())
            if media_items.device.type == "xla":
                xm.mark_step()
        latents = torch.cat(latents, dim=0)
    else:
        # 弹性运动解码器无法编码
        if is_elastic_motion:
            raise ValueError("ElasticMotionDecoder cannot encode, it's decoder-only")
        latents = vae.encode(media_items).latent_dist.sample()

    latents = normalize_latents(latents, vae, vae_per_channel_normalize)
    if is_video_shaped and not isinstance(
            vae, (VideoAutoencoder, CausalVideoAutoencoder, MotionCausalVideoAutoencoder)
    ) and not is_elastic_motion:
        latents = rearrange(latents, "(b n) c h w -> b c n h w", b=batch_size)
    return latents


def vae_decode(
        latents: Tensor,
        vae: AutoencoderKL,
        is_video: bool = True,
        split_size: int = 1,
        vae_per_channel_normalize=False,
        timestep=None,
) -> Tensor:
    """
    解码latents为图像/视频或运动参数

    Args:
        latents: 潜在表示
        vae: VAE模型
        is_video: 是否为视频
        split_size: 分批次大小
        vae_per_channel_normalize: 是否按通道归一化
        timestep: 时间步条件

    Returns:
        解码后的图像/视频或运动参数
    """
    is_video_shaped = latents.dim() == 5
    batch_size = latents.shape[0]

    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)
    
    if is_video_shaped and not isinstance(
            vae, (VideoAutoencoder, CausalVideoAutoencoder, MotionCausalVideoAutoencoder)
    ) and not is_elastic_motion:
        latents = rearrange(latents, "b c n h w -> (b n) c h w")
        
    if split_size > 1:
        if len(latents) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(latents) // split_size
        image_batch = [
            _run_decoder(
                latent_batch, vae, is_video, vae_per_channel_normalize, timestep
            )
            for latent_batch in latents.split(encode_bs)
        ]
        images = torch.cat(image_batch, dim=0)
    else:
        images = _run_decoder(
            latents, vae, is_video, vae_per_channel_normalize, timestep
        )

    if is_video_shaped and not isinstance(
            vae, (VideoAutoencoder, CausalVideoAutoencoder, MotionCausalVideoAutoencoder)
    ) and not is_elastic_motion:
        images = rearrange(images, "(b n) c h w -> b c n h w", b=batch_size)
    return images


def vae_decode_motion(
        latents: Tensor,
        vae: AutoencoderKL,
        target_frames: int,
        split_size: int = 1,
        vae_per_channel_normalize=False,
        timestep=None,
) -> Tensor:
    """
    专门解码latents为运动参数

    Args:
        latents: 潜在表示 [batch, C, T_compressed, 1, n]
        vae: 运动VAE模型 (MotionCausalVideoAutoencoder 或 ElasticMotionDecoder)
        target_frames: 目标帧数
        split_size: 分批次大小
        vae_per_channel_normalize: 是否按通道归一化
        timestep: 时间步条件

    Returns:
        解码后的运动参数 [batch, 69, target_frames, 1, n]
    """
    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)
    
    if not isinstance(vae, MotionCausalVideoAutoencoder) and not is_elastic_motion:
        raise ValueError("vae_decode_motion requires MotionCausalVideoAutoencoder or ElasticMotionDecoder model")

    is_video_shaped = latents.dim() == 5
    batch_size = latents.shape[0]

    # 验证输入形状
    if latents.dim() != 5:
        raise ValueError(f"Expected 5D tensor for motion decoding, got {latents.dim()}D")

    _, _, T_compressed, H, W = latents.shape
    if H != 1:
        raise ValueError(f"Expected height dimension to be 1 for motion data, got {H}")

    if split_size > 1:
        if len(latents) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(latents) // split_size
        motion_batch = [
            _run_motion_decoder(
                latent_batch, vae, target_frames, vae_per_channel_normalize, timestep
            )
            for latent_batch in latents.split(encode_bs)
        ]
        motions = torch.cat(motion_batch, dim=0)
    else:
        motions = _run_motion_decoder(
            latents, vae, target_frames, vae_per_channel_normalize, timestep
        )

    return motions


def _run_decoder(
        latents: Tensor,
        vae: AutoencoderKL,
        is_video: bool,
        vae_per_channel_normalize=False,
        timestep=None,
) -> Tensor:
    """
    运行通用解码器（图像/视频）
    """
    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)

    if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)) or is_elastic_motion:
        *_, fl, hl, wl = latents.shape
        temporal_scale, spatial_scale, _ = get_vae_size_scale_factor(vae)
        latents = latents.to(vae.dtype)
        vae_decode_kwargs = {}
        if timestep is not None:
            vae_decode_kwargs["timestep"] = timestep

        # 反归一化latents
        unnormalized_latents = un_normalize_latents(latents, vae, vae_per_channel_normalize)

        if is_elastic_motion:
            # 弹性运动解码器使用自己的decode方法
            if hasattr(vae, 'decode'):
                # 获取运动通道数
                motion_channels = getattr(vae, 'motion_channels_per_person', 69)
                
                # 计算目标形状
                target_frames = fl * temporal_scale if is_video else 1
                target_shape = (
                    unnormalized_latents.shape[0],
                    motion_channels,
                    target_frames,
                    hl * spatial_scale,
                    wl * spatial_scale,
                )
                
                # 解码运动参数
                image = vae.decode(
                    unnormalized_latents,
                    target_shape=target_shape,
                    timestep=timestep if 'timestep' in vae_decode_kwargs else None,
                    return_dict=False
                )
            elif hasattr(vae, 'forward'):
                # 如果是ElasticMotionDecoder，使用forward方法
                image = vae(
                    unnormalized_latents,
                    target_frames=fl * temporal_scale if is_video else 1,
                    timestep=timestep,
                    return_dict=False
                )
            else:
                # 如果是直接调用
                image = vae(unnormalized_latents)
        else:
            # 标准VAE解码
            image = vae.decode(
                unnormalized_latents,
                return_dict=False,
                target_shape=(
                    unnormalized_latents.shape[0],
                    3,
                    fl * temporal_scale if is_video else 1,
                    hl * spatial_scale,
                    wl * spatial_scale,
                ),
                **vae_decode_kwargs,
            )[0]
    elif isinstance(vae, MotionCausalVideoAutoencoder):
        # 运动VAE使用专门的解码路径
        raise ValueError("Use vae_decode_motion for MotionCausalVideoAutoencoder")
    else:
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
        )[0]
    return image


def _run_motion_decoder(
        latents: Tensor,
        vae: AutoencoderKL,
        target_frames: int,
        vae_per_channel_normalize=False,
        timestep=None,
) -> Tensor:
    """
    运行运动参数解码器

    Args:
        latents: 潜在表示 [batch, C, T_compressed, 1, n]
        vae: 运动VAE模型
        target_frames: 目标帧数
        vae_per_channel_normalize: 是否按通道归一化
        timestep: 时间步条件

    Returns:
        运动参数 [batch, 69, target_frames, 1, n]
    """
    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)
    
    if not isinstance(vae, MotionCausalVideoAutoencoder) and not is_elastic_motion:
        raise ValueError("_run_motion_decoder requires MotionCausalVideoAutoencoder or ElasticMotionDecoder")

    # 反归一化latents
    unnormalized_latents = un_normalize_latents(latents, vae, vae_per_channel_normalize)

    # 转换到正确的数据类型
    unnormalized_latents = unnormalized_latents.to(vae.dtype)

    if is_elastic_motion:
        # 弹性运动解码器使用自己的decode或forward方法
        if hasattr(vae, 'decode'):
            # 获取运动通道数
            motion_channels = getattr(vae, 'motion_channels_per_person', 69)
            
            # 计算目标形状
            target_shape = (
                unnormalized_latents.shape[0],
                motion_channels,
                target_frames,
                1,  # 高度维度为1
                unnormalized_latents.shape[4],  # 保持宽度维度（代表人数）
            )
            
            # 解码运动参数
            motion_output = vae.decode(
                unnormalized_latents,
                target_shape=target_shape,
                timestep=timestep,
                return_dict=False
            )
        elif hasattr(vae, 'forward'):
            # 如果是ElasticMotionDecoder，使用forward方法
            motion_output = vae(
                unnormalized_latents,
                target_frames=target_frames,
                timestep=timestep,
                return_dict=False
            )
        else:
            # 如果是直接调用
            motion_output = vae(
                unnormalized_latents,
                target_frames=target_frames,
                timestep=timestep
            )
    else:
        # MotionCausalVideoAutoencoder
        motion_output = vae(
            unnormalized_latents,
            target_frames=target_frames,
            timestep=timestep
        )

    return motion_output


def get_vae_size_scale_factor(vae: AutoencoderKL) -> Tuple[int, int, int]:
    """
    获取VAE的尺寸缩放因子

    Args:
        vae: VAE模型

    Returns:
        (时间缩放因子, 空间高度缩放因子, 空间宽度缩放因子)
    """
    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)

    if is_elastic_motion:
        # 弹性运动解码器使用固定的缩放因子
        # 时间：8倍下采样 (t-1)//8+1
        temporal_factor = 8

        # 空间使用32，而不是spatial_downscale_factor
        # spatial_downscale_factor=1只表示MotionDecoder内部不做空间下采样
        # 但pipeline计算latent形状时需要的是最终的缩放因子：32
        spatial_factor = 32  # 固定为32

        return (temporal_factor, spatial_factor, spatial_factor)

    if isinstance(vae, MotionCausalVideoAutoencoder):
        # 运动VAE复用基础VAE的缩放因子
        spatial = vae.spatial_downscale_factor
        temporal = vae.temporal_downscale_factor
        return (temporal, spatial, spatial)
    elif isinstance(vae, CausalVideoAutoencoder):
        spatial = vae.spatial_downscale_factor
        temporal = vae.temporal_downscale_factor
        return (temporal, spatial, spatial)
    else:
        try:
            down_blocks = len(
                [
                    block
                    for block in vae.encoder.down_blocks
                    if isinstance(block.downsample, Downsample3D)
                ]
            )

            # 安全地获取patch_size
            if hasattr(vae.config, 'patch_size'):
                patch_size = vae.config.patch_size
            elif isinstance(vae.config, dict) and 'patch_size' in vae.config:
                patch_size = vae.config['patch_size']
            else:
                # 使用默认值
                patch_size = 1
                print(f"⚠️  VAE配置中没有patch_size属性，使用默认值: {patch_size}")

            spatial = patch_size * 2 ** down_blocks

            # 安全地获取patch_size_t
            if isinstance(vae, VideoAutoencoder):
                if hasattr(vae.config, 'patch_size_t'):
                    patch_size_t = vae.config.patch_size_t
                elif isinstance(vae.config, dict) and 'patch_size_t' in vae.config:
                    patch_size_t = vae.config['patch_size_t']
                else:
                    # 使用默认值
                    patch_size_t = 1
                    print(f"⚠️  VAE配置中没有patch_size_t属性，使用默认值: {patch_size_t}")
                temporal = patch_size_t * 2 ** down_blocks
            else:
                temporal = 1

            return (temporal, spatial, spatial)

        except AttributeError as e:
            print(f"⚠️  无法获取VAE缩放因子，使用默认值 (8, 16, 16): {e}")
            # 返回默认缩放因子
            return (8, 16, 16)


def latent_to_pixel_coords(
        latent_coords: Tensor, vae: AutoencoderKL, causal_fix: bool = False
) -> Tensor:
    """
    Converts latent coordinates to pixel coordinates by scaling them according to the VAE's
    configuration.

    Args:
        latent_coords (Tensor): A tensor of shape [batch_size, 3, num_latents]
        containing the latent corner coordinates of each token.
        vae (AutoencoderKL): The VAE model
        causal_fix (bool): Whether to take into account the different temporal scale
            of the first frame. Default = False for backwards compatibility.
    Returns:
        Tensor: A tensor of pixel coordinates corresponding to the input latent coordinates.
    """

    scale_factors = get_vae_size_scale_factor(vae)

    # 检查是否需要causal_fix
    if causal_fix:
        is_causal = isinstance(vae, (CausalVideoAutoencoder, MotionCausalVideoAutoencoder))
    else:
        is_causal = False

    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)
    if is_elastic_motion:
        is_causal = False  # 弹性运动解码器不需要causal_fix

    pixel_coords = latent_to_pixel_coords_from_factors(
        latent_coords, scale_factors, is_causal
    )
    return pixel_coords


def latent_to_pixel_coords_from_factors(
        latent_coords: Tensor, scale_factors: Tuple, causal_fix: bool = False
) -> Tensor:
    pixel_coords = (
            latent_coords
            * torch.tensor(scale_factors, device=latent_coords.device)[None, :, None]
    )
    if causal_fix:
        # Fix temporal scale for first frame to 1 due to causality
        pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - scale_factors[0]).clamp(min=0)
    return pixel_coords


def normalize_latents(
        latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    """
    归一化latents

    Args:
        latents: 潜在表示
        vae: VAE模型
        vae_per_channel_normalize: 是否按通道归一化

    Returns:
        归一化后的latents
    """
    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)

    if is_elastic_motion:
        # 弹性运动解码器使用默认缩放因子
        scaling_factor = getattr(vae, 'scaling_factor', 1.0)
        return latents * scaling_factor

    if isinstance(vae, MotionCausalVideoAutoencoder):
        # 运动VAE使用基础VAE的归一化参数
        base_vae = vae.base_vae
        if hasattr(base_vae, 'std_of_means') and hasattr(base_vae, 'mean_of_means'):
            return (
                (latents - base_vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1))
                / base_vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
                if vae_per_channel_normalize
                else latents * base_vae.config.scaling_factor
            )
        else:
            return latents * base_vae.config.scaling_factor
    elif hasattr(vae, 'std_of_means') and hasattr(vae, 'mean_of_means'):
        return (
            (latents - vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1))
            / vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
            if vae_per_channel_normalize
            else latents * vae.config.scaling_factor
        )
    else:
        return latents * vae.config.scaling_factor


def un_normalize_latents(
        latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    """
    反归一化latents

    Args:
        latents: 归一化后的潜在表示
        vae: VAE模型
        vae_per_channel_normalize: 是否按通道归一化

    Returns:
        反归一化后的latents
    """
    # 检查是否为弹性运动解码器
    is_elastic_motion = is_elastic_motion_decoder(vae)

    if is_elastic_motion:
        # 弹性运动解码器使用默认缩放因子
        scaling_factor = getattr(vae, 'scaling_factor', 1.0)
        return latents / scaling_factor

    if isinstance(vae, MotionCausalVideoAutoencoder):
        # 运动VAE使用基础VAE的反归一化参数
        base_vae = vae.base_vae
        if hasattr(base_vae, 'std_of_means') and hasattr(base_vae, 'mean_of_means'):
            return (
                latents * base_vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
                + base_vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
                if vae_per_channel_normalize
                else latents / base_vae.config.scaling_factor
            )
        else:
            return latents / base_vae.config.scaling_factor
    elif hasattr(vae, 'std_of_means') and hasattr(vae, 'mean_of_means'):
        return (
            latents * vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
            + vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
            if vae_per_channel_normalize
            else latents / vae.config.scaling_factor
        )
    else:
        return latents / vae.config.scaling_factor


class MotionVAEOutput:
    """
    运动VAE输出类

    用于存储运动VAE的解码结果
    """

    def __init__(
            self,
            motion_params: torch.FloatTensor,
            latents: Optional[torch.FloatTensor] = None,
            metadata: Optional[dict] = None
    ):
        """
        初始化运动VAE输出

        Args:
            motion_params: 运动参数张量 [batch, 69, T, 1, n]
            latents: 原始latents（可选）
            metadata: 元数据（可选）
        """
        self.motion_params = motion_params
        self.latents = latents
        self.metadata = metadata or {}

        # 自动提取信息
        self.batch_size = motion_params.shape[0]
        self.num_persons = motion_params.shape[4]  # 宽度维度代表人
        self.num_frames = motion_params.shape[2]
        self.motion_channels = motion_params.shape[1] 

    def split_by_person(self) -> List[torch.FloatTensor]:
        """
        将运动输出按人分割

        Returns:
            List of [batch, T, 69] for each person
        """
        from einops import rearrange
        
        persons_motion = []
        for i in range(self.num_persons):
            # 获取第i个人的运动 [batch, 69, T, 1, 1]
            person_motion = self.motion_params[:, :, :, :, i:i + 1]

            # 重塑为 [batch, T, 69]
            person_motion = rearrange(person_motion, 'b c t 1 1 -> b t c')
            persons_motion.append(person_motion)

        return persons_motion

    def to(self, device) -> 'MotionVAEOutput':
        """
        将输出转移到指定设备

        Args:
            device: 目标设备

        Returns:
            新的MotionVAEOutput实例
        """
        motion_params = self.motion_params.to(device)
        latents = self.latents.to(device) if self.latents is not None else None
        return MotionVAEOutput(motion_params, latents, self.metadata.copy())

    def cpu(self) -> 'MotionVAEOutput':
        """
        将输出转移到CPU

        Returns:
            新的MotionVAEOutput实例
        """
        return self.to('cpu')

    def __repr__(self) -> str:
        return (f"MotionVAEOutput(batch_size={self.batch_size}, "
                f"num_persons={self.num_persons}, "
                f"num_frames={self.num_frames}, "
                f"motion_channels={self.motion_channels})")


def save_motion_params(
        motion_output: MotionVAEOutput,
        filepath: str,
        format: str = "pt"
) -> None:
    """
    保存运动参数到文件

    Args:
        motion_output: 运动VAE输出
        filepath: 文件路径
        format: 保存格式，支持 "pt" (PyTorch) 或 "npz" (NumPy)
    """
    import os
    import numpy as np

    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if format == "pt":
        # 保存为PyTorch文件
        save_dict = {
            'motion_params': motion_output.motion_params,
            'metadata': motion_output.metadata,
            'config': {
                'batch_size': motion_output.batch_size,
                'num_persons': motion_output.num_persons,
                'num_frames': motion_output.num_frames,
                'motion_channels': motion_output.motion_channels
            }
        }
        if motion_output.latents is not None:
            save_dict['latents'] = motion_output.latents

        torch.save(save_dict, filepath)

    elif format == "npz":
        # 保存为NumPy文件
        save_dict = {
            'motion_params': motion_output.motion_params.cpu().numpy(),
            'metadata': motion_output.metadata
        }
        if motion_output.latents is not None:
            save_dict['latents'] = motion_output.latents.cpu().numpy()

        np.savez_compressed(filepath, **save_dict)

    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats: 'pt', 'npz'")

    print(f"运动参数已保存至: {filepath}")