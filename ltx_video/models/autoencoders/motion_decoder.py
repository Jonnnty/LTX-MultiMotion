#!/usr/bin/env python3
"""
弹性运动解码器 - 三分支独立网络架构
支持渐进式增加深度，实现训练数据的过拟合
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict
from einops import rearrange
import numpy as np

try:
    from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
except ImportError:
    PixArtAlphaCombinedTimestepSizeEmbeddings = None

from ltx_video.models.autoencoders.conv_nd_factory import make_conv_nd
from ltx_video.models.autoencoders.pixel_norm import PixelNorm
from ltx_video.models.autoencoders.pixel_shuffle import PixelShuffleND


class ElasticMotionDecoder(nn.Module):
    """
    弹性运动解码器 - 三分支独立网络，支持深度扩展

    输入: [batch, latent_channels, T_compressed, 1, n_persons]
    输出: [batch, 69, T_target, 1, n_persons]  # trans(3) + root_orient(3) + pose_body(63)

    三个完全独立的网络分支，每个分支支持弹性增加深度
    """

    def __init__(
            self,
            latent_channels: int = 128,
            motion_channels_per_person: int = 69,
            base_channels: int = 128,
            temporal_downscale_factor: int = 8,
            spatial_downscale_factor: int = 1,
            dims: int = 3,
            norm_layer: str = "group_norm",
            causal: bool = True,
            timestep_conditioning: bool = False,
            spatial_padding_mode: str = "zeros",
            # 正则化参数
            dropout_rate: float = 0.1,
            use_weight_decay: bool = True,
            use_layer_norm: bool = False,
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            # 解码器块配置
            decoder_blocks: List[Tuple[str, int]] = [
                ("res_x", 2),
                ("compress_time", {"residual": True, "multiplier": 2}),
                ("compress_time", {"residual": True, "multiplier": 2}),
                ("compress_time", {"residual": True, "multiplier": 2}),
                ("res_x", 2),  # 这第二个res_x块将被弹性化
            ],
            # 弹性参数
            max_res_layers: int = 16,  # 每个弹性块最大层数
            initial_res_layers: int = 2,  # 初始层数
            use_elastic_depth: bool = True,  # 是否启用弹性深度
            **kwargs
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.motion_channels_per_person = motion_channels_per_person
        self.temporal_downscale_factor = temporal_downscale_factor
        self.spatial_downscale_factor = spatial_downscale_factor
        self.dims = dims
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning

        # 正则化参数
        self.dropout_rate = dropout_rate
        self.use_weight_decay = use_weight_decay
        self.use_layer_norm = use_layer_norm
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # 弹性参数
        self.max_res_layers = max_res_layers
        self.initial_res_layers = initial_res_layers
        self.use_elastic_depth = use_elastic_depth

        # 标记这是一个运动VAE
        self.is_motion_vae = True

        # 缩放因子
        self.scaling_factor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # 解码器配置
        self.decoder_blocks_desc = decoder_blocks

        # 深度历史记录
        self.depth_history = []
        self.loss_history = []
        self.current_depth = initial_res_layers  # 当前每个分支的深度

        # ========== 三个完全独立的弹性解码器 ==========
        # trans解码器 - 输出3维
        self.trans_decoder = ElasticMotionOnlyDecoder(
            dims=dims,
            in_channels=latent_channels,
            motion_channels_per_person=3,  # 输出3维平移
            blocks=decoder_blocks,
            base_channels=base_channels,
            norm_layer=norm_layer,
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_stochastic_depth=use_stochastic_depth,
            stochastic_depth_rate=stochastic_depth_rate,
            max_res_layers=max_res_layers,
            initial_res_layers=initial_res_layers,
            use_elastic_depth=use_elastic_depth,
        )

        # root_orient解码器 - 输出3维
        self.root_decoder = ElasticMotionOnlyDecoder(
            dims=dims,
            in_channels=latent_channels,
            motion_channels_per_person=3,  # 输出3维根方向
            blocks=decoder_blocks,
            base_channels=base_channels,
            norm_layer=norm_layer,
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_stochastic_depth=use_stochastic_depth,
            stochastic_depth_rate=stochastic_depth_rate,
            max_res_layers=max_res_layers,
            initial_res_layers=initial_res_layers,
            use_elastic_depth=use_elastic_depth,
        )

        # pose_body解码器 - 输出63维
        self.pose_decoder = ElasticMotionOnlyDecoder(
            dims=dims,
            in_channels=latent_channels,
            motion_channels_per_person=63,  # 输出63维身体姿势
            blocks=decoder_blocks,
            base_channels=base_channels,
            norm_layer=norm_layer,
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_stochastic_depth=use_stochastic_depth,
            stochastic_depth_rate=stochastic_depth_rate,
            max_res_layers=max_res_layers,
            initial_res_layers=initial_res_layers,
            use_elastic_depth=use_elastic_depth,
        )
        # =========================================

        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"弹性运动解码器初始化完成:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  初始深度: {self.current_depth}层/分支")
        print(f"  最大深度: {max_res_layers}层/分支")

    def forward(
            self,
            latents: torch.FloatTensor,
            target_frames: int,
            timestep: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            **kwargs
    ) -> Union[torch.FloatTensor, dict]:
        """
        解码latents为运动参数 - 三分支独立网络

        Args:
            latents: 潜在表示 [batch, latent_channels, T_compressed, 1, n_persons]
            target_frames: 目标帧数
            timestep: 可选的时间步条件
            return_dict: 是否返回字典格式

        Returns:
            运动参数 [batch, 69, target_frames, 1, n_persons]
        """
        batch_size, channels, T_compressed, H, W = latents.shape
        n_persons = W

        # 检查数据类型
        if hasattr(self.trans_decoder.conv_in, 'weight'):
            expected_dtype = self.trans_decoder.conv_in.weight.dtype
            if latents.dtype != expected_dtype:
                latents = latents.to(expected_dtype)

        # 验证输入形状
        assert H == 1, f"高度维度应为1，当前为{H}"
        assert n_persons > 0, f"宽度维度（人数）应大于0，当前为{W}"
        assert channels == self.latent_channels, \
            f"输入通道数{channels} != 预期通道数{self.latent_channels}"

        # 计算目标形状
        target_shape = (batch_size, self.motion_channels_per_person,
                        target_frames, 1, n_persons)

        # ========== 三个分支分别前向传播 ==========
        # 1. trans解码
        trans_output = self.trans_decoder(
            latents,
            target_shape=(batch_size, 3, target_frames, 1, n_persons),
            timestep=timestep
        )
        
        # 2. root_orient解码
        root_output = self.root_decoder(
            latents,
            target_shape=(batch_size, 3, target_frames, 1, n_persons),
            timestep=timestep
        )
        
        # 3. pose_body解码
        pose_output = self.pose_decoder(
            latents,
            target_shape=(batch_size, 63, target_frames, 1, n_persons),
            timestep=timestep
        )
        
        # 4. 拼接三个部分
        motion = torch.cat([trans_output, root_output, pose_output], dim=1)
        # =========================================

        # 验证输出形状
        assert motion.shape == target_shape, \
            f"输出形状{motion.shape} != 目标形状{target_shape}"

        if return_dict:
            return {
                "motion_params": motion,
                "latents": latents,
                "target_frames": target_frames,
                "num_persons": n_persons,
                "trans_output": trans_output,
                "root_output": root_output,
                "pose_output": pose_output,
                "current_depth": self.get_current_depth(),
            }
        else:
            return motion

    # ========== 弹性深度控制方法 ==========
    
    def add_depth(self, num_layers: int = 1) -> bool:
        """
        为所有三个分支增加深度
        
        Args:
            num_layers: 要增加的层数
            
        Returns:
            bool: 是否成功增加深度
        """
        if not self.use_elastic_depth:
            print("警告: 弹性深度功能未启用")
            return False
        
        success = True
        
        # 为每个分支增加深度
        trans_success = self.trans_decoder.add_res_layers(num_layers)
        root_success = self.root_decoder.add_res_layers(num_layers)
        pose_success = self.pose_decoder.add_res_layers(num_layers)
        
        # 检查是否所有分支都成功
        success = trans_success and root_success and pose_success
        
        if success:
            # 更新当前深度
            self.current_depth += num_layers
            self.depth_history.append(self.current_depth)
            print(f"弹性增加深度成功: 当前深度 {self.current_depth} 层/分支")
        else:
            print(f"增加深度失败，可能已达到最大深度 {self.max_res_layers}")
        
        return success
    
    def get_current_depth(self) -> int:
        """获取当前深度（三个分支的平均深度）"""
        depths = [
            self.trans_decoder.get_current_depth(),
            self.root_decoder.get_current_depth(),
            self.pose_decoder.get_current_depth()
        ]
        return int(sum(depths) / len(depths))
    
    def get_max_depth(self) -> int:
        """获取最大允许深度"""
        return self.max_res_layers
    
    def set_depth(self, target_depth: int) -> bool:
        """
        设置目标深度
        
        Args:
            target_depth: 目标深度
            
        Returns:
            bool: 是否成功设置
        """
        current_depth = self.get_current_depth()
        
        if target_depth == current_depth:
            return True
        
        if target_depth > self.max_res_layers:
            print(f"目标深度 {target_depth} 超过最大允许深度 {self.max_res_layers}")
            return False
        
        if target_depth < 1:
            print(f"目标深度 {target_depth} 必须大于0")
            return False
        
        # 计算需要增加的层数
        layers_to_add = target_depth - current_depth
        
        if layers_to_add > 0:
            return self.add_depth(layers_to_add)
        else:
            # 当前不支持减少深度（但可以后续添加此功能）
            print("当前不支持减少深度")
            return False
    
    def is_at_max_depth(self) -> bool:
        """检查是否已达到最大深度"""
        return self.get_current_depth() >= self.max_res_layers
    
    def record_loss(self, loss_value: float):
        """记录损失值"""
        self.loss_history.append(loss_value)
    
    def get_depth_history(self) -> List[int]:
        """获取深度历史"""
        return self.depth_history.copy()
    
    def get_loss_history(self) -> List[float]:
        """获取损失历史"""
        return self.loss_history.copy()
    
    def should_increase_depth(self, patience: int = 3, min_improvement: float = 0.01) -> bool:
        """
        判断是否应该增加深度
        
        Args:
            patience: 耐心值（连续多少个epoch无显著改善）
            min_improvement: 最小改善阈值
            
        Returns:
            bool: 是否应该增加深度
        """
        if len(self.loss_history) < patience + 1:
            return False
        
        # 检查最近patience个epoch的损失改善情况
        recent_losses = self.loss_history[-(patience + 1):]
        improvements = []
        
        for i in range(1, len(recent_losses)):
            improvement = recent_losses[i-1] - recent_losses[i]
            improvements.append(improvement)
        
        avg_improvement = sum(improvements) / len(improvements)
        
        # 如果平均改善小于阈值，考虑增加深度
        should_increase = avg_improvement < min_improvement
        
        if should_increase:
            print(f"建议增加深度: 最近{patience}个epoch平均改善 {avg_improvement:.6f} < 阈值 {min_improvement}")
        
        return should_increase

    def decode(
            self,
            latents: torch.FloatTensor,
            target_shape: Tuple[int, int, int, int, int],
            timestep: Optional[torch.Tensor] = None,
            return_dict: bool = False,
            **kwargs
    ) -> torch.FloatTensor:
        """
        解码接口，兼容vae_encode.py中的调用方式
        """
        batch_size, channels, target_frames, H, W = target_shape
        assert H == 1, "目标高度应为1"

        return self.forward(
            latents=latents,
            target_frames=target_frames,
            timestep=timestep,
            return_dict=return_dict
        )

    def split_by_person(self, motion_output: torch.FloatTensor) -> List[torch.FloatTensor]:
        """
        将运动输出按人分割
        """
        batch_size, channels, T, H, n_persons = motion_output.shape

        persons_motion = []
        for i in range(n_persons):
            person_motion = motion_output[:, :, :, :, i:i + 1]
            person_motion = rearrange(person_motion, 'b c t 1 1 -> b t c')
            persons_motion.append(person_motion)

        return persons_motion

    def get_optimizer_params(self, learning_rate: float = 1e-4):
        """获取优化器参数"""
        if self.use_weight_decay:
            weight_decay = 1e-3
        else:
            weight_decay = 1e-5

        return [
            {
                "params": self.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay
            }
        ]

    @property
    def config(self):
        """返回配置信息"""
        import types
        return types.SimpleNamespace(
            _class_name="ElasticMotionDecoder",
            latent_channels=self.latent_channels,
            motion_channels_per_person=self.motion_channels_per_person,
            temporal_downscale_factor=self.temporal_downscale_factor,
            spatial_downscale_factor=self.spatial_downscale_factor,
            dims=self.dims,
            causal=self.causal,
            timestep_conditioning=self.timestep_conditioning,
            dropout_rate=self.dropout_rate,
            use_weight_decay=self.use_weight_decay,
            use_layer_norm=self.use_layer_norm,
            use_stochastic_depth=self.use_stochastic_depth,
            stochastic_depth_rate=self.stochastic_depth_rate,
            decoder_blocks=self.decoder_blocks_desc,
            # 弹性参数
            max_res_layers=self.max_res_layers,
            initial_res_layers=self.initial_res_layers,
            use_elastic_depth=self.use_elastic_depth,
            scaling_factor=self.scaling_factor.item()
        )

    def to_json_string(self) -> str:
        """返回JSON格式的配置字符串"""
        import json
        config_dict = {
            "_class_name": "ElasticMotionDecoder",
            "latent_channels": self.latent_channels,
            "motion_channels_per_person": self.motion_channels_per_person,
            "temporal_downscale_factor": self.temporal_downscale_factor,
            "spatial_downscale_factor": self.spatial_downscale_factor,
            "dims": self.dims,
            "causal": self.causal,
            "timestep_conditioning": self.timestep_conditioning,
            "dropout_rate": self.dropout_rate,
            "use_weight_decay": self.use_weight_decay,
            "use_layer_norm": self.use_layer_norm,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "decoder_blocks": self.decoder_blocks_desc,
            # 弹性参数
            "max_res_layers": self.max_res_layers,
            "initial_res_layers": self.initial_res_layers,
            "use_elastic_depth": self.use_elastic_depth,
            "scaling_factor": float(self.scaling_factor.item())
        }
        return json.dumps(config_dict, indent=2)

    @classmethod
    def from_config(cls, config: dict):
        """从配置创建实例"""
        return cls(**config)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_path: str,
            **kwargs
    ):
        """从预训练权重加载"""
        import os
        from pathlib import Path

        pretrained_path = Path(pretrained_path)

        if pretrained_path.is_dir():
            config_path = pretrained_path / "config.json"
            weights_path = pretrained_path / "motion_decoder.pth"

            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")

            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            model = cls.from_config(config)

            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)

        elif pretrained_path.is_file():
            if str(pretrained_path).endswith('.pth') or str(pretrained_path).endswith('.pt'):
                if 'config' not in kwargs:
                    raise ValueError("从权重文件加载时需要提供config参数")

                model = cls.from_config(kwargs['config'])
                state_dict = torch.load(pretrained_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                raise ValueError(f"不支持的文件格式: {pretrained_path}")
        else:
            raise FileNotFoundError(f"路径不存在: {pretrained_path}")

        return model

    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        from pathlib import Path

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            f.write(self.to_json_string())

        weights_path = save_path / "motion_decoder.pth"
        torch.save(self.state_dict(), weights_path)
        
        # 保存深度历史
        if self.depth_history:
            depth_path = save_path / "depth_history.txt"
            with open(depth_path, 'w') as f:
                for depth in self.depth_history:
                    f.write(f"{depth}\n")
        
        print(f"模型已保存到: {save_path}")


class ElasticMotionOnlyDecoder(nn.Module):
    """
    弹性运动解码器的核心部分，支持深度扩展
    """

    def __init__(
            self,
            dims: int = 3,
            in_channels: int = 128,
            motion_channels_per_person: int = 69,
            blocks: List[Tuple[str, int]] = [
                ("res_x", 2),
                ("compress_time", {"residual": True, "multiplier": 2}),
                ("compress_time", {"residual": True, "multiplier": 2}),
                ("compress_time", {"residual": True, "multiplier": 2}),
                ("res_x", 2),  # 这第二个res_x块将被弹性化
            ],
            base_channels: int = 128,
            norm_layer: str = "group_norm",
            causal: bool = True,
            timestep_conditioning: bool = False,
            spatial_padding_mode: str = "zeros",
            # 新增正则化参数
            dropout_rate: float = 0.1,
            use_layer_norm: bool = False,
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            # 弹性参数
            max_res_layers: int = 16,
            initial_res_layers: int = 2,
            use_elastic_depth: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.in_channels = in_channels
        self.motion_channels_per_person = motion_channels_per_person
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self.blocks_desc = blocks

        # 正则化参数
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # 弹性参数
        self.max_res_layers = max_res_layers
        self.initial_res_layers = initial_res_layers
        self.use_elastic_depth = use_elastic_depth
        self.current_res_layers = initial_res_layers

        # 标记哪些块是弹性的
        self.elastic_block_indices = []
        self.elastic_blocks = []

        # conv_in
        self.conv_in = make_conv_nd(
            dims,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        # 在conv_in后添加Dropout
        if dropout_rate > 0:
            self.dropout_after_conv_in = nn.Dropout3d(p=dropout_rate)
        else:
            self.dropout_after_conv_in = nn.Identity()

        # 创建上采样块
        self.up_blocks = nn.ModuleList([])
        output_channel = base_channels

        # 逆序遍历块配置
        for block_idx, (block_name, block_params) in enumerate(list(reversed(blocks))):
            input_channel = output_channel

            if isinstance(block_params, int):
                block_params = {"num_layers": block_params}

            # 添加随机深度
            if self.use_stochastic_depth and block_idx > 0:
                survival_prob = 1.0 - stochastic_depth_rate * (block_idx / len(blocks))
                block_params["survival_prob"] = survival_prob

            if block_name == "res_x":
                # 如果是第二个res_x块（索引为0因为列表是逆序的），则使用弹性版本
                # blocks中的第二个res_x在逆序后是第一个
                is_elastic = (block_idx == 0 and self.use_elastic_depth)
                
                if is_elastic:
                    # 使用弹性块
                    block = ElasticUNetMidBlock3D(
                        dims=dims,
                        in_channels=input_channel,
                        dropout=dropout_rate,
                        initial_layers=initial_res_layers,
                        max_layers=max_res_layers,
                        resnet_eps=1e-6,
                        resnet_groups=16,
                        norm_layer=norm_layer if not use_layer_norm else "layer_norm",
                        inject_noise=block_params.get("inject_noise", False),
                        timestep_conditioning=timestep_conditioning,
                        spatial_padding_mode=spatial_padding_mode,
                        use_elastic_depth=use_elastic_depth,
                    )
                    self.elastic_block_indices.append(len(self.up_blocks))
                    self.elastic_blocks.append(block)
                else:
                    # 使用普通块
                    from ltx_video.models.autoencoders.causal_video_autoencoder import UNetMidBlock3D
                    block = UNetMidBlock3D(
                        dims=dims,
                        in_channels=input_channel,
                        dropout=dropout_rate,
                        num_layers=block_params.get("num_layers", 2),
                        resnet_eps=1e-6,
                        resnet_groups=16,
                        norm_layer=norm_layer if not use_layer_norm else "layer_norm",
                        inject_noise=block_params.get("inject_noise", False),
                        timestep_conditioning=timestep_conditioning,
                        spatial_padding_mode=spatial_padding_mode,
                    )
            elif block_name == "compress_time":
                multiplier = block_params.get("multiplier", 2)
                output_channel = output_channel // multiplier
                block = DepthToSpaceUpsample(
                    dims=dims,
                    in_channels=input_channel,
                    stride=(2, 1, 1),
                    residual=block_params.get("residual", False),
                    out_channels_reduction_factor=multiplier,
                    spatial_padding_mode=spatial_padding_mode,
                    dropout_rate=dropout_rate,
                )
            else:
                raise ValueError(f"未知的块类型: {block_name}")

            self.up_blocks.append(block)

        # norm_out
        if use_layer_norm:
            from ltx_video.models.autoencoders.causal_video_autoencoder import LayerNorm
            self.conv_norm_out = LayerNorm(output_channel, eps=1e-6)
        elif norm_layer == "group_norm":
            if output_channel % 32 != 0:
                possible_groups = [16, 8, 4, 2, 1]
                selected_group = 1
                for g in possible_groups:
                    if output_channel % g == 0:
                        selected_group = g
                        break
                num_groups = selected_group
            else:
                num_groups = 32

            self.conv_norm_out = nn.GroupNorm(
                num_channels=output_channel, num_groups=num_groups, eps=1e-6
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        elif norm_layer == "layer_norm":
            from ltx_video.models.autoencoders.causal_video_autoencoder import LayerNorm
            self.conv_norm_out = LayerNorm(output_channel, eps=1e-6)

        self.conv_act = nn.SiLU()

        # 在激活后添加Dropout
        if dropout_rate > 0:
            self.dropout_after_act = nn.Dropout3d(p=dropout_rate)
        else:
            self.dropout_after_act = nn.Identity()

        # conv_out - 输出运动参数
        self.conv_out = make_conv_nd(
            dims,
            output_channel,
            self.motion_channels_per_person,
            3,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        # 时间步条件
        if timestep_conditioning:
            assert PixArtAlphaCombinedTimestepSizeEmbeddings is not None, \
                "需要安装diffusers以使用时间步条件"
            self.timestep_scale_multiplier = nn.Parameter(
                torch.tensor(1000.0, dtype=torch.float32)
            )
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                output_channel * 2, 0
            )
            self.last_scale_shift_table = nn.Parameter(
                torch.randn(2, output_channel) / output_channel ** 0.5
            )

        self.gradient_checkpointing = False

    # ========== 弹性深度控制方法 ==========
    
    def add_res_layers(self, num_layers: int = 1) -> bool:
        """
        为弹性块增加层数
        
        Args:
            num_layers: 要增加的层数
            
        Returns:
            bool: 是否成功增加
        """
        if not self.use_elastic_depth:
            print("警告: 弹性深度功能未启用")
            return False
        
        success = True
        
        # 为所有弹性块增加层数
        for elastic_block in self.elastic_blocks:
            for _ in range(num_layers):
                if not elastic_block.add_layer():
                    success = False
                    break
        
        if success:
            self.current_res_layers += num_layers
            print(f"增加层数成功: 当前 {self.current_res_layers} 层")
        else:
            print(f"增加层数失败，可能已达到最大层数 {self.max_res_layers}")
        
        return success
    
    def get_current_depth(self) -> int:
        """获取当前深度"""
        return self.current_res_layers
    
    def get_max_depth(self) -> int:
        """获取最大深度"""
        return self.max_res_layers
    
    def set_depth(self, target_depth: int) -> bool:
        """
        设置目标深度
        
        Args:
            target_depth: 目标深度
            
        Returns:
            bool: 是否成功设置
        """
        if target_depth == self.current_res_layers:
            return True
        
        if target_depth > self.max_res_layers:
            print(f"目标深度 {target_depth} 超过最大允许深度 {self.max_res_layers}")
            return False
        
        if target_depth < 1:
            print(f"目标深度 {target_depth} 必须大于0")
            return False
        
        # 计算需要增加的层数
        layers_to_add = target_depth - self.current_res_layers
        
        if layers_to_add > 0:
            return self.add_res_layers(layers_to_add)
        else:
            # 当前不支持减少深度
            print("当前不支持减少深度")
            return False
    
    def is_at_max_depth(self) -> bool:
        """检查是否已达到最大深度"""
        return self.current_res_layers >= self.max_res_layers

    def forward(
            self,
            sample: torch.FloatTensor,
            target_shape: Tuple[int, int, int, int, int],
            timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = sample.shape[0]

        # 通过conv_in
        sample = self.conv_in(sample, causal=self.causal)

        # 应用Dropout
        if self.dropout_rate > 0 and self.training:
            sample = self.dropout_after_conv_in(sample)

        if self.timestep_conditioning:
            assert timestep is not None, "时间步条件需要提供timestep参数"
            
            # 首先确保 timestep 有正确的形状
            # 如果 timestep 是 1D [B] 或 5D [B, 1, 1, 1, 1]，确保它是 5D
            if timestep.dim() == 1:  # [batch_size]
                timestep = timestep.view(-1, 1, 1, 1, 1)
            elif timestep.dim() == 5:  # [batch_size, 1, 1, 1, 1]
                # 确保第二维是1
                if timestep.shape[1] != 1:
                    timestep = timestep.view(-1, 1, 1, 1, 1)
            else:
                # 扩展维度到5D
                while timestep.dim() < 5:
                    timestep = timestep.unsqueeze(-1)
                if timestep.shape[1] != 1:
                    timestep = timestep.view(timestep.shape[0], 1, 1, 1, 1)
            
            scaled_timestep = timestep * self.timestep_scale_multiplier

        # 通过上采样块
        for i, up_block in enumerate(self.up_blocks):
            if self.gradient_checkpointing and self.training:
                if self.timestep_conditioning and hasattr(up_block,
                                                          'timestep_conditioning') and up_block.timestep_conditioning:
                    sample = torch.utils.checkpoint.checkpoint(
                        up_block, sample, self.causal, scaled_timestep,
                        use_reentrant=False
                    )
                else:
                    sample = torch.utils.checkpoint.checkpoint(
                        up_block, sample, self.causal,
                        use_reentrant=False
                    )
            else:
                if self.timestep_conditioning and hasattr(up_block,
                                                          'timestep_conditioning') and up_block.timestep_conditioning:
                    sample = up_block(sample, causal=self.causal, timestep=scaled_timestep)
                else:
                    sample = up_block(sample, causal=self.causal)

        # norm_out
        sample = self.conv_norm_out(sample)

        # 时间步条件（在norm_out后添加）
        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=sample.shape[0],
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(
                batch_size, embedded_timestep.shape[-1], 1, 1, 1
            )
            ada_values = self.last_scale_shift_table[
                             None, ..., None, None, None
                         ] + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        # 激活和输出
        sample = self.conv_act(sample)

        # 应用Dropout在激活后
        if self.dropout_rate > 0 and self.training:
            sample = self.dropout_after_act(sample)

        sample = self.conv_out(sample, causal=self.causal)

        return sample


class ElasticUNetMidBlock3D(nn.Module):
    """
    弹性UNet中间块，支持动态增加残差层层数
    修复：ResnetBlock3D参数匹配问题和时间步嵌入问题
    """
    def __init__(
        self,
        dims,
        in_channels,
        dropout=0.0,
        initial_layers=2,
        max_layers=16,
        resnet_eps=1e-6,
        resnet_groups=16,
        norm_layer="group_norm",
        inject_noise=False,
        timestep_conditioning=False,
        spatial_padding_mode="zeros",
        use_elastic_depth=True,
    ):
        super().__init__()
        
        from ltx_video.models.autoencoders.causal_video_autoencoder import ResnetBlock3D
        
        self.in_channels = in_channels
        self.dims = dims
        self.initial_layers = initial_layers
        self.max_layers = max_layers
        self.current_layers = initial_layers
        self.timestep_conditioning = timestep_conditioning
        self.use_elastic_depth = use_elastic_depth
        
        # 时间步嵌入器（照搬标准UNetMidBlock3D）
        if timestep_conditioning:
            assert PixArtAlphaCombinedTimestepSizeEmbeddings is not None, \
                "需要安装diffusers以使用时间步条件"
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                in_channels * 4, 0
            )
        
        # 创建残差块列表
        self.resnets = nn.ModuleList()
        
        # 初始化指定数量的块
        for i in range(initial_layers):
            resnet = ResnetBlock3D(
                dims=dims,
                in_channels=in_channels,
                out_channels=in_channels,
                dropout=dropout,
                groups=resnet_groups,  # 注意：ResnetBlock3D使用groups参数
                eps=resnet_eps,
                norm_layer=norm_layer,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,  # 布尔值，不是通道数
                spatial_padding_mode=spatial_padding_mode,
            )
            self.resnets.append(resnet)
        
        # 激活状态标记
        self.active_layers = list(range(initial_layers))
        
    def add_layer(self) -> bool:
        """
        添加一个新的残差层
        
        Returns:
            bool: 是否成功添加
        """
        if not self.use_elastic_depth:
            return False
        
        if len(self.resnets) >= self.max_layers:
            print(f"已达到最大层数 {self.max_layers}")
            return False
        
        from ltx_video.models.autoencoders.causal_video_autoencoder import ResnetBlock3D
        
        # 创建新块
        new_resnet = ResnetBlock3D(
            dims=self.dims,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            dropout=0.0,  # 新块暂时不加dropout
            groups=16,  # 使用正确的参数名
            eps=1e-6,
            norm_layer="group_norm",
            inject_noise=False,
            timestep_conditioning=self.timestep_conditioning,
            spatial_padding_mode="zeros",
        )
        
        # 添加到列表
        self.resnets.append(new_resnet)
        
        # 更新激活层列表
        self.active_layers.append(len(self.resnets) - 1)
        self.current_layers = len(self.active_layers)
        
        # 初始化新块的权重（保持旧块不变）
        for name, param in new_resnet.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        print(f"弹性块添加新层: 当前 {self.current_layers}/{self.max_layers} 层")
        return True
    
    def get_current_depth(self) -> int:
        """获取当前深度"""
        return self.current_layers

    def forward(self, hidden_states, causal=True, timestep=None):
        """前向传播，支持弹性深度，照搬标准UNetMidBlock3D的时间步处理"""
        # 时间步嵌入处理（照搬标准代码）
        timestep_embed = None
        if self.timestep_conditioning:
            assert timestep is not None, "时间步条件需要提供timestep参数"
            batch_size = hidden_states.shape[0]
            
            # 使用time_embedder处理时间步
            timestep_embed = self.time_embedder(
                timestep=timestep.flatten(),  # 展平时间步
                resolution=None,
                aspect_ratio=None,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            # 将嵌入向量reshape为5D: [batch_size, channels, 1, 1, 1]
            timestep_embed = timestep_embed.view(
                batch_size, timestep_embed.shape[-1], 1, 1, 1
            )
        
        output_states = hidden_states
        
        # 使用所有激活的层
        for i, resnet in enumerate(self.resnets):
            if i in self.active_layers:
                # 直接使用残差连接，不再使用alpha参数
                output_states = resnet(output_states, causal=causal, timestep=timestep_embed)
        
        return output_states


class DepthToSpaceUpsample(nn.Module):
    """与causal_video_autoencoder.py中的DepthToSpaceUpsample相同"""

    def __init__(
            self,
            dims,
            in_channels,
            stride,
            residual=False,
            out_channels_reduction_factor=1,
            spatial_padding_mode="zeros",
            dropout_rate=0.0,
    ):
        super().__init__()
        import numpy as np

        self.stride = stride
        self.out_channels = (
                np.prod(stride) * in_channels // out_channels_reduction_factor
        )

        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.pixel_shuffle = PixelShuffleND(dims=dims, upscale_factors=stride)
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

        if dropout_rate > 0:
            self.dropout = nn.Dropout3d(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x, causal: bool = True):
        import numpy as np

        if self.residual:
            x_in = self.pixel_shuffle(x)
            num_repeat = np.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]

        x = self.conv(x, causal=causal)
        x = self.dropout(x)
        x = self.pixel_shuffle(x)

        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]

        if self.residual:
            x = x + x_in

        return x


def create_elastic_motion_decoder_config(
        latent_channels: int = 128,
        motion_channels_per_person: int = 69,
        temporal_downscale_factor: int = 8,
        spatial_downscale_factor: int = 1,
        base_channels: int = 128,
        causal: bool = True,
        timestep_conditioning: bool = False,
        # 正则化参数
        dropout_rate: float = 0.1,
        use_weight_decay: bool = True,
        use_layer_norm: bool = False,
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        # 弹性参数
        max_res_layers: int = 16,
        initial_res_layers: int = 2,
        use_elastic_depth: bool = True,
        decoder_blocks: List[Tuple[str, int]] = [
            ("res_x", 2),
            ("compress_time", {"residual": True, "multiplier": 2}),
            ("compress_time", {"residual": True, "multiplier": 2}),
            ("compress_time", {"residual": True, "multiplier": 2}),
            ("res_x", 2),  # 这第二个res_x块将被弹性化
        ]
) -> dict:
    """
    创建弹性运动解码器配置

    Returns:
        配置字典
    """
    return {
        "_class_name": "ElasticMotionDecoder",
        "latent_channels": latent_channels,
        "motion_channels_per_person": motion_channels_per_person,
        "temporal_downscale_factor": temporal_downscale_factor,
        "spatial_downscale_factor": spatial_downscale_factor,
        "base_channels": base_channels,
        "dims": 3,
        "norm_layer": "group_norm",
        "causal": causal,
        "timestep_conditioning": timestep_conditioning,
        "spatial_padding_mode": "zeros",
        "dropout_rate": dropout_rate,
        "use_weight_decay": use_weight_decay,
        "use_layer_norm": use_layer_norm,
        "use_stochastic_depth": use_stochastic_depth,
        "stochastic_depth_rate": stochastic_depth_rate,
        # 弹性参数
        "max_res_layers": max_res_layers,
        "initial_res_layers": initial_res_layers,
        "use_elastic_depth": use_elastic_depth,
        "decoder_blocks": decoder_blocks
    }


class ProgressiveTrainer:
    """
    渐进式训练器，自动管理弹性深度增加
    """
    def __init__(
        self,
        model: ElasticMotionDecoder,
        train_loader,
        val_loader=None,
        learning_rate: float = 1e-4,
        patience: int = 3,
        min_improvement: float = 0.01,
        target_loss: float = 1e-4,
        max_epochs_per_depth: int = 10,
        warmup_epochs: int = 5,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_improvement = min_improvement
        self.target_loss = target_loss
        self.max_epochs_per_depth = max_epochs_per_depth
        self.warmup_epochs = warmup_epochs
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.get_optimizer_params(learning_rate)[0]['params'],
            lr=learning_rate
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_loss_history = []
        self.val_loss_history = [] if val_loader else []
        self.epoch_counter = 0
        
    def train_until_target(self, max_total_epochs: int = 100):
        """
        训练直到达到目标损失或最大epoch数
        
        Args:
            max_total_epochs: 最大总epoch数
            
        Returns:
            bool: 是否达到目标损失
        """
        print("开始渐进式训练...")
        print(f"初始深度: {self.model.get_current_depth()}层/分支")
        print(f"目标损失: {self.target_loss}")
        
        stagnation_counter = 0
        best_loss = float('inf')
        
        while self.epoch_counter < max_total_epochs:
            epoch = self.epoch_counter + 1
            print(f"\n=== Epoch {epoch}/{max_total_epochs} ===")
            
            # 训练一个epoch
            train_loss = self._train_epoch()
            self.train_loss_history.append(train_loss)
            self.model.record_loss(train_loss)
            
            print(f"训练损失: {train_loss:.6f}")
            
            # 验证（如果有验证集）
            if self.val_loader is not None:
                val_loss = self._validate()
                self.val_loss_history.append(val_loss)
                print(f"验证损失: {val_loss:.6f}")
                current_loss = val_loss
            else:
                current_loss = train_loss
            
            # 检查是否达到目标
            if current_loss <= self.target_loss:
                print(f"\n🎉 达到目标损失 {self.target_loss}!")
                print(f"最终深度: {self.model.get_current_depth()}层/分支")
                return True
            
            # 检查是否需要增加深度（warmup后）
            if epoch > self.warmup_epochs:
                should_increase = False
                
                # 方法1: 使用模型的自动判断
                should_increase = self.model.should_increase_depth(
                    patience=self.patience,
                    min_improvement=self.min_improvement
                )
                
                # 方法2: 简单判断 - 长时间无改善
                if not should_increase and len(self.train_loss_history) >= self.patience + 1:
                    recent_losses = self.train_loss_history[-(self.patience + 1):]
                    avg_improvement = sum(recent_losses[i-1] - recent_losses[i] 
                                        for i in range(1, len(recent_losses))) / (len(recent_losses) - 1)
                    if avg_improvement < self.min_improvement * 0.5:
                        should_increase = True
                        print(f"长时间无改善，建议增加深度")
                
                # 增加深度
                if should_increase:
                    if self.model.is_at_max_depth():
                        print(f"已达到最大深度 {self.model.get_max_depth()}，无法继续增加")
                    else:
                        print("正在增加深度...")
                        success = self.model.add_depth(num_layers=1)
                        if success:
                            print(f"深度增加到 {self.model.get_current_depth()}层/分支")
                            stagnation_counter = 0
                            best_loss = float('inf')
                            
                            # 调整学习率（可选）
                            self._adjust_learning_rate(0.8)  # 暂时降低学习率
                        else:
                            print("增加深度失败")
            
            # 更新最佳损失和停滞计数器
            if current_loss < best_loss - self.min_improvement:
                best_loss = current_loss
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # 长时间停滞但无法增加深度时，考虑调整学习率
            if stagnation_counter >= self.patience * 2 and self.model.is_at_max_depth():
                print("长时间停滞且已达最大深度，调整学习率...")
                self._adjust_learning_rate(0.5)
                stagnation_counter = 0
            
            self.epoch_counter += 1
        
        print(f"\n⚠️ 达到最大epoch数 {max_total_epochs}，停止训练")
        print(f"最终深度: {self.model.get_current_depth()}层/分支")
        print(f"最终训练损失: {self.train_loss_history[-1]:.6f}")
        return False
    
    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 前向传播
            latents = batch['latents']
            target = batch['motion_params']
            target_frames = batch['target_frames']
            
            output = self.model(
                latents=latents,
                target_frames=target_frames,
                timestep=batch.get('timestep', None)
            )
            
            # 计算损失
            loss = self.criterion(output['motion_params'], target)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def _validate(self):
        """验证"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                latents = batch['latents']
                target = batch['motion_params']
                target_frames = batch['target_frames']
                
                output = self.model(
                    latents=latents,
                    target_frames=target_frames,
                    timestep=batch.get('timestep', None)
                )
                
                loss = self.criterion(output['motion_params'], target)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def _adjust_learning_rate(self, factor: float):
        """调整学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        print(f"学习率调整为: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch_counter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'depth_history': self.model.get_depth_history(),
            'current_depth': self.model.get_current_depth(),
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch_counter = checkpoint['epoch']
        self.train_loss_history = checkpoint['train_loss_history']
        self.val_loss_history = checkpoint['val_loss_history']
        print(f"检查点已加载: epoch {self.epoch_counter}, 深度 {checkpoint['current_depth']}")


def demo_elastic_motion_decoder():
    """演示弹性运动解码器"""

    # 创建配置
    config = create_elastic_motion_decoder_config(
        dropout_rate=0.1,
        use_weight_decay=True,
        use_layer_norm=False,
        use_stochastic_depth=True,
        stochastic_depth_rate=0.1,
        max_res_layers=32,  # 最大32层
        initial_res_layers=2,  # 初始2层
        use_elastic_depth=True
    )

    # 创建模型
    motion_decoder = ElasticMotionDecoder.from_config(config)

    # 测试输入
    batch_size = 1
    latent_channels = 128
    T_compressed = 16
    n_persons = 2

    latents = torch.randn(batch_size, latent_channels, T_compressed, 1, n_persons)

    # 目标帧数
    target_frames = 121

    # 初始前向传播
    motion_output = motion_decoder(
        latents=latents,
        target_frames=target_frames,
        timestep=torch.tensor([0.5, 0.5])
    )

    # 检查输出形状
    print(f"输出形状: {motion_output['motion_params'].shape}")
    print(f"应该为: [1, 69, 121, 1, 2]")
    print(f"当前深度: {motion_output['current_depth']}层/分支")
    
    # 获取三个分支的输出
    trans_out = motion_output['trans_output']
    root_out = motion_output['root_output']
    pose_out = motion_output['pose_output']
    
    print(f"\n三个分支的输出统计:")
    print(f"trans形状: {trans_out.shape}, 范围: [{trans_out.min():.3f}, {trans_out.max():.3f}]")
    print(f"root形状: {root_out.shape}, 范围: [{root_out.min():.3f}, {root_out.max():.3f}]")
    print(f"pose形状: {pose_out.shape}, 范围: [{pose_out.min():.3f}, {pose_out.max():.3f}]")
    
    # 测试增加深度
    print(f"\n=== 测试弹性增加深度 ===")
    print(f"增加深度前: {motion_decoder.get_current_depth()}层/分支")
    
    # 模拟一些训练损失
    for i in range(5):
        motion_decoder.record_loss(0.5 - i * 0.05)
    
    # 检查是否应该增加深度
    should_increase = motion_decoder.should_increase_depth(patience=3, min_improvement=0.01)
    print(f"应该增加深度: {should_increase}")
    
    # 手动增加深度
    if motion_decoder.add_depth(num_layers=1):
        print(f"增加深度后: {motion_decoder.get_current_depth()}层/分支")
        
        # 再次前向传播
        motion_output2 = motion_decoder(
            latents=latents,
            target_frames=target_frames,
            timestep=torch.tensor([0.5, 0.5])
        )
        print(f"增加深度后输出形状: {motion_output2['motion_params'].shape}")
    
    # 测试设置目标深度
    print(f"\n=== 测试设置目标深度 ===")
    target_depth = 4
    if motion_decoder.set_depth(target_depth):
        print(f"成功设置深度到: {motion_decoder.get_current_depth()}层/分支")
    else:
        print(f"设置深度失败")
    
    print(f"\n最大允许深度: {motion_decoder.get_max_depth()}层/分支")
    print(f"是否达到最大深度: {motion_decoder.is_at_max_depth()}")

    return motion_decoder


if __name__ == "__main__":
    demo_elastic_motion_decoder()