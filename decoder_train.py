#!/usr/bin/env python3
"""
弹性分支训练 - 可以指定训练root/trans/pose分支
支持断点续训功能
训练时只保存当前训练分支的权重
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import traceback
from scipy.interpolate import interp1d
import sys
import signal
import json
import math
import time

# 导入弹性运动解码器
from ltx_video.models.autoencoders.motion_decoder import ElasticMotionDecoder, create_elastic_motion_decoder_config


def interpolate_motion_to_target_frames(
        motion: np.ndarray,
        source_frames: int,
        target_frames: int,
        method: str = 'linear'
) -> np.ndarray:
    """
    将运动数据从source_frames插值到target_frames
    """
    if source_frames == target_frames:
        return motion.copy()

    source_time = np.linspace(0, 1, source_frames)
    target_time = np.linspace(0, 1, target_frames)

    interpolated = np.zeros((target_frames, motion.shape[1]), dtype=motion.dtype)

    for d in range(motion.shape[1]):
        f = interp1d(source_time, motion[:, d], kind='linear',
                     fill_value='extrapolate', bounds_error=False)
        interpolated[:, d] = f(target_time)

    return interpolated


class RootOnlyLoss(nn.Module):
    """只训练root分支的损失函数"""

    def __init__(self):
        super().__init__()
        self.position_loss = nn.SmoothL1Loss()

    def compute_velocity(self, motion):
        """计算速度（一阶差分）"""
        velocity = motion[..., 1:, :] - motion[..., :-1, :]
        return velocity

    def compute_acceleration(self, motion):
        """计算加速度（二阶差分）"""
        velocity = self.compute_velocity(motion)
        acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]
        return acceleration

    def forward(self, pred_root, target_root):
        # 1. 位置损失
        pos_loss = self.position_loss(pred_root, target_root)

        # 2. 速度损失
        pred_vel = self.compute_velocity(pred_root)
        target_vel = self.compute_velocity(target_root)
        vel_loss = self.position_loss(pred_vel, target_vel)

        # 3. 加速度损失
        pred_acc = self.compute_acceleration(pred_root)
        target_acc = self.compute_acceleration(target_root)
        acc_loss = self.position_loss(pred_acc, target_acc)

        # 总损失 - 统一权重
        total_loss = pos_loss + vel_loss + acc_loss

        return total_loss


class TransOnlyLoss(nn.Module):
    """只训练trans分支的损失函数"""

    def __init__(self):
        super().__init__()
        self.position_loss = nn.SmoothL1Loss()

    def forward(self, pred_trans, target_trans):
        # trans只有位置损失
        trans_loss = self.position_loss(pred_trans, target_trans)
        return trans_loss


class PoseOnlyLoss(nn.Module):
    """只训练pose分支的损失函数"""

    def __init__(self):
        super().__init__()
        self.position_loss = nn.SmoothL1Loss()

    def forward(self, pred_pose, target_pose):
        # pose只有位置损失
        pose_loss = self.position_loss(pred_pose, target_pose)
        return pose_loss


class LTXMotionDataset(Dataset):
    """为ltx解码器准备的数据集"""

    def __init__(
            self,
            features_dir: str,
            gt_dir: str,
            temporal_factor: int = 8,
            interpolate_method: str = 'linear'
    ):
        self.features_dir = Path(features_dir)
        self.gt_dir = Path(gt_dir)
        self.temporal_factor = temporal_factor
        self.interpolate_method = interpolate_method

        # 获取所有特征文件
        self.feature_files = sorted(list(self.features_dir.glob("*.pth")))

        if not self.feature_files:
            raise ValueError(f"在 {features_dir} 中未找到.pth特征文件")

        print(f"找到 {len(self.feature_files)} 个特征文件")

        # 匹配GT文件并过滤无效数据
        self.samples = []
        self.invalid_samples = []

        for feature_file in self.feature_files:
            file_id = feature_file.stem
            gt_file = self.gt_dir / f"{file_id}.pkl"

            if not gt_file.exists():
                self.invalid_samples.append((feature_file, f"GT文件不存在: {gt_file}"))
                continue

            try:
                # 检查特征文件
                feature_data = torch.load(feature_file, map_location='cpu', weights_only=True)
                if isinstance(feature_data, dict):
                    has_tensor = False
                    for value in feature_data.values():
                        if isinstance(value, torch.Tensor) and value.numel() > 0:
                            has_tensor = True
                            break
                    if not has_tensor:
                        self.invalid_samples.append((feature_file, "特征文件中没有有效张量"))
                        continue

                # 检查GT文件
                with open(gt_file, 'rb') as f:
                    gt_data = pickle.load(f)

                if 'person1' not in gt_data and 'person2' not in gt_data:
                    self.invalid_samples.append((gt_file, "GT文件中没有人数据"))
                    continue

                self.samples.append({
                    'feature_file': feature_file,
                    'gt_file': gt_file,
                    'file_id': file_id
                })

            except Exception as e:
                self.invalid_samples.append((feature_file, f"预检查失败: {str(e)}"))
                continue

        print(f"有效样本数: {len(self.samples)}")
        if self.invalid_samples:
            print(f"跳过无效样本数: {len(self.invalid_samples)}")
            for i, (file, reason) in enumerate(self.invalid_samples[:5]):
                print(f"  无效样本 {i + 1}: {file} - {reason}")
            if len(self.invalid_samples) > 5:
                print(f"  还有 {len(self.invalid_samples) - 5} 个无效样本未显示...")

        if not self.samples:
            raise ValueError("未找到任何有效的特征-GT对")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]

            # 加载latent特征
            feature_data = torch.load(sample['feature_file'], map_location='cpu', weights_only=True)

            # 提取latent张量
            if isinstance(feature_data, dict):
                latent = None
                for key, value in feature_data.items():
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        latent = value
                        break
                if latent is None:
                    raise ValueError(f"特征文件中未找到有效张量数据: {sample['feature_file']}")
            elif isinstance(feature_data, torch.Tensor):
                latent = feature_data
                if latent.numel() == 0:
                    raise ValueError(f"特征张量为空: {sample['feature_file']}")
            else:
                raise ValueError(f"未知的特征数据类型: {type(feature_data)}")

            # 处理维度
            if latent.dim() == 5:  # [B, C, T, H, W]
                if latent.shape[0] == 1:
                    latent = latent.squeeze(0)  # [C, T, H, W]
            elif latent.dim() == 4:  # [C, T, H, W]
                pass
            elif latent.dim() == 3:  # [C, T, D]
                latent = latent.unsqueeze(-1).unsqueeze(-1)  # [C, T, 1, 1]
            else:
                while latent.dim() < 4:
                    latent = latent.unsqueeze(-1)

            if latent.dim() != 4:
                raise ValueError(f"latent维度应为4，当前为{latent.dim()}")

            # 获取latent的帧数
            T_latent = latent.shape[1]
            if T_latent == 0:
                raise ValueError(f"latent时间维度为0: {sample['feature_file']}")

            # 计算目标帧数
            target_frames = (T_latent - 1) * self.temporal_factor + 1

            # 加载GT运动参数
            with open(sample['gt_file'], 'rb') as f:
                gt_data = pickle.load(f)

            # 检查是否有两个人的数据
            has_person1 = 'person1' in gt_data
            has_person2 = 'person2' in gt_data

            # 提取所有运动参数
            def extract_motion_params(data):
                params = {}
                for key in ['root_orient', 'pose_body', 'trans']:
                    if key in data:
                        param = data[key]
                        if isinstance(param, torch.Tensor):
                            param = param.cpu.numpy()
                        params[key] = param
                    else:
                        raise ValueError(f"缺少字段 {key} 在 {sample['file_id']}")

                return params

            # 获取两个人的运动参数
            motion_params_list = []
            original_frames_list = []

            if has_person1:
                params1 = extract_motion_params(gt_data['person1'])
                original_frames1 = params1['root_orient'].shape[0]

                # 插值到目标帧数
                interpolated_params1 = {}
                for key, value in params1.items():
                    if original_frames1 != target_frames:
                        interpolated_params1[key] = interpolate_motion_to_target_frames(
                            value,
                            original_frames1,
                            target_frames,
                            method=self.interpolate_method
                        )
                    else:
                        interpolated_params1[key] = value

                motion_params_list.append(interpolated_params1)
                original_frames_list.append(original_frames1)

            if has_person2:
                params2 = extract_motion_params(gt_data['person2'])
                original_frames2 = params2['root_orient'].shape[0]

                interpolated_params2 = {}
                for key, value in params2.items():
                    if original_frames2 != target_frames:
                        interpolated_params2[key] = interpolate_motion_to_target_frames(
                            value,
                            original_frames2,
                            target_frames,
                            method=self.interpolate_method
                        )
                    else:
                        interpolated_params2[key] = value

                motion_params_list.append(interpolated_params2)
                original_frames_list.append(original_frames2)

            if not motion_params_list:
                raise ValueError(f"GT文件中未找到人物数据: {sample['file_id']}")

            # 创建有效帧掩码
            valid_masks = []
            for i, original_frames in enumerate(original_frames_list):
                valid_mask = np.zeros(target_frames, dtype=np.float32)
                valid_mask[0] = 1.0
                valid_mask[-1] = 1.0

                if original_frames < target_frames:
                    source_time = np.linspace(0, 1, original_frames)
                    target_time = np.linspace(0, 1, target_frames)

                    for t_src in source_time:
                        idx_target = np.argmin(np.abs(target_time - t_src))
                        valid_mask[idx_target] = 1.0
                else:
                    step = original_frames / target_frames
                    for j in range(target_frames):
                        idx_original = int(j * step)
                        if idx_original < original_frames:
                            valid_mask[j] = 1.0

                valid_masks.append(valid_mask)

            # 合并每个人的数据
            if len(motion_params_list) == 2:
                # 两个人都有数据
                root_orient = np.stack([p['root_orient'] for p in motion_params_list])  # [2, T, 3]
                pose_body = np.stack([p['pose_body'] for p in motion_params_list])  # [2, T, pose_dim]
                trans = np.stack([p['trans'] for p in motion_params_list])  # [2, T, 3]
                valid_mask = np.stack(valid_masks)  # [2, T]
                num_persons = 2
            else:
                # 只有一个人的数据，复制一份作为第二个人
                root_orient = np.stack([motion_params_list[0]['root_orient'],
                                        motion_params_list[0]['root_orient']])
                pose_body = np.stack([motion_params_list[0]['pose_body'],
                                      motion_params_list[0]['pose_body']])
                trans = np.stack([motion_params_list[0]['trans'],
                                  motion_params_list[0]['trans']])
                valid_mask = np.stack([valid_masks[0], valid_masks[0]])
                num_persons = 1

            # 转换为Tensor
            root_orient_tensor = torch.from_numpy(root_orient).float()
            pose_body_tensor = torch.from_numpy(pose_body).float()
            trans_tensor = torch.from_numpy(trans).float()
            valid_mask_tensor = torch.from_numpy(valid_mask).float()

            result = {
                'latent': latent,  # [C, T_latent, 1, 1]
                'root_orient': root_orient_tensor,  # [2, target_frames, 3]
                'pose_body': pose_body_tensor,  # [2, target_frames, pose_dim]
                'trans': trans_tensor,  # [2, target_frames, 3]
                'valid_mask': valid_mask_tensor,  # [2, target_frames]
                'file_id': sample['file_id'],
                'target_frames': target_frames,
                'T_latent': T_latent,
                'num_persons': num_persons
            }

            return result

        except Exception as e:
            print(f"警告: 处理样本 {idx} 时出错: {str(e)}")
            next_idx = (idx + 1) % len(self.samples)
            if next_idx != idx:
                return self.__getitem__(next_idx)
            else:
                return self._create_placeholder()

    def _create_placeholder(self):
        """创建一个占位符样本"""
        C = 128
        T = 8
        target_frames = (T - 1) * self.temporal_factor + 1
        pose_dim = 63  # 假设pose_body维度为63

        latent = torch.randn(C, T, 1, 1, dtype=torch.float32)
        root_orient = torch.zeros(2, target_frames, 3, dtype=torch.float32)
        pose_body = torch.zeros(2, target_frames, pose_dim, dtype=torch.float32)
        trans = torch.zeros(2, target_frames, 3, dtype=torch.float32)
        valid_mask = torch.ones(2, target_frames, dtype=torch.float32)

        return {
            'latent': latent,
            'root_orient': root_orient,
            'pose_body': pose_body,
            'trans': trans,
            'valid_mask': valid_mask,
            'file_id': 'placeholder',
            'target_frames': target_frames,
            'T_latent': T,
            'num_persons': 2
        }


class ElasticBranchDecoderTrainer:
    """
    弹性分支训练器
    可以指定训练root/trans/pose分支，当训练下降缓慢时弹性增加深度
    支持断点续训功能
    训练时只保存当前训练分支的权重
    """

    def __init__(
            self,
            decoder: ElasticMotionDecoder,
            branch: str = "root",  # "root", "trans", "pose"
            device: str = "cuda",
            learning_rate: float = 1e-4,
            target_loss: float = 0.01,  # 目标损失
            patience: int = 5,  # 耐心值（多少个epoch用于计算平均改善）
            min_improvement: float = 0.001,  # 最小改善阈值
            max_depth: int = 8,  # 最大深度限制
            min_epochs_after_depth_increase: int = 30,  # 新增深度后至少训练的epoch数
            consecutive_failures_required: int = 3,  # 连续检查不达标次数要求
            checkpoint_path: str = None,  # 断点续训检查点路径
    ):
        self.branch = branch
        self.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("警告: CUDA不可用，切换到CPU")
            self.device = "cpu"

        print(f"🔧 训练器设备: {self.device}")
        print(f"🔧 训练分支: {self.branch}")

        # 将模型移动到设备并验证
        self.decoder = decoder.to(self.device)
        self._verify_model_on_device()

        self.target_loss = target_loss
        self.patience = patience
        self.min_improvement = min_improvement
        self.max_depth = max_depth
        self.min_epochs_after_depth_increase = min_epochs_after_depth_increase
        self.consecutive_failures_required = consecutive_failures_required  # 新增参数

        # 记录训练历史
        self.loss_history = []
        self.depth_history = []
        self.lr_history = []
        self.epoch_counter = 0
        self.last_depth_increase_epoch = 0

        # 弹性训练状态 - 新增连续失败计数器
        self.stagnation_counter = 0
        self.best_loss = float('inf')
        self.last_improvement_epoch = 0
        self.consecutive_check_failures = 0  # 新增：连续检查失败次数

        # 根据分支冻结其他分支
        self._freeze_other_branches()

        # 只优化当前分支的参数
        if self.branch == "root":
            branch_params = list(self.decoder.root_decoder.parameters())
            self.loss_fn = RootOnlyLoss()
        elif self.branch == "trans":
            branch_params = list(self.decoder.trans_decoder.parameters())
            self.loss_fn = TransOnlyLoss()
        elif self.branch == "pose":
            branch_params = list(self.decoder.pose_decoder.parameters())
            self.loss_fn = PoseOnlyLoss()
        else:
            raise ValueError(f"未知的分支: {self.branch}")

        print(f"🔧 {self.branch}分支参数数量: {sum(p.numel() for p in branch_params):,}")

        self.optimizer = optim.AdamW(
            branch_params,
            lr=learning_rate,
            weight_decay=1e-5
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )

        # 断点续训：如果有训练检查点，加载训练状态
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            print(f"✓ 从检查点恢复训练状态: {checkpoint_path}")

        # 中断处理
        self.interrupted = False
        self.setup_interrupt_handler()

        print(f"\n{self.branch}分支训练器初始化:")
        print(f"  设备: {self.device}")
        print(f"  学习率: {learning_rate}")
        print(f"  目标损失: {target_loss}")
        print(f"  最大深度: {max_depth}层")
        print(f"  耐心值: {patience}个epoch")
        print(f"  最小改善阈值: {min_improvement}")
        print(f"  增加深度后最少训练epoch: {min_epochs_after_depth_increase}")
        print(f"  连续失败要求: {consecutive_failures_required}次")
        print(f"  当前深度: {self._get_current_depth()}层")
        print(f"  当前epoch: {self.epoch_counter}")

    def _verify_model_on_device(self):
        """验证整个模型是否在正确的设备上"""
        device = torch.device(self.device)

        print(f"🔍 验证模型设备位置...")

        incorrect_params = []
        for name, param in self.decoder.named_parameters():
            if param.device != device:
                incorrect_params.append((name, param.device))

        if incorrect_params:
            print(f"⚠️  发现 {len(incorrect_params)} 个参数在错误的设备上:")
            for name, wrong_device in incorrect_params[:3]:
                print(f"  {name}: {wrong_device} -> 移动到 {device}")

        print(f"✓ 模型验证完成，设备: {device}")

    def _freeze_other_branches(self):
        """冻结其他分支的参数"""
        if self.branch == "root":
            # 冻结trans和pose
            for param in self.decoder.trans_decoder.parameters():
                param.requires_grad = False
            for param in self.decoder.pose_decoder.parameters():
                param.requires_grad = False
            print("✓ 冻结trans和pose分支")
        elif self.branch == "trans":
            # 冻结root和pose
            for param in self.decoder.root_decoder.parameters():
                param.requires_grad = False
            for param in self.decoder.pose_decoder.parameters():
                param.requires_grad = False
            print("✓ 冻结root和pose分支")
        elif self.branch == "pose":
            # 冻结root和trans
            for param in self.decoder.root_decoder.parameters():
                param.requires_grad = False
            for param in self.decoder.trans_decoder.parameters():
                param.requires_grad = False
            print("✓ 冻结root和trans分支")

    def _get_current_depth(self):
        """获取当前分支的深度"""
        if self.branch == "root":
            return self.decoder.root_decoder.get_current_depth()
        elif self.branch == "trans":
            return self.decoder.trans_decoder.get_current_depth()
        elif self.branch == "pose":
            return self.decoder.pose_decoder.get_current_depth()
        else:
            return 2  # 默认值

    def _is_at_max_depth(self):
        """检查是否达到最大深度"""
        current_depth = self._get_current_depth()
        return current_depth >= self.max_depth

    def _extract_branch_state_dict(self):
        """提取当前训练分支的状态字典"""
        if self.branch == "root":
            prefix = "root_decoder."
        elif self.branch == "trans":
            prefix = "trans_decoder."
        elif self.branch == "pose":
            prefix = "pose_decoder."
        else:
            return {}
        
        branch_state_dict = {}
        full_state_dict = self.decoder.state_dict()
        
        for key, value in full_state_dict.items():
            if key.startswith(prefix):
                # 移除分支前缀
                new_key = key[len(prefix):]
                branch_state_dict[new_key] = value
        
        return branch_state_dict

    def load_checkpoint(self, checkpoint_path: str):
        """从检查点加载完整的训练状态"""
        print(f"📥 从检查点加载训练状态: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # 检查分支是否匹配
        checkpoint_branch = checkpoint.get('branch', self.branch)
        if checkpoint_branch != self.branch:
            print(f"⚠️  警告: 检查点分支({checkpoint_branch})与当前分支({self.branch})不匹配!")
            print(f"⚠️  将加载检查点分支的权重，但可能无法完全兼容")

        # 检查检查点类型：完整模型还是分支模型
        is_branch_checkpoint = 'branch_state_dict' in checkpoint
        
        if is_branch_checkpoint:
            print("📦 加载分支专用检查点...")
            # 分支专用检查点
            branch_state_dict = checkpoint['branch_state_dict']
            
            # 将分支权重映射回完整模型
            if self.branch == "root":
                prefix = "root_decoder."
            elif self.branch == "trans":
                prefix = "trans_decoder."
            elif self.branch == "pose":
                prefix = "pose_decoder."
            
            full_state_dict = self.decoder.state_dict()
            loaded_count = 0
            
            for key, value in branch_state_dict.items():
                full_key = prefix + key
                if full_key in full_state_dict:
                    full_state_dict[full_key] = value
                    loaded_count += 1
            
            print(f"  ✓ 加载了 {loaded_count} 个分支参数")
            
            # 加载模型权重
            self.decoder.load_state_dict(full_state_dict, strict=False)
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
        else:
            print("📦 加载完整模型检查点...")
            # 完整模型检查点
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)

            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载训练历史
        self.loss_history = checkpoint.get('loss_history', [])
        self.depth_history = checkpoint.get('depth_history', [])
        self.lr_history = checkpoint.get('lr_history', [])

        # 加载训练状态
        self.epoch_counter = checkpoint.get('epoch_counter', 0)
        self.last_depth_increase_epoch = checkpoint.get('last_depth_increase_epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.stagnation_counter = checkpoint.get('stagnation_counter', 0)
        self.last_improvement_epoch = checkpoint.get('last_improvement_epoch', 0)
        self.consecutive_check_failures = checkpoint.get('consecutive_check_failures', 0)  # 新增

        checkpoint_target_loss = checkpoint.get('target_loss', None)
        checkpoint_max_depth = checkpoint.get('max_depth', None)
        checkpoint_consecutive_failures = checkpoint.get('consecutive_failures_required', None)

        print(f"✓ 训练状态加载完成:")
        print(f"  已训练epoch: {self.epoch_counter}")
        print(f"  最后增加深度epoch: {self.last_depth_increase_epoch}")
        print(f"  当前深度: {self._get_current_depth()}层")
        print(f"  最佳损失: {self.best_loss:.6f}")
        print(f"  连续失败次数: {self.consecutive_check_failures}")

        # 打印参数对比
        if checkpoint_target_loss is not None and checkpoint_target_loss != self.target_loss:
            print(f"  📊 目标损失: 检查点={checkpoint_target_loss}, 使用命令行参数={self.target_loss}")
        if checkpoint_max_depth is not None and checkpoint_max_depth != self.max_depth:
            print(f"  📏 最大深度: 检查点={checkpoint_max_depth}, 使用命令行参数={self.max_depth}")
        if checkpoint_consecutive_failures is not None and checkpoint_consecutive_failures != self.consecutive_failures_required:
            print(
                f"  ⚠️  连续失败要求: 检查点={checkpoint_consecutive_failures}, 使用命令行参数={self.consecutive_failures_required}")

        # 移动模型到正确的设备
        self.decoder = self.decoder.to(self.device)

    def setup_interrupt_handler(self):
        """设置中断信号处理"""

        def signal_handler(sig, frame):
            print(f"\n接收到中断信号 (Ctrl+C)")
            self.interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _add_depth(self, num_layers=1):
        """为当前分支增加深度"""
        current_depth = self._get_current_depth()
        new_depth = current_depth + num_layers

        if new_depth > self.max_depth:
            print(f"⚠️  已达到最大深度 {self.max_depth}层")
            return False

        print(f"🔧 为{self.branch}分支增加深度: {current_depth} -> {new_depth}层")

        # 调用模型的方法增加深度
        if self.branch == "root":
            success = self.decoder.root_decoder.add_res_layers(num_layers=num_layers)
        elif self.branch == "trans":
            success = self.decoder.trans_decoder.add_res_layers(num_layers=num_layers)
        elif self.branch == "pose":
            success = self.decoder.pose_decoder.add_res_layers(num_layers=num_layers)
        else:
            success = False

        if success:
            # 重置连续失败计数器
            self.consecutive_check_failures = 0
            print(f"✓ {self.branch}分支深度增加到 {new_depth}层")
            print(f"✓ 连续失败计数器已重置")

            # 重新创建优化器，包含新层的参数
            self._recreate_optimizer()

            return True
        return False

    def _recreate_optimizer(self):
        """重新创建优化器，包含新层的参数"""
        # 获取当前分支的参数
        if self.branch == "root":
            branch_params = list(self.decoder.root_decoder.parameters())
        elif self.branch == "trans":
            branch_params = list(self.decoder.trans_decoder.parameters())
        elif self.branch == "pose":
            branch_params = list(self.decoder.pose_decoder.parameters())

        # 保存当前学习率
        if self.optimizer.param_groups:
            current_lr = self.optimizer.param_groups[0]['lr']
        else:
            current_lr = 1e-4

        # 创建新的优化器
        self.optimizer = optim.AdamW(
            branch_params,
            lr=current_lr,
            weight_decay=1e-5
        )

        print(f"🔧 重新创建优化器，学习率: {current_lr:.2e}")

    def prepare_batch(self, batch):
        """准备批次数据"""
        try:
            def to_device(obj):
                if isinstance(obj, torch.Tensor):
                    if obj.dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
                        return obj.to(self.device)
                    else:
                        return obj.to(self.device).float()
                elif isinstance(obj, dict):
                    return {k: to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_device(item) for item in obj]
                else:
                    return obj

            batch = to_device(batch)

            latent = batch['latent']

            if latent.dim() == 4:
                if latent.shape[0] == 128:
                    latent = latent.unsqueeze(0)
                else:
                    latent = latent.unsqueeze(-1)
            elif latent.dim() == 3:
                latent = latent.unsqueeze(0).unsqueeze(-1)
            elif latent.dim() == 5:
                pass
            else:
                raise ValueError(f"latent维度应为5，当前为{latent.dim()}")

            latents = latent.float()

            # 根据分支获取目标数据
            if self.branch == "root":
                target = batch['root_orient'].float()
            elif self.branch == "trans":
                target = batch['trans'].float()
            elif self.branch == "pose":
                target = batch['pose_body'].float()

            valid_mask = batch.get('valid_mask', None)
            if valid_mask is not None:
                valid_mask = valid_mask.float()

            B, C, T_latent, H, W = latents.shape

            timestep = torch.rand(B, device=self.device)

            metadata = {
                'valid_mask': valid_mask,
                'num_persons': batch.get('num_persons', torch.tensor([2], device=self.device, dtype=torch.int64))
            }

            return latents, target, timestep, metadata
        except Exception as e:
            print(f"准备批次数据失败: {e}")
            traceback.print_exc()
            return None, None, None, None

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch，返回损失"""
        self.decoder.train()

        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch_counter + 1}")
        for batch_idx, batch in enumerate(pbar):
            if self.interrupted:
                print("检测到中断信号")
                return total_loss / max(num_batches, 1)

            try:
                latents, target, timestep, metadata = self.prepare_batch(batch)

                if latents is None:
                    continue

                valid_mask = metadata.get('valid_mask')

                num_persons_tensor = metadata['num_persons']
                batch_size = latents.shape[0]

                file_ids = batch.get('file_id', [])
                if any('placeholder' in fid for fid in file_ids):
                    continue

                # 计算解码器期望的目标帧数
                temporal_factor = getattr(self.decoder, 'temporal_downscale_factor', 8)
                T_latent = latents.shape[2]
                decoder_target_frames = (T_latent - 1) * temporal_factor + 1

                # 前向传播 - 只计算当前分支
                self.optimizer.zero_grad()

                if self.branch == "root":
                    output = self.decoder.root_decoder(
                        latents,
                        target_shape=(batch_size, 3, decoder_target_frames, 1, batch['num_persons'][0]),
                        timestep=timestep
                    )
                elif self.branch == "trans":
                    output = self.decoder.trans_decoder(
                        latents,
                        target_shape=(batch_size, 3, decoder_target_frames, 1, batch['num_persons'][0]),
                        timestep=timestep
                    )
                elif self.branch == "pose":
                    pose_dim = target.shape[-1]  # 从目标数据获取pose维度
                    output = self.decoder.pose_decoder(
                        latents,
                        target_shape=(batch_size, pose_dim, decoder_target_frames, 1, batch['num_persons'][0]),
                        timestep=timestep
                    )

                # 计算损失
                batch_loss = 0
                total_valid_persons = 0

                for sample_idx in range(batch_size):
                    num_persons_current = int(num_persons_tensor[sample_idx].item())

                    for person_idx in range(num_persons_current):
                        # 获取预测值
                        if self.branch == "root":
                            pred = output[sample_idx:sample_idx + 1, :, :, :, person_idx]
                            pred = pred.squeeze(-1).squeeze(-1).permute(0, 2, 1)
                        elif self.branch == "trans":
                            pred = output[sample_idx:sample_idx + 1, :, :, :, person_idx]
                            pred = pred.squeeze(-1).squeeze(-1).permute(0, 2, 1)
                        elif self.branch == "pose":
                            pred = output[sample_idx:sample_idx + 1, :, :, :, person_idx]
                            pred = pred.squeeze(-1).squeeze(-1).permute(0, 2, 1)

                        # 获取GT值
                        person_target = target[sample_idx:sample_idx + 1, person_idx, :, :]

                        # 对齐时间维度
                        if pred.shape[1] != person_target.shape[1]:
                            if pred.shape[1] > person_target.shape[1]:
                                person_target = torch.nn.functional.interpolate(
                                    person_target.permute(0, 2, 1),
                                    size=pred.shape[1],
                                    mode='linear',
                                    align_corners=False
                                ).permute(0, 2, 1)
                            else:
                                pred = torch.nn.functional.interpolate(
                                    pred.permute(0, 2, 1),
                                    size=person_target.shape[1],
                                    mode='linear',
                                    align_corners=False
                                ).permute(0, 2, 1)

                        # 调整有效掩码
                        scale_factor = 1.0
                        if valid_mask is not None:
                            if valid_mask.dim() == 3:
                                person_valid_mask = valid_mask[sample_idx:sample_idx + 1, person_idx, :]
                            else:
                                person_valid_mask = valid_mask[sample_idx:sample_idx + 1, :]

                            if person_valid_mask.shape[1] != pred.shape[1]:
                                person_valid_mask = torch.nn.functional.interpolate(
                                    person_valid_mask.unsqueeze(1),
                                    size=pred.shape[1],
                                    mode='nearest'
                                ).squeeze(1)

                            valid_ratio = person_valid_mask.mean()
                            if valid_ratio < 0.5:
                                scale_factor = valid_ratio

                        # 计算损失
                        loss = self.loss_fn(pred, person_target) * scale_factor
                        batch_loss += loss
                        total_valid_persons += 1

                # 平均损失
                if total_valid_persons > 0:
                    batch_loss = batch_loss / total_valid_persons

                    # 反向传播
                    batch_loss.backward()

                    # 梯度裁剪
                    if self.branch == "root":
                        torch.nn.utils.clip_grad_norm_(self.decoder.root_decoder.parameters(), max_norm=1.0)
                    elif self.branch == "trans":
                        torch.nn.utils.clip_grad_norm_(self.decoder.trans_decoder.parameters(), max_norm=1.0)
                    elif self.branch == "pose":
                        torch.nn.utils.clip_grad_norm_(self.decoder.pose_decoder.parameters(), max_norm=1.0)

                    # 优化器步进
                    self.optimizer.step()

                    # 更新统计
                    total_loss += batch_loss.item()
                    num_batches += 1

                    # 更新进度条
                    pbar.set_postfix({
                        f'{self.branch}_loss': batch_loss.item(),
                        'depth': self._get_current_depth()
                    })

            except Exception as e:
                print(f"\n批处理 {batch_idx} 训练失败: {e}")
                traceback.print_exc()
                continue

        if num_batches > 0:
            return total_loss / num_batches
        else:
            return 0.0

    def should_increase_depth(self, current_loss, epoch):
        """判断是否应该增加深度"""
        if current_loss <= self.target_loss:
            print(f"🎯 已达到目标损失 {current_loss:.6f} ≤ {self.target_loss}")
            return False

        if self._is_at_max_depth():
            print(f"📏 已达到最大深度 {self.max_depth}层")
            return False

        epochs_since_last_increase = epoch - self.last_depth_increase_epoch
        if epochs_since_last_increase < self.min_epochs_after_depth_increase:
            remaining = self.min_epochs_after_depth_increase - epochs_since_last_increase
            print(f"⏳ 距离上次增加深度仅 {epochs_since_last_increase} 个epoch，"
                  f"还需训练 {remaining} 个epoch才能再次增加深度")
            return False

        if len(self.loss_history) < self.patience + 1:
            return False

        recent_losses = self.loss_history[-(self.patience + 1):]
        improvements = []

        for i in range(1, len(recent_losses)):
            improvement = recent_losses[i - 1] - recent_losses[i]
            improvements.append(improvement)

        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        print(f"📊 深度判断: 当前损失 {current_loss:.6f}, "
              f"最近{self.patience}个epoch平均改善 {avg_improvement:.6f}, "
              f"阈值 {self.min_improvement}")
        print(f"📊 连续失败次数: {self.consecutive_check_failures}/{self.consecutive_failures_required}")

        # 检查改善是否达标
        improvement_met = avg_improvement >= self.min_improvement

        if not improvement_met:
            # 改善不达标，增加连续失败计数
            self.consecutive_check_failures += 1
            print(
                f"⚠️  改善不达标，连续失败次数: {self.consecutive_check_failures}/{self.consecutive_failures_required}")
        else:
            # 改善达标，重置连续失败计数
            self.consecutive_check_failures = 0
            print(f"✓ 改善达标，重置连续失败计数器")

        # 只有连续失败次数达到要求时才增加深度
        should_increase = (self.consecutive_check_failures >= self.consecutive_failures_required)

        if should_increase:
            print(f"📈 建议增加深度: 连续 {self.consecutive_check_failures} 次检查不达标")

        return should_increase

    def elastic_training_step(self, current_loss, epoch):
        """弹性训练步骤：检查是否需要增加深度"""
        self.loss_history.append(current_loss)

        if current_loss <= self.target_loss:
            print(f"\n🎉 达到目标损失: {current_loss:.6f} ≤ {self.target_loss}")
            return False

        if self.should_increase_depth(current_loss, epoch):
            if self._add_depth(num_layers=1):
                current_depth = self._get_current_depth()
                self.depth_history.append({
                    'epoch': epoch + 1,
                    'depth': current_depth,
                    'loss': current_loss
                })

                self.last_depth_increase_epoch = epoch + 1

                print(f"\n🔧 弹性增加深度: {current_depth}层 (epoch: {epoch + 1})")

                # 确保新层在正确的设备上
                self.decoder = self.decoder.to(self.device)

                # 调整学习率
                self._adjust_learning_rate(0.8)

                return True

        return False

    def _adjust_learning_rate(self, factor: float):
        """调整学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        print(f"📉 学习率调整为: {current_lr:.2e}")

    def save_checkpoint(self, save_dir: Path, epoch: int, loss: float):
        """保存检查点 - 只保存当前训练分支的权重"""
        save_dir.mkdir(parents=True, exist_ok=True)

        # 只保存当前训练分支的权重
        checkpoint_path = save_dir / f"{self.branch}_checkpoint_epoch{epoch:03d}.pt"

        original_device = self.device
        if original_device != "cpu":
            self.decoder = self.decoder.cpu()

        # 提取当前分支的状态字典
        branch_state_dict = self._extract_branch_state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'branch_state_dict': branch_state_dict,  # 只包含当前分支的权重
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'current_depth': self._get_current_depth(),
            'depth_history': self.depth_history,
            'loss_history': self.loss_history,
            'lr_history': self.lr_history,
            'target_loss': self.target_loss,
            'max_depth': self.max_depth,
            'best_loss': self.best_loss,
            'stagnation_counter': self.stagnation_counter,
            'last_improvement_epoch': self.last_improvement_epoch,
            'epoch_counter': self.epoch_counter,
            'last_depth_increase_epoch': self.last_depth_increase_epoch,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'min_epochs_after_depth_increase': self.min_epochs_after_depth_increase,
            'patience': self.patience,
            'min_improvement': self.min_improvement,
            'consecutive_failures_required': self.consecutive_failures_required,
            'consecutive_check_failures': self.consecutive_check_failures,
            'branch': self.branch,
            'is_branch_checkpoint': True,  # 标记为分支检查点
        }

        torch.save(checkpoint, checkpoint_path)

        if original_device != "cpu":
            self.decoder = self.decoder.to(original_device)

        # 保存配置信息
        config_path = save_dir / f"{self.branch}_config_epoch{epoch:03d}.json"
        with open(config_path, 'w') as f:
            config_data = {
                'epoch': epoch,
                'loss': loss,
                'current_depth': self._get_current_depth(),
                'depth_history': self.depth_history,
                'target_loss': self.target_loss,
                'max_depth': self.max_depth,
                'min_epochs_after_depth_increase': self.min_epochs_after_depth_increase,
                'patience': self.patience,
                'min_improvement': self.min_improvement,
                'consecutive_failures_required': self.consecutive_failures_required,
                'consecutive_check_failures': self.consecutive_check_failures,
                'branch': self.branch,
                'is_branch_checkpoint': True,
                'checkpoint_type': 'branch_only',
                'model_config': self.decoder.to_json_string() if hasattr(self.decoder, 'to_json_string') else {},
            }
            json.dump(config_data, f, indent=2)

        print(f"\n💾 保存检查点: epoch {epoch}")
        print(f"  📍 路径: {checkpoint_path}")
        print(f"  📊 {self.branch}损失: {loss:.6f}")
        print(f"  📏 当前深度: {self._get_current_depth()}层")
        print(f"  ⚠️  连续失败次数: {self.consecutive_check_failures}/{self.consecutive_failures_required}")
        print(f"  📊 分支参数数量: {len(branch_state_dict)}层")
        
        return checkpoint_path

    def train(
            self,
            train_loader: DataLoader,
            num_epochs: int,
            save_dir: str,
            save_freq: int = 5,
            start_epoch: int = None
    ):
        """主训练循环 - 只保存当前训练分支的权重"""
        print(f"\n🎯 开始弹性训练{self.branch}分支")
        print(f"📊 训练集: {len(train_loader.dataset)} 样本")
        print(f"💾 保存模式: 只保存{self.branch}分支权重")

        if start_epoch is None:
            start_epoch = self.epoch_counter
        actual_epochs = num_epochs - start_epoch

        print(f"⏳ 总epoch数: {num_epochs} (从{start_epoch}开始，实际训练{actual_epochs}个epoch)")
        print(f"🎯 目标{self.branch}损失: {self.target_loss}")
        print(f"📏 最大深度: {self.max_depth}层")
        print(f"⏱️  增加深度后最少训练epoch: {self.min_epochs_after_depth_increase}")
        print(f"⚠️  连续失败要求: {self.consecutive_failures_required}次")
        print(f"💾 保存频率: 每{save_freq}个epoch")
        print("=" * 60)

        save_dir = Path(save_dir)

        try:
            for epoch in range(start_epoch, num_epochs):
                self.epoch_counter = epoch + 1

                if self.interrupted:
                    print("检测到中断信号，保存检查点...")
                    if len(self.loss_history) > 0:
                        last_loss = self.loss_history[-1]
                        self.save_checkpoint(save_dir, epoch + 1, last_loss)
                    break

                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch + 1}/{num_epochs} (全局: {self.epoch_counter})")
                print(f"训练分支: {self.branch}")
                print(f"当前深度: {self._get_current_depth()}层")
                print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"连续失败次数: {self.consecutive_check_failures}/{self.consecutive_failures_required}")
                if self.last_depth_increase_epoch > 0:
                    epochs_since_increase = (epoch + 1) - self.last_depth_increase_epoch
                    print(f"距离上次增加深度: {epochs_since_increase}个epoch")
                print(f"{'=' * 50}")

                # 训练一个epoch
                current_loss = self.train_epoch(train_loader)

                improvement = 0.0
                if len(self.loss_history) > 0:
                    last_loss = self.loss_history[-1]
                    improvement = last_loss - current_loss

                print(f"\n📊 训练结果:")
                print(f"  {self.branch}损失: {current_loss:.6f}")
                if len(self.loss_history) > 0:
                    print(f"  📈 改善值: {improvement:.6f} ({'+' if improvement > 0 else ''}{improvement:.6f})")

                # 弹性训练步骤
                depth_increased = self.elastic_training_step(current_loss, epoch)

                # 更新学习率调度器
                self.scheduler.step(current_loss)

                # 更新最佳损失
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.last_improvement_epoch = epoch + 1

                # 保存检查点
                save_this_epoch = (epoch + 1) % save_freq == 0 or depth_increased
                if save_this_epoch:
                    self.save_checkpoint(save_dir, epoch + 1, current_loss)

                # 保存最新检查点
                latest_checkpoint = save_dir / f"latest_{self.branch}_checkpoint.pt"
                if latest_checkpoint.exists():
                    latest_checkpoint.unlink()
                latest_checkpoint_path = self.save_checkpoint(save_dir, epoch + 1, current_loss)
                if latest_checkpoint_path:
                    # 重命名为latest
                    Path(latest_checkpoint_path).rename(latest_checkpoint)
                    print(f"  🔄 更新最新检查点: {latest_checkpoint}")

                # 检查是否达到目标
                if current_loss <= self.target_loss:
                    print(f"\n{'=' * 50}")
                    print(f"🎉 训练完成！达到目标{self.branch}损失")
                    print(f"📊 最终{self.branch}损失: {current_loss:.6f}")
                    print(f"📏 最终深度: {self._get_current_depth()}层")
                    print(f"📅 训练总epoch: {self.epoch_counter}")
                    print(f"{'=' * 50}")

                    # 保存最终检查点
                    final_path = self.save_checkpoint(save_dir, epoch + 1, current_loss)
                    final_checkpoint = save_dir / f"final_{self.branch}_checkpoint.pt"
                    if final_checkpoint.exists():
                        final_checkpoint.unlink()
                    Path(final_path).rename(final_checkpoint)
                    print(f"💾 保存最终检查点: {final_checkpoint}")

                    break

                # 检查是否达到最大epoch
                if epoch + 1 >= num_epochs:
                    print(f"\n{'=' * 50}")
                    print(f"⚠️ 达到最大epoch数")
                    print(f"📊 最终{self.branch}损失: {current_loss:.6f}")
                    print(f"📏 最终深度: {self._get_current_depth()}层")
                    print(f"📅 训练总epoch: {self.epoch_counter}")
                    print(f"{'=' * 50}")

                    self.save_checkpoint(save_dir, epoch + 1, current_loss)

                # 检查中断
                if self.interrupted:
                    print("检测到中断信号，保存检查点...")
                    self.save_checkpoint(save_dir, epoch + 1, current_loss)
                    break

        except KeyboardInterrupt:
            print("\n用户中断训练")
            if len(self.loss_history) > 0:
                last_loss = self.loss_history[-1]
                self.save_checkpoint(save_dir, self.epoch_counter, last_loss)

        except Exception as e:
            print(f"\n训练过程中发生错误: {e}")
            traceback.print_exc()
            if len(self.loss_history) > 0:
                last_loss = self.loss_history[-1]
                self.save_checkpoint(save_dir, self.epoch_counter, last_loss)


def create_elastic_decoder_from_config(
        max_depth: int = 8,
        initial_depth: int = 2,
        branch: str = "root"
):
    """从配置创建全新的弹性解码器"""
    print(f"📥 创建全新的弹性解码器")

    config = create_elastic_motion_decoder_config(
        latent_channels=128,
        motion_channels_per_person=69,
        base_channels=128,
        causal=True,
        timestep_conditioning=True,
        dropout_rate=0.1,
        use_weight_decay=True,
        use_layer_norm=False,
        use_stochastic_depth=True,
        stochastic_depth_rate=0.1,
        max_res_layers=max_depth,
        initial_res_layers=initial_depth,
        use_elastic_depth=True,
    )

    decoder = ElasticMotionDecoder.from_config(config)

    print(f"✓ 创建弹性解码器成功")
    print(f"  📏 最大深度: {max_depth}层")
    print(f"  📐 初始深度: {initial_depth}层")
    print(f"  🔧 训练分支: {branch}")

    return decoder


def create_elastic_trainer_from_checkpoint(
        checkpoint_path: str,
        branch: str = "root",
        device: str = "cuda",
        train_loader: DataLoader = None,
        # 新增：允许传递命令行参数来覆盖检查点中的值
        max_depth: int = None,
        target_loss: float = None,
        patience: int = None,
        min_improvement: float = None,
        min_epochs_after_depth_increase: int = None,
        consecutive_failures_required: int = None,
        learning_rate: float = None
):
    """从检查点创建弹性训练器（用于断点续训）"""
    print(f"🔄 从检查点创建{branch}训练器: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # 检查检查点类型
    is_branch_checkpoint = checkpoint.get('is_branch_checkpoint', False)
    print(f"  📦 检查点类型: {'分支专用' if is_branch_checkpoint else '完整模型'}")

    # 从检查点获取分支信息
    checkpoint_branch = checkpoint.get('branch', branch)
    if checkpoint_branch != branch:
        print(f"⚠️  警告: 检查点分支({checkpoint_branch})与指定分支({branch})不匹配!")
        print(f"⚠️  将使用检查点分支: {checkpoint_branch}")
        branch = checkpoint_branch

    current_depth = checkpoint.get('current_depth', 2)

    # 优先使用命令行参数，如果没有则使用检查点中的值
    effective_max_depth = max_depth if max_depth is not None else checkpoint.get('max_depth', 8)
    effective_target_loss = target_loss if target_loss is not None else checkpoint.get('target_loss', 0.01)
    effective_patience = patience if patience is not None else checkpoint.get('patience', 5)
    effective_min_improvement = min_improvement if min_improvement is not None else checkpoint.get('min_improvement',
                                                                                                   0.001)
    effective_min_epochs = min_epochs_after_depth_increase if min_epochs_after_depth_increase is not None else checkpoint.get(
        'min_epochs_after_depth_increase', 30)
    effective_consecutive_failures = consecutive_failures_required if consecutive_failures_required is not None else checkpoint.get(
        'consecutive_failures_required', 3)
    effective_learning_rate = learning_rate if learning_rate is not None else checkpoint.get('learning_rate', 1e-4)

    # 直接从检查点恢复模型
    config = create_elastic_motion_decoder_config(
        latent_channels=128,
        motion_channels_per_person=69,
        base_channels=128,
        causal=True,
        timestep_conditioning=True,
        dropout_rate=0.1,
        use_weight_decay=True,
        use_layer_norm=False,
        use_stochastic_depth=True,
        stochastic_depth_rate=0.1,
        max_res_layers=effective_max_depth,  # 使用有效的最大深度
        initial_res_layers=current_depth,
        use_elastic_depth=True,
    )

    decoder = ElasticMotionDecoder.from_config(config)

    # 创建训练器
    trainer = ElasticBranchDecoderTrainer(
        decoder=decoder,
        branch=branch,
        device=device,
        learning_rate=effective_learning_rate,
        target_loss=effective_target_loss,
        patience=effective_patience,
        min_improvement=effective_min_improvement,
        max_depth=effective_max_depth,
        min_epochs_after_depth_increase=effective_min_epochs,
        consecutive_failures_required=effective_consecutive_failures,
        checkpoint_path=checkpoint_path,
    )

    print(f"✓ {branch}训练器创建成功")
    print(f"  当前深度: {trainer._get_current_depth()}层")
    print(f"  已训练epoch: {trainer.epoch_counter}")
    print(f"  连续失败次数: {trainer.consecutive_check_failures}/{effective_consecutive_failures}")
    print(f"  使用的最大深度: {effective_max_depth}层 (命令行参数覆盖)")

    return trainer


def collate_fn(batch):
    """批次整理函数"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return _create_placeholder_batch()

    max_T_latent = max([item['T_latent'] for item in batch])
    max_target_frames = max([item['root_orient'].shape[1] for item in batch])

    batched = {}

    # 处理latent
    latents = []
    for item in batch:
        latent = item['latent']
        C, T, H, W = latent.shape

        if T < max_T_latent:
            pad_size = max_T_latent - T
            padding = torch.zeros(C, pad_size, H, W, dtype=latent.dtype)
            latent = torch.cat([latent, padding], dim=1)

        latents.append(latent)

    batched['latent'] = torch.stack(latents)

    # 处理所有运动参数
    root_orients = []
    pose_bodies = []
    trans_list = []
    valid_masks = []
    T_latents = []
    target_frames = []
    num_persons = []

    for item in batch:
        root_orient = item['root_orient']
        pose_body = item['pose_body']
        trans = item['trans']
        num_persons_current = root_orient.shape[0]

        # 处理root_orient
        T_root, D_root = root_orient.shape[1], root_orient.shape[2]
        if T_root < max_target_frames:
            pad_size = max_target_frames - T_root
            padding = torch.zeros(num_persons_current, pad_size, D_root, dtype=root_orient.dtype)
            root_orient = torch.cat([root_orient, padding], dim=1)
        root_orients.append(root_orient)

        # 处理pose_body
        T_pose, D_pose = pose_body.shape[1], pose_body.shape[2]
        if T_pose < max_target_frames:
            pad_size = max_target_frames - T_pose
            padding = torch.zeros(num_persons_current, pad_size, D_pose, dtype=pose_body.dtype)
            pose_body = torch.cat([pose_body, padding], dim=1)
        pose_bodies.append(pose_body)

        # 处理trans
        T_trans, D_trans = trans.shape[1], trans.shape[2]
        if T_trans < max_target_frames:
            pad_size = max_target_frames - T_trans
            padding = torch.zeros(num_persons_current, pad_size, D_trans, dtype=trans.dtype)
            trans = torch.cat([trans, padding], dim=1)
        trans_list.append(trans)

        if 'valid_mask' in item:
            valid_mask = item['valid_mask']
            if valid_mask.shape[1] < max_target_frames:
                pad_size = max_target_frames - valid_mask.shape[1]
                padding = torch.zeros(num_persons_current, pad_size, dtype=valid_mask.dtype)
                valid_mask = torch.cat([valid_mask, padding], dim=1)
            valid_masks.append(valid_mask)

        T_latents.append(item['T_latent'])
        target_frames.append(item['target_frames'])
        num_persons.append(num_persons_current)

    batched['root_orient'] = torch.stack(root_orients)
    batched['pose_body'] = torch.stack(pose_bodies)
    batched['trans'] = torch.stack(trans_list)

    if valid_masks:
        batched['valid_mask'] = torch.stack(valid_masks)

    batched['T_latent'] = torch.tensor(T_latents, dtype=torch.int64)
    batched['target_frames'] = torch.tensor(target_frames, dtype=torch.int64)
    batched['num_persons'] = torch.tensor(num_persons, dtype=torch.int64)
    batched['file_id'] = [item['file_id'] for item in batch]

    return batched


def _create_placeholder_batch():
    """创建一个占位符批次"""
    C = 128
    T_latent = 8
    T_target = (T_latent - 1) * 8 + 1
    pose_dim = 63

    return {
        'latent': torch.randn(1, C, T_latent, 1, 1),
        'root_orient': torch.zeros(1, 2, T_target, 3),
        'pose_body': torch.zeros(1, 2, T_target, pose_dim),
        'trans': torch.zeros(1, 2, T_target, 3),
        'valid_mask': torch.ones(1, 2, T_target),
        'T_latent': torch.tensor([T_latent], dtype=torch.int64),
        'target_frames': torch.tensor([T_target], dtype=torch.int64),
        'num_persons': torch.tensor([2], dtype=torch.int64),
        'file_id': ['placeholder_batch']
    }


def main():
    parser = argparse.ArgumentParser(description="弹性训练指定分支，当训练下降缓慢时增加深度，支持断点续训")

    # 数据参数
    parser.add_argument("--features_dir", type=str, required=True,
                        help="latent特征文件目录")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="GT运动参数目录")
    parser.add_argument("--interpolate_method", type=str, default="linear",
                        choices=["linear", "cubic"], help="插值方法")

    # 训练分支选择
    parser.add_argument("--branch", type=str, required=True,
                        choices=["root", "trans", "pose"],
                        help="训练的分支：root/trans/pose")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="训练epoch数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载工作进程数")

    # 弹性训练参数
    parser.add_argument("--target_loss", type=float, default=0.01,
                        help="目标损失（当达到此值时停止训练）")
    parser.add_argument("--patience", type=int, default=5,
                        help="耐心值（多少个epoch用于计算平均改善）")
    parser.add_argument("--min_improvement", type=float, default=0.001,
                        help="最小改善阈值（小于此值认为停滞）")
    parser.add_argument("--max_depth", type=int, default=8,
                        help="最大深度限制")
    parser.add_argument("--initial_depth", type=int, default=2,
                        help="初始深度")
    parser.add_argument("--min_epochs_after_depth_increase", type=int, default=30,
                        help="增加深度后至少训练的epoch数")
    parser.add_argument("--consecutive_failures_required", type=int, default=3,
                        help="连续检查不达标次数要求（默认3次）")

    # 保存参数
    parser.add_argument("--save_dir", type=str, default="./decoder_training",
                        help="保存目录")
    parser.add_argument("--save_freq", type=int, default=5,
                        help="保存频率（epoch）")

    # 断点续训参数
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练，例如: ./decoder_training/latest_root_checkpoint.pt")

    # 设备
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备（cuda/cpu）")

    args = parser.parse_args()

    print("=" * 60)
    print(f"弹性{args.branch}分支训练 - 支持断点续训")
    print("=" * 60)
    print(f"训练分支: {args.branch}")
    print(f"特征目录: {args.features_dir}")
    print(f"GT目录: {args.gt_dir}")
    if args.resume:
        print(f"恢复检查点: {args.resume}")
    print(f"批大小: {args.batch_size}")
    print(f"Epoch数: {args.num_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"\n🎯 训练目标:")
    print(f"  训练分支: {args.branch}")
    print(f"  目标损失: {args.target_loss}")
    print(f"  最大深度: {args.max_depth}层")
    print(f"  初始深度: {args.initial_depth}层")
    print(f"  耐心值: {args.patience}个epoch")
    print(f"  最小改善阈值: {args.min_improvement}")
    print(f"  增加深度后最少训练epoch: {args.min_epochs_after_depth_increase}")
    print(f"  连续失败要求: {args.consecutive_failures_required}次")
    print(f"📊 保存配置:")
    print(f"  保存目录: {args.save_dir}")
    print(f"  保存频率: 每{args.save_freq}个epoch")
    print(f"  检查点类型: 只保存{args.branch}分支权重")  # 修复这里，使用args.branch
    print("=" * 60)

    # 设置设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = "cpu"

    # 创建数据集
    print("\n创建数据集...")
    dataset = LTXMotionDataset(
        features_dir=args.features_dir,
        gt_dir=args.gt_dir,
        temporal_factor=8,
        interpolate_method=args.interpolate_method
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    print(f"训练集: {len(dataset)} 样本")

    # 检查是否从检查点恢复训练
    if args.resume and os.path.exists(args.resume):
        print(f"\n🔄 从检查点恢复训练: {args.resume}")
        trainer = create_elastic_trainer_from_checkpoint(
            checkpoint_path=args.resume,
            branch=args.branch,
            device=device,
            train_loader=train_loader,
            max_depth=args.max_depth,
            target_loss=args.target_loss,
            patience=args.patience,
            min_improvement=args.min_improvement,
            min_epochs_after_depth_increase=args.min_epochs_after_depth_increase,
            consecutive_failures_required=args.consecutive_failures_required,
            learning_rate=args.learning_rate
        )

        # 计算剩余的训练epoch数
        remaining_epochs = max(0, args.num_epochs - trainer.epoch_counter)
        if remaining_epochs == 0:
            print(f"⚠️  检查点已经训练了 {trainer.epoch_counter} 个epoch，已达到目标 {args.num_epochs}")
            print("如果要继续训练，请增加 --num_epochs 参数")
            return

        print(f"继续训练剩余 {remaining_epochs} 个epoch (总计: {args.num_epochs})")

    else:
        # 创建全新的弹性解码器
        print("\n创建全新的弹性解码器...")
        decoder = create_elastic_decoder_from_config(
            max_depth=args.max_depth,
            initial_depth=args.initial_depth,
            branch=args.branch
        )

        # 统计参数
        total_params = sum(p.numel() for p in decoder.parameters())
        root_params = sum(p.numel() for p in decoder.root_decoder.parameters())
        trans_params = sum(p.numel() for p in decoder.trans_decoder.parameters())
        pose_params = sum(p.numel() for p in decoder.pose_decoder.parameters())

        print(f"\n📊 模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  root解码器: {root_params:,}")
        print(f"  trans解码器: {trans_params:,}")
        print(f"  pose解码器: {pose_params:,}")
        print(f"  当前训练分支: {args.branch}")
        print(f"  当前深度: {decoder.get_current_depth()}层")

        # 创建训练器
        print(f"\n创建{args.branch}训练器...")
        trainer = ElasticBranchDecoderTrainer(
            decoder=decoder,
            branch=args.branch,
            device=device,
            learning_rate=args.learning_rate,
            target_loss=args.target_loss,
            patience=args.patience,
            min_improvement=args.min_improvement,
            max_depth=args.max_depth,
            min_epochs_after_depth_increase=args.min_epochs_after_depth_increase,
            consecutive_failures_required=args.consecutive_failures_required,
            checkpoint_path=None,
        )

    # 开始训练
    try:
        trainer.train(
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            save_dir=args.save_dir,
            save_freq=args.save_freq,
            start_epoch=trainer.epoch_counter
        )

        print(f"\n训练完成！结果保存在: {args.save_dir}")

        # 打印最终统计
        if trainer.depth_history:
            print(f"\n📈 深度历史记录:")
            for record in trainer.depth_history:
                print(f"  Epoch {record['epoch']}: {record['depth']}层, "
                      f"{args.branch}损失: {record['loss']:.6f}")

        print(f"\n📋 训练总结:")
        print(f"  训练分支: {args.branch}")
        print(f"  训练总epoch数: {trainer.epoch_counter}")
        print(f"  最终{args.branch}损失: {trainer.loss_history[-1] if trainer.loss_history else 'N/A':.6f}")
        print(f"  最终深度: {trainer._get_current_depth()}层")
        print(f"  最佳损失: {trainer.best_loss:.6f}")
        print(f"  深度增加次数: {len(trainer.depth_history)}次")
        print(f"  连续失败最大次数: {trainer.consecutive_check_failures}次")
        print(f"  保存模式: 只保存{args.branch}分支权重")  # 修复这里，使用args.branch

        # 输出文件列表
        print(f"\n📁 生成的文件:")
        save_dir = Path(args.save_dir)
        if save_dir.exists():
            checkpoint_files = list(save_dir.glob("*checkpoint*.pt"))
            config_files = list(save_dir.glob("*config*.json"))
            
            if checkpoint_files:
                print(f"  检查点文件 ({len(checkpoint_files)}个):")
                for file in sorted(checkpoint_files)[-5:]:  # 显示最后5个文件
                    file_size = file.stat().st_size / (1024 * 1024)  # MB
                    print(f"    - {file.name} ({file_size:.2f} MB)")
            
            if config_files:
                print(f"  配置文件 ({len(config_files)}个):")
                for file in sorted(config_files)[-3:]:  # 显示最后3个文件
                    print(f"    - {file.name}")

    except Exception as e:
        print(f"\n主训练过程中发生错误: {e}")
        traceback.print_exc()
        print("程序退出")


if __name__ == "__main__":
    main()