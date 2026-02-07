#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np

from ltx_video.latent_prepare import infer, InferenceConfig, get_saved_features, get_last_feature


@dataclass
class BatchInferenceConfig:
    """批量处理的配置"""
    prompts_dir1: str = field(
        default="./separate_annots/text1/",
        metadata={"help": "第一个提示文本目录"}
    )
    prompts_dir2: str = field(
        default="./separate_annots/text2/",
        metadata={"help": "第二个提示文本目录"}
    )
    motions_dir: str = field(
        default="./motions/",
        metadata={"help": "动作文件目录"}
    )
    output_base_dir: str = field(
        default="./features_output/",
        metadata={"help": "特征输出基目录"}
    )
    height: int = field(
        default=32,
        metadata={"help": "视频高度"}
    )
    width: int = field(
        default=64,
        metadata={"help": "视频宽度"}
    )
    pipeline_config: str = field(
        default="configs/ltxv-2b-0.9.8-distilled.yaml",
        metadata={"help": "管道配置文件"}
    )
    base_prompt1: str = field(
        default="the first person",
        metadata={"help": "第一个人基础提示"}
    )
    base_prompt2: str = field(
        default="the second person",
        metadata={"help": "第二个人基础提示"}
    )


def read_pkl_frames(pkl_path: str) -> int:
    """读取pkl文件中的帧数"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
            # 方法1: 直接读取'frames'键
            if 'frames' in data:
                frames = data['frames']
                print(f"  从'frames'键读取到帧数: {frames}")
                return frames
            
            # 方法2: 从person1的trans数据形状读取帧数
            elif 'person1' in data and 'trans' in data['person1']:
                trans_data = data['person1']['trans']
                if hasattr(trans_data, 'shape') and len(trans_data.shape) >= 1:
                    frames = trans_data.shape[0]
                    print(f"  从person1.trans形状读取到帧数: {frames}")
                    return frames
            
            # 方法3: 尝试其他可能包含帧数的键
            elif 'person1' in data:
                # 检查person1字典中是否有时间序列数据
                for key, value in data['person1'].items():
                    if isinstance(value, np.ndarray) and len(value.shape) >= 1:
                        frames = value.shape[0]
                        print(f"  从person1.{key}形状读取到帧数: {frames}")
                        return frames
            
            print(f"  警告: 无法从 {pkl_path} 读取帧数，使用默认值121")
            return 121
            
    except Exception as e:
        print(f"  错误: 读取 {pkl_path} 失败: {e}")
        return 121


def find_matching_files(dir1: Path, dir2: Path) -> list:
    """找到两个目录中匹配的文件"""
    # 获取所有txt文件
    files1 = sorted(dir1.glob("*.txt"))
    files2 = sorted(dir2.glob("*.txt"))
    
    print(f"目录1中找到 {len(files1)} 个txt文件")
    print(f"目录2中找到 {len(files2)} 个txt文件")
    
    matching_files = []
    
    # 找出两个目录中都存在的文件名
    for f1 in files1:
        for f2 in files2:
            if f1.name == f2.name:
                matching_files.append((f1, f2))
                break
    
    print(f"找到 {len(matching_files)} 个匹配的文件对")
    return matching_files


def read_text_file(file_path: Path) -> str:
    """读取文本文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 移除可能的换行符和多余空格
            content = ' '.join(content.split())
            return content
    except Exception as e:
        print(f"  错误: 读取 {file_path} 失败: {e}")
        return ""


def create_prompt(base1: str, text1: str, base2: str, text2: str) -> str:
    """创建组合提示"""
    # 清理文本，确保没有多余空格
    text1_clean = text1.strip()
    text2_clean = text2.strip()
    base1_clean = base1.strip()
    base2_clean = base2.strip()
    
    # 构建提示，确保格式正确
    prompt = f"{base1_clean} {text1_clean} and {base2_clean} {text2_clean}"
    return prompt


def save_feature_to_pth(feature, output_path: Path, txt_filename: str, prompt: str, num_frames: int):
    """保存特征到pth文件，使用txt文件名"""
    if feature is not None:
        # 创建目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存特征
        torch.save({
            'feature': feature,
            'prompt': prompt,
            'source_file': txt_filename,
            'num_frames': num_frames,
            'shape': feature.shape,
            'dtype': str(feature.dtype)
        }, output_path)
        print(f"  ✓ 特征已保存到: {output_path}")
        print(f"    特征形状: {feature.shape}")
        return True
    else:
        print(f"  ✗ 未捕获到特征，跳过保存 {output_path}")
        return False


def batch_infer(config: BatchInferenceConfig):
    """批量推理函数"""
    # 转换为Path对象
    dir1 = Path(config.prompts_dir1)
    dir2 = Path(config.prompts_dir2)
    motions_dir = Path(config.motions_dir)
    output_base_dir = Path(config.output_base_dir)
    
    # 检查目录是否存在
    if not dir1.exists():
        print(f"错误: 目录不存在: {dir1}")
        return
    if not dir2.exists():
        print(f"错误: 目录不存在: {dir2}")
        return
    if not motions_dir.exists():
        print(f"错误: 动作目录不存在: {motions_dir}")
        return
    
    # 创建输出目录
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 找到匹配的文件
    matching_files = find_matching_files(dir1, dir2)
    
    if not matching_files:
        print("错误: 未找到匹配的文件对")
        return
    
    print(f"\n{'='*60}")
    print(f"开始批量处理，共 {len(matching_files)} 个文件对")
    print(f"{'='*60}")
    
    # 处理每个文件对
    processed_count = 0
    for idx, (file1, file2) in enumerate(matching_files, 1):
        print(f"\n[处理 {idx}/{len(matching_files)}] {'-'*40}")
        print(f"  文件1: {file1.name}")
        print(f"  文件2: {file2.name}")
        
        # 读取文本内容
        text1 = read_text_file(file1)
        text2 = read_text_file(file2)
        
        if not text1:
            print(f"  警告: {file1.name} 内容为空，跳过")
            continue
        if not text2:
            print(f"  警告: {file2.name} 内容为空，跳过")
            continue
        
        print(f"  文本1: {text1[:50]}..." if len(text1) > 50 else f"  文本1: {text1}")
        print(f"  文本2: {text2[:50]}..." if len(text2) > 50 else f"  文本2: {text2}")
        
        # 创建组合提示
        prompt = create_prompt(config.base_prompt1, text1, config.base_prompt2, text2)
        print(f"  组合提示: {prompt[:80]}..." if len(prompt) > 80 else f"  组合提示: {prompt}")
        
        # 读取对应的pkl文件获取帧数
        pkl_name = file1.stem + ".pkl"  # 例如 1.txt -> 1.pkl
        pkl_path = motions_dir / pkl_name
        
        if pkl_path.exists():
            num_frames = read_pkl_frames(str(pkl_path))
        else:
            print(f"  警告: {pkl_name} 不存在，使用默认帧数121")
            num_frames = 121
        
        # 创建输出子目录
        output_subdir = output_base_dir / f"batch_{idx}"
        output_subdir.mkdir(exist_ok=True)
        
        # 设置随机种子（使用文件索引作为种子的一部分）
        seed = 171198 + idx
        
        try:
            # 创建InferenceConfig，指定特征保存目录和文件名
            features_dir = output_base_dir / "decoder_features"
            feature_filename = f"{file1.stem}.pth"  # 使用txt文件名，如 1.txt -> 1.pth
            
            inference_config = InferenceConfig(
                prompt=prompt,
                output_path=str(output_subdir),
                features_output_dir=str(features_dir),  # 指定统一特征目录
                feature_filename=feature_filename,      # 指定与txt同名的文件名
                pipeline_config=config.pipeline_config,
                seed=seed,
                height=config.height,
                width=config.width,
                num_frames=num_frames,
                frame_rate=30,
                offload_to_cpu=False,
                negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                enable_second_stage=False,
                save_first_stage_video=False
            )
            
            # 执行推理
            print(f"  开始推理... (帧数: {num_frames}, 种子: {seed})")
            print(f"  特征将保存为: {features_dir}/{feature_filename}")
            infer(config=inference_config)
            
            # 获取保存的特征（现在由infer函数直接保存，这里只做验证）
            saved_features = get_saved_features()
            if saved_features:
                last_feature = get_last_feature()
                
                # 验证特征文件是否已保存
                feature_path = features_dir / feature_filename
                if feature_path.exists():
                    print(f"  ✓ 验证: 特征文件已保存: {feature_path}")
                    processed_count += 1
                else:
                    print(f"  ✗ 警告: 特征文件未找到: {feature_path}")
            else:
                print(f"  警告: 未捕获到特征")
                
        except Exception as e:
            print(f"  错误: 推理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"  ✓ 第 {idx} 对文件处理完成 (特征: {feature_filename})")
    
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"成功处理了 {processed_count}/{len(matching_files)} 对文件")
    print(f"特征输出目录: {output_base_dir}/decoder_features/")
    print(f"特征文件命名: 与txt文件同名，如 1.txt -> 1.pth")
    print(f"{'='*60}")


def main():
    # 使用标准argparse而不是HfArgumentParser，以便更好地控制参数
    parser = argparse.ArgumentParser(description="LTX Video 批量推理")
    
    # 批量处理参数
    parser.add_argument("--batch_mode", action="store_true", 
                       help="启用批量处理模式")
    parser.add_argument("--prompts_dir1", type=str, 
                       default="./separate_annots/text1/",
                       help="第一个提示文本目录")
    parser.add_argument("--prompts_dir2", type=str,
                       default="./separate_annots/text2/",
                       help="第二个提示文本目录")
    parser.add_argument("--motions_dir", type=str,
                       default="./motions/",
                       help="动作文件目录")
    parser.add_argument("--output_base_dir", type=str,
                       default="./features_output/",
                       help="特征输出基目录")
    parser.add_argument("--base_prompt1", type=str,
                       default="the first person",
                       help="第一个人基础提示")
    parser.add_argument("--base_prompt2", type=str,
                       default="the second person",
                       help="第二个人基础提示")
    
    # 单次推理参数（保持兼容性）
    parser.add_argument("--prompt", type=str, default="", help="单次推理的提示")
    parser.add_argument("--height", type=int, default=32, help="视频高度")
    parser.add_argument("--width", type=int, default=64, help="视频宽度")
    parser.add_argument("--num_frames", type=int, default=121, help="帧数")
    parser.add_argument("--pipeline_config", type=str, 
                       default="configs/ltxv-2b-0.9.8-distilled.yaml",
                       help="管道配置文件")
    parser.add_argument("--seed", type=int, default=171198, help="随机种子")
    
    args = parser.parse_args()
    
    if args.batch_mode:
        # 批量处理模式
        print("="*60)
        print("LTX Video 批量处理模式")
        print("="*60)
        
        batch_config = BatchInferenceConfig(
            prompts_dir1=args.prompts_dir1,
            prompts_dir2=args.prompts_dir2,
            motions_dir=args.motions_dir,
            output_base_dir=args.output_base_dir,
            height=args.height,
            width=args.width,
            pipeline_config=args.pipeline_config,
            base_prompt1=args.base_prompt1,
            base_prompt2=args.base_prompt2
        )
        
        batch_infer(batch_config)
    else:
        # 单次推理模式（保持原有功能）
        print("进入单次推理模式")
        
        # 为了保持兼容性，使用HfArgumentParser
        hf_parser = HfArgumentParser(InferenceConfig)
        hf_config = hf_parser.parse_args_into_dataclasses()[0]
        infer(config=hf_config)


if __name__ == "__main__":
    main()