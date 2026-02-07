"""
使用LLaMA-Factory微调MRFO专家模型 - 改进版
增强训练参数,提升微调效果
"""
import os
import torch

def check_environment():
    """检查环境"""
    print("=" * 70)
    print("🔍 环境检查")
    print("=" * 70)

    # GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  未检测到GPU,将使用CPU(会很慢)")

    # 数据
    if os.path.exists("mrfo_training_data_complete.json"):
        import json
        with open("mrfo_training_data_complete.json", encoding='utf-8') as f:  # ⬅️ 修复编码
            data = json.load(f)
        print(f"✅ 训练数据: {len(data)} 条")
    else:
        print("❌ 未找到训练数据文件!")
        return False

    # LLaMA-Factory
    try:
        from llamafactory.train.tuner import run_exp
        print("✅ LLaMA-Factory已安装")
    except ImportError:
        print("❌ LLaMA-Factory未安装!")
        print("   请运行: pip install llamafactory")
        return False

    print("=" * 70)
    return True


def train(config_level="balanced"):
    """
    执行训练

    Args:
        config_level: 配置级别
            - "quick": 快速改进(5轮,学习率8e-5)
            - "balanced": 平衡改进(8轮,学习率1e-4) [推荐]
            - "aggressive": 激进改进(10轮,学习率1.5e-4)
    """
    from llamafactory.train.tuner import run_exp

    print("\n" + "=" * 70)
    print(f"🚀 开始微调MRFO专家模型 - {config_level.upper()}模式")
    print("=" * 70)
    print()

    # 根据配置级别设置参数
    if config_level == "quick":
        epochs = 30
        lr = 5e-4
        lora_rank = 16
        lora_alpha = 32
        grad_accum = 8
        warmup = 0.15
        print("📋 配置: 快速改进")
        print("   - 适合: 效果略有改善但不够明显")
        print("   - 预计时间: 12-15分钟")

    elif config_level == "aggressive":
        epochs = 10
        lr = 1.5e-4
        lora_rank = 32
        lora_alpha = 64
        grad_accum = 8
        warmup = 0.2
        print("📋 配置: 激进改进")
        print("   - 适合: 确保完全记住所有训练数据")
        print("   - 预计时间: 35-40分钟")
        print("   ⚠️  显存可能接近4GB上限")

    else:  # balanced (默认推荐)
        epochs = 8
        lr = 1e-4
        lora_rank = 16
        lora_alpha = 32
        grad_accum = 8
        warmup = 0.15
        print("📋 配置: 平衡改进 [推荐]")
        print("   - 适合: 大多数情况")
        print("   - 预计时间: 20-25分钟")

    # 训练参数
    args = {
        # 模型
        "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "quantization_bit": 4,
        "quantization_method": "bitsandbytes",

        # LoRA (改进)
        "finetuning_type": "lora",
        "lora_rank": lora_rank,           # ⬅️ 从8改为16/32
        "lora_alpha": lora_alpha,         # ⬅️ 从16改为32/64
        "lora_dropout": 0.05,
        "lora_target": "all",

        # 数据
        "dataset": "mrfo_dataset",
        "dataset_dir": "./",
        "template": "qwen",
        "cutoff_len": 512,
        "val_size": 0.1,
        "overwrite_cache": True,

        # 训练 (改进)
        "stage": "sft",
        "do_train": True,
        "output_dir": f"./saves/mrfo_lora_{config_level}",  # 不同配置保存到不同文件夹
        "overwrite_output_dir": True,

        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,  # ⬅️ 从4改为8
        "learning_rate": lr,                        # ⬅️ 从5e-5改为更高
        "num_train_epochs": epochs,                 # ⬅️ 从3改为5/8/10

        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": warmup,                     # ⬅️ 从0.1改为0.15/0.2

        "logging_steps": 5,
        "save_steps": 50,
        "save_total_limit": 2,

        "fp16": True,
        "report_to": "none",
        "seed": 42,
    }

    print(f"\n📊 详细配置:")
    print(f"   模型: Qwen2.5-1.5B-Instruct")
    print(f"   数据: 56条")
    print(f"   训练轮数: {epochs} epochs")
    print(f"   学习率: {lr}")
    print(f"   LoRA Rank: {lora_rank}")
    print(f"   Batch size: 1 × {grad_accum}(累积) = {grad_accum}")
    print(f"   Warmup: {warmup*100:.0f}%")
    print()

    try:
        # 执行训练
        run_exp(args)

        print("\n" + "=" * 70)
        print("✅ 训练完成!")
        print("=" * 70)
        print(f"📁 模型保存在: ./saves/mrfo_lora_{config_level}")
        print()
        print("💡 查看训练效果:")
        print("   - 如果Loss < 1.5: 基本成功")
        print("   - 如果Loss < 1.0: 很好")
        print("   - 如果Loss < 0.5: 完美!")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    # 1. 检查环境
    if not check_environment():
        return

    # 2. 选择配置级别
    print("\n" + "=" * 70)
    print("🎛️  选择训练配置")
    print("=" * 70)
    print()
    print("1. 快速改进 (5轮, ~15分钟)")
    print("   - 适合: 第一次微调效果略有改善")
    print("   - Loss目标: < 1.5")
    print()
    print("2. 平衡改进 (8轮, ~25分钟) [⭐推荐]")
    print("   - 适合: 大多数情况,成功率高")
    print("   - Loss目标: < 1.0")
    print()
    print("3. 激进改进 (10轮, ~40分钟)")
    print("   - 适合: 确保完全记住训练数据")
    print("   - Loss目标: < 0.5")
    print("   - 注意: 显存占用接近4GB")
    print()

    choice = input("请选择配置 (1/2/3, 默认2): ").strip()

    config_map = {
        "1": "quick",
        "2": "balanced",
        "3": "aggressive",
        "": "balanced"  # 默认
    }

    config_level = config_map.get(choice, "balanced")

    # 3. 确认开始
    print("\n" + "=" * 70)
    print(f"准备开始训练 - {config_level.upper()}模式")
    print("=" * 70)
    print()
    print("训练过程中你会看到:")
    print("  ✅ Loss逐渐下降")
    print("  ✅ 显存占用稳定")
    print("  ⏱️  进度条显示剩余时间")
    print()

    input("按Enter开始训练...")

    # 4. 开始训练
    train(config_level)

    print("\n" + "=" * 70)
    print("🎯 下一步: 测试微调效果")
    print("=" * 70)
    print()
    print("运行测试:")
    print(f"   python test_finetuned_model.py")
    print()
    print("注意: 测试时需要修改模型路径为:")
    print(f"   lora_path = './saves/mrfo_lora_{config_level}'")


if __name__ == "__main__":
    main()