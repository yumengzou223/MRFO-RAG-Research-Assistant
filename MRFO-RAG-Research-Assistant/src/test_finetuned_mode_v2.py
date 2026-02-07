"""
测试微调后的MRFO专家模型 - 改进版
可以选择测试不同版本的微调模型
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


class FinetunedModelTester:
    def __init__(self, lora_path):
        print("=" * 70)
        print("🧪 加载微调后的模型")
        print("=" * 70)
        print(f"📁 LoRA路径: {lora_path}")

        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        # 1. 加载tokenizer
        print("\n🔄 加载Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

        # 2. 加载基础模型(4bit)
        print("🔄 加载基础模型...")
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 3. 加载LoRA权重
        print("🔄 加载LoRA微调权重...")
        self.model = PeftModel.from_pretrained(base_model, lora_path)

        print("✅ 微调模型加载完成!")
        print("=" * 70)

    def generate(self, question: str, max_new_tokens: int = 256) -> str:
        """生成回答"""
        messages = [
            {"role": "user", "content": question}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        return response


def test_critical_questions(model):
    """
    测试关键问题(之前容易出错的)
    """
    print("\n" + "=" * 70)
    print("🎯 关键问题测试")
    print("=" * 70)

    critical_tests = [
        {
            "question": "什么是MRFO算法?",
            "check_keywords": ["蝠鲼", "Manta Ray", "觅食优化"],
            "avoid_keywords": ["Multi-Objective"],
            "note": "检查是否正确理解MRFO"
        },
        {
            "question": "在复杂场景下,DLM MRFO相比MRFO算法降低了多少成本?",
            "check_keywords": ["5.89%"],
            "avoid_keywords": ["53.29%", "7.89%"],
            "note": "检查数字准确性"
        },
        {
            "question": "MRFO算法的三种觅食策略是什么?",
            "check_keywords": ["链式", "螺旋", "翻滚"],
            "avoid_keywords": [],
            "note": "检查基础概念"
        },
        {
            "question": "DLM MRFO引入了哪些改进机制?",
            "check_keywords": ["离散", "动态权重", "长时记忆", "变异", "PAR"],
            "avoid_keywords": [],
            "note": "检查多点记忆(至少包含3个)"
        },
        {
            "question": "什么是峰值平均比PAR?",
            "check_keywords": ["Peak-to-Average", "峰值负载", "平均负载"],
            "avoid_keywords": [],
            "note": "检查术语理解"
        }
    ]

    total_score = 0
    max_score = 0

    for i, test in enumerate(critical_tests, 1):
        print(f"\n{'=' * 70}")
        print(f"测试 {i}/{len(critical_tests)}: {test['note']}")
        print(f"{'=' * 70}")
        print(f"\n❓ 问题: {test['question']}")
        print("-" * 70)

        answer = model.generate(test['question'], max_new_tokens=250)
        print(f"\n💡 回答:\n{answer}")

        # 评分
        score = 0
        check_count = 0

        print(f"\n📊 评估:")

        # 检查必须包含的关键词
        for keyword in test['check_keywords']:
            if keyword in answer:
                print(f"  ✅ 包含关键词: {keyword}")
                score += 1
            else:
                print(f"  ❌ 缺少关键词: {keyword}")
            check_count += 1

        # 检查不应包含的关键词
        for keyword in test['avoid_keywords']:
            if keyword in answer:
                print(f"  ⚠️  包含错误内容: {keyword}")
                score -= 0.5
            check_count += 0.5

        total_score += score
        max_score += len(test['check_keywords'])

        print(f"\n得分: {score}/{len(test['check_keywords'])}")
        print("-" * 70)
        input("按Enter继续...")

    print("\n" + "=" * 70)
    print("📊 总体评分")
    print("=" * 70)
    accuracy = (total_score / max_score * 100) if max_score > 0 else 0
    print(f"✅ 准确率: {accuracy:.1f}% ({total_score:.1f}/{max_score})")

    if accuracy >= 80:
        print("🎉 优秀! 微调效果很好!")
    elif accuracy >= 60:
        print("👍 良好,但还有改进空间")
    else:
        print("⚠️  效果不够理想,建议用更激进的配置重新训练")

    return accuracy


def main():
    print("🎓 微调模型测试程序 v2.0")
    print()

    # 1. 选择要测试的模型版本
    print("可用的微调模型:")
    models = []

    if os.path.exists("./saves/mrfo_lora_quick"):
        models.append(("quick", "./saves/mrfo_lora_quick"))
        print("  1. 快速改进版")

    if os.path.exists("./saves/mrfo_lora_balanced"):
        models.append(("balanced", "./saves/mrfo_lora_balanced"))
        print("  2. 平衡改进版")

    if os.path.exists("./saves/mrfo_lora_aggressive"):
        models.append(("aggressive", "./saves/mrfo_lora_aggressive"))
        print("  3. 激进改进版")

    if os.path.exists("./saves/mrfo_lora"):
        models.append(("original", "./saves/mrfo_lora"))
        print("  4. 原始版本(第一次训练)")

    if not models:
        print("❌ 未找到任何微调模型!")
        print("请先运行训练: python run_training_v2.py")
        return

    print()
    choice = input(f"选择要测试的模型 (1-{len(models)}): ").strip()

    try:
        idx = int(choice) - 1
        model_name, model_path = models[idx]
    except:
        print("无效选择,使用最新的模型")
        model_name, model_path = models[-1]

    print(f"\n✅ 将测试: {model_name} ({model_path})")

    # 2. 加载模型
    model = FinetunedModelTester(model_path)

    # 3. 运行测试
    accuracy = test_critical_questions(model)

    # 4. 建议
    print("\n" + "=" * 70)
    print("💡 改进建议")
    print("=" * 70)

    if accuracy < 60:
        print("\n效果不够理想,建议:")
        print("  1. 使用更激进的配置重新训练")
        print("     python run_training_v2.py")
        print("     选择 '3. 激进改进'")
        print()
        print("  2. 或者增加训练数据")
        print("     从论文中提取更多QA对")
    elif accuracy < 80:
        print("\n效果良好但可以更好,建议:")
        print("  - 如果用的是'快速'配置,试试'平衡'或'激进'")
        print("  - 检查训练Loss是否充分下降(< 1.0)")
    else:
        print("\n✅ 微调效果很好!")
        print("可以进入下一步: 整合到RAG系统")


if __name__ == "__main__":
    main()