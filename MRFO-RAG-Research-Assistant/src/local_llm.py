"""
本地LLM封装: 支持Qwen、GLM等开源模型
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Optional, List, Dict


class LocalLLM:
    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
            use_4bit: bool = True,
            device: str = "auto"
    ):
        """
        初始化本地LLM

        Args:
            model_name: 模型名称或路径
            use_4bit: 是否使用4bit量化(节省显存)
            device: 设备("auto", "cuda", "cpu")
        """
        print("=" * 60)
        print("🚀 初始化本地大语言模型")
        print("=" * 60)
        print(f"📦 模型: {model_name}")
        print(f"⚙️  量化: {'4-bit' if use_4bit else '16-bit'}")
        print(f"🖥️  设备: {device}")
        print()

        self.model_name = model_name

        # 配置量化(重要!4bit量化能把显存占用从14GB降到4GB)
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 4bit量化
                bnb_4bit_compute_dtype=torch.float16,  # 计算用float16
                bnb_4bit_quant_type="nf4",  # 量化类型
                bnb_4bit_use_double_quant=True,  # 双重量化
            )
            print("🔧 使用4-bit量化配置")
        else:
            quantization_config = None

        # 加载Tokenizer
        print("🔄 加载Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✅ Tokenizer加载完成")

        # 加载模型(这一步会下载,第一次很慢)
        print("\n🔄 加载模型(首次运行会下载,约4-7GB,请耐心等待)...")
        print("💡 提示: 下载后会缓存,下次启动很快\n")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,  # 自动分配到GPU
            trust_remote_code=True,
            torch_dtype=torch.float16 if not use_4bit else "auto"
        )

        print("\n✅ 模型加载完成!")

        # 显示显存占用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"📊 显存占用: {memory_allocated:.2f} GB (已分配)")
            print(f"📊 显存预留: {memory_reserved:.2f} GB (已预留)")

        print("=" * 60)
        print()

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            do_sample: bool = True,
            system_prompt: Optional[str] = None
    ) -> str:
        """
        生成回答

        Args:
            prompt: 用户问题
            max_new_tokens: 最大生成token数
            temperature: 温度(越高越随机,0.1-1.0)
            top_p: 核采样概率
            do_sample: 是否采样(False=贪婪解码)
            system_prompt: 系统提示词

        Returns:
            生成的文本
        """
        # 构建消息(符合Qwen的chat格式)
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # 应用chat模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt")

        # 移到GPU
        if torch.cuda.is_available():
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码(只取新生成的部分)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def chat(self, history: List[Dict[str, str]], user_message: str) -> str:
        """
        多轮对话

        Args:
            history: 对话历史 [{"role": "user", "content": "..."}, ...]
            user_message: 新的用户消息

        Returns:
            助手回复
        """
        # 添加新消息
        messages = history + [{"role": "user", "content": user_message}]

        # 应用模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


# ========== 测试代码 ==========
def test_basic_generation():
    """
    测试基础生成能力
    """
    print("\n" + "=" * 60)
    print("🧪 测试1: 基础问答")
    print("=" * 60)

    # 初始化模型(第一次会下载,要等一会儿)
    llm = LocalLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        use_4bit=True  # 使用4bit量化
    )

    # 简单问答
    questions = [
        "什么是大语言模型?用一句话解释。",
        "Python和Java有什么区别?",
        "解释一下什么是Attention机制。"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n❓ 问题{i}: {question}")
        print("-" * 60)

        answer = llm.generate(
            prompt=question,
            max_new_tokens=200,
            temperature=0.7
        )

        print(f"💡 回答: {answer}")
        print()


def test_with_system_prompt():
    """
    测试系统提示词
    """
    print("\n" + "=" * 60)
    print("🧪 测试2: 使用系统提示词")
    print("=" * 60)

    llm = LocalLLM(use_4bit=True)

    system_prompt = """你是一个MRFO算法专家,专门研究元启发式优化算法。
请用简洁专业的语言回答问题。"""

    question = "MRFO算法的核心思想是什么?"

    print(f"\n🎭 系统提示: {system_prompt}")
    print(f"\n❓ 问题: {question}")
    print("-" * 60)

    answer = llm.generate(
        prompt=question,
        system_prompt=system_prompt,
        temperature=0.5  # 更专业,降低温度
    )

    print(f"💡 回答: {answer}")


def test_multi_turn_chat():
    """
    测试多轮对话
    """
    print("\n" + "=" * 60)
    print("🧪 测试3: 多轮对话")
    print("=" * 60)

    llm = LocalLLM(use_4bit=True)

    # 对话历史
    history = []

    # 第一轮
    print("\n👤 用户: 什么是RAG?")
    response1 = llm.chat(history, "什么是RAG?")
    print(f"🤖 助手: {response1}")

    history.append({"role": "user", "content": "什么是RAG?"})
    history.append({"role": "assistant", "content": response1})

    # 第二轮
    print("\n👤 用户: 它有什么优势?")
    response2 = llm.chat(history, "它有什么优势?")
    print(f"🤖 助手: {response2}")


def main():
    """
    主函数
    """
    print("🚀 LocalLLM 测试程序")
    print()

    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("⚠️  GPU不可用,将使用CPU(会很慢)")

    print()

    # 运行测试
    try:
        test_basic_generation()
        test_with_system_prompt()  # 取消注释来测试
        test_multi_turn_chat()      # 取消注释来测试

        print("\n" + "=" * 60)
        print("✅ 所有测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
