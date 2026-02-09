"""
完整的RAG系统 - 使用微调后的MRFO专家模型
"""
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from document_processor import DocumentProcessor


class FinalRAGSystem:
    def __init__(
            self,
            finetuned_model_path: str = "./saves/mrfo_lora_quick",
            embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        初始化完整RAG系统(带微调模型)
        """
        print("=" * 70)
        print("🚀 初始化完整RAG系统 - 微调版")
        print("=" * 70)

        # 1. 文档处理器
        print("\n📄 初始化文档处理器...")
        self.doc_processor = DocumentProcessor(chunk_size=150, overlap=30)

        # 2. Embedding模型
        print("🔄 加载Embedding模型...")
        self.embedding_model = SentenceTransformer(embedding_model)

        # 3. 向量数据库
        print("💾 初始化向量数据库...")
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        try:
            self.collection = self.chroma_client.get_collection("mrfo_final")
        except:
            self.collection = self.chroma_client.create_collection("mrfo_final")

        # 4. 加载微调后的LLM
        print("\n🤖 加载微调后的MRFO专家模型...")
        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.llm = PeftModel.from_pretrained(base_model, finetuned_model_path)

        print("✅ 微调模型加载完成!")
        print("=" * 70)

    def add_pdf(self, pdf_path: str):
        """添加PDF到知识库"""
        print(f"\n📚 处理PDF: {pdf_path}")
        chunks = self.doc_processor.process_pdf(pdf_path, method='sentences')

        print(f"🔄 向量化 {len(chunks)} 个chunks...")
        for i, chunk_data in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk_data['text']).tolist()
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk_data['text']],
                metadatas=[chunk_data['metadata']],
                ids=[f"{os.path.basename(pdf_path)}_chunk_{i}"]
            )

        print(f"✅ 已添加 {len(chunks)} 个chunks")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关文档"""
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)

            retrieved_docs.append({
                'text': results['documents'][0][i],
                'similarity': similarity
            })

        return retrieved_docs

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        """使用微调模型生成答案"""
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

        with torch.no_grad():
            outputs = self.llm.generate(
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

    def query(self, question: str, use_rag: bool = True) -> Dict:
        """
        RAG查询

        Args:
            question: 用户问题
            use_rag: 是否使用检索增强(False=直接问模型)
        """
        print(f"\n❓ 问题: {question}")
        print("-" * 70)

        if use_rag:
            # RAG模式
            print("🔍 检索相关文档...")
            docs = self.retrieve(question, top_k=3)

            # 构建prompt
            context = "\n\n".join([
                f"【参考{i + 1}】{doc['text']}"
                for i, doc in enumerate(docs)
            ])

            prompt = f"""你是蝠鲼觅食优化算法（MRFO）的研究助手，请根据以下参考资料回答问题。

{context}

【问题】{question}

【要求】严格基于参考资料进行回答，不要混淆不同算法的功能或者作用，回答应该尽量包含关键信息。
【重要规则】
1. **只能**使用上述参考资料中明确提到的信息
2. **禁止**添加任何参考资料中没有的内容
3. **禁止**推测、猜测或联想
4. 如果参考资料不足,必须说"参考资料中没有足够信息回答这个问题"
5. 直接引用关键原文,可以用自己的话简洁总结，但是不能编造概念。
6.不确定的内容请直接引用原文，尤其是涉及具体数字的回答，有关精确实验数据之类的的请直接引用相关原文，不能自己胡乱填写或者混淆数据或者混淆数据对应的情境.
7.术语请保证中英文对应
8.不要混淆原算法和基于此改进的算法之间的功能和作用，两者的名字缩写有点像
请回答:"""

            print("📋 使用RAG模式")
        else:
            # 纯模型模式
            prompt = question
            print("🤖 使用纯模型模式")

        # 生成答案
        print("💡 生成答案...")
        answer = self.generate(prompt)

        print(f"\n✅ 回答:\n{answer}")

        return {
            'question': question,
            'answer': answer,
            'mode': 'RAG' if use_rag else 'Direct'
        }


def demo():
    """演示完整系统"""
    print("🎯 完整RAG系统演示")
    print()

    # 1. 初始化
    rag = FinalRAGSystem()

    # 2. 添加知识库
    pdf_path = "基于多策略改进MRFO算法的家庭能源调度优化2025-11-10.pdf"
    if os.path.exists(pdf_path):
        rag.add_pdf(pdf_path)

    # 3. 对比测试: RAG vs 纯模型
    print("\n" + "=" * 70)
    print("🧪 对比测试: RAG模式 vs 纯模型模式")
    print("=" * 70)

    test_questions = [
        "MRFO算法是什么?",  # 训练数据有
        "DLMMRFO在复杂场景下的优化效果如何?",  # 需要检索具体数字
        "本文的实验中用了哪些对比算法?",  # 训练数据可能没有,需要RAG
    ]

    for q in test_questions:
        print(f"\n{'=' * 70}")
        print(f"问题: {q}")
        print(f"{'=' * 70}")

        # 纯模型
        print("\n【方式1: 纯微调模型】")
        rag.query(q, use_rag=False)

        # RAG
        print("\n【方式2: RAG + 微调模型】")
        rag.query(q, use_rag=True)

        print("\n" + "-" * 70)
        input("按Enter继续...")

    print("\n" + "=" * 70)
    print("✅ 演示完成!")
    print("=" * 70)
    print("\n💡 总结:")
    print("  - 训练数据覆盖的问题: 微调模型直接回答准确")
    print("  - 需要文档细节的问题: RAG提供精确信息")
    print("  - 两者结合 = 完美的研究助手!")


if __name__ == "__main__":
    demo()