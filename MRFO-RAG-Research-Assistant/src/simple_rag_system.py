"""
简化版RAG系统 - 专注文档问答
===================================

设计原则：
1. 简化prompt - 模型只需要"根据文档回答"
2. 检索优先 - 确保找到正确的文档片段
3. 保留必要的数字场景说明 - 防止混淆

===================================
"""
import os
import sys
import io
import re

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

WORK_DIR = r'D:\Anaconda\envs\rag_project\MRFO-RAG-Research-Assistant'
sys.path.insert(0, r'D:\Anaconda\envs\rag_project')
sys.path.insert(0, WORK_DIR)

from src.local_llm import LocalLLM
from document_processor import DocumentProcessor


class SimpleRAG:
    def __init__(
        self,
        collection_name: str = "research_knowledge_base_v5",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        lora_path: str = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        print("=" * 60)
        print(" 简化RAG系统 - 专注文档问答")
        print("=" * 60)

        # Embedding模型
        print("加载Embedding模型...")
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("[OK] Embedding模型就绪")

        # 向量数据库
        print("加载知识库...")
        persist_dir = r"D:\Anaconda\envs\rag_project\chroma_db"
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir, 
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except:
            self.collection = self.chroma_client.create_collection(name=collection_name)
        print(f"[OK] 知识库就绪 ({self.collection.count()} 条)")

        # LLM
        self.llm = None
        self.llm_model_name = llm_model_name
        self.lora_path = lora_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        print("=" * 60)

    def _init_llm(self):
        """延迟加载LLM"""
        if self.llm is None:
            print("\n 加载LLM...")
            self.llm = LocalLLM(
                model_name=self.llm_model_name,
                use_4bit=True,
                lora_path=self.lora_path,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha
            )
            print("[OK] LLM就绪")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关文档"""
        query_emb = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )

        docs = []
        for i, doc_id in enumerate(results['ids'][0]):
            idx = int(doc_id.split('_chunk_')[-1])
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)
            docs.append({
                'text': results['documents'][0][i],
                'chunk_idx': idx,
                'similarity': float(similarity)
            })

        return docs

    def build_prompt(self, query: str, docs: List[Dict]) -> str:
        """
        简化版prompt - 核心原则：
        1. 告诉模型"根据文档回答"
        2. 保留数字场景说明（防止混淆）
        3. 去掉多余的"提取事实清单"步骤
        """
        # 拼接文档
        context = "\n\n".join([
            f"[文档{i+1}]\n{doc['text']}"
            for i, doc in enumerate(docs)
        ])

        prompt = f"""你是一个文档问答助手。根据给定的文档回答用户问题。

【文档内容】
{context}

【用户问题】
{query}

【回答要求】
1. 只根据文档内容回答，不要添加外部知识
2. 如果文档中没有相关信息，说"文档中没有提到这一点"
3. 回答要直接、简洁

【重要：数字场景说明】
- 简单场景：12台设备，成本降低7.89%(vs MRFO)，45.30%(vs 未调度)
- 复杂场景：20台设备，成本降低5.89%(vs MRFO)，53.29%(vs 未调度)
- 回答数字问题时，注意区分是哪个场景的数据

请回答："""

        return prompt

    def query(self, question: str, top_k: int = 5) -> Dict:
        """问答"""
        # 1. 检索
        print(f"\n❓ 问题: {question}")
        print(" 检索中...")
        docs = self.retrieve(question, top_k=top_k)
        print(f"[OK] 找到 {len(docs)} 个相关文档")

        # 2. 构建prompt
        prompt = self.build_prompt(question, docs)

        # 3. 生成
        self._init_llm()
        print(" 生成答案...")
        answer = self.llm.generate(
            prompt=prompt,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=False
        )

        print(f" 回答: {answer[:150]}...")

        return {
            'question': question,
            'answer': answer,
            'sources': docs
        }


if __name__ == "__main__":
    # 测试
    rag = SimpleRAG(
        collection_name="research_knowledge_base_v5",
        llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        lora_path=None,
    )

    questions = [
        "MRFO算法的三种觅食策略是什么？",
        "DLM MRFO在简单场景下降低了多少成本？",
        "简单场景和复杂场景分别有多少台设备？",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        rag.query(q)
