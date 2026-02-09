"""
改进版RAG系统 - 修复相似度计算和减少幻觉
"""
import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from document_processor import DocumentProcessor
from local_llm import LocalLLM


class AdvancedRAGv2:
    def __init__(
            self,
            embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
            collection_name: str = "research_knowledge_base_v2"
    ):
        """
        初始化改进版RAG系统
        """
        print("=" * 70)
        print("🚀 初始化改进版RAG系统 v2.0")
        print("=" * 70)

        # 1. 初始化文档处理器(更小的chunk)
        print("\n📄 初始化文档处理器...")
        self.doc_processor = DocumentProcessor(
            chunk_size=300,  # ⬅️ 改小
            overlap=80  # ⬅️ 增加
        )
        print("✅ 文档处理器就绪(chunk_size=300, overlap=80)")

        # 2. 初始化Embedding模型
        print("\n🔄 加载Embedding模型...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("✅ Embedding模型加载完成")

        # 3. 初始化向量数据库
        print("\n💾 初始化向量数据库...")
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"✅ 已加载现有知识库: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(name=collection_name)
            print(f"✅ 已创建新知识库: {collection_name}")

        # 4. 初始化LLM
        print("\n🤖 初始化本地LLM...")
        self.llm = LocalLLM(
            model_name=llm_model_name,
            use_4bit=True
        )

        print("\n" + "=" * 70)
        print("✅ RAG系统初始化完成!")
        print("=" * 70)
        print()

    def add_documents_from_pdf(self, pdf_path: str) -> int:
        """从PDF添加文档到知识库"""
        print(f"\n📚 正在处理PDF: {pdf_path}")
        print("-" * 70)

        chunks = self.doc_processor.process_pdf(pdf_path, method='sentences')

        if not chunks:
            print("❌ PDF处理失败")
            return 0

        print(f"\n🔄 正在为 {len(chunks)} 个chunks生成向量...")

        for i, chunk_data in enumerate(chunks):
            text = chunk_data['text']
            metadata = chunk_data['metadata']

            embedding = self.embedding_model.encode(text).tolist()
            doc_id = f"{os.path.basename(pdf_path)}_chunk_{i}"

            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )

            if (i + 1) % 10 == 0:
                print(f"  已处理: {i + 1}/{len(chunks)} chunks")

        print(f"\n✅ 成功添加 {len(chunks)} 个chunks到知识库!")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关文档(修复了相似度计算)
        """
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i] if results['distances'] else 0

            # ⬇️ 修复: 正确计算相似度
            similarity = 1 / (1 + distance)

            retrieved_docs.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'similarity': similarity,
                'distance': distance
            })

        return retrieved_docs

    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        构建严格的prompt,减少幻觉
        """
        context = "\n\n".join([
            f"【参考资料{i + 1}】\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        # ⬇️ 更严格的prompt
        prompt = f"""你是一个严谨的研究助手。请**严格**根据以下参考资料回答问题。

【参考资料】
{context}

【用户问题】
{query}

【重要规则】
1. **只能**使用上述参考资料中明确提到的信息
2. **禁止**添加任何参考资料中没有的内容
3. **禁止**推测、猜测或联想
4. 如果参考资料不足,必须说"参考资料中没有足够信息回答这个问题"
5. 直接引用关键原文,用自己的话简洁总结
6.不确定的内容请直接引用原文，尤其是涉及具体数字的回答，不能自己胡乱填写或者混淆数据
7.术语请保证中英文对应

请严格遵守规则回答:"""

        return prompt

    def query(
            self,
            question: str,
            top_k: int = 5,
            show_sources: bool = True,
            temperature: float = 0.3
    ) -> Dict:
        """
        RAG查询(改进版)
        """
        print(f"\n❓ 用户问题: {question}")
        print("-" * 70)

        # 1. 检索
        print(f"🔍 正在检索相关文档(Top {top_k})...")
        retrieved_docs = self.retrieve(question, top_k)

        if show_sources:
            print(f"\n📋 检索到 {len(retrieved_docs)} 个相关文档:")
            for i, doc in enumerate(retrieved_docs):
                # ⬇️ 显示正确的相似度
                print(f"\n  [{i + 1}] 相似度: {doc['similarity']:.3f} | 距离: {doc['distance']:.2f}")
                print(f"      {doc['text'][:200]}...")

        # 2. 构建prompt
        prompt = self.build_prompt(question, retrieved_docs)

        # 3. 生成
        print(f"\n🤖 LLM正在生成答案(temperature={temperature})...")
        answer = self.llm.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=temperature,  # ⬅️ 更保守
            do_sample=True
        )

        print(f"\n💡 回答:\n{answer}")

        return {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs
        }


# ========== 演示代码 ==========
def demo():
    """完整演示"""
    print("🎯 改进版RAG系统演示\n")

    # 1. 初始化
    rag = AdvancedRAGv2()

    # 2. 添加知识
    pdf_path = "基于多策略改进MRFO算法的家庭能源调度优化 (已自动恢复).pdf"

    if os.path.exists(pdf_path):
        print("\n" + "=" * 70)
        print("📚 Step 1: 构建知识库")
        print("=" * 70)

        num_chunks = rag.add_documents_from_pdf(pdf_path)
        print(f"\n✅ 知识库已包含 {num_chunks} 个文档片段")
    else:
        print(f"⚠️  未找到PDF: {pdf_path}")
        return

    # 3. 测试查询
    print("\n" + "=" * 70)
    print("🧪 Step 2: 测试RAG查询")
    print("=" * 70)

    test_questions = [
        "MRFO算法的三种觅食策略是什么?",  # ⬅️ 更具体的问题
        "DLM MRFO算法引入了哪些改进机制?",
        "在复杂场景下,DLM MRFO算法相比MRFO算法降低了多少成本?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"测试问题 {i}/{len(test_questions)}")
        print(f"{'=' * 70}")

        result = rag.query(
            question=question,
            top_k=5,  # ⬅️ 检索5个文档
            show_sources=True,
            temperature=0.3  # ⬅️ 更保守
        )

        print("\n" + "-" * 70)
        input("按Enter继续...")

    print("\n" + "=" * 70)
    print("✅ 演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
