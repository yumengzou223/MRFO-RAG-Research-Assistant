"""
RAG v3.0 测试脚本
测试混合搜索 + Cross-Encoder重排序的效果
"""
import sys
import os
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 确保能导入src模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

WORK_DIR = r"D:\Anaconda\envs\rag_project"
os.chdir(WORK_DIR)

# 用Python找PDF
import glob
pdf_files = glob.glob(os.path.join(WORK_DIR, "*.pdf"))
for pf in pdf_files:
    print(f"Found PDF: {pf}")

pdf_path = None
for pf in pdf_files:
    if "2025-11-10" in pf or "MRFO" in pf:
        pdf_path = pf
        break

if not pdf_path:
    print("No MRFO PDF found!")
    sys.exit(1)

print(f"\nUsing PDF: {pdf_path}\n")

from src.advanced_rag_system_v2 import AdvancedRAGv2


def test_retriever_only():
    """只测试检索器，不调用LLM（省显存省时间）"""
    print("=" * 70)
    print("🧪 测试检索器 (不调用LLM)")
    print("=" * 70)

    rag = AdvancedRAGv2(
        use_hybrid=True,
        use_rerank=True,
        vector_top_k=20,
        bm25_top_k=20,
        final_top_k=5,
        rerank_top_k=10
    )

    # 检查/构建知识库
    stats = rag.get_stats()
    print(f"\n📊 知识库状态: {stats}")

    if stats["total_chunks"] == 0:
        print("\n📚 知识库为空，正在从PDF构建...")
        rag.add_documents_from_pdf(pdf_path)
        stats = rag.get_stats()
        print(f"✅ 知识库构建完成: {stats}")

    print("\n" + "=" * 70)
    print("🔍 检索效果对比测试")
    print("=" * 70)

    test_queries = [
        "MRFO算法的三种觅食策略是什么?",
        "DLM MRFO算法引入了哪些改进机制?",
        "在复杂场景下,DLM MRFO相比于MRFO降低了多少成本?降低了多少PAR?",
        "长时记忆机制的实现细节",
        "PAR专项优化策略的意义",
        "实验中日总成本是多少?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"❓ [{i}/{len(test_queries)}] {query}")
        print("-" * 70)

        start = time.time()
        results = rag.retrieve(query, top_k=5)
        elapsed = time.time() - start

        print(f"   检索耗时: {elapsed:.2f}s | 结果数: {len(results)}")
        for j, r in enumerate(results, 1):
            cross_info = ""
            if 'cross_score' in r:
                cross_info = f" | 交叉分:{r['cross_score']:.3f} | 综合:{r['combined_score']:.3f}"
            print(f"   [{j}] 得分:{r['similarity']:.3f}{cross_info}")
            print(f"       {r['text'][:150]}...")

    print("\n" + "=" * 70)
    print("✅ 检索测试完成!")
    print("=" * 70)
    return rag


def test_full_rag(rag):
    """完整RAG测试 (含LLM生成)"""
    print("\n" + "=" * 70)
    print("🤖 完整RAG测试 (含LLM生成)")
    print("=" * 70)

    test_questions = [
        "MRFO算法的三种觅食策略是什么?",
        "DLM MRFO算法在复杂场景下相比于MRFO降低了多少成本?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"❓ [{i}/{len(test_questions)}] {question}")
        print("-" * 70)
        result = rag.query(question, show_sources=True, temperature=0.3)
        print(f"\n📝 回答: {result['answer'][:200]}")
        print("-" * 70)


if __name__ == "__main__":
    print("🚀 开始RAG v3.0 测试\n")

    # Step 1: 只测检索器
    rag = test_retriever_only()

    # Step 2: 确认硬件状态
    import torch
    if torch.cuda.is_available():
        print(f"\n📊 GPU显存占用: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
        print(f"📊 GPU显存预留: {torch.cuda.memory_reserved()/(1024**3):.2f} GB")

    # Step 3: 完整RAG测试 (可选，按需取消注释)
    # print("\n是否运行完整RAG测试 (含LLM生成)? 需要约2-3GB显存")
    # response = input("按Enter继续，输入n跳过: ")
    # if response.strip().lower() != 'n':
    #     test_full_rag(rag)

    print("\n✅ 所有测试完成!")
