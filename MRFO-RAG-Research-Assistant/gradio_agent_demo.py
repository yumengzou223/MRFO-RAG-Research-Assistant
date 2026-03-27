"""
MRFO研究助手 - Agent智能路由版本
===================================

## 核心功能

Agent会自动判断问题类型，选择最合适的模型：
- 问题含"原MRFO"、"原始MRFO"→ 用基础模型回答
- 问题含"DLM MRFO"、"深度学习增强"→ 用LoRA模型回答
- 问"MRFO算法原理"→ 先用基础模型，再用LoRA补充

## 运行方法
python gradio_agent_demo.py
浏览器访问 http://localhost:7860

===================================
"""
import os
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

WORK_DIR = r'D:\Anaconda\envs\rag_project\MRFO-RAG-Research-Assistant'
PARENT_DIR = r'D:\Anaconda\envs\rag_project'
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, WORK_DIR)

import re
import gradio as gr
from src.advanced_rag_system_v2 import AdvancedRAGv2

print("=" * 60)
print("MRFO研究助手 - Agent智能路由版本")
print("=" * 60)

# ============================================================
# 初始化两个RAG系统
# ============================================================
print("\n[1/4] 初始化RAG系统（基础模型 - 用于原MRFO）...")
rag_base = AdvancedRAGv2(
    collection_name="research_knowledge_base_v5",
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_path=None,  # 基础模型，不加载LoRA
    use_hybrid=True,
    use_rerank=True,
    vector_top_k=20,
    bm25_top_k=20,
    final_top_k=8,
    rerank_top_k=12
)
print("✅ 基础模型就绪")

print("\n[2/4] 初始化RAG系统（LoRA模型 - 用于DLM MRFO）...")
rag_lora = AdvancedRAGv2(
    collection_name="research_knowledge_base_v5",
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_path=r"D:\Anaconda\envs\rag_project\saves\mrfo_lora_aggressive",
    lora_rank=32,
    lora_alpha=64,
    use_hybrid=True,
    use_rerank=True,
    vector_top_k=20,
    bm25_top_k=20,
    final_top_k=8,
    rerank_top_k=12
)
print("✅ LoRA模型就绪")

print("\n[3/4] Agent路由就绪")
print("=" * 60)

# ============================================================
# Agent路由函数
# ============================================================
def classify_question(question: str) -> str:
    """
    判断问题类型，决定用哪个模型
    
    返回值：
    - "base": 用基础模型（原MRFO问题）
    - "lora": 用LoRA模型（DLM MRFO问题）
    - "both": 两个都用，合并答案
    """
    q = question.lower()
    
    # 明确指明原MRFO → 基础模型
    if any(kw in q for kw in ['原mrfo', '原始mrfo', '原始的mrfo', 'base mrfo']):
        return "base"
    
    # 明确指明DLM MRFO → LoRA模型
    if any(kw in q for kw in ['dlm mrfo', '深度学习增强mrfo', '改进mrfo']):
        return "lora"
    
    # 问"核心思想"、"原理"、"三种觅食" → 基础模型
    if any(kw in q for kw in ['核心思想', '原理', '三种觅食', '链式', '螺旋', '翻滚']):
        return "base"
    
    # 问"改进"、"增强"、"优化" → LoRA模型
    if any(kw in q for kw in ['改进', '增强', '优化', 'dlm']):
        return "lora"
    
    # 问"成本", "par", "数字" → 两个都用
    if any(kw in q for kw in ['成本', 'par', '降低', '百分比', '数字', '结果']):
        return "both"
    
    # 默认用基础模型
    return "base"

def agent_chat(question: str, history: list) -> str:
    """
    Agent核心：根据问题类型路由到合适的模型
    """
    print(f"\n收到问题: {question}")
    
    # Step 1: 分类判断
    question_type = classify_question(question)
    print(f"Agent判断: {question_type} (base=原MRFO, lora=DLM MRFO, both=合并)")
    
    try:
        if question_type == "base":
            # 用基础模型回答
            print("→ 使用基础模型（原MRFO）")
            result = rag_base.query(question, show_sources=False)
            answer = result['answer']
            
        elif question_type == "lora":
            # 用LoRA模型回答
            print("→ 使用LoRA模型（DLM MRFO）")
            result = rag_lora.query(question, show_sources=False)
            answer = result['answer']
            
        else:  # both
            # 两个模型都回答，合并结果
            print("→ 使用双模型（合并答案）")
            
            # 先用基础模型
            print("  [1/2] 基础模型回答中...")
            result_base = rag_base.query(question, show_sources=False)
            answer_base = result_base['answer']
            
            # 再用LoRA模型
            print("  [2/2] LoRA模型回答中...")
            result_lora = rag_lora.query(question, show_sources=False)
            answer_lora = result_lora['answer']
            
            # 合并答案
            answer = f"""【综合分析】

【基础模型观点】（关于原MRFO）：
{answer_base}

---
【LoRA模型观点】（关于DLM MRFO）：
{answer_lora}

---
注：综合两个模型的回答，可以更全面地理解MRFO和DLM MRFO的关系。"""
        
        print(f"回答: {answer[:80]}...")
        return answer
        
    except Exception as e:
        error_msg = f"出错了: {str(e)}"
        print(error_msg)
        return error_msg

# ============================================================
# 创建Gradio界面
# ============================================================
print("\n[4/4] 创建界面...")

demo = gr.ChatInterface(
    fn=agent_chat,
    title="🔍 MRFO研究助手 (Agent智能版)",
    description="""
    <b>智能问答系统</b> - Agent会自动判断问题类型选择最合适的模型：
    <br>• 问<font color="blue">原MRFO算法原理</font>→ 基础模型回答
    <br>• 问<font color="green">DLM MRFO改进效果</font>→ LoRA模型回答
    <br>• 问<font color="purple">成本/数字比较</font>→ 综合两个模型
    """,
    
    textbox=gr.Textbox(
        label="输入问题",
        placeholder="例如：原MRFO算法的核心思想是什么？DLM MRFO改进了什么？",
        lines=3
    ),
    
    theme="soft",
    
    examples=[
        ["MRFO算法的三种觅食策略是什么？"],
        ["DLM MRFO在简单场景下相比MRFO降低了多少成本？"],
        ["原MRFO算法的核心思想是什么？"],
        ["DLM MRFO相比原始MRFO做了哪些改进？"],
        ["简单场景和复杂场景分别有多少台设备？"],
    ]
)

print("✅ 界面创建完成!")
print("\n" + "=" * 60)
print("🎉 访问地址: http://localhost:7860")
print("=" * 60)

demo.launch(
    server_name="0.0.0.0",
    share=False,
    inbrowser=True
)
