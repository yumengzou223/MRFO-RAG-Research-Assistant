"""
Agent路由系统 - 完整测试
===================================
使用双模型智能路由：
- 基础模型：回答MRFO概念问题
- LoRA模型：回答DLM MRFO数字问题
- 合并模式：两者都用，综合回答

测试所有问题并评估结果
===================================
"""
import os
import sys
import re

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

WORK_DIR = r'D:\Anaconda\envs\rag_project\MRFO-RAG-Research-Assistant'
sys.path.insert(0, r'D:\Anaconda\envs\rag_project')
sys.path.insert(0, WORK_DIR)

from src.advanced_rag_system_v2 import AdvancedRAGv2

print("=" * 60)
print("MRFO Agent - 双模型智能路由测试")
print("=" * 60)

# ============================================================
# 初始化两个RAG系统
# ============================================================
print("\n[1/3] 初始化基础模型...")
rag_base = AdvancedRAGv2(
    collection_name="research_knowledge_base_v5",
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_path=None,
    use_hybrid=True, use_rerank=True
)
print("    [OK] 基础模型就绪")

print("\n[2/3] 初始化LoRA模型...")
rag_lora = AdvancedRAGv2(
    collection_name="research_knowledge_base_v5",
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_path=r"D:\Anaconda\envs\rag_project\saves\mrfo_lora_aggressive",
    lora_rank=32, lora_alpha=64,
    use_hybrid=True, use_rerank=True
)
print("    [OK] LoRA模型就绪")

# ============================================================
# Agent路由函数
# ============================================================
def classify_question(question: str) -> str:
    """判断问题类型"""
    q = question.lower()
    
    # 明确指明原MRFO基础概念
    if any(kw in q for kw in ['原mrfo', '原始mrfo', '原始的mrfo']):
        return "base"
    
    # 明确指明DLM MRFO
    if any(kw in q for kw in ['dlm mrfo', '深度学习增强', '改进mrfo']):
        return "lora"
    
    # 问基础概念
    if any(kw in q for kw in ['核心思想', '原理', '三种觅食', '链式', '螺旋', '翻滚', '觅食策略']):
        return "base"
    
    # 问改进内容
    if any(kw in q for kw in ['改进', '增强', '优化', '做了哪些']):
        return "lora"
    
    # 问数字/成本
    if any(kw in q for kw in ['成本', 'par', '降低', '百分比', '数字', '结果']):
        return "both"
    
    # 默认用基础模型
    return "base"

def agent_query(question: str) -> str:
    """Agent查询"""
    qtype = classify_question(question)
    
    if qtype == "base":
        print(f"    [路由 -> 基础模型]")
        result = rag_base.query(question, show_sources=False)
        return result['answer'], "基础模型"
        
    elif qtype == "lora":
        print(f"    [路由 -> LoRA模型]")
        result = rag_lora.query(question, show_sources=False)
        return result['answer'], "LoRA模型"
        
    else:  # both
        print(f"    [路由 -> 双模型合并]")
        r1 = rag_base.query(question, show_sources=False)
        r2 = rag_lora.query(question, show_sources=False)
        combined = f"""【基础模型回答】
{r1['answer']}

【LoRA模型回答】
{r2['answer']}"""
        return combined, "双模型合并"

# ============================================================
# 测试题目
# ============================================================
print("\n[3/3] 开始测试...")
print("=" * 60)

test_cases = [
    # 数字类问题
    {
        "id": "N-01",
        "q": "DLM MRFO在简单场景下相比MRFO算法降低了多少成本？",
        "expected": ["7.89%"],
        "type": "lora"
    },
    {
        "id": "N-02",
        "q": "DLM MRFO在复杂场景下相比MRFO算法降低了多少成本？",
        "expected": ["5.89%"],
        "type": "lora"
    },
    {
        "id": "N-03",
        "q": "DLM MRFO在简单场景下相比未调度基准降低了多少成本？",
        "expected": ["45.30%"],
        "type": "lora"
    },
    {
        "id": "N-04",
        "q": "DLM MRFO在复杂场景下相比未调度基准降低了多少成本？",
        "expected": ["53.29%"],
        "type": "lora"
    },
    {
        "id": "N-05",
        "q": "简单场景和复杂场景分别有多少台设备？",
        "expected": ["12", "20"],
        "type": "both"
    },
    {
        "id": "N-06",
        "q": "PAR专项优化策略的触发条件是什么？",
        "expected": ["PAR", "式(35)"],
        "type": "lora"
    },
    
    # 概念类问题
    {
        "id": "C-01",
        "q": "MRFO算法的核心思想是什么？",
        "expected": ["蝠鲼", "觅食"],
        "type": "base"
    },
    {
        "id": "C-02",
        "q": "MRFO算法的三种觅食策略分别是什么？",
        "expected": ["链式", "螺旋", "翻滚"],
        "type": "base"
    },
    {
        "id": "C-03",
        "q": "DLM MRFO相比原始MRFO做了哪些改进？",
        "expected": ["离散", "长时记忆", "PAR"],
        "type": "lora"
    },
]

# ============================================================
# 执行测试
# ============================================================
def check_answer(answer: str, expected: list) -> tuple:
    """检查答案是否正确"""
    found = sum(1 for kw in expected if kw in answer)
    recall = found / len(expected) if expected else 1.0
    return recall >= 0.5, recall

results = []
for tc in test_cases:
    print(f"\n[{tc['id']}] {tc['q']}")
    print(f"    期望: {tc['expected']}")
    
    try:
        answer, model_used = agent_query(tc['q'])
        ok, recall = check_answer(answer, tc['expected'])
        
        status = "[PASS]" if ok else "[FAIL]"
        print(f"    {status} (使用: {model_used}, 召回: {recall:.0%})")
        print(f"    答案: {answer[:100]}...")
        
        results.append({
            'id': tc['id'],
            'question': tc['q'],
            'expected': tc['expected'],
            'answer': answer[:200],
            'model': model_used,
            'pass': ok,
            'recall': recall
        })
    except Exception as e:
        print(f"    [ERROR] {e}")
        results.append({
            'id': tc['id'],
            'question': tc['q'],
            'expected': tc['expected'],
            'answer': '',
            'model': 'ERROR',
            'pass': False,
            'recall': 0
        })

# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("测试结果汇总")
print("=" * 60)

passed = sum(1 for r in results if r['pass'])
total = len(results)

# 按类型统计
base_results = [r for r in results if r['model'] == '基础模型']
lora_results = [r for r in results if r['model'] == 'LoRA模型']
both_results = [r for r in results if r['model'] == '双模型合并']

print(f"\n总体: {passed}/{total} ({passed/total*100:.0f}%)")

if base_results:
    base_pass = sum(1 for r in base_results if r['pass'])
    print(f"  - 基础模型: {base_pass}/{len(base_results)} ({base_pass/len(base_results)*100:.0f}%)")

if lora_results:
    lora_pass = sum(1 for r in lora_results if r['pass'])
    print(f"  - LoRA模型: {lora_pass}/{len(lora_results)} ({lora_pass/len(lora_results)*100:.0f}%)")

if both_results:
    both_pass = sum(1 for r in both_results if r['pass'])
    print(f"  - 双模型合并: {both_pass}/{len(both_results)} ({both_pass/len(both_results)*100:.0f}%)")

print("\n详细结果:")
for r in results:
    status = "PASS" if r['pass'] else "FAIL"
    print(f"  [{status}] {r['id']} ({r['model']}): {r['question'][:40]}...")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
