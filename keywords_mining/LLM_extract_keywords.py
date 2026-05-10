import json
import os
from openai import OpenAI

# ================= 配置区域 =================
# 1. 之前统计生成的原始词表路径
RAW_KEYWORDS_FILE = "/home/zimo/projects/tcm-workplace/new_pipe/keywords/妇科_gold_standard.json" 
# 2. 精选后保存的路径
REFINED_OUTPUT_FILE = "/home/zimo/projects/tcm-workplace/new_pipe/keywords/妇科_expert_refined.json"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), 
    base_url="https://api.chatanywhere.tech/v1"
)
MODEL_NAME = "gpt-5" # 4o-mini 足够胜任逻辑分类任务
# ===========================================

def expert_refinement_prompt(specialty, raw_list):
    return f"""你是一位资深中医临床专家与知识工程专家。你擅长从中医医案中提取最具分科特征的“金标准”术语。

# 任务目标
我提供了一组通过统计学方法从【{specialty}】训练集中提取的高频候选词。这些词在“内科”语料中出现的频率几乎为零。
请你对这些词进行二次精炼，选出真正具备“一票定性”能力的专业关键词，并剔除无效噪音。

# 待审计词表
{json.dumps(raw_list, ensure_ascii=False)}

# 审计准则
1. **医学特异性 (Precision)**：该词是否几乎只在{specialty}出现？（如：“生化汤”是金标准；“植物”是噪音）。
2. **语义完整性**：剔除带有具体剂量的词（如“二钱川断”），保留其核心药名（“川断”）。
3. **剔除统计偏差**：
   - 剔除食材/补品：如“鲍鱼”、“麻雀”、“鱼鳔”，除非它们是专科极度特异的药。
   - 剔除泛化词：如“动物”、“植物”、“炎症”、“隐患”、“专科”、“按期”、“炎症”。
   - 剔除人名/医生名/地名：如“克震”、“玉璋”、“黄先”、“琴溪”。
4. **语义加长 (消歧)**：如果单字词有歧义，请结合中医常识建议更精准的表达。

# 输出要求
请直接输出一个 JSON 对象，结构如下：
{{
  "gold_standard": {{
    "formulas": ["特有的方剂名"],
    "symptoms_diseases": ["特有的病证名"],
    "anatomy_physiology": ["特有的解剖或生理术语"]
  }},
  "discarded": {{
    "noise_or_general": ["被剔除的通用词/噪音/人名"],
    "dietary": ["被剔除的补品/食材"]
  }},
  "refined_suggestions": [
    {{"original": "单字词", "suggested": "更精准的词", "reason": "理由"}}
  ]
}}
"""

def main():
    # 1. 加载原始统计词表
    if not os.path.exists(RAW_KEYWORDS_FILE):
        print(f"❌ 找不到文件: {RAW_KEYWORDS_FILE}")
        return
    
    with open(RAW_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
        raw_list = json.load(f)

    # 如果原始词表太长（超过200个），建议分批次处理
    # 这里我们处理前 150 个最具代表性的词
    sample_list = raw_list[:150]

    print(f"🚀 正在启动 Agent 专家审计，处理 {len(sample_list)} 个候选词...")

    # 2. 调用 LLM 进行专家审计
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严谨的中医知识图谱工程师，只输出 JSON 格式。"},
                {"role": "user", "content": expert_refinement_prompt("妇科", sample_list)}
            ],
            temperature=0.1,
            response_format={ "type": "json_object" }
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # 3. 保存精选后的金标准库
        with open(REFINED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"✅ 精选完成！结果已存至: {REFINED_OUTPUT_FILE}")
        
        # 4. 打印审计简报
        gold = result.get("gold_standard", {})
        total_gold = len(gold.get("formulas", [])) + len(gold.get("symptoms_diseases", [])) + len(gold.get("anatomy_physiology", []))
        print(f"📊 审计摘要:")
        print(f"   - 入选金标准词: {total_gold} 个")
        print(f"   - 剔除噪音/食材: {len(result.get('discarded', {}).get('noise_or_general', [])) + len(result.get('discarded', {}).get('dietary', []))} 个")

    except Exception as e:
        print(f"⚠️ Agent 审计出错: {e}")

if __name__ == "__main__":
    main()