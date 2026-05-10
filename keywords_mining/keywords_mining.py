import json
import jieba
from collections import Counter
from tqdm import tqdm

# ================= 配置区域 =================
TRAIN_DATA_FILE = "/home/zimo/projects/tcm-glm4-sft/venv_broken_backup/LLaMA-Factory/data/train_glm.json"
TARGET_LABEL = "外科"  # 你要挖掘的科室：妇科、儿科、外科、耳鼻喉科
MIN_FREQ = 3         # 在专科中至少出现3次
# ===========================================

def get_clean_full_text(item):
    """
    完全模拟推理时的文本清洗逻辑：
    合并 instruction 和 input，剔除前缀，不截断。
    """
    prefix = "给定下面的中医医案，请从（内科|外科|妇科|耳鼻喉科|儿科）选择一个作为这个医案的标签：\n"
    full_text = item.get("instruction", "") + item.get("input", "")
    # 剔除指令前缀
    clean_text = full_text.replace(prefix, "").strip()
    # 移除末尾可能的管道符
    if clean_text.endswith("|"):
        clean_text = clean_text[:-1]
    return clean_text

def main():
    # 1. 分类加载数据
    target_docs = []
    internal_docs = []
    
    print(f"📂 正在加载并预处理数据...")
    with open(TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            label = item.get("output", "").strip()
            clean_text = get_clean_full_text(item)
            
            if label == TARGET_LABEL:
                target_docs.append(clean_text)
            elif label == "内科":
                internal_docs.append(clean_text)

    print(f"✅ 加载完成：【{TARGET_LABEL}】{len(target_docs)}篇，【内科】{len(internal_docs)}篇")

    # 2. 统计专科候选词
    print(f"🧠 正在提取【{TARGET_LABEL}】高频词候选池...")
    candidate_counter = Counter()
    for doc in target_docs:
        # 只提取长度大于等于2的词
        words = [w for w in jieba.lcut(doc) if len(w) >= 2]
        candidate_counter.update(words)
    
    # 初步筛选高频词
    initial_candidates = [word for word, count in candidate_counter.items() if count >= MIN_FREQ]
    print(f"🔍 发现初步候选词 {len(initial_candidates)} 个")

    # 3. 核心环节：全量内科原文“毒性测试”
    print(f"🛡️ 正在进行内科语料过滤 (基于全文本检索)...")
    gold_standard = []
    
    # 遍历每一个候选词
    for word in tqdm(initial_candidates):
        is_toxic = False
        # 在每一篇内科原文中暴力检索该字符串
        for int_doc in internal_docs:
            if word in int_doc:
                is_toxic = True
                break
        
        # 只有在 1700 条内科原文中完全没出现过的词，才保留
        if not is_toxic:
            gold_standard.append((word, candidate_counter[word]))

    # 4. 排序并保存
    # 按专科频次降序排列
    gold_standard.sort(key=lambda x: x[1], reverse=True)
    
    final_list = [item[0] for item in gold_standard]
    
    print("\n" + "="*40)
    print(f"🏆 提纯完成！共获得 【{TARGET_LABEL}】 金标准词 {len(final_list)} 个")
    print(f"这些词在 1700 篇内科训练集中出现频次为：绝对 0")
    print("="*40)

    # 输出前 100 个
    output_top = final_list[:100]
    print(f"\n🚀 建议加入词表的 Top 100 词汇：")
    print(json.dumps(output_top, ensure_ascii=False))

    # 保存到本地方便查看
    output_filename = f"/home/zimo/projects/tcm-workplace/new_pipe/keywords/{TARGET_LABEL}_gold_standard.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=4)
    print(f"\n💾 全量纯净词表已保存至: {output_filename}")

if __name__ == "__main__":
    main()