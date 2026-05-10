import json
import os
import argparse
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from collections import defaultdict

# ================= 默认配置 =================
DEFAULT_CONFIG = {
    "s1_file": "/home/zimo/projects/tcm-workplace/stage1/results/inference_results_glm-4-9b-stage1_binary_final.json",
    "s2_file": "/home/zimo/projects/tcm-workplace/stage1/results/inference_results_glm-4-9b-stage2_5_kinds.json",
    "test_file": "/home/zimo/projects/tcm-glm4-sft/venv_broken_backup/LLaMA-Factory/data/tcm_test.json",
    "keywords_file": "/home/zimo/projects/tcm-workplace/new_pipe/keywords/final_expert_rules.json",
    "output_file": "/home/zimo/projects/tcm-workplace/new_pipe/results/final_hybrid_report_keywords.json",
    "audit_file": "/home/zimo/projects/tcm-workplace/new_pipe/results/keyword_performance_audit.csv"
}

# ===========================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="中医医案分类 - 混合推理集成系统评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python merge_results.py  # 使用默认配置
  python merge_results.py --s1-file path/to/stage1.json --s2-file path/to/stage2.json
  python merge_results.py --config config.json  # 从配置文件读取
        """
    )
    
    parser.add_argument("--s1-file", type=str, default=DEFAULT_CONFIG["s1_file"],
                        help="Stage1 二分类推理结果文件路径")
    parser.add_argument("--s2-file", type=str, default=DEFAULT_CONFIG["s2_file"],
                        help="Stage2 五分类推理结果文件路径")
    parser.add_argument("--test-file", type=str, default=DEFAULT_CONFIG["test_file"],
                        help="原始测试集文件路径（包含真实标签）")
    parser.add_argument("--keywords-file", type=str, default=DEFAULT_CONFIG["keywords_file"],
                        help="关键词规则字典文件路径")
    parser.add_argument("--output-file", type=str, default=DEFAULT_CONFIG["output_file"],
                        help="评估报告输出文件路径")
    parser.add_argument("--audit-file", type=str, default=DEFAULT_CONFIG["audit_file"],
                        help="关键词审计表输出文件路径")
    parser.add_argument("--config", type=str, default=None,
                        help="从JSON配置文件读取所有参数（优先级高于命令行参数）")
    
    return parser.parse_args()

def load_config_from_file(config_path):
    """从配置文件加载参数"""
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return {}

def load_data_robust(path):
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    return data

def get_pure_text(item):
    full_text = item.get("instruction", "") + item.get("input", "")
    prefix = "给定下面的中医医案，请从（内科|外科|妇科|耳鼻喉科|儿科）选择一个作为这个医案的标签：\n"
    content = full_text.replace(prefix, "").split("|")[0].strip()
    return content

def main(args):
    print("🧠 正在通过【全量统计词表仲裁】合并 Stage 1 和 Stage 2 结果...")
    print(f"📂 配置信息:")
    print(f"   S1文件: {args.s1_file}")
    print(f"   S2文件: {args.s2_file}")
    print(f"   测试集: {args.test_file}")
    print(f"   关键词: {args.keywords_file}\n")
    
    # 1. 加载资源
    s1_preds = load_data_robust(args.s1_file)
    s2_preds_pool = load_data_robust(args.s2_file)
    with open(args.test_file, 'r', encoding='utf-8') as f:
        orig_test = json.load(f)
    with open(args.keywords_file, 'r', encoding='utf-8') as f:
        keywords_dict = json.load(f)

    # 2. 准备 Stage 2 推理池
    s2_queue = [item["predicted_label"] for item in s2_preds_pool]
    s2_ptr = 0

    y_true, y_pred = [], []
    
    # 统计项
    stats = {
        "rule_hit_spec": 0,
        "rule_hit_int": 0,
        "sft_decided": 0
    }
    
    # --- 新增：关键词表现统计字典 ---
    # 结构: {word: {"correct": 0, "wrong": 0, "category": ""}}
    keyword_perf = defaultdict(lambda: {"correct": 0, "wrong": 0, "category": ""})

    categories = ["内科", "外科", "妇科", "耳鼻喉科", "儿科"]

    # 3. 执行集成推理逻辑
    for i, s1_item in enumerate(s1_preds):
        text = get_pure_text(orig_test[i])
        target_true = orig_test[i]["output"].strip()
        s1_label = s1_item["predicted_label"]

        if s1_label == "内科":
            final_p = "内科"
        else:
            initial_s2_p = s2_queue[s2_ptr] if s2_ptr < len(s2_queue) else "内科"
            s2_ptr += 1
            
            final_p = None
            hit_word = None
            hit_cat = None
            
            # B1. 尝试匹配专科关键词
            for cat in ["外科", "妇科", "儿科", "耳鼻喉科"]:
                # 寻找具体是哪个词命中了
                matched_words = [word for word in keywords_dict.get(cat, []) if word in text]
                if matched_words:
                    hit_word = matched_words[0] # 记录命中的第一个词
                    hit_cat = cat
                    break
            
            if hit_word:
                final_p = hit_cat
                stats["rule_hit_spec"] += 1
                # 记录该词的表现
                keyword_perf[hit_word]["category"] = hit_cat
                if final_p == target_true:
                    keyword_perf[hit_word]["correct"] += 1
                else:
                    keyword_perf[hit_word]["wrong"] += 1
            else:
                # B2. 没中专科词，尝试匹配内科关键词 (纠偏)
                matched_int_words = [word for word in keywords_dict.get("内科", []) if word in text]
                if matched_int_words:
                    final_p = "内科"
                    hit_word = matched_int_words[0]
                    stats["rule_hit_int"] += 1
                    # 记录内科词表现
                    keyword_perf[hit_word]["category"] = "内科"
                    if final_p == target_true:
                        keyword_perf[hit_word]["correct"] += 1
                    else:
                        keyword_perf[hit_word]["wrong"] += 1
                else:
                    # B3. 规则均未命中
                    final_p = initial_s2_p
                    stats["sft_decided"] += 1

        y_true.append(target_true)
        y_pred.append(final_p)

    # ================= 结果报告 =================
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print("\n" + "█"*65)
    print(f"🏆 集成架构评估报告")
    print(f"🥇 总体准确率 (Accuracy): {acc:.4f}")
    print(f"📊 总体宏平均 F1 分数: {macro_f1:.4f}")
    print("-" * 65)
    print(f"📊 决策分布: 专科规则命中 {stats['rule_hit_spec']} | 内科纠偏命中 {stats['rule_hit_int']} | SFT五分类{stats['sft_decided']}")
    print("█"*65)


    # 混淆矩阵
    print("\n最终混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    print(pd.DataFrame(cm, index=[f"True:{c}" for c in categories], columns=[f"Pred:{c}" for c in categories]))

if __name__ == "__main__":
    args = parse_args()
    
    # 如果指定了配置文件，优先使用配置文件
    if args.config:
        file_config = load_config_from_file(args.config)
        for key, value in file_config.items():
            setattr(args, key, value)
    
    main(args)