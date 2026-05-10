"""
中医医案分类 - 混合推理集成系统
集成 Stage1 二分类 + Stage2 五分类 + 关键词规则仲裁
"""

import json
import torch
import os
import argparse
import pandas as pd
import difflib
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
from loguru import logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="中医医案分类 - 混合推理集成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python inference_hybrid_ensemble.py  # 使用默认配置
  python inference_hybrid_ensemble.py --config config.json  # 从配置文件读取
        """
    )
    
    # ===== Stage1 二分类配置 =====
    parser.add_argument("--stage1_model_path", type=str, 
                        default="/home/zimo/projects/tcm-workplace/saves/glm4-9b-chat/lora/merged_sft_binary_stage1_9616",
                        help="Stage1 二分类模型路径")
    
    # ===== Stage2 五分类配置 =====
    parser.add_argument("--stage2_model_path", type=str,
                        default="/home/zimo/projects/tcm-glm4-sft/zmenv/LLaMA-Factory/saves/glm-4-9b-chat/lora/sft_with_added_data_wo_external_2_merged",
                        help="Stage2 五分类模型路径")
    
    # ===== 数据和规则配置 =====
    parser.add_argument("--data_path", type=str,
                        default="/home/zimo/projects/tcm-glm4-sft/venv_broken_backup/LLaMA-Factory/data/tcm_test.json",
                        help="测试数据文件路径")
    parser.add_argument("--keywords_file", type=str,
                        default="/home/zimo/projects/tcm-workplace/new_pipe/keywords/final_expert_rules.json",
                        help="关键词规则文件路径")
    
    # ===== 输出配置 =====
    parser.add_argument("--output_dir", type=str,
                        default="/home/zimo/projects/tcm-workplace/tcm-upload-files/results",
                        help="输出目录")
    parser.add_argument("--output_file", type=str,
                        default="hybrid_ensemble_results.json",
                        help="最终结果输出文件名")
    parser.add_argument("--audit_file", type=str,
                        default="keyword_performance_audit.csv",
                        help="关键词审计表输出文件名")
    
    # ===== 推理参数配置 =====
    parser.add_argument("--gpu", type=str, default="7",
                        help="Stage1 使用的 GPU 设备号（双 GPU 模式下为第一张）")
    parser.add_argument("--gpu_stage2", type=str, default="6",
                        help="Stage2 使用的 GPU 设备号（双 GPU 模式下为第二张）")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批处理大小")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="生成最大token数")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="生成温度参数")
    parser.add_argument("--config", type=str, default=None,
                        help="从JSON配置文件读取所有参数")
    
    return parser.parse_args()


def load_config_from_file(config_path):
    """从配置文件加载参数"""
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def load_json_or_jsonl(file_path):
    """加载JSON或JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def post_process_answer(generated_text, categories):
    """后处理生成的文本"""
    cleaned_text = generated_text.strip().replace("。", "").replace(" ", "")
    matches = difflib.get_close_matches(cleaned_text, categories, n=1, cutoff=0.3)
    return matches[0] if matches else cleaned_text


def batch_infer_stage1(model, tokenizer, prompts, labels, categories, batch_size=8, max_new_tokens=10, temperature=0.01):
    """Stage1 二分类推理"""
    all_predictions = []
    output_data = []
    
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    do_sample_flag = True if temperature > 0.1 else False
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Stage1 二分类推理"):
        batch_prompts = prompts[i:i + batch_size]
        batch_labels = labels[i:i + len(batch_prompts)]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample_flag,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        
        generated_tokens = outputs[:, input_len:]
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for j, gen_text in enumerate(decoded_texts):
            true_label = batch_labels[j]
            predicted_label = post_process_answer(gen_text, categories)
            
            all_predictions.append(predicted_label)
            output_data.append({
                "sample_id": i + j,
                "true_label": true_label,
                "generated_raw": gen_text.strip(),
                "predicted_label": predicted_label
            })
    
    return all_predictions, output_data


def batch_infer_stage2(model, tokenizer, prompts, labels, categories, batch_size=8, max_new_tokens=10, temperature=1.0):
    """Stage2 五分类推理"""
    all_predictions = []
    output_data = []
    
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    do_sample_flag = True if temperature > 0.1 else False
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Stage2 五分类推理"):
        batch_prompts = prompts[i:i + batch_size]
        batch_labels = labels[i:i + len(batch_prompts)]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample_flag,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        
        generated_tokens = outputs[:, input_len:]
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for j, gen_text in enumerate(decoded_texts):
            true_label = batch_labels[j]
            predicted_label = post_process_answer(gen_text, categories)
            
            all_predictions.append(predicted_label)
            output_data.append({
                "sample_id": i + j,
                "true_label": true_label,
                "generated_raw": gen_text.strip(),
                "predicted_label": predicted_label
            })
    
    return all_predictions, output_data


def get_pure_text(item):
    """提取医案纯文本内容"""
    full_text = item.get("instruction", "") + item.get("input", "")
    prefix = "给定下面的中医医案，请从（内科|外科|妇科|耳鼻喉科|儿科）选择一个作为这个医案的标签：\n"
    content = full_text.replace(prefix, "").split("|")[0].strip()
    return content


def hybrid_ensemble_inference(stage1_preds, stage2_preds_all, test_data, keywords_dict):
    """
    混合推理集成：Stage1 + Stage2 + 关键词规则仲裁
    
    决策流程:
    1. 如果Stage1预测为"内科" → 直接返回"内科"
    2. 否则 (Stage1预测为"专科"):
       a. 尝试匹配专科关键词 (外科、妇科、儿科、耳鼻喉科)
       b. 如果未匹配，尝试匹配内科关键词（纠偏）
       c. 如果仍未匹配，使用Stage2的预测结果
    """
    
    final_predictions = []
    ensemble_output_data = []
    
    stats = {
        "stage1_direct": 0,           # Stage1直接返回内科
        "rule_hit_spec": 0,            # 专科关键词命中
        "rule_hit_int": 0,             # 内科关键词命中（纠偏）
        "stage2_fallback": 0           # Stage2兜底
    }
    
    keyword_perf = defaultdict(lambda: {"correct": 0, "wrong": 0, "category": ""})
    
    categories = ["内科", "外科", "妇科", "耳鼻喉科", "儿科"]
    stage2_ptr = 0  # 指向Stage2预测队列的指针
    
    logger.info("开始混合推理集成...")
    
    for i, stage1_pred in enumerate(tqdm(stage1_preds, desc="集成推理")):
        text = get_pure_text(test_data[i])
        true_label = test_data[i]["output"].strip()
        
        # ===== 决策流程 =====
        if stage1_pred == "内科":
            # A. Stage1直接预测为内科
            final_pred = "内科"
            stats["stage1_direct"] += 1
            decision_path = "Stage1_Direct"
        else:
            # B. Stage1预测为专科，进入Stage2 + 规则仲裁流程
            final_pred = None
            hit_word = None
            hit_cat = None
            
            # B1. 尝试匹配专科关键词
            for cat in ["外科", "妇科", "儿科", "耳鼻喉科"]:
                matched_words = [word for word in keywords_dict.get(cat, []) if word in text]
                if matched_words:
                    hit_word = matched_words[0]
                    hit_cat = cat
                    break
            
            if hit_word:
                # B1命中：使用专科关键词预测
                final_pred = hit_cat
                stats["rule_hit_spec"] += 1
                decision_path = f"SpecRule({hit_word})"
                
                keyword_perf[hit_word]["category"] = hit_cat
                if final_pred == true_label:
                    keyword_perf[hit_word]["correct"] += 1
                else:
                    keyword_perf[hit_word]["wrong"] += 1
            else:
                # B2. 没有专科关键词，尝试匹配内科关键词（纠偏）
                matched_int_words = [word for word in keywords_dict.get("内科", []) if word in text]
                if matched_int_words:
                    final_pred = "内科"
                    hit_word = matched_int_words[0]
                    stats["rule_hit_int"] += 1
                    decision_path = f"IntRule({hit_word})"
                    
                    keyword_perf[hit_word]["category"] = "内科"
                    if final_pred == true_label:
                        keyword_perf[hit_word]["correct"] += 1
                    else:
                        keyword_perf[hit_word]["wrong"] += 1
                else:
                    # B3. 规则均未命中，使用Stage2预测
                    stage2_pred = stage2_preds_all[stage2_ptr] if stage2_ptr < len(stage2_preds_all) else "内科"
                    stage2_ptr += 1
                    final_pred = stage2_pred
                    stats["stage2_fallback"] += 1
                    decision_path = f"Stage2({stage2_pred})"
        
        final_predictions.append(final_pred)
        ensemble_output_data.append({
            "sample_id": i,
            "true_label": true_label,
            "stage1_pred": stage1_pred,
            "stage2_pred": stage2_preds_all[stage2_ptr-1] if stage2_ptr > 0 and stage2_ptr <= len(stage2_preds_all) else "N/A",
            "final_pred": final_pred,
            "decision_path": decision_path,
            "correct": final_pred == true_label
        })
    
    return final_predictions, ensemble_output_data, stats, keyword_perf, categories


def main():
    args = parse_args()
    
    # ===== 加载配置文件 =====
    if args.config:
        file_config = load_config_from_file(args.config)
        for key, value in file_config.items():
            setattr(args, key.replace("-", "_"), value)
    
    # 固定为双GPU并行模式（Stage1 -> 第一张；Stage2 -> 第二张）
    logger.info(f"使用双GPU并行模式 (Stage1: GPU {args.gpu}, Stage2: GPU {args.gpu_stage2})")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu},{args.gpu_stage2}"
    device_stage1 = "cuda:0"
    device_stage2 = "cuda:1"
    
    # ===== 加载数据 =====
    logger.info(f"加载测试数据: {args.data_path}")
    test_data = load_json_or_jsonl(args.data_path)
    logger.info(f"共加载 {len(test_data)} 条数据")
    
    # ===== 加载关键词规则 =====
    logger.info(f"加载关键词规则: {args.keywords_file}")
    with open(args.keywords_file, 'r', encoding='utf-8') as f:
        keywords_dict = json.load(f)
    
    # ===== 准备输出目录 =====
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    audit_path = os.path.join(args.output_dir, args.audit_file)
    
    # ===== Stage1: 二分类推理 =====
    logger.info("\n" + "="*70)
    logger.info("STAGE 1: 二分类推理 (内科 vs 专科)")
    logger.info("="*70)
    
    stage1_categories = ["内科", "专科"]
    stage1_prompts, stage1_labels = [], []
    system_prompt_stage1 = "你是一个中医分诊专家。请分析医案内容，判断其属于【内科】还是涉及外科、妇科、儿科或耳鼻喉科的【专科】。你只能回答【内科】或【专科】这两个词。"
    
    logger.info(f"加载Stage1模型: {args.stage1_model_path}")
    tokenizer_s1 = AutoTokenizer.from_pretrained(args.stage1_model_path, trust_remote_code=True)
    tokenizer_s1.padding_side = "left"
    if tokenizer_s1.pad_token is None:
        tokenizer_s1.pad_token = tokenizer_s1.eos_token
    
    # 双GPU：将 Stage1 模型加载到指定的 GPU（device_stage1）
    model_s1 = AutoModelForCausalLM.from_pretrained(
        args.stage1_model_path,
        trust_remote_code=True,
        device_map=device_stage1,
        torch_dtype=torch.bfloat16
    )
    model_s1.eval()
    
    for item in test_data:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        content = f"{instruction}\n{input_text}\n结论："
        
        messages = [
            {"role": "system", "content": system_prompt_stage1},
            {"role": "user", "content": content}
        ]
        
        formatted_prompt = tokenizer_s1.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        stage1_prompts.append(formatted_prompt)
        stage1_labels.append(item["output"].strip())
    
    stage1_preds, stage1_output_data = batch_infer_stage1(
        model_s1, tokenizer_s1, stage1_prompts, stage1_labels, stage1_categories,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    stage1_acc = accuracy_score(stage1_labels, stage1_preds)
    stage1_f1 = f1_score(stage1_labels, stage1_preds, labels=stage1_categories, average='macro', zero_division=0)
    logger.info(f"Stage1 准确率: {stage1_acc:.4f} | F1: {stage1_f1:.4f}")
    
    # ===== 单GPU模式下释放Stage1模型显存 =====
    if args.gpu_mode == "single":
        logger.info("释放Stage1模型显存...")
        del model_s1
        del tokenizer_s1
        torch.cuda.empty_cache()
    
    # ===== Stage2: 五分类推理 (仅针对非内科样本) =====
    logger.info("\n" + "="*70)
    logger.info("STAGE 2: 五分类推理 (仅处理Stage1预测为专科的样本)")
    logger.info("="*70)
    
    stage2_categories = ["内科", "外科", "妇科", "耳鼻喉科", "儿科"]
    stage2_prompts, stage2_labels_filtered, stage2_indices = [], [], []
    system_prompt_stage2 = "你是一个专业的中医专家。请仅从以下选项中选择一个回答：内科、外科、妇科、耳鼻喉科、儿科。"
    
    logger.info(f"加载Stage2模型: {args.stage2_model_path}")
    tokenizer_s2 = AutoTokenizer.from_pretrained(args.stage2_model_path, trust_remote_code=True)
    tokenizer_s2.padding_side = "left"
    if tokenizer_s2.pad_token is None:
        tokenizer_s2.pad_token = tokenizer_s2.eos_token
    
    # 根据GPU模式选择加载策略
    if args.gpu_mode == "single":
        model_s2 = AutoModelForCausalLM.from_pretrained(
            args.stage2_model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:  # dual mode
        model_s2 = AutoModelForCausalLM.from_pretrained(
            args.stage2_model_path,
            trust_remote_code=True,
            device_map=device_stage2,
            torch_dtype=torch.bfloat16
        )
    model_s2.eval()
    
    # 只为Stage1预测为"专科"的样本准备Stage2推理
    for i, (s1_pred, item) in enumerate(zip(stage1_preds, test_data)):
        if s1_pred == "专科":  # 只有非内科才需要Stage2推理
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            content = f"{instruction}\n{input_text}\n"
            
            messages = [
                {"role": "system", "content": system_prompt_stage2},
                {"role": "user", "content": content}
            ]
            
            formatted_prompt = tokenizer_s2.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            stage2_prompts.append(formatted_prompt)
            stage2_labels_filtered.append(item["output"].strip())
            stage2_indices.append(i)
    
    stage2_preds = []
    if stage2_prompts:
        stage2_preds, _ = batch_infer_stage2(
            model_s2, tokenizer_s2, stage2_prompts, stage2_labels_filtered, stage2_categories,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        stage2_acc = accuracy_score(stage2_labels_filtered, stage2_preds)
        stage2_f1 = f1_score(stage2_labels_filtered, stage2_preds, labels=stage2_categories, average='macro', zero_division=0)
        logger.info(f"Stage2 准确率: {stage2_acc:.4f} | F1: {stage2_f1:.4f}")
    
    # ===== 混合推理集成 =====
    logger.info("\n" + "="*70)
    logger.info("STAGE 3: 混合推理集成 (Stage1 + Stage2 + 关键词规则)")
    logger.info("="*70)
    
    final_preds, ensemble_output, stats, keyword_perf, categories = hybrid_ensemble_inference(
        stage1_preds, stage2_preds, test_data, keywords_dict
    )
    
    # ===== 评估指标 =====
    true_labels = [item["output"].strip() for item in test_data]
    
    acc = accuracy_score(true_labels, final_preds)
    macro_f1 = f1_score(true_labels, final_preds, labels=categories, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, final_preds, labels=categories, average='weighted', zero_division=0)
    
    print("\n" + "█"*70)
    print("🏆 混合推理集成 - 最终评估报告")
    print("█"*70)
    print(f"🥇 总体准确率 (Accuracy): {acc:.4f}")
    print(f"📊 宏平均 F1 分数: {macro_f1:.4f}")
    print(f"📊 加权 F1 分数: {weighted_f1:.4f}")
    print("-" * 70)
    print(f"📊 决策分布:")
    print(f"   Stage1直接返回内科: {stats['stage1_direct']}")
    print(f"   专科关键词命中: {stats['rule_hit_spec']}")
    print(f"   内科关键词命中（纠偏）: {stats['rule_hit_int']}")
    print(f"   Stage2兜底: {stats['stage2_fallback']}")
    print("█"*70)
    
    # 混淆矩阵
    print("\n最终混淆矩阵:")
    cm = confusion_matrix(true_labels, final_preds, labels=categories)
    cm_df = pd.DataFrame(cm, index=[f"True:{c}" for c in categories], columns=[f"Pred:{c}" for c in categories])
    print(cm_df)
    
    # 分类报告
    print("\n分类详细报告:")
    print(classification_report(true_labels, final_preds, labels=categories, digits=4))
    
    # ===== 保存结果 =====
    logger.info(f"保存结果到: {output_path}")
    
    final_output = {
        "evaluation_metrics": {
            "accuracy": float(acc),
            "f1_score_macro": float(macro_f1),
            "f1_score_weighted": float(weighted_f1),
            "stage1_accuracy": float(stage1_acc),
            "stage1_f1": float(stage1_f1),
            "stage2_accuracy": float(stage2_acc) if stage2_preds else 0,
            "stage2_f1": float(stage2_f1) if stage2_preds else 0
        },
        "decision_stats": {
            "stage1_direct": int(stats['stage1_direct']),
            "rule_hit_spec": int(stats['rule_hit_spec']),
            "rule_hit_int": int(stats['rule_hit_int']),
            "stage2_fallback": int(stats['stage2_fallback'])
        },
        "confusion_matrix": cm.tolist(),
        "predictions": ensemble_output
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    logger.success(f"✅ 结果已保存到: {output_path}")
    
    # ===== 关键词审计表 =====
    logger.info(f"保存关键词审计表到: {audit_path}")
    
    audit_list = []
    for word, metrics in keyword_perf.items():
        total = metrics["correct"] + metrics["wrong"]
        audit_list.append({
            "关键词": word,
            "所属科室": metrics["category"],
            "命中总数": total,
            "判断正确": metrics["correct"],
            "判断错误": metrics["wrong"],
            "准确率": f"{(metrics['correct']/total):.2%}" if total > 0 else "0%"
        })
    
    if audit_list:
        audit_df = pd.DataFrame(audit_list)
        audit_df = audit_df.sort_values(by="判断错误", ascending=False)
        audit_df.to_csv(audit_path, index=False, encoding='utf_8_sig')
        logger.success(f"✅ 审计表已保存到: {audit_path}")
        
        print("\n⚠️ 表现最差的关键词 (建议剔除或精炼):")
        bad_keywords = audit_df[audit_df["判断错误"] > 0].head(10)
        if not bad_keywords.empty:
            print(bad_keywords.to_string(index=False))
    
    logger.info("\n✅ 混合推理集成完成！")


if __name__ == "__main__":
    main()
