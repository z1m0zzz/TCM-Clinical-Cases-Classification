import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import difflib
import os
import argparse
from sklearn.metrics import f1_score
from tqdm import tqdm
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description="GLM-4-9B Stage 1 二分类推理脚本")
    # 修改默认路径为你的二分类模型和二分类测试集
    parser.add_argument("--base_model_path", type=str, default="/home/zimo/projects/tcm-workplace/saves/glm4-9b-chat/lora/merged_sft_binary_stage1_4393")
    parser.add_argument("--data_path", type=str, default="/home/zimo/projects/tcm-workplace/stage1/test_stage1_binary.json")
    parser.add_argument("--output_path", type=str, default="/home/zimo/projects/tcm-workplace/stage1/results/inference_results_glm-4-9b-stage1_binary_final_4393.json")
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.01, help="生成时的温度参数") 
    return parser.parse_args()

def load_json_or_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]

def post_process_answer(generated_text, categories):
    cleaned_text = generated_text.strip().replace("。", "").replace(" ", "")
    # 只在 ["内科", "专科"] 中寻找最接近的匹配
    matches = difflib.get_close_matches(cleaned_text, categories, n=1, cutoff=0.3)
    return matches[0] if matches else cleaned_text

def batch_infer(model, tokenizer, prompts, labels, categories, output_path,
                batch_size=1, max_new_tokens=10, temperature=1.0):
    all_predictions = []
    output_data = []

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # 注意：如果 do_sample=False, temperature 是不生效的
    # 分类建议用 False 追求稳定，如果想用温度调节，请把下方的 do_sample 改为 True
    do_sample_flag = True if temperature > 0.1 else False

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Stage 1 Inferencing"):
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
                "sample_id": i + j + 1,
                "true_label": true_label,
                "generated_raw": gen_text.strip(),
                "predicted_label": predicted_label
            })

    correct_count = sum(1 for true, pred in zip(labels, all_predictions) if true == pred)
    accuracy = correct_count / len(labels)
    # 核心修改：F1 计算只针对这两个类
    f1 = f1_score(labels, all_predictions, labels=categories, average='macro', zero_division=0)

    logger.info(f"\n[Stage 1 Binary Result]\nAccuracy: {accuracy:.4f}\nMacro F1: {f1:.4f}")

    final_output = {
        "evaluation_metrics": {"accuracy": accuracy, "f1_score_macro": f1},
        "predictions": output_data
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # --- 修改 1: 类别定义 ---
    categories = ["内科", "专科"]

    logger.info(f"加载模型: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    data_list = load_json_or_jsonl(args.data_path)
    prompts, labels = [], []

    # --- 修改 2: 二分类专用 System Prompt ---
    system_prompt = "你是一个中医分诊专家。请分析医案内容，判断其属于【内科】还是涉及外科、妇科、儿科或耳鼻喉科的【专科】。你只能回答“内科”或“专科”这两个词，不要说任何废话。"

    for item in data_list:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        content = f"{instruction}\n{input_text}\n结论："
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)
        labels.append(item["output"].strip())

    batch_infer(
        model, tokenizer, prompts, labels, categories,
        args.output_path, 
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature 
    )

if __name__ == "__main__":
    main()