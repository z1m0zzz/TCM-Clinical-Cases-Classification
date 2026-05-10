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
    parser = argparse.ArgumentParser(description="GLM-4-9B Chat 分类推理脚本")
    parser.add_argument("--base_model_path", type=str, default="/home/zimo/projects/tcm-glm4-sft/zmenv/LLaMA-Factory/saves/glm-4-9b-chat/lora/dpo_mixed_with_added_data_0.5-merged_cp100")
    parser.add_argument("--data_path", type=str, default="/home/zimo/projects/tcm-glm4-sft/venv_broken_backup/LLaMA-Factory/data/tcm_test.json")
    parser.add_argument("--output_path", type=str, default="/home/zimo/projects/qwen3-8b-dpo/dpo_results/inference_results_glm-4-9b-on_test_data_dpo.json")
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    # 1. 这里你已经加好了，保持不动
    parser.add_argument("--temperature", type=float, default=1.0, help="生成时的温度参数") 
    return parser.parse_args()

def load_json_or_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]

def post_process_answer(generated_text, categories):
    cleaned_text = generated_text.strip()
    matches = difflib.get_close_matches(cleaned_text, categories, n=1, cutoff=0.5)
    return matches[0] if matches else cleaned_text

# 2. 修改函数签名，增加 temperature 参数
def batch_infer(model, tokenizer, prompts, labels, categories, output_path,
                batch_size=1, max_new_tokens=10, temperature=1.0):
    all_predictions = []
    output_data = []

    if tokenizer.padding_side != 'left':
        logger.warning(f"强制将 tokenizer.padding_side 从 {tokenizer.padding_side} 改为 left")
        tokenizer.padding_side = 'left'

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Inferencing (T={temperature})"):
        batch_prompts = prompts[i:i + batch_size]
        batch_labels = labels[i:i + len(batch_prompts)]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 必须为 True，temperature 才会生效
                # 3. 这里修改为使用函数参数传入的 temperature
                temperature=temperature, 
                #top_p=0.9,       # 建议加上 top_p 防止采样到极低概率的词，配合 temperature 使用效果更好
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False 
            )

        generated_tokens = outputs[:, input_len:]
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for j, gen_text in enumerate(decoded_texts):
            true_label = batch_labels[j]
            clean_gen_text = gen_text.strip()
            predicted_label = post_process_answer(clean_gen_text, categories)

            all_predictions.append(predicted_label)
            output_data.append({
                "sample_id": i + j + 1,
                "true_label": true_label,
                "generated_raw": clean_gen_text,
                "predicted_label": predicted_label
            })

    correct_count = sum(1 for true, pred in zip(labels, all_predictions) if true == pred)
    accuracy = correct_count / len(labels)
    f1 = f1_score(labels, all_predictions, labels=categories, average='macro', zero_division=0)

    logger.info(f"\nTemperature: {temperature}\nAccuracy: {accuracy:.4f}\nMacro F1: {f1:.4f}")

    final_output = {
        "evaluation_metrics": {"accuracy": accuracy, "f1_score_macro": f1, "temperature": temperature},
        "predictions": output_data
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    logger.success(f"推理结果已保存到: {output_path}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger.info(f"使用 GPU: {args.gpu} | Temperature: {args.temperature}")

    categories = ["内科", "外科", "妇科", "耳鼻喉科", "儿科"]
    #categories = ["0", "1", "3", "5", "6"]

    logger.info(f"加载模型: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    tokenizer.padding_side = "left"
    
    # GLM-4 推荐的 pad token 处理方式
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    logger.info("加载数据")
    data_list = load_json_or_jsonl(args.data_path)
    logger.info(f"共加载 {len(data_list)} 条")

    prompts, labels = [], []
    # 稍微加强了一点 System Prompt，防止在 Temp 高的时候太放飞自我
    system_prompt = "你是一个专业的中医专家。请仅从以下选项中选择一个回答：内科、外科、妇科、耳鼻喉科、儿科。" 

    for item in data_list:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        content = f"{instruction}\n{input_text}\n"
        
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

    # 4. 在调用时传入 args.temperature
    batch_infer(
        model, tokenizer, prompts, labels, categories,
        args.output_path, 
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature 
    )

if __name__ == "__main__":
    main()