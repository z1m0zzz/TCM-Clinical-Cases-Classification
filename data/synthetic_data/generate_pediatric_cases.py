import re
import json
import random
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 1. API 配置
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://api.chatanywhere.tech/v1"
MODEL_NAME = "gpt-4o-mini"  

# 2. 文件路径
INPUT_FILE = "/home/zimo/projects/qwen3-8b-dpo/expand_data/儿科知识库.txt"
TEMP_OUTPUT_FILE = "/home/zimo/projects/qwen3-8b-dpo/expand_data/sft_pediatric_temp_progress_test.jsonl"
FINAL_OUTPUT_FILE = "/home/zimo/projects/qwen3-8b-dpo/kb_new/儿科_scaling_data.jsonl"

# 3. 生成设置
NUM_CASES_PER_KNOWLEDGE = 20 
MAX_WORKERS = 10

# 4. === Few-Shot 示例库 (请在这里填入你的示例) ===
# 这些示例将教会模型：什么是半文言风格？什么是你想要的医案结构？
FEW_SHOT_EXAMPLES = """
【范例1】
王春元二令郎，年甫七岁。久患赤痢，消导削积之剂已服过多，后转下白如涕，浑无粪。诊之，浮中沉六脉俱虚无神，三五不调；外症手足俱冷且硬，面浮，齿白，懒语，此阳气虚寒之症。宜温补脾胃以生肺金，用补中益气加炮姜、官桂各二分，其间人参止用三分，且陈腐不堪。服四剂，手足略软，言语亦健，第未温耳，其下白仍不减，亦虚寒滑脱危症，宜补、宜涩、宜温，复用前药加好参五分、大附二分半、御米壳一分。服一剂，则足已温，大便即有粪，白退十八，自兹手足俱温软，泄自全止，还服前方，去御米壳、附子二味。予归，属以如身中已温暖，姜、桂亦去，后服参苓白术散以培中气。使来岁乙巳厥阴风木之气不能制，饮食尤宜慎之。|

【范例2】
陈女孩，年二岁，苏州人。|病名：春温夹痰喘。（俗名肺风痰喘，实则肺闭）|原因：痰热内蕴，又感风温。|症候：壮热有汗，神识昏蒙，微咳喘急，喉有痰声漉漉，便溏溺少。|诊断：纹淡紫，舌苔厚白，脉来细数，已服过麻杏甘膏汤，无效，风痰热交结上焦，肺气将闭，襁褓肺弱，防涌塞骤变，勉拟轻清开泄，以尽医力。|疗法：肺位最高而司呼吸，喉为肺之外候，射干、牛蒡、甘、桔，利肺开喉为君，苏、葶、莱菔子，豁痰宣降为臣，更以杏仁、枳壳、前胡、郁金，宽胸宣郁为佐使也。病在上焦，药用轻清，仿徐之才轻可去实之义。|处方：炒牛蒡三钱生甘草四分广郁金钱半莱菔子三钱甜葶苈一钱前胡钱半泡射干八分苦桔梗五分白杏仁二钱炙苏子钱半|生枳壳钱半|次诊：喘势较平，小溲稍长，热灼之势亦缓，咳嗽痰多，便溏甚粘，痰邪已由肺入胃肠而下行，脉细较扬，右部濡滑数，关纹隐而不显，痰热尚充斥肺胃，质小病重，防喘塞骤变，治再清宣。|次方：炒牛蒡三钱生甘草四分广郁金钱半炙苏子钱半泡射干八分苦桔梗五分白杏仁二钱，勿研甜葶苈一钱生枳壳一钱嫩前胡钱半白通草一钱广橘白一钱|三诊：喘平，咳声亦松，肺气已得宣利，热退身凉，微微自汗，大便溏薄，溺多而黄，舌苔腻薄，脉象濡滑数，病情已入坦途，治再清肺，顺气化痰。|三方：熟牛蒡二钱象贝三钱炙苏子钱半冬瓜子四钱囫囵杏仁二钱去皮尖炒蒌皮三钱连翘壳三钱通草一钱生枳壳一钱前胡钱半莱菔子三钱|效果：服二剂后，诸恙均和，惟尚咳嗽有痰，仍宜清肺化痰，又服二剂全愈。|廉按：邪闭在肺，势极危险，而对症发药，不旬日已全者，因小儿脏腑嫩薄，易入亦易出，所以效力神速也。|

【范例3】
陈荷官，病痞积腹胀，发热干呛，善食而黄瘦，便溏溺赤。儿科药广服无功，已将绝望。孟英闻而怜之曰；吾于幼科虽未讨论，姑赠一方，或有生机也。以黄连、白芍、牡、鳖甲、鸡肫皮、木瓜、山楂、楝实、橘皮、桔梗、霞天曲、旋覆、栀子、丹皮、五谷虫等药，一剂知，旬余愈。|

【范例4】
一小儿，发热廿一日不退，每日寒热往来，清晨更甚，腹痛口渴，手足时厥，胸满，腹胀，脐痛，每食辄胀，时而头痛甚剧，热甚时则脉弦数甚，热缓脉亦较为缓和，小便短黄，大便尚通。其舌苔花白，应以虫病为主，而外邪未解亦须兼顾。以乌梅丸意立方治之。|乌梅灰二钱吴萸炒黄连四分川椒四分黄芩一钱五分银柴胡一钱五分炒老米四钱茯苓二钱橘饼四钱谷芽三钱苏梗一钱五分|服两剂热退病减，后以甘淡之药调治，遂渐痊愈。|

【范例5】
抱灵居士|次甥，发热、痰喘、足冷、面赤，以参苏饮二剂不应；或以升麻、葛根反呕，腹胀，咳汗，便秘；以藿香正气散加山楂一剂，泻后足温；以保和丸调姜汤，热退，咳痰清涕，胸澎齿衄，舌黄人倦；以香砂六君子汤加枳、桔，便秘，痰喘，舌黄，齿血；以凉膈散去硝、黄，加陈、半一剂，血止，咳甚；以华盖散一剂而愈。|

【范例6】
汪文斗乃侄女，甫生三日，忽不乳而脐腹胀大，延予过视，按之坚硬如石。曰：肠胃脆|弱，胎毒内攻，兼有秽血，据理无可生之机，喜其禀受苍实，乃试与利惊丹一服，行下积秽若干，其腹渐宽而饮乳，今已数岁矣。虽然，此亦偶中之耳，然不用此药似无别法，不知高明者又将何药而治之？|
"""
# ===========================================

SURNAMES = [
    "张", "王", "李", "赵", "陈", "刘", "周", "吴", "郑", "孙", 
    "朱", "马", "胡", "郭", "林", "何", "高", "罗", "沈", "韩",
    "钱", "陆", "徐", "杨", "顾", "叶", "方", "金", "袁", "曹",
    "邓", "许", "傅", "沈", "曾", "彭", "吕", "苏", "卢", "蒋", 
    "魏", "江", "谢", "邹", "喻", "柏", "水", "窦", "章", "戚",
    "尤", "毕", "屈", "裴", "武", "荀", "谈", "邵", "仲", "缪"
]

# 2. 复杂关系后缀 (80% 概率)
IDENTITY_TEMPLATES = [
    "{s}氏令爱", "{s}翁之孙", "{s}家稚子", "{s}姓幼女", "{s}君乃郎", 
    "{s}员外幼子", "{s}孟小", "{s}仲童", "{s}氏小儿", "{s}家乳儿", 
    "邻儿{s}姓", "{s}氏外孙", "{s}处士之子", "{s}二官", "{s}九小姐"
]

# 3. 简单通用称谓 (20% 概率 - 新增)
SIMPLE_IDENTITIES = [
    "某童", "某幼", "某小", "某孩", 
    "本城某童", "邻童", "一幼童", "一孩"
]

# 4. 年龄/阶段代称
AGE_TERMS = [
    # 极低龄 (针对新生儿类)
    "生三朝", "甫生三日", "弥月", "百日内", "襁褓中", "尚未断乳", "吮乳之年", 
    # 幼儿期
    "年二岁", "年三岁", "五岁稚龄", "六岁小儿", "七岁孩童", "年方八岁", 
    # 少年期
    "龆龄之年", "垂髫之时", "十岁内外", "年甫十二", "年十三", "十五岁以下", 
    # 书面雅称
    "稚年", "稚齿", "童蒙", "髫年", "弱小", "质弱"
]


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
write_lock = threading.Lock()
def get_random_identity(disease_name=""):
    surname = random.choice(SURNAMES)
    
    # 逻辑：如果是新生儿病，强制低龄
    newborn_keywords = ["新生儿", "胎", "脐", "初生", "吮乳", "婴儿"]
    is_newborn = any(kw in disease_name for kw in newborn_keywords)
    
    if is_newborn:
        age_desc = random.choice(["生三朝", "甫生三日", "弥月", "百日内", "襁褓中"])
        identity = f"{surname}氏乳儿"
    else:
        # 80% 复杂称谓，20% 简单
        if random.random() < 0.8:
            template = random.choice(IDENTITY_TEMPLATES)
            identity = template.format(s=surname)
        else:
            identity = random.choice(SIMPLE_IDENTITIES).replace("某", surname)
        
        # 增加“左/右”标识的概率（中医传统：左男右女）
        if random.random() < 0.3:
            identity += random.choice(["左", "右"])
            
        age_desc = random.choice(AGE_TERMS)
    
    return identity, age_desc

def parse_knowledge_base(file_path):
    """
    鲁棒性更强的知识库解析函数
    能自动识别多种分隔符，并打印调试信息
    """
    knowledge_list = []
    print(f"📖 正在读取知识库: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件不存在！请检查路径：{file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📄 文件共 {len(lines)} 行。正在分析前 3 行格式...")
    for i, line in enumerate(lines[:3]):
        print(f"   [Line {i+1}]: {line.strip()[:100]}...") # 只打印前100字

    success_count = 0
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line: continue

        disease = ""
        definition = ""
        category = "儿科" # 默认

        # --- 尝试解析策略 1: 标准正则 (兼容中英文冒号) ---
        # 匹配 "病名...分类...定义..." 这种结构
        try:
            # 1. 提取病名
            # 查找 "病名" 或 "名称" 后面，直到句号或逗号或空格
            match_name = re.search(r"(病名|名称)[:：]\s*(.*?)[。；;,\t]", line)
            if match_name:
                disease = match_name.group(2).strip()
            
            # 2. 提取定义
            # 查找 "定义" 或 "特征" 后面，直到行尾
            match_def = re.search(r"(定义|特征)[:：]\s*(.*)", line)
            if match_def:
                definition = match_def.group(2).strip()
        except: pass

        # --- 尝试解析策略 2: 简单分割 (如果是 Excel 导出的 Tab 分隔或逗号分隔) ---
        if not disease or not definition:
            parts = re.split(r'[\t,，]', line) # 按 Tab 或 逗号 切分
            if len(parts) >= 2:
                # 假设第一列是病名，最后一列是定义（常见 Excel 结构）
                # 简单清洗一下，去掉可能的 "中医病名：" 前缀
                raw_name = parts[0].replace("中医病名", "").replace("：", "").replace(":", "").strip()
                # 过滤掉表头
                if "病名" not in raw_name and len(raw_name) > 1: 
                    disease = raw_name
                    # 假设定义在最后或是最长的一段
                    longest_part = max(parts, key=len)
                    if len(longest_part) > 10:
                        definition = longest_part

        # --- 结果判定 ---
        if disease and definition:
            # 清洗一下定义中的前缀（如果有）
            definition = definition.replace("该病的定义和临床特征是", "").replace("：", "").replace(":", "").strip()
            
            knowledge_list.append({
                "disease": disease,
                "category": "儿科", # 既然是儿科知识库，直接定死
                "definition": definition
            })
            success_count += 1
        else:
            # 只在解析失败时打印前几行错误，防止刷屏
            if success_count < 3 and line_num < 5: 
                print(f"⚠️ 第 {line_num+1} 行解析失败，未找到病名或定义。")

    print(f"✅ 成功解析 {len(knowledge_list)} 条知识点。")
    
    if len(knowledge_list) == 0:
        print("❌ 警告：一条都没解析出来！请检查：")
        print("1. 文件是否为空？")
        print("2. 格式是否与预期完全不同？(建议贴一行原始内容给我看)")
        
    return knowledge_list

def get_existing_progress(temp_file):
    """断点续传检查"""
    existing_counts = {}
    if not os.path.exists(temp_file):
        return existing_counts
    
    print("🔄 正在恢复进度...")
    with open(temp_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                disease = data.get('meta', {}).get('disease', 'unknown')
                existing_counts[disease] = existing_counts.get(disease, 0) + 1
            except: continue
    return existing_counts

def generate_single_case(knowledge_item):
    """单个生成任务 (含 Few-Shot Prompt + 动态身份)"""
    disease = knowledge_item['disease']
    definition = knowledge_item['definition']
    
    # --- 修改点：调用身份生成器 ---
    patient_name, patient_age = get_random_identity()


    
    # 如果是“乳儿”、“襁褓”，逻辑上年龄不能太大，做个简单修正
    if "乳" in patient_name or "襁褓" in patient_name:
        patient_age = "数月"
    

    is_simple_style = random.random() < 0.3 
    
    if is_simple_style:
        style_instruction = """
        【写做要求】：采用【极简叙述流】。
        1. 仅保留：患者信息、核心症状（含一两个儿科特征词）、处方。
        2. 严禁使用“辨证”、“病因”、“方解”等标签。
        3. 总字数控制在 50-100 字以内。
        """
    else:
        style_instruction = """
        【写做要求】：采用【深度结构化流】。
        1. 包含：辨证、病因、症候、治法、处方、方解。
        2. 使用“|”分隔各段。
        3. 详细描述儿科特异性体征。
        """
    prompt = f"""
    你是一位经验丰富的中医儿科专家。请基于以下【参考范例】的风格，撰写一个真实的**【儿科】**医案。

    ### 参考范例（文风参考）：
    {FEW_SHOT_EXAMPLES}

    ### 核心任务信息：
    - 病名：{disease}
    - 临床特征：{definition}
    
    {style_instruction}
    ### 写作强制要求（提分关键）：
    1. **患者身份**：必须以 **“{patient_name}，{patient_age}。”** 开头。
    2. **特异性查体**：文中必须包含儿科特有的望诊描述。例如：**“指纹色紫/滞/直透气关”**、**“囟门平陷/隆起”**、**“啼哭无泪”** 或 **“乳食不化/呕吐乳块”**。
    3. **用药剂量（极重要）**：处方剂量必须符合儿科“轻灵”特色。每味药通常在 **0.3克至6克** 之间（或使用“分”、“钱”），**绝对禁止**出现内科常见的20克、30克大剂量。
    4. **对抗逻辑**：即便患者的表现是普通的感冒、咳嗽或腹泻（极易混淆为内科），也要强调其病机在于“脏腑娇嫩、脾常不足”或“胎毒/惊恐”，以此将分类锁定在儿科。
    5. **风格与格式**：半文言，古朴简练。
    """

    try:
        for _ in range(3): 
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "你是一个精通中医儿科的AI助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.95, # 稍微调高温度，增加多样性
                    max_tokens=600,
                    timeout=60
                )
                content = response.choices[0].message.content.strip()
                if len(content) > 10:
                    return {"content": content, "success": True, "disease": disease}
            except Exception as e:
                time.sleep(2)
                continue
        return {"success": False, "error": "Retries exhausted"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    knowledge_base = parse_knowledge_base(INPUT_FILE)
    existing_counts = get_existing_progress(TEMP_OUTPUT_FILE)
    
    tasks = []
    for item in knowledge_base:
        disease = item['disease']
        current_count = existing_counts.get(disease, 0)
        needed = NUM_CASES_PER_KNOWLEDGE - current_count
        if needed > 0:
            for _ in range(needed): tasks.append(item)
    
    print(f"📊 需生成任务数: {len(tasks)}")
    
    if not tasks:
        print("🎉 所有任务已完成。")
        convert_to_final_json()
        return

    print(f"🚀 启动线程池 (Max Workers: {MAX_WORKERS})...")
    
    with open(TEMP_OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(generate_single_case, t): t for t in tasks}
            pbar = tqdm(total=len(tasks), desc="Generating")
            
            for future in as_completed(future_to_task):
                result = future.result()
                if result['success']:
                    entry = {
                        "instruction": "",
                        "input": result['content'],
                        "output": "儿科",
                        "meta": {"disease": result['disease']}
                    }
                    with write_lock:
                        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        f_out.flush()
                pbar.update(1)
            pbar.close()

    print("✅ 生成结束。")
    convert_to_final_json()

def convert_to_final_json():
    print("🔄 转换格式中...")
    final_data = []
    if os.path.exists(TEMP_OUTPUT_FILE):
        with open(TEMP_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'meta' in obj: del obj['meta']
                    final_data.append(obj)
                except: pass
    
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"🎉 最终文件保存至: {FINAL_OUTPUT_FILE} (共 {len(final_data)} 条)")

if __name__ == "__main__":
    main()