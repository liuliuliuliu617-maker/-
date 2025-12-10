import os
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
import time
import argparse

# ============================ 配置参数 ============================

# 1. 路径配置
BASE_MODEL_PATH = r"models\Qwen2.5-3B-Instruct-Local"

# 可选板块以及对应的 LoRA 模型目录（位于当前文件夹下 models/）
SECTOR_NAMES = [
    "APEX",
    "cos",
    "FPS",
    "PUBG",
    "历史品鉴",
    "三角洲",
    "时政",
    "英雄联盟",
]
DEFAULT_SECTOR = "英雄联盟"
SECTOR_MODEL_MAP = {name: os.path.join("models", name) for name in SECTOR_NAMES}

# 输入文件夹的基础路径 (你只需要输入文件名，代码会自动拼上这个路径)
BASE_INPUT_DIR = os.path.join("output", "航空母舰")

# 输出文件夹
OUTPUT_DIR = "航空母舰_最终分类结果"

# 2. 性能参数
# 批大小不要太大，避免在 CPU 上一次算太多导致非常慢
BATCH_SIZE = 6
# Windows 环境下先关闭 4bit 量化，避免依赖 bitsandbytes
USE_4BIT = True 

# 关键词规则
KEYWORD_RULES = {
    3: ['加微信', 'vx', 'q群', '私信我', '主页有', '进群', '联系方式', '送福利', '优惠券', '秒杀', '限时', '免费领', '搭子', '互关', 'BV'],
    4: ['@'],
    2: ['洗澡狗', 'gsl', '猪杂', '皇军', '僵尸', 'sgjj', '水鬼', '÷', '出生', '死妈', '杂交', '盗版', '虚空'],
    5: ['经验+3', '水贴', '插眼', 'v我50']
}

TEXT_KEY = "text"
ID_MAP = {1: "正常", 2: "争论", 3: "广告", 4: "@某人", 5: "无意义"}

# ============================ 核心函数 ============================

def load_model(sector: str | None = None):
    """根据板块加载对应的 LoRA 模型。

    模型目录约定为 models/<板块名>，例如 models/英雄联盟。
    当传入未知板块或对应 LoRA 目录不存在时，会回退到默认板块 DEFAULT_SECTOR。
    """

    original_sector = sector
    # 先按名称检查
    if sector not in SECTOR_MODEL_MAP:
        sector = DEFAULT_SECTOR

    lora_path = SECTOR_MODEL_MAP[sector]

    # 如果对应目录不存在，则回退到默认板块
    if not os.path.isdir(lora_path):
        print(
            f"警告: 板块 '{original_sector}' 的 LoRA 路径 '{lora_path}' 不存在，"
            f"将使用默认板块 '{DEFAULT_SECTOR}' 的模型。"
        )
        sector = DEFAULT_SECTOR
        lora_path = SECTOR_MODEL_MAP[sector]

    print(f"当前实际使用板块: {sector}")
    print(f"Loading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    print(f"Loading model (4-bit={USE_4BIT})...")
    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # 为避免自动 offload 带来的 LoRA 适配 KeyError，我们显式选择单一设备加载
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map=None,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    print(f"Loading LoRA adapter from: {lora_path}")
    # 同样不使用 offload，直接在同一设备上加载 LoRA
    model = PeftModel.from_pretrained(base_model, lora_path, device_map=None)
    model.to(device)
    # 兼容后续 model.device 的用法
    model.device = torch.device(device)
    model.eval() 
    
    return model, tokenizer

def check_keywords(text):
    text_lower = text.lower()
    for category_id, keywords in KEYWORD_RULES.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return category_id, keyword
    return None, None

def predict_batch_llm(texts, model, tokenizer):
    prompts = []
    for text in texts:
        prompt = f"<|im_start|>system\n你是一个分类助手。<|im_end|>\n<|im_start|>user\n请对以下评论进行分类，只输出类别ID。\n评论内容：{text}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2,
            temperature=0.1,  
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    results = []
    for output_text in decoded_outputs:
        match = re.search(r'\d+', output_text)
        if match:
            results.append(int(match.group(0)))
        else:
            results.append(1)
    return results

def classify_items(data, model, tokenizer):
    """对一批评论数据进行分类，返回带预测结果的列表。

    data: List[dict]，每项至少包含 TEXT_KEY 对应的文本字段，例如 {"bv": ..., "text": ...}
    """
    results = []
    batch_indices = []
    batch_texts = []
    
    start_time = time.time()
    
    # 1. 预处理与关键词匹配
    for i, item in enumerate(data):
        text = item.get(TEXT_KEY, "").strip()
        item['classification_method'] = "empty"
        item['predicted_id'] = None
        
        if not text:
            results.append(item)
            continue
            
        clean_text = text.replace('\n', ' ').replace('\r', '')
        kw_id, hit_word = check_keywords(clean_text)
        
        if kw_id is not None:
            item['predicted_id'] = kw_id
            item['classification_method'] = f"keyword ({hit_word})"
            item['predicted_label'] = ID_MAP.get(kw_id, "未知")
            results.append(item)
        else:
            batch_indices.append(i)
            batch_texts.append(clean_text)
            results.append(item)
    
    # 2. 批量模型预测（对所有需模型判断的评论全部调用 LLM）
    if batch_texts:
        total_to_predict = len(batch_texts)
        texts_for_llm = batch_texts
        indices_for_llm = batch_indices

        total_batches = (total_to_predict + BATCH_SIZE - 1) // BATCH_SIZE
        # 注意：Windows 控制台默认使用 GBK 编码，无法显示 emoji，会导致 UnicodeEncodeError
        print(f"正在调用 GPU 预测 {total_to_predict} 条评论 (共 {total_batches} 批)...")

        for i in tqdm(range(0, total_to_predict, BATCH_SIZE), leave=False):
            current_texts = texts_for_llm[i : i + BATCH_SIZE]
            current_indices = indices_for_llm[i : i + BATCH_SIZE]

            pred_ids = predict_batch_llm(current_texts, model, tokenizer)

            for idx, pred_id in zip(current_indices, pred_ids):
                results[idx]['predicted_id'] = pred_id
                results[idx]['predicted_label'] = ID_MAP.get(pred_id, "未知")
                results[idx]['classification_method'] = "model_batch"


    return results


def process_single_file(filename, model, tokenizer):
    """处理单个文件的核心逻辑（交互模式下使用 BASE_INPUT_DIR/OUTPUT_DIR）。"""
    input_file = os.path.join(BASE_INPUT_DIR, filename)
    output_file = os.path.join(OUTPUT_DIR, f"classified_{filename}")

    if not os.path.exists(input_file):
        print(f"❌ 错误：在 '{BASE_INPUT_DIR}' 下找不到文件 '{filename}'")
        return

    print(f"📖 正在读取 {filename} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("❌ 错误：JSON 文件格式损坏。")
            return

    if not data:
        print("⚠️  警告：文件为空。")
        return

    start_time = time.time()

    results = classify_items(data, model, tokenizer)

    # 3. 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    elapsed = time.time() - start_time
    print(f"✅ 处理完成！耗时: {elapsed:.2f}s")
    print(f"📂 结果已保存至: {output_file}")


# ============================ 主程序 (交互循环) ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评论分类（按板块选择模型）")
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="板块名称，如: APEX / cos / FPS / PUBG / 历史品鉴 / 三角洲 / 时政 / 英雄联盟",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        type=str,
        default=None,
        help="输入 JSON 路径（服务模式，配合 --out 使用）",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=str,
        default=None,
        help="输出 JSON 路径（服务模式，配合 --in 使用）",
    )

    args = parser.parse_args()

    sector = args.sector
    service_mode = bool(args.input_path and args.output_path)

    # 决定板块
    if not sector:
        if service_mode:
            # 服务模式下未显式指定，使用默认板块
            sector = DEFAULT_SECTOR
        else:
            print("可选板块：" + " / ".join(SECTOR_NAMES))
            sector_input = input(f"请输入板块名称（默认 {DEFAULT_SECTOR}）: ").strip()
            sector = sector_input or DEFAULT_SECTOR

    if sector not in SECTOR_MODEL_MAP:
        print(f"未识别板块 '{sector}'，使用默认 '{DEFAULT_SECTOR}'")
        sector = DEFAULT_SECTOR

    # 0. 准备输出目录（仅交互模式使用）
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 启动时加载模型 (只做一次)
    print("\n" + "="*50)
    print(f"正在初始化 Qwen2.5-3B 模型（板块：{sector}），请稍候...")
    print("这可能需要 10-20 秒，取决于你的硬盘速度。")
    print("="*50 + "\n")
    
    model, tokenizer = load_model(sector)
    
    # 服务模式：供后端调用，读取 --in，写入 --out
    if service_mode:
        in_path = args.input_path
        out_path = args.output_path

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 无法读取输入 JSON: {e}", file=sys.stderr)
            sys.exit(1)

        if not isinstance(data, list):
            print("❌ 输入 JSON 格式错误，期望为数组", file=sys.stderr)
            sys.exit(1)

        results = classify_items(data, model, tokenizer)

        # 转为后端期望的结构：
        # [{ original_comment_data: <原始数据+预测字段>, predicted_label_id, predicted_label_text }, ...]
        out_items = []
        for item in results:
            out_items.append({
                "original_comment_data": item,
                "predicted_label_id": item.get("predicted_id"),
                "predicted_label_text": item.get("predicted_label"),
            })

        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_items, f, ensure_ascii=False, indent=4)

        print(f"✅ 服务模式完成：读取 {len(data)} 条评论，结果已保存至 {out_path}")
        sys.exit(0)

    # 交互模式：沿用原来的行为
    print("\n" + "="*50)
    print("🎉 模型加载完毕！系统已就绪。")
    print(f"📂 默认输入目录: {BASE_INPUT_DIR}")
    print("="*50)

    # 2. 进入交互循环
    while True:
        print("\n" + "-"*30)
        user_input = input("请输入 JSON 文件名 (例如: 1.json) | 输入 q 退出: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("再见！")
            break
            
        if not user_input:
            continue
            
        # 容错：如果你不小心输入了全路径，代码尝试提取文件名
        if os.path.sep in user_input:
            user_input = os.path.basename(user_input)
            
        # 容错：如果你忘了加 .json 后缀
        if not user_input.endswith('.json'):
            user_input += '.json'
            
        process_single_file(user_input, model, tokenizer)
