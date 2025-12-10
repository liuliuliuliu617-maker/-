import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
import time

# ============================ 配置参数 ============================

# 1. 路径配置
BASE_MODEL_PATH = "Qwen2.5-3B-Instruct-Local"
LORA_PATH = "qwen2.5_3b_qlora_output(时政)" 

# 输入文件夹的基础路径 (你只需要输入文件名，代码会自动拼上这个路径)
BASE_INPUT_DIR = os.path.join("output", "航空母舰")

# 输出文件夹
OUTPUT_DIR = "航空母舰_最终分类结果"

# 2. 性能参数
BATCH_SIZE = 16
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

def load_model():
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

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        quantization_config=bnb_config,
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
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

def process_single_file(filename, model, tokenizer):
    """处理单个文件的核心逻辑"""
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
    
    # 2. 批量模型预测
    if batch_texts:
        total_batches = (len(batch_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"🤖 正在调用 GPU 预测 {len(batch_texts)} 条评论 (共 {total_batches} 批)...")
        
        for i in tqdm(range(0, len(batch_texts), BATCH_SIZE), leave=False):
            current_texts = batch_texts[i : i + BATCH_SIZE]
            current_indices = batch_indices[i : i + BATCH_SIZE]
            
            pred_ids = predict_batch_llm(current_texts, model, tokenizer)
            
            for idx, pred_id in zip(current_indices, pred_ids):
                results[idx]['predicted_id'] = pred_id
                results[idx]['predicted_label'] = ID_MAP.get(pred_id, "未知")
                results[idx]['classification_method'] = "model_batch"

    # 3. 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    elapsed = time.time() - start_time
    print(f"✅ 处理完成！耗时: {elapsed:.2f}s")
    print(f"📂 结果已保存至: {output_file}")


# ============================ 主程序 (交互循环) ============================
if __name__ == "__main__":
    # 0. 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 启动时加载模型 (只做一次)
    print("\n" + "="*50)
    print("正在初始化 Qwen2.5-3B 模型，请稍候...")
    print("这可能需要 10-20 秒，取决于你的硬盘速度。")
    print("="*50 + "\n")
    
    model, tokenizer = load_model()
    
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
