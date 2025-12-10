import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================ 配置参数 ============================

# 1. 原版模型路径 (指向包含 .safetensors 的文件夹)
MODEL_PATH = "Qwen2.5-3B-Instruct-Local" 

# 2. 输入/输出路径
INPUT_DIR = os.path.join("output", "wp7")
OUTPUT_DIR = "txt训练(wp7)"

# 3. 性能参数
BATCH_SIZE = 32  # 3B模型在4060上，批大小设为32通常很稳且快

# 4. JSON 键名
TEXT_KEY = "text"

# ============================ 航母吧专用规则 ============================
KEYWORD_RULES = {
    3: ['加微信', 'vx', 'q群', '私信', '主页', '领资料', '兼职', '代做', '互关', 'BV', '拼夕夕', '砍一刀'],
    4: ['@'],
    5: ['经验+3', '水贴', '插眼', 'v我50', '886', '围观', '前排', '挽尊', 'dd', '顶'],
    2: [
        '美狗', '殖人', '耗材', '1450', '蛙', '呆湾', 
        '俄孝子', '黄俄', '乌贼',
        '赢麻了', '偷着乐', '下大棋', '这就是', '反思',
        '神神', '兔兔', '小粉红', '纳粹,' '鬼子', '棒子',
        'nmsl', '死全家', '孤儿', '脑瘫', '弱智', '傻逼', 'sb', 'cnm', 'nt'
    ]
}

# ============================ 核心函数 ============================

def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # 必须设置 padding_side='left' 才能进行批量生成
        tokenizer.padding_side = 'left' 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型到 GPU，使用 BF16 精度节省显存
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        print("模型加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"Error: 加载失败。请检查路径 {MODEL_PATH} 是否正确。\n详细错误: {e}")
        exit()

def check_keywords(text):
    """关键词快速筛查"""
    text_lower = text.lower()
    for cat_id, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                return cat_id
    return None

def batch_predict(texts, model, tokenizer):
    """批量推理函数"""
    # 1. 构建 Prompt
    prompts = []
    system_prompt = "你是一个贴吧评论审核员。请将评论分类为：1(正常，包括长篇分析等，出现卖惨的长篇大论可分为广告), 2(争论，只有带有明显的人身攻击、政治引战（灯塔、乌友、美狗等）、嘲讽语气的才归为), 3(广告,注意不要把有内容的长文本分到这里), 4(@某人，出现@归为), 5(无意义,如只有一个表情包，或只包含特殊字符，如+3，已三连等)。\n**只输出一个数字ID。**"

    for t in texts:
        # 使用 Qwen2.5 的对话模板
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"评论：{t}"}
        ]
        # apply_chat_template 会自动添加 <|im_start|> 等标记
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text_input)

    # 2. 批量编码
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    # 3. 批量生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2, # 只需要生成一个数字
            temperature=0.1,  # 低温，稳定
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # 4. 批量解码
    # 只解码新生成的部分
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # 5. 解析结果
    results = []
    for res in responses:
        match = re.search(r'\d+', res)
        if match:
            results.append(int(match.group(0)))
        else:
            results.append(1) # 默认正常
            
    return results

# ============================ 主程序 ============================
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Error: 输入目录 {INPUT_DIR} 不存在")
        exit()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载模型
    model, tokenizer = load_model()

    # 2. 扫描文件
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"找到 {len(files)} 个文件。")

    for json_file in tqdm(files, desc="总进度"):
        input_path = os.path.join(INPUT_DIR, json_file)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(json_file)[0] + ".txt")

        # 读取数据
        with open(input_path, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: continue

        file_lines = []     # 最终要写入的行
        batch_texts = []    # 待预测文本缓存
        batch_indices = []  # 待预测文本在原始列表中的索引
        
        # 预填充结果列表，长度与数据一致，先填 None
        results_map = [None] * len(data) 

        # --- 第一轮：关键词筛选 & 准备 Batch ---
        for i, item in enumerate(data):
            text = item.get(TEXT_KEY, "").strip()
            if not text:
                results_map[i] = "" # 空行标记
                continue
                
            clean_text = text.replace('\n', ' ').replace('\r', '')
            
            # 关键词检查
            kw_id = check_keywords(clean_text)
            
            if kw_id:
                # 命中关键词，直接定性
                results_map[i] = f"{i+1}. {clean_text}*{kw_id}"
            else:
                # 未命中，加入 GPU 处理队列
                batch_texts.append(clean_text)
                batch_indices.append(i)

        # --- 第二轮：GPU 批量预测 ---
        if batch_texts:
            # 按 BATCH_SIZE 切片处理
            for k in range(0, len(batch_texts), BATCH_SIZE):
                current_batch = batch_texts[k : k + BATCH_SIZE]
                current_indices = batch_indices[k : k + BATCH_SIZE]
                
                # 核心预测调用
                pred_ids = batch_predict(current_batch, model, tokenizer)
                
                # 回填结果
                for idx, pred_id in zip(current_indices, pred_ids):
                    original_text = batch_texts[k + current_indices.index(idx)]
                    results_map[idx] = f"{idx+1}. {original_text}*{pred_id}"

        # --- 第三轮：写入文件 ---
        valid_lines = [line for line in results_map if line] # 过滤掉空行
        if valid_lines:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write('\n'.join(valid_lines))

    print(f"\n处理完成！结果在: {OUTPUT_DIR}")
