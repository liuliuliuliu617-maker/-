import os
import re
import gc # 【新增】引入垃圾回收
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    prepare_model_for_kbit_training
)
from datasets import Dataset

# ================= 配置参数 =================
# 1. 基础模型路径
MODEL_PATH = "Qwen2.5-3B-Instruct-Local"

# 训练超参数 (全局通用)
MAX_LEN = 256           
BATCH_SIZE = 1          
GRAD_ACCUMULATION = 8   
EPOCHS = 3              
LEARNING_RATE = 2e-4    

KEYWORD_RULES = {
    # ... (保持原有的关键词规则不变，此处省略以节省篇幅) ...
    'cjb': (2, '争论'),
    '僵尸': (2, '争论'),
    '送姜': (2, '争论'),
    '断手': (2, '争论'),
    '研发i': (2, '争论'),
    '水鬼': (2, '争论'),
    'sgjj': (2, '争论'),
    '暴毙': (2, '争论'),
    '4396': (2, '争论'),
    '猪杂': (2, '争论'),
    '科杂': (2, '争论'),
    '皇军': (2, '争论'),
    '杂交': (2, '争论'),
    '拉夫': (2, '争论'),
    '懦手': (2, '争论'),
    '销户': (2, '争论'),
    '÷': (2, '争论'),
    '出生': (2, '争论'),
    '死妈': (2, '争论'),
    '孤儿': (2, '争论'),
    '户口': (2, '争论')
}

# ================= 数据加载函数 (保持不变) =================
def load_data(data_dir):
    """读取 txt 文件并转换为带有【关键词提示】的指令微调格式"""
    data = []
    pattern = re.compile(r'^\s*\d+\.\s*(.*?)\*(\d+)', re.MULTILINE)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return []

    print(f"Loading data from {data_dir}...")
    rule_hit_count = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    matches = pattern.finditer(content)
                    for match in matches:
                        text = match.group(1).strip()
                        label_id = match.group(2).strip()
                        
                        if not text or not label_id:
                            continue

                        found_keyword = None
                        target_class_info = None
                        
                        for kw, (cid, cname) in KEYWORD_RULES.items():
                            if kw in text:
                                found_keyword = kw
                                target_class_info = (str(cid), cname)
                                break
                        
                        base_instruction = "请对以下评论进行分类，只输出类别ID。"
                        
                        if found_keyword:
                            rule_hit_count += 1
                            hint_instruction = (
                                f"{base_instruction}\n"
                                f"注意：评论中包含关键词“{found_keyword}”，通常属于【{target_class_info[1]}】类。"
                            )
                            sample = {
                                "instruction": hint_instruction,
                                "input": f"评论内容：{text}",
                                "output": label_id
                            }
                            # 过采样
                            data.append(sample)
                            data.append(sample)
                            data.append(sample)
                        else:
                            data.append({
                                "instruction": base_instruction,
                                "input": f"评论内容：{text}",
                                "output": label_id
                            })

            except Exception as e:
                print(f"Skipping {filename}: {e}")
    
    print(f"Loaded {len(data)} training samples.")
    print(f"Keyword rules triggered {rule_hit_count} times (before oversampling).")
    return data

# ================= 数据预处理 (保持不变) =================
def process_func(example, tokenizer):
    messages = [
        {"role": "system", "content": "你是一个专业的文本分类助手。"},
        {"role": "user", "content": example['instruction'] + "\n" + example['input']}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_full = text_input + example['output'] + tokenizer.eos_token
    tokenized = tokenizer(text_full, truncation=True, max_length=MAX_LEN, padding="max_length")
    
    input_ids = tokenized["input_ids"]
    labels = [-100] * len(input_ids)
    
    input_len = len(tokenizer(text_input, truncation=True, max_length=MAX_LEN)["input_ids"])
    if input_len < len(input_ids):
        for i in range(input_len, len(input_ids)):
            if input_ids[i] == tokenizer.pad_token_id:
                break
            labels[i] = input_ids[i]
            
    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# ================= 【核心修改】单个任务训练函数 =================
def train_one_task(data_dir, output_dir):
    """
    针对单个数据文件夹执行完整的训练流程
    """
    print(f"\n{'='*20} 开始处理任务 {'='*20}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 1. 检查是否已经训练完成
    # 检查输出目录中是否有 adapter_config.json，如果有则视为已完成
    if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        print(f"检测到 {output_dir} 已存在完成的模型文件，跳过训练。")
        return

    # 2. 准备数据
    raw_data = load_data(data_dir)
    if not raw_data:
        print(f"目录 {data_dir} 中没有有效数据，跳过。")
        return

    # 3. 加载 Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. 配置量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 
    )

    # 5. 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 6. 预处理模型
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()

    # 7. 配置 LoRA
    print("Configuring LoRA...")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        inference_mode=False,
        r=16,            
        lora_alpha=32,
        lora_dropout=0.05
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 8. 处理数据集
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(lambda x: process_func(x, tokenizer), remove_columns=dataset.column_names)

    # 9. 训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        logging_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,       
        optim="paged_adamw_32bit", 
        save_strategy="epoch",
        warmup_ratio=0.03,
        gradient_checkpointing=True, 
        remove_unused_columns=False,
        save_total_limit=2 
    )

    # 10. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    # 11. 训练或断点续训
    model.config.use_cache = False 
    
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
    
    if last_checkpoint is not None:
        print(f"检测到断点: {last_checkpoint}，继续训练...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("开始新训练...")
        trainer.train()

    # 12. 保存
    print(f"Saving model to {output_dir}")
    model.config.use_cache = True
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # ================= 【重要】清理显存 =================
    print("Cleaning up GPU memory...")
    del model
    del trainer
    del tokenizer
    del dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(f"任务 {data_dir} 完成。\n")


# ================= 主程序入口 =================
if __name__ == "__main__":
    # 获取当前目录下所有文件夹
    all_items = os.listdir('.')
    
    # 筛选出以 "txt训练" 开头的文件夹
    # 例如: "txt训练(PUBG)", "txt训练(LOL)"
    target_dirs = [d for d in all_items if os.path.isdir(d) and d.startswith("txt训练")]
    
    print(f"扫描到 {len(target_dirs)} 个待训练文件夹: {target_dirs}")
    
    for data_dir in target_dirs:
        # 自动生成输出目录名称
        # 例如: "txt训练(PUBG)" -> "qwen2.5_3b_qlora_output(PUBG)"
        
        # 提取括号里的内容，或者直接替换前缀
        # 简单替换策略：
        dir_suffix = data_dir.replace("txt训练", "") # 得到 "(PUBG)"
        output_dir = f"qwen2.5_3b_qlora_output{dir_suffix}"
        
        try:
            train_one_task(data_dir, output_dir)
        except Exception as e:
            print(f"训练 {data_dir} 时发生严重错误: {e}")
            print("尝试清理显存并继续下一个任务...")
            torch.cuda.empty_cache()
            continue
            
    print("\n所有任务处理完毕！")
