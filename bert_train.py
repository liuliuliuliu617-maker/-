import os
import re
import json
import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================ 中文显示配置 ============================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================ 配置参数 ============================
MODEL_NAME = "bert-base-chinese"
LOCAL_MODEL_DIR = "./bert-base-chinese-local"
SAVE_MODEL_DIR = "model_saved_with_keywords(英雄联盟)" # 修改保存路径名以区分
CLASS_FILE = "class.txt"
TRAIN_DATA_DIR = "txt训练(英雄联盟)" # 请确保这是你的数据目录名

MAX_LEN = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EPOCHS = 10
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.1
VAL_SPLIT_RATIO = 0.1

# ============================ 关键词配置 ============================
# !!! IMPORTANT: 这里的 KEYWORD_CONFIG_TEMPLATE 仅用于定义关键词和权重。
# 它的键现在将是“文本标签”，而不是 0-based ID，以便在主程序中根据实际加载的类别进行映射。
# 请重新审视并精确你的关键词列表。避免过于宽泛的词语。
RAW_KEYWORD_CONFIG_TEMPLATE = {
    '争论': { 
        'keywords': ['艹', '香草','假','超市','唐','卖','麦麸','逆天神人','烧鸡','性压抑','有股味',"幽默",'pc',"诗人",'bot','入机','抄袭','小孩','吵','大份','屎','低智','诸如','÷女','拉踩','招笑','妈的','÷'],
        'weight': 1.0 # 匹配到关键词时，该特征的强度
    },
    '广告': { # 请根据 class.txt 确保 '广告' 是实际的文本标签
       'keywords': ['加微信', 'vx', 'q群', '私信', '主页有', '私我','进群', '联系方式','送福利', '优惠券', '秒杀', '限时', '免费领', '搭子','找到男朋友','找女朋友','找男朋友','互关','BV','私信UP','聊天记录'], # 精简广告词
       'weight': 1.0
    },
    '@某人': {
       'keywords': ['@'],
       'weight': 1.0
    }
    # 如果有更多类别需要关键词增强，可以在这里添加，键为文本标签
}
# KEYWORD_FEATURE_DIM_PER_CLASS 将在主程序中被 num_labels 覆盖，保持一致。


# ============================ 自定义分类器模型 (修改版：直接注入法) ============================
class BertForSequenceClassificationWithKeywords(nn.Module):
    def __init__(self, bert_model, num_labels, keyword_feature_dim, class_weights=None):
        super().__init__()
        self.bert = bert_model
        self.num_labels = num_labels
        # keyword_feature_dim 必须等于 num_labels，用于直接相加
        self.keyword_feature_dim = keyword_feature_dim 
        self.class_weights = class_weights

        # 【修改点1】分类器只接收 BERT 的输出维度 (768维)
        classifier_input_dim = self.bert.config.hidden_size

        # 自定义分类头：一个简单的 MLP，现在只处理 BERT 特征
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_labels)
        )

    def forward(self, input_ids, attention_mask, keyword_features=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        bert_output = outputs.pooler_output # (batch_size, 768)

        # 【修改点2】不再拼接，而是直接让 BERT 输出通过分类器，得到基础 Logits
        bert_logits = self.classifier(bert_output) # (batch_size, num_labels)

        # 【修改点3】处理关键词特征，并直接加到 Logits 上
        if keyword_features is not None:
            # 确保 keyword_features 的维度是 (batch_size, self.num_labels)
            # 这里的 keyword_features 就是你生成的包含 [0, 1000.0, 0...] 的向量
            if keyword_features.size(1) != self.num_labels:
                 # (保留之前的维度检查和修复逻辑，略...)
                 print(f"Warning: Keyword features dimension mismatch. Adjusting...")
                 if keyword_features.size(1) < self.num_labels:
                    padding = torch.zeros(keyword_features.size(0), self.num_labels - keyword_features.size(1), device=keyword_features.device)
                    keyword_features = torch.cat([keyword_features, padding], dim=1)
                 else:
                    keyword_features = keyword_features[:, :self.num_labels]

            # --- 核心修改：直接相加 ---
            # 最终得分 = BERT判断的得分 + 关键词强制赋予的得分
            # 这里的加法会广播到 batch 中的每一个样本
            final_logits = bert_logits + keyword_features
        else:
            final_logits = bert_logits

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            # 使用最终的 logits 计算损失
            loss = loss_fct(final_logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": final_logits}


# ============================ 辅助函数 ============================

def load_classes(class_file_path):
    """
    从 class.txt 文件加载类别名称和映射。
    文件格式: ID: 类别名称 (例如: 1: 正常)
    返回：
    1. id_to_label_0_based_for_report (0-based ID -> 文本标签，用于报告)
    2. label_text_to_0_based_id (文本标签 -> 0-based ID，用于数据加载)
    3. original_id_from_file_to_label_text (class.txt 中原始 ID -> 文本标签，用于数据加载)
    4. num_labels (类别总数)
    """
    original_id_from_file_to_label_text = {}
    
    with open(class_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':', 1)
            if len(parts) == 2:
                class_id = int(parts[0].strip())
                label = parts[1].strip()
                original_id_from_file_to_label_text[class_id] = label
            else:
                print(f"Warning: Class file line format not recognized: '{line}'. Please check 'class.txt'. Skipping.")

    if not original_id_from_file_to_label_text:
        print("Error: No classes parsed from class.txt. Please ensure file exists and uses 'ID: Label' format with semi-colon.")
        return {}, {}, {}, 0

    sorted_original_ids = sorted(original_id_from_file_to_label_text.keys())
    
    id_to_label_0_based_for_report = {}
    label_text_to_0_based_id = {}

    for i, original_id in enumerate(sorted_original_ids):
        label_text = original_id_from_file_to_label_text[original_id]
        id_to_label_0_based_for_report[i] = label_text
        label_text_to_0_based_id[label_text] = i
    
    num_labels = len(id_to_label_0_based_for_report)

    print(f"Loaded classes (0-based ID -> Label Text): {id_to_label_0_based_for_report}")
    return id_to_label_0_based_for_report, label_text_to_0_based_id, original_id_from_file_to_label_text, num_labels


def generate_keyword_features(text, keyword_config_0_based_map, num_labels):
    """
    根据文本内容和关键词配置生成关键词特征。
    返回的特征向量维度为 num_labels。
    """
    feature_vector = [0.0] * num_labels
    
    for label_0_based_id, config in keyword_config_0_based_map.items():
        if 0 <= label_0_based_id < num_labels: # 确保ID在有效范围内
            for keyword in config['keywords']:
                if keyword.lower() in text.lower():
                    feature_vector[label_0_based_id] = config['weight']
                    break
    return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)


def load_data_from_txt(data_dir, original_id_to_label_text_map, label_text_to_0_based_id_map, 
                       keyword_config_0_based_map, num_labels):
    """
    从指定目录下的TXT文件加载文本和标签。支持多行内容，以 `*类别ID` 结尾。
    文件格式: 序号. 内容*类别ID
    并生成关键词特征。
    """
    texts = []
    labels = []
    all_original_texts = []
    keyword_features_list = []
    skipped_count = 0
    print(f"Loading data from: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: Training data directory '{data_dir}' not found.")
        exit()
    
    sample_pattern = re.compile(
        r'^\s*(\d+)\.\s*'          
        r'([\s\S]*?)'             
        r'\*(\d+)'                
        r'(?=\s*(?:^\s*\d+\.|\Z))'
        , re.MULTILINE | re.DOTALL
    )

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                full_content = f.read()
            
            matches = sample_pattern.finditer(full_content)

            found_in_file = 0
            for match in matches:
                text_content = match.group(2).strip()
                raw_label_id_from_txt = int(match.group(3).strip())
                
                if text_content and raw_label_id_from_txt is not None:
                    if raw_label_id_from_txt in original_id_to_label_text_map:
                        label_text = original_id_to_label_text_map[raw_label_id_from_txt]
                        if label_text in label_text_to_0_based_id_map:
                            standard_label_id = label_text_to_0_based_id_map[label_text]
                            texts.append(text_content)
                            labels.append(standard_label_id)
                            all_original_texts.append(text_content)
                            
                            # 生成关键词特征
                            keyword_features_list.append(generate_keyword_features(
                                text_content, keyword_config_0_based_map, num_labels
                            ))
                            found_in_file += 1
                        else:
                            print(f"Warning: Label text '{label_text}' (orig ID {raw_label_id_from_txt}) not in 0-based mapping. Skipping. Content: '{text_content[:50]}...' in '{filename}'")
                            skipped_count += 1
                    else:
                        print(f"Warning: Original label ID '{raw_label_id_from_txt}' from TXT not in class file. Skipping. Content: '{text_content[:50]}...' in '{filename}'")
                        skipped_count += 1
                else:
                    print(f"Warning: Extracted empty text or label from match in '{filename}'. Skipping.")
                    skipped_count += 1
            
            if found_in_file == 0:
                print(f"Warning: No valid samples found matching '序号. 内容*类别ID' pattern in '{filename}'. Please check file content and format.")
    
    print(f"Loaded {len(texts)} samples. Skipped {skipped_count} invalid lines.")
    return texts, labels, all_original_texts, keyword_features_list

def flat_accuracy(preds, labels):
    """计算预测准确率"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# (predict_new_text 函数省略，不需要修改)

def train_and_evaluate_model(model, train_dataloader, val_dataloader, test_dataloader,
                             optimizer, scheduler, device, epochs,
                             id_to_label_map_for_report, num_labels,
                             test_original_texts, tokenizer_obj, KEYWORD_CONFIG_0_BASED_LOCAL): # 传入tokenizer_obj for prediction
    """训练和评估模型的主函数"""
    best_val_accuracy = 0
    model.to(device)
    
    if os.path.exists(SAVE_MODEL_DIR) and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(SAVE_MODEL_DIR)):
        print(f"Found existing model in '{SAVE_MODEL_DIR}'. Starting validation accuracy from 0 for current run.")

    print("\nStarting training...")
    for epoch_i in range(epochs):
        print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
        print("Training...")

        total_loss = 0
        model.train() # 设置模型为训练模式

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_keyword_features = batch[2].to(device)
            b_labels = batch[3].to(device)

            model.zero_grad()

            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            keyword_features=b_keyword_features,
                            labels=b_labels)
            
            loss = outputs["loss"]
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if step % 50 == 0 and not step == 0:
                print(f"  Batch {step:>5,}  Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"\n  Average training loss: {avg_train_loss:.2f}")

        print("Running Validation...")
        model.eval()

        eval_accuracy = 0
        nb_eval_steps = 0
        
        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_keyword_features = batch[2].to(device)
            b_labels = batch[3].to(device)

            with torch.no_grad():
                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                keyword_features=b_keyword_features,
                                labels=b_labels)
            
            logits = outputs["logits"].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print(f"  Validation Accuracy: {eval_accuracy/nb_eval_steps:.2f}")
        
        current_val_accuracy = eval_accuracy/nb_eval_steps
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            if not os.path.exists(SAVE_MODEL_DIR):
                os.makedirs(SAVE_MODEL_DIR)
            
            torch.save(model.state_dict(), os.path.join(SAVE_MODEL_DIR, "pytorch_model.bin"))
            model.bert.save_pretrained(SAVE_MODEL_DIR)
            tokenizer_obj.save_pretrained(SAVE_MODEL_DIR)
            # 额外保存 KEYWORD_CONFIG_0_BASED 到模型目录
            with open(os.path.join(SAVE_MODEL_DIR, "keyword_config.json"), 'w', encoding='utf-8') as f:
                json.dump(KEYWORD_CONFIG_0_BASED_LOCAL, f, ensure_ascii=False, indent=4)

            print(f"  >>> Saved best model (state_dict) and keyword config to '{SAVE_MODEL_DIR}' with accuracy {best_val_accuracy:.4f} <<<")

    print("\nTraining complete!")

    # ==================== 测试集评估 (与之前相同，只是使用保存的最佳模型) ====================
    print("\nRunning Test Evaluation...")
    
    bert_base_model_eval = BertModel.from_pretrained(SAVE_MODEL_DIR)
    model_eval = BertForSequenceClassificationWithKeywords(
        # 注意：评估时不一定需要传入 class_weights，因为 CrossEntropyLoss 只在训练时需要。
        # 如果为了保持一致性传入，确保权重在正确的设备上。这里简单起见不传入。
        bert_base_model_eval, num_labels, num_labels 
    )
    if os.path.exists(os.path.join(SAVE_MODEL_DIR, "pytorch_model.bin")):
        model_eval.load_state_dict(torch.load(os.path.join(SAVE_MODEL_DIR, "pytorch_model.bin"), map_location=device))
        print(f"Loaded best model state_dict from '{SAVE_MODEL_DIR}' for final test evaluation...")
    else:
        print(f"Error: No best model state_dict found in '{SAVE_MODEL_DIR}'. Cannot perform final test evaluation with best model.")
        return
        
    model_eval.to(device)
    model_eval.eval()
    
    predictions, true_labels_eval = [], []
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_keyword_features = batch[2].to(device)
        b_labels = batch[3].to(device)

        with torch.no_grad():
            outputs = model_eval(input_ids=b_input_ids,
                                 attention_mask=b_input_mask,
                                 keyword_features=b_keyword_features)
            
        logits = outputs["logits"].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels_eval.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels_eval = np.concatenate(true_labels_eval, axis=0)

    pred_labels = np.argmax(predictions, axis=1).flatten()
    target_names = [id_to_label_map_for_report[i] for i in range(num_labels)]
    
    print("\n--- Classification Report ---")
    print(classification_report(true_labels_eval, pred_labels, target_names=target_names))

    cm = confusion_matrix(true_labels_eval, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_plot_path = os.path.join(SAVE_MODEL_DIR, "confusion_matrix_with_keywords.png")
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix saved to '{cm_plot_path}'")
    plt.show()

    # ==================== 保存分类结果到CSV文件 ====================
    if len(test_original_texts) != len(true_labels_eval):
        print("Warning: Length of test_original_texts does not match true_labels. Cannot generate full CSV report.")
    else:
        true_label_texts = [id_to_label_map_for_report[label] for label in true_labels_eval]
        predicted_label_texts = [id_to_label_map_for_report[label] for label in pred_labels]

        results_df = pd.DataFrame({
            'Original_Text': test_original_texts,
            'True_Label_ID': true_labels_eval,
            'True_Label_Text': true_label_texts,
            'Predicted_Label_ID': pred_labels,
            'Predicted_Label_Text': predicted_label_texts
        })

        csv_output_path = os.path.join(SAVE_MODEL_DIR, "classified_comments_with_keywords.csv")
        results_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        print(f"\nTest set classification results saved to '{csv_output_path}'")


# ============================ 主程序 ============================
if __name__ == "__main__":
    # ============================ 模型下载/加载检查 (修改版) ============================
    # 必须存在的分词器文件
    required_tokenizer_files = ["vocab.txt", "tokenizer_config.json"]
    # 必须存在的模型配置文件
    required_config_file = "config.json"
    # 权重文件，二者有一即可
    possible_weight_files = ["pytorch_model.bin", "model.safetensors"]

    local_model_files_exist = True
    
    # 1. 检查目录是否存在
    if not os.path.exists(LOCAL_MODEL_DIR):
        local_model_files_exist = False
        print(f"Directory '{LOCAL_MODEL_DIR}' does not exist.")
    else:
        # 2. 检查必需的文件 (config 和 tokenizer)
        for f_name in required_tokenizer_files + [required_config_file]:
            if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, f_name)):
                print(f"Missing required file: {f_name}")
                local_model_files_exist = False
                break
        
        # 3. 检查权重文件 (二选一)
        if local_model_files_exist: # 只有前面的文件都存在才检查权重
            weight_found = False
            for w_file in possible_weight_files:
                if os.path.exists(os.path.join(LOCAL_MODEL_DIR, w_file)):
                    weight_found = True
                    print(f"Found model weights file: {w_file}")
                    break
            if not weight_found:
                print(f"No valid weight file found. Looked for: {possible_weight_files}")
                local_model_files_exist = False

    if not local_model_files_exist:
        # (这里是下载代码，如果你的环境无法联网，这段代码触发了也会报错，但现在应该不会触发了)
        print(f"Local BERT pre-trained model files incomplete in '{LOCAL_MODEL_DIR}'. Attempting to download...")
        try:
            # 尝试使用镜像站环境变量，以防万一
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
            
            tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_MODEL_DIR)
            model = BertModel.from_pretrained(MODEL_NAME, cache_dir=LOCAL_MODEL_DIR)
            
            # 强制保存到指定目录，确保下次能找到
            tokenizer.save_pretrained(LOCAL_MODEL_DIR)
            model.save_pretrained(LOCAL_MODEL_DIR)
            print(f"Model downloaded and saved to '{LOCAL_MODEL_DIR}'.")
        except Exception as e:
            print(f"\nCritical Error: Failed to download model. Details: {e}")
            print("Please ensure you have a working internet connection or manually place the required files.")
            exit()
    else:
        print(f"Local BERT pre-trained model files validated in '{LOCAL_MODEL_DIR}'. Proceeding...")

    # ============================ 实际主程序逻辑 ============================

    # 1. 加载类别
    id_to_label_map_for_report, label_text_to_0_based_id, original_id_from_file_to_label_text, num_labels = load_classes(CLASS_FILE)

    if not num_labels:
        print("Error: No classes loaded. Exiting.")
        exit()

    # 2. 准备 KEYWORD_CONFIG 映射，使用 0-based ID
    KEYWORD_CONFIG_0_BASED = {}
    # (关键词配置代码保持不变...)
    for label_text, config in RAW_KEYWORD_CONFIG_TEMPLATE.items():
        if label_text in label_text_to_0_based_id:
            target_0_based_id_for_keywords = label_text_to_0_based_id[label_text]
            KEYWORD_CONFIG_0_BASED[target_0_based_id_for_keywords] = config
            print(f"Keyword feature configured for 0-based ID {target_0_based_id_for_keywords} (Label: '{label_text}').")
        else:
            print(f"Warning: Label '{label_text}' from RAW_KEYWORD_CONFIG_TEMPLATE not found in class.txt. Keyword feature for this label will not be applied.")

    KEYWORD_FEATURE_DIM_AT_INIT = num_labels 
    
    # 3. 加载数据
    texts, labels, all_original_texts, keyword_features_list = load_data_from_txt(
        TRAIN_DATA_DIR, original_id_from_file_to_label_text, label_text_to_0_based_id, 
        KEYWORD_CONFIG_0_BASED, num_labels
    )

    if not texts:
        print("Error: No training data loaded. Exiting.")
        exit()

    # 4. 加载Tokenizer (模型初始化移到后面)
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)

    # 5. 数据预处理和划分
    input_ids = []
    attention_masks = []

    for text in texts:
        # (Tokenization 代码保持不变...)
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = MAX_LEN,
                            padding = 'max_length',
                            truncation = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        current_input_ids = encoded_dict['input_ids']
        current_attention_mask = encoded_dict['attention_mask']
        # (处理长度不匹配的代码保持不变...)
        if current_input_ids.size(1) != MAX_LEN:
             if current_input_ids.size(1) > MAX_LEN:
                 current_input_ids = current_input_ids[:, :MAX_LEN]
                 current_attention_mask = current_attention_mask[:, :MAX_LEN]
             else:
                 padding_needed = MAX_LEN - current_input_ids.size(1)
                 current_input_ids = torch.cat([current_input_ids, torch.full((1, padding_needed), tokenizer.pad_token_id, dtype=torch.long)], dim=1)
                 current_attention_mask = torch.cat([current_attention_mask, torch.full((1, padding_needed), 0, dtype=torch.long)], dim=1)

        input_ids.append(current_input_ids)
        attention_masks.append(current_attention_mask)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    keyword_features = torch.cat(keyword_features_list, dim=0)

    # 数据划分
    temp_train_inputs, test_inputs, \
    temp_train_masks, test_masks, \
    temp_train_keyword_features, test_keyword_features, \
    temp_train_labels, test_labels, \
    temp_train_original_texts, test_original_texts = train_test_split(
        input_ids, attention_masks, keyword_features, labels, np.array(all_original_texts), 
        random_state=RANDOM_SEED,
        test_size=TEST_SPLIT_RATIO,
        stratify=labels
    )
    
    train_inputs, val_inputs, \
    train_masks, val_masks, \
    train_keyword_features, val_keyword_features, \
    train_labels, val_labels, \
    train_original_texts, val_original_texts = train_test_split(
        temp_train_inputs, temp_train_masks, temp_train_keyword_features, temp_train_labels, temp_train_original_texts,
        random_state=RANDOM_SEED,
        test_size=VAL_SPLIT_RATIO,
        stratify=temp_train_labels
    )

    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")
    print(f"Test samples: {len(test_inputs)}")

    train_dataset = TensorDataset(train_inputs, train_masks, train_keyword_features, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_keyword_features, val_labels)
    test_dataset = TensorDataset(test_inputs, test_masks, test_keyword_features, test_labels)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

    # 6. 设置设备 (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==================== 新增：计算训练集类别权重 ====================
    print("\n[Class Balancing] Calculating class weights based on training data distribution...")
    # 将 tensor 转换为 numpy 数组用于 sklearn 计算
    train_labels_np = train_labels.cpu().numpy()
    
    # 获取训练集中实际存在的所有唯一类别标签
    classes_present = np.unique(train_labels_np)

    # 使用 'balanced' 模式计算权重
    # 公式大致为: n_samples / (n_classes * np.bincount(y))
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=classes_present,
        y=train_labels_np
    )

    # 创建一个长度为 num_labels 的完整的权重数组，默认值为1.0
    # 确保权重数组的长度与类别总数一致，即使某些类别在训练集中碰巧没有出现（虽然 stratify 应该避免这种情况）
    full_class_weights_np = np.ones(num_labels, dtype=np.float32)
    for cls_idx, weight in zip(classes_present, class_weights_np):
        full_class_weights_np[cls_idx] = weight

    # 转换为 PyTorch Tensor 并移动到相应的设备 (GPU/CPU)
    class_weights_tensor = torch.tensor(full_class_weights_np, dtype=torch.float).to(device)
    
    print(f"Class counts in training set: {np.bincount(train_labels_np, minlength=num_labels)}")
    print(f"Calculated balanced class weights (on {device}):\n{class_weights_tensor}")
    # =================================================================

    # 7. (移动到这里) 初始化模型并传入权重
    model_saved_files_exist = True
    if not os.path.exists(SAVE_MODEL_DIR):
        model_saved_files_exist = False
    else:
        if not os.path.exists(os.path.join(SAVE_MODEL_DIR, "pytorch_model.bin")):
            model_saved_files_exist = False
    
    # 注意：tokenizer 已经在上面加载了

    if model_saved_files_exist:
        print(f"Loading custom model from '{SAVE_MODEL_DIR}' for continued training/evaluation.")
        bert_base_model = BertModel.from_pretrained(LOCAL_MODEL_DIR)
        # ==================== 修改模型初始化，传入权重 ====================
        model = BertForSequenceClassificationWithKeywords(
            bert_base_model, num_labels, KEYWORD_FEATURE_DIM_AT_INIT,
            class_weights=class_weights_tensor  # 传入计算好的权重张量
        )
        # =================================================================
        model.load_state_dict(torch.load(os.path.join(SAVE_MODEL_DIR, "pytorch_model.bin")))
        # 如果加载了旧模型，确保权重在正确的设备上 (虽然上面已经to(device)，这里是双重保险)
        model.class_weights = model.class_weights.to(device)
        print(f"Custom model loaded from '{SAVE_MODEL_DIR}'.")
    else:
        print(f"No previous best model found in '{SAVE_MODEL_DIR}'. Loading initial BERT encoder from '{LOCAL_MODEL_DIR}' and initializing custom classifier.")
        bert_base_model = BertModel.from_pretrained(LOCAL_MODEL_DIR)
        # ==================== 修改模型初始化，传入权重 ====================
        model = BertForSequenceClassificationWithKeywords(
            bert_base_model, num_labels, KEYWORD_FEATURE_DIM_AT_INIT,
            class_weights=class_weights_tensor # 传入计算好的权重张量
        )
        # =================================================================
        print("BERT encoder and custom classification head loaded successfully with class weights.")


    # 8. 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 9. 训练和评估模型
    train_and_evaluate_model(model, train_dataloader, val_dataloader, test_dataloader,
                             optimizer, scheduler, device, EPOCHS,
                             id_to_label_map_for_report, num_labels,
                             test_original_texts.tolist(), tokenizer, KEYWORD_CONFIG_0_BASED)
    
    print("\nModel training and evaluation finished. Best model saved to directory.")
    print("Classification report and Confusion Matrix also generated.")
    print("Test set classification results saved to CSV.")
