import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# =================配置区域=================
# 输入文件路径
JSON_FILE_PATH = '航空母舰_最终分类结果\\classified_10278672846.json'

# 本地Qwen模型路径 (请修改为你截图中的实际文件夹路径)
# 例如: "E:/Models/Qwen2.5-3B-Instruct-..."
MODEL_PATH = "Qwen2.5-3B-Instruct-Local"

# 中文字体路径 (必须设置，否则云图和图表中文会显示乱码)
# Windows常见路径: "C:/Windows/Fonts/msyh.ttf" (微软雅黑)
# Linux/Mac请换成系统有的字体
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
# ==========================================

def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_pie_chart(df):
    """绘制分类占比饼图"""
    plt.figure(figsize=(10, 6))
    
    # 解决Matplotlib中文乱码
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=FONT_PATH)
    plt.rcParams['font.sans-serif'] = [font.get_name()] 
    plt.rcParams['axes.unicode_minus'] = False

    label_counts = df['predicted_label'].value_counts()
    
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
            startangle=140, textprops={'fontproperties': font})
    plt.title('评论分类分布图', fontproperties=font)
    plt.axis('equal') 
    plt.show()

def plot_word_clouds(df):
    """为每个类别生成词云"""
    categories = df['predicted_label'].unique()
    
    # 停用词简单的列表，可根据需要扩展
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '去', '能', '着', '看', '自', '之', '后'}

    for category in categories:
        subset = df[df['predicted_label'] == category]
        text_content = " ".join(subset['text'].tolist())
        
        # 结巴分词
        cut_text = " ".join([word for word in jieba.cut(text_content) if word not in stopwords and len(word) > 1])
        
        if not cut_text.strip():
            print(f"类别 [{category}] 没有足够的文本生成词云。")
            continue

        # 生成词云
        wc = WordCloud(
            font_path=FONT_PATH,
            background_color='white',
            width=800,
            height=600,
            max_words=100
        ).generate(cut_text)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'类别: {category} - 词云图', fontproperties=FontProperties(fname=FONT_PATH, size=15))
        plt.show()

def qwen_summarize(df, model_path):
    """调用本地Qwen模型进行总结"""
    print("\n正在加载本地Qwen模型，请稍候...")
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16, # 显存不够可改为 "auto" 或不设置
            trust_remote_code=True
        )
    except Exception as e:
        print(f"模型加载失败，请检查路径是否正确: {e}")
        return

    # 准备所有评论文本
    all_comments = "\n".join([f"- {t}" for t in df['text'].tolist()])
    
    # 构建提示词 (Prompt)
    prompt = f"""
    以下是从网络收集的一组评论数据：
    
    {all_comments}
    
    请根据上述评论内容，完成以下任务：
    1. 总结讨论的主要核心话题。
    2. 根据评论分类结果(predicted_label)分布及内容分析评论倾向。
    3. 根据分类结果，分析哪些话题更受网民关注，哪些话题更容易引发争议。
    4. 给出对未来趋势的简要预测和建议。

    请输出一份简练的分析报告。
    """
    
    messages = [
        {"role": "system", "content": "你是一个专业的数据舆情分析师。"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("正在生成智能总结...")
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("-" * 30)
    print("【Qwen智能总结报告】")
    print("-" * 30)
    print(response)

# =================主程序=================
if __name__ == "__main__":
    # 1. 检查数据文件是否存在，如果不存在则创建一个临时的用于测试
    if not os.path.exists(JSON_FILE_PATH):
        print(f"未找到 {JSON_FILE_PATH}，正在创建示例文件...")
        # (这里使用你问题中提供的JSON内容)
        sample_data = [
            {"text": "最近好像经济上没太大动作，军事上给的压力？", "predicted_label": "正常"},
            {"text": "轮流来，过几天就是抛一点日债，再过几天撞海警船...", "predicted_label": "正常"},
            # ... (为了节省空间，假设这里是你提供的完整JSON)
            {"text": "主要是日本没啥好制裁的，不太依靠日本", "predicted_label": "正常"}
        ]
        # 注意：为了让代码能跑，如果你本地没有文件，请确保你已经把你的JSON保存为文件
        # 或者使用下面这行代码把你的数据写入文件：
        # with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f: json.dump(your_data_list, f)
        print("请确保目录下存在json文件。")
    
    if os.path.exists(JSON_FILE_PATH):
        # 加载数据
        df = load_data(JSON_FILE_PATH)
        print(f"成功加载 {len(df)} 条评论数据。")
        
        # 2. 绘制饼图
        print("生成分类饼图...")
        plot_pie_chart(df)
        
        # 3. 绘制词云
        print("生成词云图...")
        from matplotlib.font_manager import FontProperties # 再次导入以防作用域问题
        plot_word_clouds(df)
        
        # 4. Qwen模型总结
        # 检查模型路径是否存在
        if os.path.exists(MODEL_PATH):
            qwen_summarize(df, MODEL_PATH)
        else:
            print(f"警告：找不到模型路径 '{MODEL_PATH}'，跳过智能总结步骤。请修改代码中的 MODEL_PATH 变量。")
    else:
        print("错误：无法找到数据文件。")
