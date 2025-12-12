import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
from collections import Counter
import re
import jieba
from 预测 import load_model, classify_items, DEFAULT_SECTOR, ID_MAP
from db import (
    get_user_by_account_and_password,
    create_user,
    get_user_by_account,
    insert_video,
    insert_comments_bulk,
    get_comments_by_bv,
    insert_keyword,
    get_keywords_by_user_and_sector,
    delete_keyword,
    get_user_video_count,
    user_has_video,
    get_videos_by_user,
    delete_video_and_comments,
)
import json
import subprocess
import tempfile
from io import BytesIO
import base64
from wordcloud import WordCloud


app = Flask(__name__, static_folder='front', static_url_path='/front')
# 在测试开发时打开 CORS，方便跨域请求；生产请根据需要配置
CORS(app)


# ======================== 大模型预加载与缓存 ========================

_model_cache = {}
_model_lock = threading.Lock()

# 每个用户允许爬取的视频上限
MAX_VIDEO_PER_USER = 5


def get_model_for_sector(sector: str | None):
    """按板块获取已加载的大模型，没有则加载并缓存。

    为了避免每次请求都重复加载权重，这里做一个简单的进程内缓存。
    """
    # 未指定板块时使用默认板块
    key = (sector or DEFAULT_SECTOR).strip() or DEFAULT_SECTOR

    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]

        # load_model 内部已处理未知板块回退为 DEFAULT_SECTOR
        model, tokenizer = load_model(key)
        _model_cache[key] = (model, tokenizer)
        return model, tokenizer


def parse_bv_from_url(url: str | None) -> str | None:
    """从 B 站视频 URL 中提取 BV 号，未匹配到则返回 None。"""
    if not url:
        return None
    try:
        m = re.search(r"BV[0-9A-Za-z]{10}", url)
        return m.group(0) if m else None
    except Exception:
        return None


def build_wordcloud_and_summary(preview):
    """根据预览评论构建词云数据和简单文字总结。

    - 词云数据: [{"text": 词语, "weight": 频次}, ...]
    - 总结示例: "本次共爬取 X 条评论，其中 正常 Y 条，争论 Z 条 ... 高频词包括：..."
    """
    if not preview:
        return [], "暂无评论可供总结。"

    texts = []
    tags = []
    for item in preview:
        t = (item.get('text') or '').strip()
        if t:
            texts.append(t)
        try:
            tags.append(int(item.get('tag') or 0))
        except Exception:
            pass

    if not texts:
        return [], "暂无评论可供总结。"

    all_text = "\n".join(texts)
    # 仅保留中英文和数字
    cleaned = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", " ", all_text)

    # jieba 分词
    words = [w.strip() for w in jieba.lcut(cleaned) if w.strip()]

    # 简单停用词表
    stopwords = {
        '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
        '到', '说', '去', '能', '着', '看', '自', '之', '后', '你', '他', '她', '它', '啊', '呢', '吧',
        '呀', '吗', '还', '没', '这个', '那个', '我们', '你们', '他们', '不是', '没有'
    }

    filtered = [w for w in words if w not in stopwords and len(w) > 1]
    counter = Counter(filtered)
    most_common = counter.most_common(50)
    wordcloud_data = [
        {"text": word, "weight": int(cnt)}
        for word, cnt in most_common
        if word
    ]

    total = len(preview)
    tag_counter = Counter(tags)
    label_map = {
        1: "正常",
        2: "争论",
        3: "广告",
        4: "@某人",
        5: "无意义",
    }

    parts = []
    for tag_id in range(1, 6):
        cnt = tag_counter.get(tag_id, 0)
        if cnt:
            parts.append(f"{label_map.get(tag_id, tag_id)} {cnt} 条")

    summary = f"本次共爬取 {total} 条评论。" if total else "暂无评论。"
    if parts:
        summary += " 其中 " + "，".join(parts) + "。"

    if most_common:
        top_words = [w for w, _ in most_common[:10]]
        summary += " 高频词包括：" + "、".join(top_words) + "。"

    return wordcloud_data, summary

def build_wordcloud_image_from_data(wordcloud_data):
    """
    根据 build_wordcloud_and_summary 生成的 wordcloud_data
    构造整体词云图片，并返回 base64 字符串（不带 data:image 前缀）。
    """
    if not wordcloud_data:
        return None

    # 按照 可视化.py 里的逻辑，用词频重复拼接成文本
    words = []
    for item in wordcloud_data:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        try:
            cnt = int(item.get("weight") or 1)
        except Exception:
            cnt = 1
        cnt = max(cnt // 2, 1)
        words.append((text, cnt))

    if not words:
        return None

    text_for_wc = " ".join(
        [((word + " ") * cnt) for word, cnt in words]
    )

    if not text_for_wc.strip():
        return None

    # 和 可视化.py 保持一致的 font_path（按你自己机器调整）
    FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

    wc = WordCloud(
        font_path=FONT_PATH,
        background_color="white",
        width=1000,
        height=600,
        max_words=150,
    ).generate(text_for_wc)

    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    img_bytes = buf.getvalue()
    buf.close()

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

@app.route('/api/login', methods=['POST'])
def login():
    # 支持 JSON 或表单提交
    data = request.get_json(silent=True) or request.form
    account = data.get('account')
    password = data.get('password')

    if not account or not password:
        return jsonify({'success': False, 'message': 'account and password required'}), 400

    user = get_user_by_account_and_password(account, password)
    if user:
        # 不返回 password
        return jsonify({'success': True, 'user': {'id': user['id'], 'name': user['name'], 'account': user['account']}})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or request.form
    name = (data.get('name') or '').strip()
    account = (data.get('account') or '').strip()
    password = (data.get('password') or '').strip()

    if not name or not account or not password:
        return jsonify({'success': False, 'message': 'name, account, password required'}), 400

    # 账号重复校验
    if get_user_by_account(account):
        return jsonify({'success': False, 'message': 'Account already exists'}), 409

    user_id = create_user(name, account, password)
    if user_id:
        return jsonify({'success': True, 'user': {'id': user_id, 'name': name, 'account': account}}), 201
    else:
        return jsonify({'success': False, 'message': 'Register failed'}), 500


# 提供静态文件访问： /front/<path> 和根目录直接返回登录页
@app.route('/front/<path:filename>')
def front_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/')
def index():
    # 默认打开你的登录页
    return send_from_directory(app.static_folder, '登录页.html')

@app.route('/register')
def register_page():
    return send_from_directory(app.static_folder, 'register.html')


@app.route('/process')
def process_page():
    return send_from_directory(app.static_folder, 'process.html')


@app.route('/user_videos')
def user_videos_page():
    """简单的用户视频管理页面，用于测试视频数量限制与删除功能。"""
    return send_from_directory(app.static_folder, 'user_videos.html')
@app.route('/api/comments/by_bv', methods=['GET'])
def comments_by_bv_api():
    """根据 BV 获取最新评论（用于前端刷新首页评论显示）."""
    bv = (request.args.get('bv') or '').strip()
    if not bv:
        return jsonify({'success': False, 'message': 'bv required'}), 400

    try:
        rows = get_comments_by_bv(bv, limit=None)
        preview = []
        for r in rows:
            preview.append({
                'BV': r.get('BV'),
                'text': r.get('context'),
                'tag': r.get('tag'),
            })
        return jsonify({'success': True, 'bv': bv, 'preview': preview})
    except Exception as e:
        print('comments_by_bv_api error:', e)
        return jsonify({'success': False, 'message': 'get comments failed'}), 500


@app.route('/api/keywords', methods=['GET', 'POST', 'DELETE'])
def keywords_api():
    """关键词库管理接口。

    GET: /api/keywords?user_id=1&sector=APEX
         -> 返回当前用户 + 板块下的关键词列表

    POST: JSON { word, tag, sector, user_id }
         -> 新增或更新关键词
    """
    if request.method == 'GET':
        try:
            user_id = request.args.get('user_id', default='1')
            sector = (request.args.get('sector') or '').strip()
            try:
                user_id_int = int(user_id)
            except Exception:
                user_id_int = 1

            if not sector:
                return jsonify({'success': False, 'message': 'sector required'}), 400

            rows = get_keywords_by_user_and_sector(user_id_int, sector)
            return jsonify({'success': True, 'data': rows})
        except Exception as e:
            print('keywords_api GET error:', e)
            return jsonify({'success': False, 'message': 'get keywords failed'}), 500
    elif request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        word = (data.get('word') or '').strip()
        sector = (data.get('sector') or '').strip()
        tag = data.get('tag')
        user_id = data.get('user_id')

        try:
            tag_int = int(tag)
        except Exception:
            tag_int = None
        try:
            user_id_int = int(user_id) if user_id is not None else 1
        except Exception:
            user_id_int = 1

        if not word or tag_int is None or not sector:
            return jsonify({'success': False, 'message': 'word, tag, sector required'}), 400

        try:
            ok = insert_keyword(word, tag_int, user_id_int, sector)
            if not ok:
                return jsonify({'success': False, 'message': 'insert keyword failed'}), 500
            return jsonify({'success': True})
        except Exception as e:
            print('keywords_api POST error:', e)
            return jsonify({'success': False, 'message': 'insert keyword error'}), 500
    elif request.method == 'DELETE':
        data = request.get_json(silent=True) or request.form or request.args
        word = (data.get('word') or '').strip() if data.get('word') is not None else ''
        sector = (data.get('sector') or '').strip() if data.get('sector') is not None else ''
        user_id = data.get('user_id')

        try:
            user_id_int = int(user_id) if user_id is not None else 1
        except Exception:
            user_id_int = 1

        if not word or not sector:
            return jsonify({'success': False, 'message': 'word and sector required'}), 400

        try:
            ok = delete_keyword(word, user_id_int, sector)
            if not ok:
                return jsonify({'success': False, 'message': 'delete keyword failed'}), 500
            return jsonify({'success': True})
        except Exception as e:
            print('keywords_api DELETE error:', e)
            return jsonify({'success': False, 'message': 'delete keyword error'}), 500


@app.route('/api/user/videos', methods=['GET', 'DELETE'])
def user_videos_api():
    """用户已爬取视频的查询和删除接口。

    GET  /api/user/videos?user_id=1
        -> 返回该用户已爬取视频列表及当前数量

    DELETE /api/user/videos {"user_id": 1, "bv": "BVxxxx"}
        -> 删除该用户的某个视频及其评论
    """
    if request.method == 'GET':
        user_id = request.args.get('user_id')
        account = (request.args.get('account') or '').strip()
        user_id_int = None

        # 优先使用 user_id，其次根据 account 查询
        if user_id is not None:
            try:
                user_id_int = int(user_id)
            except Exception:
                user_id_int = None
        elif account:
            user = get_user_by_account(account)
            if user:
                user_id_int = user.get('id')

        if user_id_int is None:
            return jsonify({'success': False, 'message': 'user_id or account required'}), 400

        try:
            videos = get_videos_by_user(user_id_int)
            # 顺便同步并返回当前数量
            count = get_user_video_count(user_id_int)
            return jsonify({
                'success': True,
                'data': videos,
                'count': count,
                'limit': MAX_VIDEO_PER_USER,
            })
        except Exception as e:
            print('user_videos_api GET error:', e)
            return jsonify({'success': False, 'message': 'get user videos failed'}), 500

    # DELETE
    data = request.get_json(silent=True) or request.form or request.args
    user_id = data.get('user_id')
    account = (data.get('account') or '').strip()
    bv = (data.get('bv') or '').strip()

    user_id_int = None
    if user_id is not None:
        try:
            user_id_int = int(user_id)
        except Exception:
            user_id_int = None
    elif account:
        user = get_user_by_account(account)
        if user:
            user_id_int = user.get('id')

    if user_id_int is None or not bv:
        return jsonify({'success': False, 'message': 'user_id/account and bv required'}), 400

    try:
        ok = delete_video_and_comments(user_id_int, bv)
        if not ok:
            return jsonify({'success': False, 'message': 'delete video failed'}), 400

        # 删除后返回最新数量，方便前端刷新
        count = get_user_video_count(user_id_int)
        return jsonify({
            'success': True,
            'count': count,
            'limit': MAX_VIDEO_PER_USER,
        })
    except Exception as e:
        print('user_videos_api DELETE error:', e)
        return jsonify({'success': False, 'message': 'delete video error'}), 500


@app.route('/api/user/video_comments', methods=['GET'])
def user_video_comments_api():
    """查询某个用户已爬取视频的全部评论。"""
    bv = (request.args.get('bv') or '').strip()
    user_id = request.args.get('user_id')
    account = (request.args.get('account') or '').strip()

    user_id_int = None
    if user_id is not None:
        try:
            user_id_int = int(user_id)
        except Exception:
            user_id_int = None
    elif account:
        user = get_user_by_account(account)
        if user:
            user_id_int = user.get('id')

    if user_id_int is None or not bv:
        return jsonify({'success': False, 'message': 'user_id/account and bv required'}), 400

    try:
        # 只允许查看自己已爬取的视频
        if not user_has_video(user_id_int, bv):
            return jsonify({'success': False, 'message': 'video not found for this user'}), 404

        rows = get_comments_by_bv(bv, limit=None)
        comments = [
            {
                'BV': r.get('BV'),
                'text': r.get('context'),
                'tag': r.get('tag'),
            }
            for r in rows
        ]
        return jsonify({'success': True, 'bv': bv, 'comments': comments})
    except Exception as e:
        print('user_video_comments_api error:', e)
        return jsonify({'success': False, 'message': 'get comments failed'}), 500


@app.route('/api/process', methods=['POST'])
def process_video():
    """
    接收 { url, user_id?, sector? }：
    1) 运行 spider.py 爬取评论生成临时 json
    2) 运行 预测.py 按板块模型标注评论，输出 json
    3) 保存视频信息到 video
    4) 保存评论到 comment
    返回插入条数与 BV
    """
    data = request.get_json(silent=True) or request.form
    url = (data.get('url') or '').strip()
    sector = (data.get('sector') or '').strip() or None
    user_id = data.get('user_id')
    try:
        user_id = int(user_id) if user_id is not None else None
    except Exception:
        user_id = None

    if not url:
        return jsonify({ 'success': False, 'message': 'url required' }), 400

    # 若提供了用户信息，则在爬取前检查是否超过可爬取视频上限
    if user_id is not None:
        bv_from_url = parse_bv_from_url(url)

        # 如果该 BV 对该用户是新视频，则需要检查当前已爬取数量
        is_existing_video = False
        if bv_from_url:
            try:
                is_existing_video = user_has_video(user_id, bv_from_url)
            except Exception as e:
                print('process_video check user_has_video error:', e)

        if not is_existing_video:
            try:
                current_cnt = get_user_video_count(user_id)
            except Exception as e:
                print('process_video get_user_video_count error:', e)
                current_cnt = 0

            if current_cnt >= MAX_VIDEO_PER_USER:
                return jsonify({
                    'success': False,
                    'code': 'VIDEO_LIMIT_REACHED',
                    'message': f'每个用户最多只能爬取 {MAX_VIDEO_PER_USER} 个不同的视频，请先删除已有视频后再继续。',
                }), 403

    # 若为登录用户，可先记录视频信息（用户输入阶段就保存），避免后续爬取失败导致丢失
    # BV 暂未知，先用 url 作为兜底标识；成功爬取后会用真实 BV 更新
    if user_id is not None:
        insert_video(url, sector, user_id)

    # 1. 运行 spider.py -> 输出原始评论 json
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, 'raw.json')

        try:
            # 使用当前正在运行 server.py 的解释器来调用 spider.py
            spider_path = os.path.join(os.path.dirname(__file__), 'spider.py')
            subprocess.run(
                [sys.executable, spider_path, '--url', url, '--out', raw_path],
                check=True
            )
        except subprocess.CalledProcessError as e:
            return jsonify({'success': False, 'message': f'spider failed: {e}'}), 500

        # 2. 在当前进程中调用大模型，对原始评论打标签
        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            return jsonify({ 'success': False, 'message': f'load raw json failed: {e}' }), 500

        if not isinstance(raw_data, list):
            return jsonify({ 'success': False, 'message': 'raw json format error: expect list' }), 500

        try:
            model, tokenizer = get_model_for_sector(sector)
            classified_items = classify_items(raw_data, model, tokenizer)
        except Exception as e:
            return jsonify({ 'success': False, 'message': f'predict failed (in-process): {e}' }), 500

        # ====== 根据关键词库对模型分类结果进行覆盖 ======
        # 若当前登录用户在该板块配置了关键词，包含该词的评论会被强制归到对应标签
        try:
            if sector and user_id is not None:
                keyword_rows = get_keywords_by_user_and_sector(user_id, sector)
                if keyword_rows:
                    id_to_label = ID_MAP
                    for item in classified_items:
                        # 尽量从多种字段中取到评论文本
                        oc = item.get('original_comment_data') or {}
                        text = (
                            item.get('text')
                            or item.get('context')
                            or oc.get('text')
                            or oc.get('context')
                            or ''
                        )
                        if not text:
                            continue
                        for kw in keyword_rows:
                            w = kw.get('word') or ''
                            t = kw.get('tag')
                            if not w or t is None:
                                continue
                            if w in text:
                                item['predicted_id'] = t
                                # 将 ID 映射回标签文本，若映射失败则保留原值
                                try:
                                    item['predicted_label'] = id_to_label.get(
                                        t, item.get('predicted_label')
                                    )
                                except Exception:
                                    pass
                                break
        except Exception as e:
            # 关键词库逻辑不应影响主流程，出错仅记录日志
            print('apply keyword override error:', e)

        # 转成与原先 预测.py 服务模式相同的结构
        comments = []
        for item in classified_items:
            comments.append({
                'original_comment_data': item,
                'predicted_label_id': item.get('predicted_id'),
                'predicted_label_text': item.get('predicted_label'),
            })

        # 3. 从评论中提取 BV
        bv = None
        for c in comments:
            oc = c.get('original_comment_data') or {}
            bv = oc.get('bv') or oc.get('BV') or bv
            if bv:
                break

        # 若无法从评论中获取 BV，可退化从 URL 解析（此处简化为原样 URL）
        if not bv:
            bv = url

        # 登录用户：保存或更新视频记录和评论到数据库
        if user_id is not None:
            # 保存或更新视频记录（之前以 URL 写入，这里用真实 BV 覆盖 sector/user_id）
            insert_video(bv, sector, user_id)

            # 4. 保存评论（映射成 tag/context/BV）
            inserted = insert_comments_bulk(comments)

            # 从数据库按 BV 拉取全部评论作为页面预览
            preview_rows = get_comments_by_bv(bv, limit=None)
            preview = []
            for r in preview_rows:
                preview.append({'BV': r.get('BV'), 'text': r.get('context'), 'tag': r.get('tag')})
        else:
            # 未登录用户：不写入数据库，仅基于本次分类结果构造预览
            inserted = 0
            preview = []
            for c in comments:
                oc = c.get('original_comment_data') or {}
                text = (
                    c.get('context')
                    or oc.get('context')
                    or oc.get('text')
                    or c.get('text')
                    or ''
                )
                tag = c.get('predicted_label_id')
                if not text or tag is None:
                    continue
                preview.append({'BV': bv, 'text': text, 'tag': tag})

        # 基于预览数据构建词云和文字总结
        try:
            wordcloud_data, summary_text = build_wordcloud_and_summary(preview)
        except Exception as e:
            print('build_wordcloud_and_summary error:', e)
            wordcloud_data, summary_text = [], ""
        try:
            wordcloud_img = build_wordcloud_image_from_data(wordcloud_data)
        except Exception as e:
            print('build_wordcloud_image_from_data error:', e)
            wordcloud_img = None

    return jsonify({
        'success': True,
        'bv': bv,
        'inserted': inserted,
        'preview': preview,
        'sector': sector,
        'wordcloud': wordcloud_data,
        'wordcloud_img': wordcloud_img,
        'summary': summary_text,
    })


if __name__ == '__main__':
    # 开发模式，便于调试；此时不在启动阶段预加载大模型，避免潜在崩溃导致 Flask 无法启动
    # 大模型会在第一次调用 /api/process 时，通过 get_model_for_sector 延迟加载并缓存在进程内
    app.run(debug=True)