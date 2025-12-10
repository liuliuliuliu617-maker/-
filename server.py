from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
from 预测 import load_model, classify_items, DEFAULT_SECTOR
from db import (
    get_user_by_account_and_password,
    create_user,
    get_user_by_account,
    insert_video,
    insert_comments_bulk,
    get_comments_by_bv,
)
import json
import subprocess
import tempfile

app = Flask(__name__, static_folder='front', static_url_path='/front')
# 在测试开发时打开 CORS，方便跨域请求；生产请根据需要配置
CORS(app)


# ======================== 大模型预加载与缓存 ========================

_model_cache = {}
_model_lock = threading.Lock()


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

    # 先记录视频信息（用户输入阶段就保存），避免后续爬取失败导致丢失
    # BV 暂未知，先用 url 作为兜底标识；成功爬取后会用真实 BV 更新
    insert_video(url, sector, user_id)

    # 1. 运行 spider.py -> 输出原始评论 json
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, 'raw.json')

        # spider.py 支持: python spider.py --url <url> --out <raw.json>
        try:
            subprocess.run([
                'python', 'spider.py', '--url', url, '--out', raw_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({ 'success': False, 'message': f'spider failed: {e}' }), 500

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

        # 转成与原先 预测.py 服务模式相同的结构
        comments = []
        for item in classified_items:
            comments.append({
                'original_comment_data': item,
                'predicted_label_id': item.get('predicted_id'),
                'predicted_label_text': item.get('predicted_label'),
            })

        # 3. 从评论中提取 BV 并保存 video
        bv = None
        for c in comments:
            oc = c.get('original_comment_data') or {}
            bv = oc.get('bv') or oc.get('BV') or bv
            if bv:
                break

        # 若无法从评论中获取 BV，可退化从 URL 解析（此处简化为原样 URL）
        if not bv:
            bv = url

        # 保存或更新视频记录（之前以 URL 写入，这里用真实 BV 覆盖 sector/user_id）
        insert_video(bv, sector, user_id)

        # 4. 保存评论（映射成 tag/context/BV）
        inserted = insert_comments_bulk(comments)

        # 从数据库按 BV 拉取全部评论作为页面预览
        preview_rows = get_comments_by_bv(bv, limit=None)
        preview = []
        for r in preview_rows:
            preview.append({'BV': r.get('BV'), 'text': r.get('context'), 'tag': r.get('tag')})

    return jsonify({ 'success': True, 'bv': bv, 'inserted': inserted, 'preview': preview, 'sector': sector })


if __name__ == '__main__':
    # 开发模式，便于调试；此时不在启动阶段预加载大模型，避免潜在崩溃导致 Flask 无法启动
    # 大模型会在第一次调用 /api/process 时，通过 get_model_for_sector 延迟加载并缓存在进程内
    app.run(debug=True)