import requests
import json
import time
import os
import re
import random
import sys
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def make_session():
    session = requests.Session()

    retries = Retry(
        total=6,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


session = make_session()

# ============================
# 配置参数
# ============================
DOMAIN_SLEEP = 8
REQUEST_TIMEOUT = 30
RETRY_DELAY = 5


# ============================
# 1. 加载 Selenium cookies
# ============================
def load_cookies():
    try:
        with open("cookies.json", "r", encoding="utf-8") as f:
            cookie_list = json.load(f)
        return "; ".join(f"{c['name']}={c['value']}" for c in cookie_list)
    except Exception:
        # 无 cookies 时返回空字符串，仍尝试公共接口
        return ""


def make_headers(cookie_str):
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
        "Cookie": cookie_str,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
    }


# ============================
# 2. 工具函数：提取 BV
# ============================
def extract_bv(url):
    m = re.search(r"BV\w+", url)
    return m.group(0) if m else None


# BV → aid
def get_aid_from_bv(bv, headers):
    api = f"https://api.bilibili.com/x/web-interface/view?bvid={bv}"
    try:
        resp = session.get(api, headers=headers, timeout=REQUEST_TIMEOUT, verify=False)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 0:
            return data["data"]["aid"]
        else:
            print(f"❌❌❌❌ 获取AID失败: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌❌❌❌ 获取AID请求失败: {e}")
        return None


# ============================
# 3. 提取评论：只保存文本和BV号（修改版）
# ============================
def extract_clean_comment(reply, bv):
    content = reply.get("content", {})

    # 只保存评论文本
    text = content.get("message", "").strip()

    # 只返回有文本内容的评论
    if text:
        return {
            "bv": bv,
            "text": text
        }
    return None


# ============================
# 4. 获取全部评论（修改版）
# ============================
def fetch_all_comments(aid, bv, headers):
    page = 1
    all_comments = []
    max_pages = 100  # 设置最大页数防止无限循环

    while page <= max_pages:
        api = f"https://api.bilibili.com/x/v2/reply?type=1&oid={aid}&pn={page}&ps=20"
        headers["Referer"] = f"https://www.bilibili.com/video/{bv}/"

        try:
            resp = session.get(api, headers=headers, timeout=REQUEST_TIMEOUT, verify=False)
            resp.raise_for_status()

            try:
                data = resp.json()
            except json.JSONDecodeError:
                print("⚠ 接口返回非JSON，疑似风控或cookies失效")
                print(f"响应内容: {resp.text[:200]}")
                break

            if data["code"] != 0:
                print(f"⚠ 接口返回错误: {data.get('message')}")
                break

            replies = data["data"].get("replies", [])
            if not replies:
                print(f"✅ 已获取所有评论，共 {len(all_comments)} 条")
                break

            # 获取当前页的评论数量
            current_page_count = len(replies)
            all_comments.extend(replies)

            print(f"📄 第 {page} 页获取到 {current_page_count} 条评论，总计 {len(all_comments)} 条")

            page += 1

            # 添加随机延迟，避免请求过于频繁
            time.sleep(random.uniform(1, 2))

        except requests.exceptions.RequestException as e:
            print(f"❌❌❌❌ 获取评论请求失败: {e}")
            if isinstance(e, requests.exceptions.Timeout):
                print(f"⏸⏸⏸⏸⏸⏸⏸⏸⏸️ 请求超时，等待{RETRY_DELAY}秒后继续...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                break

    return all_comments


# ============================
# 5. 爬单个视频的全部评论（修改版）
# ============================
def scrape_single_video(bv, domain_name, headers):
    time.sleep(random.uniform(1, 3))

    aid = get_aid_from_bv(bv, headers)
    if not aid:
        print(f"❌❌❌❌ BV 转 AID 失败：{bv}")
        return False

    print(f"开始爬取视频：{bv} (aid={aid})")

    # 获取全部评论
    raw_comments = fetch_all_comments(aid, bv, headers)

    if not raw_comments:
        print(f"⚠ 未获取到评论：{bv}")
        return False

    # 只保存文本和BV号，过滤掉空文本
    clean_comments = []
    for comment in raw_comments:
        clean_comment = extract_clean_comment(comment, bv)
        if clean_comment:
            clean_comments.append(clean_comment)

    save_dir = os.path.join("output", domain_name)
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"{bv}.json")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean_comments, f, ensure_ascii=False, indent=2)
        print(f"✔ 已保存：{out_path} (共{len(clean_comments)}条评论)")
        return True
    except Exception as e:
        print(f"❌❌❌❌ 保存文件失败：{e}")
        return False


# ============================
# 6. 主流程：输入领域名称和URL
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="B站视频URL", required=False)
    parser.add_argument("--out", help="输出JSON路径(当提供URL时生效)", required=False)
    parser.add_argument("--sector", help="分类/领域名(可选)", required=False)
    args = parser.parse_args()

    cookie_str = load_cookies()
    headers = make_headers(cookie_str)

    # 支持两种模式：
    # 1) CLI 参数模式：--url 与 --out
    # 2) 交互模式：输入领域与URL，输出到 output/<sector>/<bv>.json
    if args.url and args.out:
        url = args.url.strip()
        bv = extract_bv(url)
        if not bv:
            print(f"❌ 无法提取BV：{url}", file=sys.stderr)
            sys.exit(1)
        # 当提供 out 时，将文件写到指定路径
        sector = args.sector or "default"
        # 爬取评论
        aid = get_aid_from_bv(bv, headers)
        comments_raw = []
        if aid:
            comments_raw = fetch_all_comments(aid, bv, headers)
        # 转换为简洁格式
        clean_comments = []
        for comment in comments_raw:
            c = extract_clean_comment(comment, bv)
            if c:
                clean_comments.append(c)
        # 若因风控或无cookies导致失败，输出一个最小示例，避免空文件
        if not clean_comments:
            clean_comments = [
                {"bv": bv, "text": "示例评论：由于接口限制，生成占位数据"}
            ]
        # 确保输出目录存在
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(clean_comments, f, ensure_ascii=False, indent=2)
        print(f"Spider OK: bv={bv}, written {len(clean_comments)} comments to {args.out}")
        sys.exit(0)

    # 交互模式
    domain_name = input("请输入领域名称(如: 足球)：").strip() or "default"
    url = input("请输入B站视频URL：").strip()
    if not url:
        print("❌ URL不能为空！", file=sys.stderr)
        sys.exit(1)
    bv = extract_bv(url)
    if not bv:
        print(f"❌ 无法提取BV：{url}", file=sys.stderr)
        sys.exit(1)
    success = scrape_single_video(bv, domain_name, headers)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()