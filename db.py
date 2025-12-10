"""
db.py - MySQL 版的数据访问封装（使用 PyMySQL）

说明:
- 默认连接到本地 MySQL: host=127.0.0.1, port=3306, user=root, password=root
- 默认数据库名为: testdb （请根据你的实际数据库名修改 DB_NAME 或通过环境变量设置）
- 需要安装依赖: pip install PyMySQL

函数:
- get_connection() -> 返回 PyMySQL 连接（DictCursor）
- get_user_by_account_and_password(account, password) -> 根据 account/password 查询 user 表，返回 dict 或 None
"""

import os
from typing import Optional, Dict
import pymysql
from pymysql.cursors import DictCursor
from pymysql import IntegrityError

# 可通过环境变量覆盖这些默认值
DB_HOST = os.environ.get('DB_HOST', '127.0.0.1')
DB_PORT = int(os.environ.get('DB_PORT', 3306))
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '123iop12')
DB_NAME = os.environ.get('DB_NAME', 'back')  # <-- 请改为你的数据库名

def get_connection():
    """
    返回一个新的 PyMySQL 连接，使用 DictCursor 以便返回 dict 而不是 tuple。
    使用完成后请调用 conn.close()。
    """
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8mb4',
        cursorclass=DictCursor,
        autocommit=True
    )
    return conn

def get_user_by_account_and_password(account: str, password: str) -> Optional[Dict]:
    """
    参数化查询：根据 account 和 password 返回用户（不包含 password 字段）
    返回示例: {'id': 1, 'name': 'Alice', 'account': 'alice'}
    如果未找到，返回 None。
    """
    if account is None or password is None:
        return None

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = "SELECT id, name, account FROM user WHERE account = %s AND password = %s LIMIT 1"
            cur.execute(sql, (account, password))
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as e:
        # 在开发阶段打印错误，生产环境请改为记录到日志系统
        print("db.get_user_by_account_and_password error:", e)
        return None
    finally:
        if conn:
            conn.close()

def get_user_by_account(account: str) -> Optional[Dict]:
    """
    根据账号查询用户，返回 dict 或 None。
    """
    if not account:
        return None

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = "SELECT id, name, account FROM user WHERE account = %s LIMIT 1"
            cur.execute(sql, (account,))
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as e:
        print("db.get_user_by_account error:", e)
        return None
    finally:
        if conn:
            conn.close()

def create_user(name: str, account: str, password: str) -> Optional[int]:
    """
    创建用户：插入到 user(name, account, password) 表。
    返回新插入的用户 id，失败返回 None。
    """
    if not name or not account or not password:
        return None

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = "INSERT INTO user(name, account, password) VALUES(%s, %s, %s)"
            cur.execute(sql, (name, account, password))
            return cur.lastrowid
    except IntegrityError as e:
        # 一般是唯一约束冲突
        print("db.create_user integrity error:", e)
        return None
    except Exception as e:
        print("db.create_user error:", e)
        return None
    finally:
        if conn:
            conn.close()

def insert_video(bv: str, sector: Optional[str], user_id: Optional[int]) -> bool:
    """
    插入视频记录到 video(BV, sector, user_id)。若已存在可选择忽略或更新，此处简单插入。
    返回 True/False 表示是否成功。
    """
    if not bv:
        return False
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # 若主键(BV)已存在则更新 sector/user_id，避免重复错误 1062
            sql = (
                "INSERT INTO video(BV, sector, user_id) VALUES(%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE sector=VALUES(sector), user_id=VALUES(user_id)"
            )
            cur.execute(sql, (bv, sector, user_id))
            return True
    except Exception as e:
        print("db.insert_video error:", e)
        return False
    finally:
        if conn:
            conn.close()

def insert_comments_bulk(comments: list) -> int:
    """
    批量插入评论到 comment(id, tag, context, BV)。
    期望每条为 dict，包含 keys: tag, context, BV；id 由数据库自增。
    返回成功插入条数。
    """
    if not comments:
        return 0
    conn = None
    inserted = 0
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # 表名为 comments；tag 为整型，映射 predicted_label_id
            sql = "INSERT INTO comments(tag, context, BV) VALUES(%s, %s, %s)"
            data = []
            for c in comments:
                # 适配 forecast 输出结构：
                # { original_comment_data: { bv, text, ... }, predicted_label_id, predicted_label_text }
                tag = c.get('predicted_label_id')
                oc = c.get('original_comment_data') or {}
                context = c.get('context') or oc.get('context') or oc.get('text') or c.get('text')
                bv = c.get('BV') or c.get('bv') or oc.get('BV') or oc.get('bv')
                if tag is None:
                    # 兼容文本标签，进行映射
                    text_label = (c.get('predicted_label_text') or c.get('label') or '').strip()
                    mapping = {
                        '正常': 0,
                        '争论': 1,
                        '广告': 2,
                        '@某人': 3,
                        '无意义': 4,
                    }
                    tag = mapping.get(text_label)
                # 默认无标签归为 0（正常），避免整型列插入失败
                if tag is None:
                    tag = 0
                if context and bv:
                    data.append((tag, context, bv))
            if data:
                cur.executemany(sql, data)
                inserted = len(data)
        return inserted
    except Exception as e:
        print("db.insert_comments_bulk error:", e)
        return inserted
    finally:
        if conn:
            conn.close()

def get_comments_by_bv(bv: str, limit: Optional[int] = 20) -> list:
    """
    按 BV 查询最近的评论，返回 list[dict]，包含 tag, context, BV。

    limit 为 None 时不加 LIMIT，返回该 BV 下的所有评论。
    """
    if not bv:
        return []
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            if limit is not None:
                sql = "SELECT tag, context, BV FROM comments WHERE BV = %s ORDER BY id DESC LIMIT %s"
                cur.execute(sql, (bv, limit))
            else:
                sql = "SELECT tag, context, BV FROM comments WHERE BV = %s ORDER BY id DESC"
                cur.execute(sql, (bv,))
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]
    except Exception as e:
        print("db.get_comments_by_bv error:", e)
        return []
    finally:
        if conn:
            conn.close()