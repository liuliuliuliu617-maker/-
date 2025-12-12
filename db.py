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
from typing import Optional, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import IntegrityError
import os
from typing import Optional, Dict, List
import pymysql
from pymysql.cursors import DictCursor
from pymysql import IntegrityError


def get_connection():
    conn = psycopg2.connect(
        database="back",  # ← 你的库名
        user="joe",
        password="!Aa20050715",
        host="123.249.109.214",
        port="26000",
        sslmode="disable",
        cursor_factory=RealDictCursor,  # ★ 关键：使用 RealDictCursor
    )
    conn.autocommit = True
    return conn

def _ensure_user_vcnt_column(conn) -> None:
    """确保 user 表中存在 v_cnt 字段"""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'user' AND column_name = 'v_cnt'")
            row = cur.fetchone()
            if not row:
                cur.execute("ALTER TABLE user ADD COLUMN v_cnt INT NOT NULL DEFAULT 0")
    except Exception as e:
        print("db._ensure_user_vcnt_column warn:", e)

def get_user_by_account_and_password(account: str, password: str) -> Optional[Dict]:
    if account is None or password is None:
        return None

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = 'SELECT id, name, account FROM "user" WHERE account = %s AND password = %s LIMIT 1'
            cur.execute(sql, (account, password))
            row = cur.fetchone()
            print("Query Result:", row)  # 打印查询结果，确认是否返回了 id
            return dict(row) if row else None
    except Exception as e:
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
    if not name or not account or not password:
        return None

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = """
            INSERT INTO "user"(name, account, password) 
            VALUES (%s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (name, account, password))
            row = cur.fetchone()
            return row['id'] if row else None
    except IntegrityError as e:
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
                    # 兼容仅有文本标签的情况，映射为与模型一致的 1~5 编码
                    text_label = (c.get('predicted_label_text') or c.get('label') or '').strip()
                    mapping = {
                        '正常': 1,
                        '争论': 2,
                        '广告': 3,
                        '@某人': 4,
                        '无意义': 5,
                    }
                    tag = mapping.get(text_label)
                # 默认无标签归为 1（正常），避免整型列插入失败
                if tag is None:
                    tag = 1
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


def get_user_video_count(user_id: int) -> int:
    """返回指定用户已成功爬取的视频数量，并同步更新 user.v_cnt。

    仅统计 video 表中属于该用户且 BV 看起来是正式 B 站 BV 号的记录，
    避免把早期以 URL 作为 BV 的占位记录计入统计。
    """
    if user_id is None:
        return 0

    conn = None
    count = 0
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = (
                "SELECT COUNT(*) AS c FROM video "
                "WHERE user_id = %s AND BV LIKE 'BV%%%%%%%%%%'"
            )
            cur.execute(sql, (user_id,))
            row = cur.fetchone() or {}
            try:
                count = int(row.get('c') or 0)
            except Exception:
                count = 0

        # 同步写回到 user.v_cnt（若列不存在会自动尝试新增）
        try:
            _ensure_user_vcnt_column(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE `user` SET `v_cnt` = %s WHERE id = %s",
                    (count, user_id),
                )
        except Exception as e:
            print("db.get_user_video_count update v_cnt warn:", e)

        return count
    except Exception as e:
        print("db.get_user_video_count error:", e)
        return count
    finally:
        if conn:
            conn.close()


def user_has_video(user_id: int, bv: str) -> bool:
    """判断指定用户是否已爬取过某个 BV 视频。"""
    if user_id is None or not bv:
        return False

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = "SELECT 1 FROM video WHERE user_id = %s AND BV = %s LIMIT 1"
            cur.execute(sql, (user_id, bv))
            row = cur.fetchone()
            return bool(row)
    except Exception as e:
        print("db.user_has_video error:", e)
        return False
    finally:
        if conn:
            conn.close()


def get_videos_by_user(user_id: int) -> List[Dict]:
    """获取某个用户已爬取的全部视频列表。

    仅返回该用户、且 BV 为标准 BV 号的视频记录。
    每条记录至少包含: {"BV": str, "sector": str | None, "user_id": int}
    """
    if user_id is None:
        return []

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = (
                "SELECT BV, sector, user_id FROM video "
                "WHERE user_id = %s AND BV LIKE 'BV%%%%%%%%%%' "
                "ORDER BY BV DESC"
            )
            cur.execute(sql, (user_id,))
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]
    except Exception as e:
        print("db.get_videos_by_user error:", e)
        return []
    finally:
        if conn:
            conn.close()


def delete_video_and_comments(user_id: int, bv: str) -> bool:
    """删除指定用户的某个视频及其所有评论。

    仅当该 BV 属于该用户时才会执行删除，并在成功后
    重新计算并更新 user.v_cnt。
    """
    if user_id is None or not bv:
        return False

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # 确认视频属于该用户
            cur.execute("SELECT user_id FROM video WHERE BV = %s", (bv,))
            row = cur.fetchone()
            if not row:
                return False
            owner_id = row.get('user_id')
            if owner_id != user_id:
                return False

            # 删除该视频下的所有评论
            cur.execute("DELETE FROM comments WHERE BV = %s", (bv,))
            # 删除视频记录
            cur.execute("DELETE FROM video WHERE BV = %s", (bv,))

        # 重新计算并更新 v_cnt
        try:
            get_user_video_count(user_id)
        except Exception as e:
            print("db.delete_video_and_comments update v_cnt warn:", e)

        return True
    except Exception as e:
        print("db.delete_video_and_comments error:", e)
        return False
    finally:
        if conn:
            conn.close()


def _ensure_keyword_table_exists(conn):
    """确保 keyword 表存在（openGauss / PostgreSQL 写法）"""
    with conn.cursor() as cur:
        sql = """
        CREATE TABLE IF NOT EXISTS keyword (
            id SERIAL PRIMARY KEY,
            word VARCHAR(255) NOT NULL,
            tag INT NOT NULL,
            "user" INT NOT NULL,
            sector VARCHAR(255) NOT NULL,
            UNIQUE (word, "user", sector)
        );
        """
        cur.execute(sql)


def insert_keyword(word: str, tag: int, user_id: int, sector: str) -> bool:
    if not word or user_id is None or not sector:
        return False

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # 1. 尝试更新 (注意列名 "user" 必须加双引号)
            sql_update = """
            UPDATE keyword 
            SET tag = %s 
            WHERE word = %s AND "user" = %s AND sector = %s
            """
            cur.execute(sql_update, (tag, word, user_id, sector))

            # 2. 如果影响行数为0，说明不存在，执行插入
            if cur.rowcount == 0:
                try:
                    sql_insert = """
                    INSERT INTO keyword(word, tag, "user", sector)
                    VALUES (%s, %s, %s, %s)
                    """
                    cur.execute(sql_insert, (word, tag, user_id, sector))
                except IntegrityError:
                    # 并发情况下可能已被别人插入，忽略错误
                    pass
            return True
    except Exception as e:
        print("db.insert_keyword error:", e)
        return False
    finally:
        if conn:
            conn.close()


def get_keywords_by_user_and_sector(user_id: int, sector: str):
    if user_id is None or not sector:
        return []

    conn = None
    try:
        conn = get_connection()
        _ensure_keyword_table_exists(conn)
        with conn.cursor() as cur:
            sql = """
            SELECT word, tag, sector 
            FROM keyword 
            WHERE "user" = %s AND sector = %s
            """
            cur.execute(sql, (user_id, sector))
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]
    except Exception as e:
        print("db.get_keywords_by_user_and_sector error:", e)
        return []
    finally:
        if conn:
            conn.close()



def delete_keyword(word: str, user_id: int, sector: str) -> bool:
    if not word or user_id is None or not sector:
        return False

    conn = None
    try:
        conn = get_connection()
        _ensure_keyword_table_exists(conn)
        with conn.cursor() as cur:
            sql = """
            DELETE FROM keyword
            WHERE word = %s AND "user" = %s AND sector = %s
            """
            cur.execute(sql, (word, user_id, sector))
            return True
    except Exception as e:
        print("db.delete_keyword error:", e)
        return False
    finally:
        if conn:
            conn.close()
