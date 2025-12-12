import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./UserCenter.css";

// 和 Home 页类似的一个分类归一化函数
function normalizeCategory(tag) {
  // 先打印一下原始 tag，调试用：上线前你可以删掉
  console.log("raw tag =>", tag, typeof tag);

  // 1）tag 完全没填：当成“正常”
  if (tag === null || tag === undefined || tag === "") {
    return "正常";
  }

  // 2）先按“中文字符串”处理一遍
  const str = String(tag);

  if (str.includes("争论") || str.includes("吵架")) return "争论";
  if (str.includes("广告")) return "广告";
  if (str.includes("无意义") || str.includes("水") || str.includes("灌水"))
    return "无意义";
  if (str.includes("@")) return "@某人";
  if (str.includes("正常")) return "正常";

  // 3）再按“数字编码”处理（支持 1 / "1"）
  const n = Number(tag);
  if (!Number.isNaN(n)) {
    switch (n) {
      case 1:
        return "正常";
      case 2:
        return "争论";
      case 3:
        return "广告";
      case 4:
        return "@某人";
      case 5:
        return "无意义";
      default:
        // 未知数字，保守给正常
        return "正常";
    }
  }

  // 4）既不是常见中文，也不是我们定义的数字，就原样返回，方便你发现问题
  return str;
}

export default function UserCenter() {
  const navigate = useNavigate();

  const [userId, setUserId] = useState(null);
  const [videos, setVideos] = useState([]);
  const [count, setCount] = useState(0);
  const [limit, setLimit] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // 评论相关 state
  const [activeComments, setActiveComments] = useState({ bv: null, list: [] });
  const [commentLoadingBV, setCommentLoadingBV] = useState("");
  const [commentsError, setCommentsError] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem("user");
    if (!stored) {
      setError("尚未登录，无法查看浏览历史。");
      setLoading(false);
      return;
    }

    let uid = 1;
    try {
      const user = JSON.parse(stored);
      if (user && user.id) {
        uid = user.id;
      }
    } catch {
      // 解析失败就用默认 ID
    }
    setUserId(uid);

    // ⭐ 一定要访问后端 5000 端口，而不是相对路径
    fetch(`http://127.0.0.1:5000/api/user/videos?user_id=${uid}`)
      .then(async (res) => {
        if (!res.ok) {
          const text = await res.text();
          console.error("user videos error body:", text);
          throw new Error("HTTP " + res.status);
        }
        return res.json();
      })
      .then((data) => {
        console.log("user videos resp:", data);
        if (!data.success) {
          setError(data.message || "获取浏览历史失败");
        } else {
          setVideos(data.data || []);
          setCount(data.count || 0);
          setLimit(data.limit || 0);
        }
      })
      .catch((err) => {
        console.error("user videos fetch error:", err);
        setError("请求失败，请稍后重试");
      })
      .finally(() => setLoading(false));
  }, []);

  const handleBackHome = () => {
    navigate("/home");
  };

  // 点击「查看已爬取的评论」
  const handleToggleComments = (bv) => {
    // 如果已经展开，再点一次就是收起
    if (activeComments.bv === bv) {
      setActiveComments({ bv: null, list: [] });
      setCommentsError("");
      return;
    }

    if (!userId) {
      setCommentsError("尚未登录，无法查看评论。");
      return;
    }

    setCommentLoadingBV(bv);
    setCommentsError("");

    fetch(
      `http://127.0.0.1:5000/api/user/video_comments?user_id=${userId}&bv=${encodeURIComponent(
        bv
      )}`
    )
      .then(async (res) => {
        const bodyText = await res.text();
        let data;
        try {
          data = JSON.parse(bodyText);
        } catch {
          console.error("user_video_comments_api invalid json:", bodyText);
          throw new Error("解析返回数据失败");
        }

        if (!res.ok || !data.success) {
          const msg =
            data && data.message
              ? data.message
              : `请求失败（HTTP ${res.status}）`;
          throw new Error(msg);
        }

        return data;
      })
      .then((data) => {
        const list = (data.comments || []).map((c) => ({
          ...c,
          category: normalizeCategory(c.tag),
        }));
        setActiveComments({ bv, list });
      })
      .catch((err) => {
        console.error("fetch video_comments error:", err);
        setCommentsError(err.message || "获取评论失败");
        setActiveComments({ bv, list: [] });
      })
      .finally(() => {
        setCommentLoadingBV("");
      });
  };

  return (
    <div className="uc-page">
      <div className="uc-header">
        <div>
          <h1 className="uc-title">用户中心</h1>
          <p className="uc-subtitle">查看你已经爬取过的视频记录</p>
        </div>
        <button className="uc-back-btn" onClick={handleBackHome}>
          返回首页
        </button>
      </div>

      {loading && <div className="uc-status">加载中...</div>}

      {error && !loading && (
        <div className="uc-status uc-error">{error}</div>
      )}

      {!loading && !error && (
        <>
          <div className="uc-summary">
            已爬取 <span className="uc-summary-number">{count}</span> /{" "}
            <span className="uc-summary-number">{limit}</span> 个视频
          </div>

          {videos.length === 0 ? (
            <div className="uc-empty">
              还没有任何浏览历史，去爬取一个视频试试吧～
            </div>
          ) : (
            <div className="uc-video-list">
              {videos.map((v, idx) => {
                const bv = v.BV || v.bv || "未知BV";
                const sector = v.sector || "默认板块";
                const url = v.url || v.link || "";
                const createdAt =
                  v.created_at || v.createdAt || v.timestamp || "";

                const isActive = activeComments.bv === bv;

                return (
                  <div className="uc-video-card" key={idx}>
                    <div className="uc-video-main">
                      <div className="uc-video-bv">BV：{bv}</div>
                      <div className="uc-video-meta">
                        <span className="uc-video-sector">
                          板块：{sector || "未标注"}
                        </span>
                        {createdAt && (
                          <span className="uc-video-time">
                            爬取时间：{createdAt}
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="uc-video-actions">
                      {url && (
                        <a
                          className="uc-video-link"
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          在 B 站中打开
                        </a>
                      )}

                      <button
                        className="uc-comment-btn"
                        onClick={() => handleToggleComments(bv)}
                        disabled={commentLoadingBV === bv}
                      >
                        {commentLoadingBV === bv
                          ? "评论加载中..."
                          : isActive
                          ? "收起评论"
                          : "查看已爬取的评论"}
                      </button>
                    </div>

                    {isActive && (
                      <div className="uc-comments">
                        {commentsError && (
                          <div className="uc-status uc-error">
                            {commentsError}
                          </div>
                        )}

                        {!commentsError &&
                          (activeComments.list.length === 0 ? (
                            <div className="uc-comments-empty">
                              这个视频暂时没有评论记录。
                            </div>
                          ) : (
                            activeComments.list.map((c, i2) => (
                              <div
                                className="uc-comment-item"
                                key={`${bv}-${i2}`}
                              >
                                <span className="uc-comment-tag">
                                  {c.category}
                                </span>
                                <span className="uc-comment-text">
                                  {c.text}
                                </span>
                              </div>
                            ))
                          ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
