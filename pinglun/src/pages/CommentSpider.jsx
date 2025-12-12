import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./CommentSpider.css";

// ✅ 换成你自己的 GIF 路径
import HamsterGif from "../assets/hamster.gif";

export default function CommentSpider() {
  const [url, setUrl] = useState(""); // 视频URL
  const [sector, setSector] = useState(""); // 领域/标签（下拉选择）
  const [loading, setLoading] = useState(false); // 加载状态
  const [error, setError] = useState(""); // 错误信息
  const [preview, setPreview] = useState([]); // 评论预览
  const [progress, setProgress] = useState(0); // 进度条（0-100）
  const [showProgress, setShowProgress] = useState(false); // 是否显示进度

  const navigate = useNavigate();

  // 获取当前用户ID，若未登录，使用默认 ID（例如：1）
  let userId = 1;
  try {
    const stored = localStorage.getItem("user");
    if (stored) {
      const user = JSON.parse(stored);
      if (user && (user.id || user.user_id)) {
        userId = user.id || user.user_id;
      }
    }
  } catch (e) {
    // 解析失败就用默认 ID
  }

  // 提交表单，爬取视频评论
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!url.trim()) {
      setError("请输入视频URL");
      return;
    }

    if (!sector.trim()) {
      setError("请选择领域");
      return;
    }

    setError("");
    setLoading(true);
    setPreview([]);
    setShowProgress(true);
    setProgress(10);

    let timer;
    try {
      // 模拟进度变化（从 10% 到 90%）
      timer = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          const next = prev + Math.random() * 10;
          return next > 90 ? 90 : next;
        });
      }, 400);

      // 发送请求到后端爬取评论数据
      const resp = await fetch("http://127.0.0.1:5000/api/process", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url,
          user_id: userId,
          sector, // 领域/标签传给后端
        }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        console.error("process error body:", text);
        setError("服务端错误，状态码：" + resp.status);
        setShowProgress(false);
        setProgress(0);
        return;
      }

      const data = await resp.json();
      console.log("process resp:", data);

      // 处理错误
      if (!data.success) {
        setError(data.message || "爬取失败");
        setShowProgress(false);
        setProgress(0);
      } else {
        // 处理成功的结果
        const previewData = data.preview || [];
        setPreview(previewData);
        setProgress(100); // 进度到 100%

        // ⭐ 把后端返回的“所有内容 + 前端补充的信息”都存起来（按用户区分）
        const payload = {
          ...data, // 后端返回的所有字段
          url, // 记录本次爬取使用的原始链接
          sector: data.sector || sector, // 以后端为准，没给就用当前输入
        };

        const key = `lastVideoResult_user_${userId}`;
        localStorage.setItem(key, JSON.stringify(payload));

        // 延时 800ms 后跳转到首页
        setTimeout(() => {
          setShowProgress(false);
          navigate("/home");
        }, 800);
      }
    } catch (err) {
      console.error("process fetch error:", err);
      setError("网络错误：" + err.message);
      setShowProgress(false);
      setProgress(0);
    } finally {
      if (timer) clearInterval(timer);
      setLoading(false);
    }
  };

  // 为了安全，进度条宽度和小仓鼠位置都做个 0-100 的限制
  const safeProgress = Math.max(0, Math.min(progress, 100));

  return (
    <div className="spider-page">
      <div className="spider-card">
        <h1 className="spider-title">B站视频评论爬取</h1>
        <p className="spider-subtitle">
          输入视频链接，一键抓取并分析弹幕与评论
        </p>

        <form className="spider-form" onSubmit={handleSubmit}>
          {/* 视频 URL */}
          <label className="spider-label">视频 URL</label>
          <div className="spider-input-row">
            <input
              className="spider-input"
              placeholder="例如：https://www.bilibili.com/video/BVxxxxxxxxx"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
          </div>

          {/* 领域/标签：改成下拉选择 */}
          <label className="spider-label">领域</label>
          <div className="spider-input-row">
            <select
              className="spider-input"
              value={sector}
              onChange={(e) => setSector(e.target.value)}
            >
              <option value="">请选择领域</option>
              <option value="APEX">APEX</option>
              <option value="ASMR">ASMR</option>
              <option value="COS">COS</option>
              <option value="FPS">FPS</option>
              <option value="LPL">LPL</option>
              <option value="PUBG">PUBG</option>
              <option value="三角洲">三角洲</option>
              <option value="乙游">乙游</option>
              <option value="二创">二创</option>
              <option value="历史评鉴">历史评鉴</option>
              <option value="时政">时政</option>
              <option value="英雄联盟">英雄联盟</option>
            </select>
          </div>

          <div className="spider-input-row">
            <button className="spider-btn" disabled={loading}>
              {loading ? "爬取中..." : "开始爬取"}
            </button>
          </div>

          <div className="spider-tip">
            支持标准 B 站视频链接，建议先完成登录再进行爬取。
          </div>

          {/* ✅ 页面常驻提示 */}
          <div className="spider-first-tip">
            第一次爬取时间可能稍长，请耐心等待～
          </div>
        </form>

        {/* ✅ 仓鼠骑车进度条：gif 在进度条上前进 */}
        {showProgress && (
          <div className="spider-progress">
            <div className="hamster-progress-bar">
              {/* 进度条轨道 */}
              <div className="hamster-progress-track">
                {/* 已完成区域 */}
                <div
                  className="hamster-progress-fill"
                  style={{ width: `${safeProgress}%` }}
                />
              </div>

              {/* 仓鼠 GIF，随进度移动 */}
              <img
                src={HamsterGif}
                alt="正在努力爬取中..."
                className="hamster-progress-gif"
                style={{ left: `${safeProgress}%` }}
              />
            </div>

            <div className="spider-progress-text">
              {safeProgress < 100
                ? `正在爬取评论... 第一次爬取时间可能稍长，请耐心等待（${Math.round(
                    safeProgress
                  )}%）`
                : "爬取完成，正在跳转首页..."}
            </div>
          </div>
        )}

        {error && <div className="spider-error">{error}</div>}

        {preview.length > 0 && (
          <div className="spider-result">
            <div className="spider-result-header">
              爬取成功，以下为部分评论预览（最多展示 20 条）：
            </div>
            <div className="spider-list">
              {preview.map((item, idx) => (
                <div className="comment-item" key={idx}>
                  <div className="comment-tag">{item.tag ?? "未标注"}</div>
                  <div className="comment-text">{item.text}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
