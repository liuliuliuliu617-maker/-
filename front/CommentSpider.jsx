import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./CommentSpider.css"; // 样式文件

// 标签ID到中文标签的映射
const TAG_LABEL_MAP = {
  1: "正常",
  2: "争论",
  3: "广告",
  4: "@某人",
  5: "无意义",
};

export default function CommentSpider() {
  const [url, setUrl] = useState("");       // 视频URL
  const [loading, setLoading] = useState(false); // 加载状态
  const [error, setError] = useState("");      // 错误信息
  const [preview, setPreview] = useState([]);  // 评论预览
  const [progress, setProgress] = useState(0); // 进度条
  const [showProgress, setShowProgress] = useState(false); // 是否显示进度
  const [sector, setSector] = useState("");   // 板块

  const navigate = useNavigate();

  // 获取当前用户ID，若未登录，使用默认 ID（例如：1）
  let userId = 1;
  try {
    const stored = localStorage.getItem("user");
    if (stored) {
      const user = JSON.parse(stored);
      if (user && user.id) {
        userId = user.id;
      }
    }
  } catch (e) {
    // 如果没有获取到用户信息，继续使用默认 ID
  }

  // 提交表单，爬取视频评论
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!url.trim()) {
      setError("请输入视频URL");
      return;
    }

     if (!sector) {
      setError("请选择板块");
      return;
    }

    setError(""); // 清除错误
    setLoading(true);  // 开始加载
    setPreview([]);    // 清空之前的评论
    setShowProgress(true);  // 显示进度条
    setProgress(10);   // 设置进度初值为 10%

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
          user_id: userId,  // 用户 ID
          sector,  // 用户选择的板块
        }),
      });

      const data = await resp.json();  // 获取返回的 JSON 数据

      // 处理错误
      if (!data.success) {
        setError(data.message || "爬取失败");
        setShowProgress(false);  // 隐藏进度条
        setProgress(0);  // 重置进度
      } else {
        // 处理成功的结果
        const previewData = data.preview || [];
        setPreview(previewData);
        setProgress(100);  // 进度条达到 100%

        // 保存爬取结果到 localStorage，供其他页面使用
        const payload = {
          preview: previewData,
          bv: data.bv,
          url,
          sector: data.sector || sector,
        };
        localStorage.setItem("lastVideoResult", JSON.stringify(payload));

        // 延时 800ms 后跳转到首页
        setTimeout(() => {
          setShowProgress(false);  // 隐藏进度条
          navigate("/home");  // 跳转到首页
        }, 800);
      }
    } catch (err) {
      setError("网络错误：" + err.message);
      setShowProgress(false);  // 隐藏进度条
      setProgress(0);  // 重置进度
    } finally {
      if (timer) clearInterval(timer);  // 清除进度模拟计时器
      setLoading(false);  // 停止加载状态
    }
  };

  return (
    <div className="spider-page">
      <div className="spider-card">
        <h1 className="spider-title">B站视频评论爬取</h1>
        <p className="spider-subtitle">
          输入视频链接，一键抓取并分析弹幕与评论
        </p>

        <form className="spider-form" onSubmit={handleSubmit}>
          <label className="spider-label">板块</label>
          <div className="spider-select-row">
            <select
              className="spider-select"
              value={sector}
              onChange={(e) => setSector(e.target.value)}
              disabled={loading}
            >
              <option value="">请选择板块</option>
              <option value="APEX">APEX</option>
              <option value="cos">cos</option>
              <option value="FPS">FPS</option>
              <option value="PUBG">PUBG</option>
              <option value="历史品鉴">历史品鉴</option>
              <option value="三角洲">三角洲</option>
              <option value="时政">时政</option>
              <option value="英雄联盟">英雄联盟</option>
            </select>
          </div>

          <label className="spider-label">视频 URL</label>
          <div className="spider-input-row">
            <input
              className="spider-input"
              placeholder="例如：https://www.bilibili.com/video/BVxxxxxxxxx"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <button className="spider-btn" disabled={loading}>
              {loading ? "爬取中..." : "开始爬取"}
            </button>
          </div>
          <div className="spider-tip">
            支持标准 B 站视频链接，建议先完成登录再进行爬取。
          </div>
        </form>

        {showProgress && (
          <div className="spider-progress">
            <div className="spider-progress-bar">
              <div
                className="spider-progress-inner"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="spider-progress-text">
              {progress < 100
                ? `正在爬取评论... ${Math.round(progress)}%`
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
                  <div className="comment-tag">
                    {TAG_LABEL_MAP[item.tag] || item.tag || "未标注"}
                  </div>
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
