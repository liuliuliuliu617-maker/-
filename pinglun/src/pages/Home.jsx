import React, { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import "./Home.css";
import WordCloud from "react-d3-cloud";

// 引入你的 logo（路径按你的项目实际来，如果不在这个位置就自己改一下）
import SummaryLogo from "../assets/logo.png";

const CATEGORIES = [
  { key: "正常", label: "正常" },
  { key: "争论", label: "争论" },
  { key: "@某人", label: "@某人" },
  { key: "广告", label: "广告" },
  { key: "无意义", label: "无意义" },
];

const CATEGORY_COLORS = {
  正常: "#4f46e5",
  争论: "#f97316",
  广告: "#22c55e",
  "@某人": "#0ea5e9",
  无意义: "#9ca3af",
};

// 归一化 tag
function normalizeCategory(tag) {
  if (tag == null || tag === "") return "正常";

  const str = String(tag);

  // 先按中文关键字
  if (str.includes("争论") || str.includes("吵架")) return "争论";
  if (str.includes("广告")) return "广告";
  if (str.includes("无意义") || str.includes("水") || str.includes("灌水"))
    return "无意义";
  if (str.includes("@")) return "@某人";
  if (str.includes("正常")) return "正常";

  // 再按数字编码（兼容 "1"）
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
        return "正常";
    }
  }

  return "正常";
}

// 获取当前登录用户 ID，没有登录返回 null
function getCurrentUserId() {
  try {
    const stored = localStorage.getItem("user");
    if (!stored) return null;
    const user = JSON.parse(stored);
    if (user && (user.id || user.user_id)) {
      return user.id || user.user_id;
    }
    return null;
  } catch {
    return null;
  }
}

export default function Home() {
  const [comments, setComments] = useState([]);
  const [videoInfo, setVideoInfo] = useState(null);
  const [activeCats, setActiveCats] = useState([]);
  const [loadingComments, setLoadingComments] = useState(true);
  const [error, setError] = useState("");

  // 总结 & 词云
  const [summary, setSummary] = useState("");
  const [wordcloudImg, setWordcloudImg] = useState("");
  const [wordcloudWords, setWordcloudWords] = useState([]);

  const navigate = useNavigate();

  useEffect(() => {
    const userId = getCurrentUserId();

    // 没有登录用户，或者拿不到 id，当作无爬取记录
    if (!userId) {
      setComments([]);
      setSummary("");
      setWordcloudWords([]);
      setWordcloudImg("");
      setVideoInfo({
        bv: "",
        url: "",
        sector: "",
        title: "暂无爬取记录",
        cover: "",
        tname: "",
      });
      setLoadingComments(false);
      return;
    }

    const key = `lastVideoResult_user_${userId}`;
    const raw = localStorage.getItem(key);

    if (!raw) {
      // 当前用户从未爬取过视频
      setComments([]);
      setSummary("");
      setWordcloudWords([]);
      setWordcloudImg("");
      setVideoInfo({
        bv: "",
        url: "",
        sector: "",
        title: "暂无爬取记录",
        cover: "",
        tname: "",
      });
      setLoadingComments(false);
      return;
    }

    // 有爬取记录
    try {
      const parsed = JSON.parse(raw);
      const preview = parsed.preview || [];

      setSummary(parsed.summary || "");
      setWordcloudWords(parsed.wordcloud || []);

      const img =
        parsed.wordcloud_img ||
        parsed.wordcloudImage ||
        parsed.wordcloud_url ||
        "";
      if (img) {
        if (img.startsWith("http") || img.startsWith("data:")) {
          setWordcloudImg(img);
        } else {
          setWordcloudImg(`data:image/png;base64,${img}`);
        }
      } else {
        setWordcloudImg("");
      }

      // 获取自定义词库
      const savedDict = JSON.parse(localStorage.getItem("keywords") || "[]");

      // 处理评论并应用自定义词库
      const existingComments = preview.map((item) => {
        // 使用自定义词库更新标签
        for (const keyword of savedDict) {
          if (item.text.includes(keyword.word)) {
            item.tag = keyword.tag;
            break;
          }
        }
        return {
          ...item,
          category: normalizeCategory(item.tag),
        };
      });

      // 去重评论
      const uniqueComments = Array.from(
        new Map(existingComments.map((item) => [item.text + item.BV, item])).values()
      );

      setComments(uniqueComments);

      setVideoInfo({
        bv: parsed.bv || "",
        url: parsed.url || "",
        sector: parsed.sector || "",
        title: parsed.title || "最近爬取的视频",
        cover: parsed.cover || "",
        tname: parsed.tname || "",
      });
    } catch (err) {
      console.error(err);
      setError("本地记录读取失败");
      setComments([]);
      setSummary("");
      setWordcloudWords([]);
      setWordcloudImg("");
      setVideoInfo({
        bv: "",
        url: "",
        sector: "",
        title: "暂无爬取记录",
        cover: "",
        tname: "",
      });
    }

    setLoadingComments(false);
  }, []);

  // 分类数量
  const stats = useMemo(() => {
    const s = { 正常: 0, 争论: 0, 广告: 0, "@某人": 0, 无意义: 0 };
    for (const c of comments) {
      const cat = c.category || "正常";
      s[cat] = (s[cat] || 0) + 1;
    }
    return s;
  }, [comments]);

  // 饼图数据
  const pieData = Object.entries(stats).map(([name, value]) => ({
    name,
    value,
  }));

  const filteredComments = useMemo(
    () =>
      activeCats.length
        ? comments.filter((c) => activeCats.includes(c.category))
        : comments,
    [comments, activeCats]
  );

  const toggleCategory = (key) => {
    setActiveCats((prev) =>
      prev.includes(key) ? prev.filter((i) => i !== key) : [...prev, key]
    );
  };

  const imgSrc = wordcloudImg ? wordcloudImg : "";

  // ===== 顶部导航按钮的事件 =====
  const handleStartCrawl = () => {
    navigate("/comment-spider");
  };

  const handleLogout = () => {
    localStorage.removeItem("user");
    window.location.href = "/login";
  };

  const handleGoUserCenter = () => {
    navigate("/user-center");
  };

  const handleGoCustomDict = () => {
    navigate("/custom-dict");
  };
  // ===============================

  return (
    <div className="home-page">
      {/* 顶部导航 */}
      <header className="home-nav">
        <div className="home-nav-left">
          <div className="home-logo-circle">B</div>
          <div className="home-logo-text">
            <div className="home-logo-title">评论智能分析系统</div>
            <div className="home-logo-subtitle">B站视频 · 评论分析</div>
          </div>
        </div>

        <div className="home-nav-right">
          <button className="home-nav-link" onClick={handleGoUserCenter}>
            用户中心
          </button>
          <button className="home-nav-link" onClick={handleGoCustomDict}>
            自定义词库
          </button>

          <button className="home-nav-btn" onClick={handleLogout}>
            退出登录
          </button>
          <button className="home-nav-btn" onClick={handleStartCrawl}>
            开始爬取新视频
          </button>
        </div>
      </header>

      <main className="home-main">
        {/* 左侧：视频卡片 + 分类统计 + 概况 */}
        <section className="home-left">
          {/* 视频卡片 */}
          <div className="video-card">
            {videoInfo?.cover ? (
              <img src={videoInfo.cover} className="video-cover" alt="封面" />
            ) : (
              <div className="video-cover placeholder">暂无封面</div>
            )}

            <div className="video-info">
              <div className="video-title">
                {videoInfo?.title || "暂无爬取记录"}
              </div>
              <div className="video-meta">
                {videoInfo?.tname && (
                  <span className="video-tag">{videoInfo.tname}</span>
                )}
                {videoInfo?.sector && (
                  <span className="video-tag">领域：{videoInfo.sector}</span>
                )}
                {videoInfo?.bv && (
                  <span className="video-bv">BV：{videoInfo.bv}</span>
                )}
              </div>
              {videoInfo?.url && (
                <a
                  className="video-link"
                  href={videoInfo.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  在 B 站中打开视频
                </a>
              )}
            </div>
          </div>

          {/* 分类统计 */}
          <div className="category-panel">
            <div className="category-title">评论类别统计</div>
            <div className="category-buttons">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat.key}
                  className={
                    "category-btn" +
                    (activeCats.includes(cat.key) ? " category-btn-active" : "")
                  }
                  onClick={() => toggleCategory(cat.key)}
                >
                  <span>{cat.label}</span>
                  <span className="category-count">{stats[cat.key]}</span>
                </button>
              ))}
            </div>
            <div className="category-tip">
              · 不选任何按钮 = 显示全部评论 · 单击 = 只显示该类别（可多选）
              · 再次单击 = 取消该类别
            </div>
          </div>

          {/* 概况卡片 */}
          <div className="overview-card">
            <div className="overview-title">评论概况</div>

            <div className="overview-top">
              <div className="overview-pie">
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      outerRadius={80}
                      label
                    >
                      {pieData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={CATEGORY_COLORS[entry.name]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* 总结区域 + logo */}
              <div className="overview-summary">
                <div className="overview-summary-header">
                  <div className="overview-summary-title">总结</div>
                  {/* 总结里的 logo */}
                  <img
                    src={SummaryLogo}
                    alt="summary logo"
                    className="overview-summary-logo"
                  />
                </div>
                <p className="overview-summary-text">
                  {summary || "暂无数据，请先爬取视频。"}
                </p>
              </div>
            </div>

            <div className="overview-wordcloud-section">
              <div className="overview-wordcloud-title">高频词云</div>

              {imgSrc ? (
                <img
                  src={imgSrc}
                  alt="评论词云"
                  className="overview-wordcloud-img"
                />
              ) : (
                <div className="overview-empty">暂无词云数据</div>
              )}
            </div>
          </div>

          {error && <div className="home-error">{error}</div>}
        </section>

        {/* 右侧：评论列表 */}
        <section className="home-right">
          <div className="comment-header">
            <div className="comment-header-title">评论列表</div>
            <div className="comment-header-sub">
              共 {comments.length} 条评论，当前显示 {filteredComments.length} 条
            </div>
          </div>

          <div className="comment-list">
            {loadingComments ? (
              <div className="comment-empty">评论加载中...</div>
            ) : filteredComments.length === 0 ? (
              <div className="comment-empty">暂无评论数据</div>
            ) : (
              filteredComments.map((c, i) => (
                <div className="comment-item" key={i}>
                  <span className={`comment-tag tag-${c.category}`}>
                    {c.category}
                  </span>
                  <span className="comment-text">{c.text}</span>
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
