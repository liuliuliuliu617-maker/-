import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./CommentOverview.css";

export default function CommentOverview() {
  const navigate = useNavigate();
  const [summary, setSummary] = useState("");
  const [wordcloud, setWordcloud] = useState([]);
  const [previewCount, setPreviewCount] = useState(0);

  useEffect(() => {
    const raw = localStorage.getItem("lastVideoResult");
    if (!raw) return;

    try {
      const parsed = JSON.parse(raw);
      setSummary(parsed.summary || "");
      setWordcloud(parsed.wordcloud || []);
      setPreviewCount((parsed.preview || []).length);
    } catch {
      // ignore
    }
  }, []);

  const maxWeight = useMemo(() => {
    if (!wordcloud || wordcloud.length === 0) return 1;
    return Math.max(...wordcloud.map((w) => w.weight || 1));
  }, [wordcloud]);

  const handleBackHome = () => {
    navigate("/home");
  };

  return (
    <div className="ov-page">
      <div className="ov-header">
        <div>
          <h1 className="ov-title">评论概况</h1>
          <p className="ov-subtitle">
            基于最近一次爬取的视频评论，展示整体情绪、类别分布与高频词
          </p>
        </div>
        <button className="ov-back-btn" onClick={handleBackHome}>
          返回首页
        </button>
      </div>

      {!summary && wordcloud.length === 0 && (
        <div className="ov-empty">
          暂无分析数据，请先在首页爬取一个视频。
        </div>
      )}

      {(summary || previewCount) && (
        <section className="ov-section ov-summary-card">
          <h2 className="ov-section-title">总体结论</h2>
          {previewCount ? (
            <div className="ov-summary-count">
              当前共分析{" "}
              <span className="ov-highlight-number">{previewCount}</span> 条评论
            </div>
          ) : null}
          <p className="ov-summary-text">
            {summary || "当前暂无总结。"}
          </p>
        </section>
      )}

      {wordcloud.length > 0 && (
        <section className="ov-section">
          <h2 className="ov-section-title">词云概览</h2>
          <p className="ov-section-tip">
            字越大、越深的词，出现在评论中的频次越高。
          </p>
          <div className="ov-wordcloud">
            {wordcloud.map((w, idx) => {
              const weight = w.weight || 1;
              const ratio = weight / maxWeight;
              const fontSize = 14 + ratio * 22; // 14–36px
              const opacity = 0.45 + ratio * 0.55; // 0.45–1
              const rotate = (idx % 5) * 3 - 6; // 微微旋转一点点

              return (
                <span
                  key={idx}
                  className="ov-word"
                  style={{
                    fontSize: `${fontSize}px`,
                    opacity,
                    transform: `rotate(${rotate}deg)`,
                  }}
                >
                  {w.text}
                </span>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
