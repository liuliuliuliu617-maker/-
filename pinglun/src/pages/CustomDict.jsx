import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./CustomDict.css";

const TAG_OPTIONS = [
  { value: 1, label: "正常" },
  { value: 2, label: "争论" },
  { value: 4, label: "@某人" },
  { value: 3, label: "广告" },
  { value: 5, label: "无意义" },
];

// 领域下拉选项
const SECTOR_OPTIONS = [
  "APEX",
  "ASMR",
  "COS",
  "FPS",
  "LPL",
  "PUBG",
  "三角洲",
  "乙游",
  "二创",
  "历史评鉴",
  "时政",
  "英雄联盟",
];

const getTagLabel = (value) => {
  const found = TAG_OPTIONS.find((t) => String(t.value) === String(value));
  return found ? found.label : value;
};

export default function CustomDict() {
  const navigate = useNavigate();
  const [sector, setSector] = useState(""); // 用户选择领域
  const [tag, setTag] = useState("");
  const [word, setWord] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [msg, setMsg] = useState("");

  // 所有关键词数据（从 localStorage 读取）
  const [keywords, setKeywords] = useState([]);

  // 组件挂载时加载本地关键词
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("keywords") || "[]");
      setKeywords(saved);
    } catch {
      setKeywords([]);
    }
  }, []);

  // 根据当前 sector / tag 过滤出要展示的关键词
  const filteredKeywords = keywords.filter((k) => {
    if (!sector || !tag) return false;
    return (
      (k.sector || "") === sector.trim() &&
      String(k.tag) === String(tag)
    );
  });

  // 获取当前用户 ID
  const getUserId = () => {
    const stored = localStorage.getItem("user");
    let userId = 1;
    if (stored) {
      try {
        const user = JSON.parse(stored);
        if (user.id) userId = user.id;
      } catch {}
    }
    return userId;
  };

  // 更新 lastVideoResult 的公共方法（新增或删除时都用）
  const updateLastVideoResultAfterChange = (newDict, changedWord) => {
    const lastRaw = localStorage.getItem("lastVideoResult");
    if (!lastRaw) return;

    try {
      const lastResult = JSON.parse(lastRaw);
      const updatedComments = (lastResult.preview || []).map((item) => {
        const text = item.text || "";
        // 优先使用仍然存在的自定义词库
        const matched = newDict.find((dict) => text.includes(dict.word));
        if (matched) {
          return { ...item, tag: matched.tag };
        }

        // 如果这条评论只命中了刚刚删除的那个词，就把 tag 置为“正常”（1）
        if (changedWord && text.includes(changedWord)) {
          return { ...item, tag: 1 };
        }

        return item;
      });

      localStorage.setItem(
        "lastVideoResult",
        JSON.stringify({
          ...lastResult,
          preview: updatedComments,
        })
      );
    } catch (err) {
      console.error("更新 lastVideoResult 失败:", err);
    }
  };

  // 提交保存词库
  const handleSubmit = async (e) => {
    e.preventDefault();
    setMsg("");

    if (!sector.trim()) {
      setMsg("请选择领域（sector）");
      return;
    }

    if (!tag) {
      setMsg("请选择类别");
      return;
    }

    if (!word.trim()) {
      setMsg("请输入自定义词语");
      return;
    }

    const userId = getUserId();
    setSubmitting(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/api/keywords", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          word: word.trim(),
          tag: Number(tag),
          sector: sector.trim(),
          user_id: userId,
        }),
      });

      if (!res.ok) {
        const t = await res.text();
        console.error("server error:", t);
        setMsg("服务端错误，HTTP 状态：" + res.status);
        return;
      }

      const data = await res.json();
      if (!data.success) {
        setMsg(data.message || "提交失败");
        return;
      }

      alert("自定义词库已保存！");

      // 更新 localStorage 中的词库
      const savedDict = JSON.parse(localStorage.getItem("keywords") || "[]");
      savedDict.push({
        word: word.trim(),
        tag: Number(tag),
        sector: sector.trim(),
      });
      localStorage.setItem("keywords", JSON.stringify(savedDict));
      setKeywords(savedDict); // 刷新表格

      // 更新 lastVideoResult
      updateLastVideoResultAfterChange(savedDict, null);

      navigate("/home");
    } catch (err) {
      console.error("fetch error:", err);
      setMsg("请求失败：" + err.message);
    } finally {
      setSubmitting(false);
    }
  };

  // 公共删除逻辑（给“删除该词”按钮和表格里的删除按钮共用）
  const deleteKeyword = async ({ word: targetWord, tag: targetTag, sector: targetSector }) => {
    setMsg("");

    if (!targetSector || !String(targetTag) || !targetWord) {
      setMsg("删除参数不完整，请检查领域、类别和词语");
      return;
    }

    const userId = getUserId();
    setDeleting(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/api/keywords", {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          word: targetWord.trim(),
          tag: Number(targetTag),
          sector: targetSector.trim(),
          user_id: userId,
        }),
      });

      if (!res.ok) {
        const t = await res.text();
        console.error("server error:", t);
        setMsg("删除失败，HTTP 状态：" + res.status);
        return;
      }

      const data = await res.json();
      if (!data.success) {
        setMsg(data.message || "删除失败");
        return;
      }

      alert(`自定义词条「${targetWord}」已删除！`);

      // 1）更新 localStorage 中的 keywords
      const savedDict = JSON.parse(localStorage.getItem("keywords") || "[]");
      const newDict = savedDict.filter(
        (k) =>
          !(
            k.word === targetWord.trim() &&
            String(k.tag) === String(targetTag) &&
            (k.sector || "") === targetSector.trim()
          )
      );
      localStorage.setItem("keywords", JSON.stringify(newDict));
      setKeywords(newDict); // 刷新表格

      // 2）更新 lastVideoResult 里评论的 tag
      updateLastVideoResultAfterChange(newDict, targetWord.trim());
    } catch (err) {
      console.error("delete error:", err);
      setMsg("删除请求失败：" + err.message);
    } finally {
      setDeleting(false);
    }
  };

  // 删除词库词条（使用当前输入框）
  const handleDelete = async () => {
    if (!sector.trim()) {
      setMsg("请选择要删除词条的领域（sector）");
      return;
    }

    if (!tag) {
      setMsg("请选择要删除词条的类别");
      return;
    }

    if (!word.trim()) {
      setMsg("请输入要删除的自定义词语");
      return;
    }

    await deleteKeyword({
      word: word.trim(),
      tag,
      sector: sector.trim(),
    });
  };

  // 表格里的“删除”按钮
  const handleDeleteRow = async (item) => {
    if (!window.confirm(`确认删除关键词「${item.word}」吗？`)) return;

    await deleteKeyword({
      word: item.word,
      tag: item.tag,
      sector: item.sector,
    });
  };

  return (
    <div className="cd-page">
      <div className="cd-card">
        <div className="cd-header">
          <div>
            <h1 className="cd-title">自定义词库</h1>
            <p className="cd-subtitle">
              为指定类别添加关键词，匹配到该词的评论会优先归类到此类别
            </p>
          </div>
          <button className="cd-back-btn" onClick={() => navigate("/home")}>
            返回首页
          </button>
        </div>

        <form className="cd-form" onSubmit={handleSubmit}>
          {/* 领域下拉框 */}
          <div className="cd-form-row">
            <label className="cd-label">请选择领域（sector）</label>
            <select
              className="cd-select"
              value={sector}
              onChange={(e) => setSector(e.target.value)}
            >
              <option value="">请选择领域</option>
              {SECTOR_OPTIONS.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          {/* 分类下拉框 */}
          <div className="cd-form-row">
            <label className="cd-label">请选择类别</label>
            <select
              className="cd-select"
              value={tag}
              onChange={(e) => setTag(e.target.value)}
            >
              <option value="">请选择类别</option>
              {TAG_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* 自定义关键词 */}
          <div className="cd-form-row">
            <label className="cd-label">自定义词语</label>
            <input
              className="cd-input"
              type="text"
              placeholder="例如：跑路、暴雷、官宣..."
              value={word}
              onChange={(e) => setWord(e.target.value)}
            />
          </div>

          {msg && <div className="cd-message">{msg}</div>}

          <div className="cd-actions">
            <button
              className="cd-submit-btn"
              type="submit"
              disabled={submitting}
            >
              {submitting ? "提交中..." : "确认添加"}
            </button>

            <button
              className="cd-delete-btn"
              type="button"
              onClick={handleDelete}
              disabled={deleting}
            >
              {deleting ? "删除中..." : "删除该词"}
            </button>
          </div>
        </form>

        {/* 关键词表格区域 */}
        {sector && tag && (
          <div className="cd-keywords-section">
            <div className="cd-keywords-header">
              <h2 className="cd-keywords-title">已添加关键词</h2>
              <p className="cd-keywords-subtitle">
                当前领域：<span>{sector}</span>，类别：
                <span>{getTagLabel(tag)}</span>
              </p>
            </div>

            {filteredKeywords.length === 0 ? (
              <div className="cd-table-empty">
                当前领域和类别下还没有添加关键词。
              </div>
            ) : (
              <div className="cd-table-wrapper">
                <table className="cd-table">
                  <thead>
                    <tr>
                      <th>关键词</th>
                      <th>类别</th>
                      <th>操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredKeywords.map((item, index) => (
                      <tr
                        key={`${item.word}-${item.tag}-${item.sector}-${index}`}
                      >
                        <td>{item.word}</td>
                        <td>
                          <span className="cd-tag-pill">
                            {getTagLabel(item.tag)}
                          </span>
                        </td>
                        <td>
                          <button
                            type="button"
                            className="cd-table-delete-btn"
                            onClick={() => handleDeleteRow(item)}
                            disabled={deleting}
                          >
                            删除
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
