import React, { useState } from "react";
import { useNavigate } from "react-router-dom";  // 导入 useNavigate 钩子
import "./Register.css";

export default function Register() {
  const [name, setName] = useState("");       // 昵称/用户名（后端要求的 name）
  const [account, setAccount] = useState(""); // 登录账号（后端要求 account）
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");

  const navigate = useNavigate();  // 用于页面跳转

  // 注册接口调用
  const handleRegister = async () => {
    if (!name.trim() || !account.trim() || !password.trim()) {
      setError("请输入完整信息");
      return;
    }

    if (password !== confirm) {
      setError("两次密码输入不一致");
      return;
    }

    setError("");  // 清除错误信息

    try {
      // 调用实际的后端注册接口
      const resp = await fetch("http://127.0.0.1:5000/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,         // 后端字段名
          account,      // 后端字段名
          password,     // 后端字段名
        }),
      });

      const data = await resp.json();  // 解析后端返回的 JSON 数据

      if (!data.success) {
        setError(data.message || "注册失败");
        return;
      }

      alert("注册成功，请登录");
      navigate("/login");  // 注册成功后跳转到登录页
    } catch (err) {
      setError("网络错误：" + err.message);
    }
  };

  // 返回登录页
  const handleBack = () => {
    navigate("/login");  // 跳转到登录页
  };

  return (
    <div className="login-container">
      <img src="/images/登录页/u0.png" alt="bg" className="login-bg" />

      <div className="login-box">
        <h2 className="login-title">注册账号</h2>

        {/* 错误信息（你 CSS 里可以自己定义颜色） */}
        {error && <div className="error-text">{error}</div>}

        <label className="login-label">用户名：</label>
        <input
          className="login-input"
          type="text"
          placeholder="请输入用户名"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        <label className="login-label">账号：</label>
        <input
          className="login-input"
          type="text"
          placeholder="请输入账号（用于登录）"
          value={account}
          onChange={(e) => setAccount(e.target.value)}
        />

        <label className="login-label">密码：</label>
        <input
          className="login-input"
          type="password"
          placeholder="请输入密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <label className="login-label">确认密码：</label>
        <input
          className="login-input"
          type="password"
          placeholder="重复输入密码"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
        />

        <div className="btn-group">
          {/* 提交注册按钮 */}
          <button className="login-btn" onClick={handleRegister}>
            提交注册
          </button>

          {/* 返回登录按钮 */}
          <button className="register-btn" onClick={handleBack}>
            返回登录
          </button>
        </div>
      </div>
    </div>
  );
}
