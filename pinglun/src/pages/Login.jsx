import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    if (!username.trim() || !password.trim()) {
      setError("请输入用户名和密码");
      return;
    }

    setError("");

    try {
      const resp = await fetch("http://127.0.0.1:5000/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account: username,
          password: password,
        }),
      });

      const data = await resp.json();

      if (!data.success) {
        setError(data.message || "登录失败");
        return;
      }

      // 保存用户信息
      localStorage.setItem("user", JSON.stringify(data.user));

      // ⭐⭐⭐ 登录成功后 → 跳首页
      navigate("/home");

    } catch (err) {
      setError("网络错误：" + err.message);
    }
  };

  const handleRegisterClick = () => {
    navigate("/register");
  };

  return (
    <div className="login-container">
      <img className="login-bg" src="/images/登录页/u0.png" alt="background" />

      <div className="login-box">
        <h2 className="login-title">登录</h2>

        <label className="login-label">用户名：</label>
        <input
          className="login-input"
          type="text"
          placeholder="请输入用户名"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <label className="login-label">密码：</label>
        <input
          className="login-input"
          type="password"
          placeholder="请输入密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {error && <div className="error-text">{error}</div>}

        <div className="btn-group">
          <button className="login-btn" onClick={handleLogin}>登录</button>
          <button className="register-btn" onClick={handleRegisterClick}>注册</button>
        </div>
      </div>
    </div>
  );
}
