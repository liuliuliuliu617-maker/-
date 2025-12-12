// src/App.js
import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import Login from "./pages/Login";
import Register from "./pages/Register";
import CommentSpider from "./pages/CommentSpider";
import Home from "./pages/Home";

// 新增的三个页面
import UserCenter from "./pages/UserCenter";
import CustomDict from "./pages/CustomDict";
import CommentOverview from "./pages/CommentOverview";

function App() {
  return (
    <Routes>
      {/* 默认访问 / 时跳到登录页 */}
      <Route path="/" element={<Navigate to="/login" replace />} />

      {/* 登录页 */}
      <Route path="/login" element={<Login />} />

      {/* 注册页 */}
      <Route path="/register" element={<Register />} />

      {/* 爬取页（你的 Home.jsx 里写的是 /comment-spider） */}
      <Route path="/comment-spider" element={<CommentSpider />} />

      {/* 首页 */}
      <Route path="/home" element={<Home />} />

      {/* 新增：用户中心 */}
      <Route path="/user-center" element={<UserCenter />} />

      {/* 新增：自定义词库 */}
      <Route path="/custom-dict" element={<CustomDict />} />

      {/* 新增：评论概况 */}
      <Route path="/overview" element={<CommentOverview />} />

      {/* 兜底：任何未知路径 → 登录页 */}
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  );
}

export default App;
