// front/login.js - 已修复 escapeHtml 语法错误并保留调试日志
// 说明：向 /api/login 发送 POST 请求（JSON），成功显示“登录成功”，失败显示错误信息。

(function(){
  document.addEventListener('DOMContentLoaded', () => {
    console.log('login.js: DOMContentLoaded');

    const accountEl = document.getElementById('account');
    const passwordEl = document.getElementById('password');
    const btn = document.getElementById('loginBtn');
    const msgEl = document.getElementById('msg');

    if (!accountEl || !passwordEl || !btn || !msgEl) {
      console.error('login.js: required DOM elements not found. Ensure IDs: account, password, loginBtn, msg');
      return;
    }

    function setLoading(loading) {
      btn.disabled = loading;
      btn.textContent = loading ? '登录中...' : '登录';
    }

    function showMessage(text, type) {
      msgEl.textContent = text || '';
      msgEl.className = type === 'success' ? 'success' : (type === 'error' ? 'error' : '');
    }

    // 修复：callback 接收参数 s 并返回映射值
    function escapeHtml(str) {
      const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '/': '&#x2F;',
        '`': '&#x60;',
        '=': '&#x3D;'
      };
      return String(str).replace(/[&<>"'`=\/]/g, function(s) {
        return map[s] || s;
      });
    }

    async function doLogin() {
      console.log('doLogin: start');
      const account = (accountEl.value || '').trim();
      const password = (passwordEl.value || '').trim();

      if (!account) {
        showMessage('请输入账号', 'error');
        accountEl.focus();
        return;
      }
      if (!password) {
        showMessage('请输入密码', 'error');
        passwordEl.focus();
        return;
      }

      setLoading(true);
      showMessage('', '');

      try {
        console.log('doLogin: post /api/login', { account });
        const res = await fetch('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ account, password })
        });

        console.log('doLogin: response status', res.status);
        const text = await res.text().catch(() => '');
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch(e) { console.warn('doLogin: response not JSON', text); }

        if (res.ok && data && data.success) {
          console.log('doLogin: success', data);
          // 显示登录成功界面
          document.documentElement.scrollTop = 0;
          const name = data.user && data.user.name ? escapeHtml(data.user.name) : '';
          document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;background:#f5f7fb;"><div style="text-align:center;font-family:Arial,sans-serif;"><h1 style="color:#2f9e44;margin:0;font-size:28px;">登录成功</h1><p style="margin-top:10px;color:#333;font-size:16px;">欢迎，' + name + '</p></div></div>';
        } else {
          const msg = (data && data.message) ? data.message : '登录失败，请检查账号或密码';
          console.warn('doLogin: fail', { status: res.status, msg });
          showMessage(msg, 'error');
        }
      } catch (err) {
        console.error('doLogin: request error', err);
        showMessage('请求失败，请稍后重试', 'error');
      } finally {
        setLoading(false);
      }
    }

    // 绑定点击与回车提交
    btn.addEventListener('click', doLogin);
    [accountEl, passwordEl].forEach(el => el.addEventListener('keydown', function(e){
      if (e.key === 'Enter') doLogin();
    }));

    console.log('login.js: initialized');
  });
})();