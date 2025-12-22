<<<<<<< HEAD
项目网址:http://1.94.247.8/（由于公网ip是租用华为云的，无法连接可能是云服务器到期问题）
# -
实践代码
=======
由于models较大，无法直接上传到github中，若需要运行系统请点击.gitattributes，使用git lfs依次将相应文件下载到本地并存放在相应文件夹中，位置如.gitattributes中链接所示。

--Qwen2.5_train.py：模型训练代码，可使用本地数据对模型进行训练

--pinglun:前端代码文件夹，使用了一个基于 React 和 React Router 的前端应用结构。主要功能包括登录、注册、用户中心、评论概览等模块，每个功能通过独立的组件进行管理。使用 React Router 实现页面导航，使得在单页应用中可以动态加载不同页面，提升用户体验。组件化设计使得每个模块独立且可复用，便于维护和扩展。每个组件配有独立的 CSS 文件，实现样式的局部管理，避免样式冲突。通过 Navigate 组件实现路径重定向，确保访问根路径时会自动跳转到登录页。使用时请确保使用pycharm打开并在本地下载node.js

--models:模型存放处

--训练文本:对模型进行微调的训练数据存放处

后端：

--db.py ：所有数据库访问逻辑（用户、视频、评论、关键词、v_cnt 统计）。

--server.py ：Flask 服务入口，登录/注册接口、爬取接口 /api/process、关键词接口 /api/keywords、用户视频管理接口 /api/user/videos、/api/user/video_comments，以及页面路由。

--back.sql：数据库 back 的表结构（user / video / comments / keyword），要和db.py配套。

--预测.py：加载本地 Qwen 模型并对评论进行分类（结合关键词规则和大模型），输出类别 ID 和中文标签，供后端入库和展示使用。
