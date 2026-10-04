# invest-agent 投资学习 Agent

问一句话，看它一步步查数据、读新闻，再给出有依据的回答。

比如问「澳元最近怎么样，要不要换？」，agent 会自己决定：先查实时汇率，再查近 30 天走势，再搜相关新闻，最后综合回答并写出依据。网页上会实时显示每一步调用了什么工具、拿到了什么数据。

> 仅用于学习投资知识，不做真实交易，回答不构成投资建议。
> 它的前身是 [fx-bot](https://github.com/Szt-123-coder/fx-bot)（汇率提醒机器人）。

## 能做什么

| 功能 | 例子 |
| --- | --- |
| 多步分析 | 澳元最近怎么样，要不要换？ / 英伟达和特斯拉最近哪个涨得多？ |
| 查价格和走势 | 美元现在多少人民币？ / 茅台最近走势怎么样？ |
| 读新闻 | 澳元为什么最近涨了？ |
| 价位提醒 | 美元跌到 7.0 提醒我 |
| 波动提醒 | 澳元跌了就提醒我（从高点每跌 0.2% 提醒一次） |
| 关注和偏好 | 帮我关注英镑 / 记住，我只关心换汇 |

## 架构

```
网页 / 评测脚本 / 定时任务
          │
     FastAPI 服务 ── 流式推送每一步，并存进数据库
          │
   Agent（LangChain create_agent，ReAct 循环）◄──► 大模型（默认 DeepSeek，可替换）
          │
  ┌───────┼──────────┬──────────────┐
查行情    搜新闻      提醒与关注      记忆读写
Yahoo    Tavily       └──── SQLite ────┘
```

| 部分 | 选择 | 为什么 |
| --- | --- | --- |
| Agent | LangChain / LangGraph | 主流框架；另有 [手写版循环](app/react_by_hand.py) 对照学习原理 |
| 大模型 | DeepSeek，兼容 OpenAI 接口的都能换 | 工具调用稳定、便宜；换模型只改环境变量 |
| 新闻 | Tavily 搜索 API | 专为 agent 设计 |
| 行情 | Yahoo Finance | 汇率和全球股票都有 |
| 数据库 | SQLite + SQLAlchemy | 一个文件即可；换 PostgreSQL 只改 `DATABASE_URL` |
| 网页 | FastAPI + 原生 HTML，Server-Sent Events | agent 每走一步就推给页面 |

设计取舍写在 [docs/design](docs/design)。

## 评测

`evals/cases.jsonl` 里是测试题，每题写明期望调用的工具和参数、不该调用的工具、回答的评分标准。

```bash
python -m evals.run_eval
```

- **规则评测**：检查 agent 做的事。工具和参数对不对，修改失败时有没有假装成功
- **模型裁判**：另一个模型按评分标准给回答打 1–5 分（有密钥时才跑）
- **人工抽检**：每次评测导出 `evals/reports/<编号>.csv`，人工在 `human_score` 列打分，和裁判对比

结果显示在网页的「评测面板」（`/eval`）。

## 本地运行

```bash
pip install -r requirements-dev.txt
cp .env.example .env      # 不填密钥就是演示模式，不花钱
uvicorn app.main:app --reload
# 打开 http://localhost:8000
```

不配置任何密钥时，项目用「演示模式」运行：一个按关键词规则调用工具的假模型，配合示例行情和示例新闻。页面和步骤都是真的在跑，只是「思考」部分是规则，不是 AI。

测试：`pytest -q`

## 部署

```bash
docker build -t invest-agent .
docker run -d -p 8000:8000 --env-file .env -v invest-data:/app/data invest-agent
```

设了 `ACCESS_PASSWORD` 后，只有在页面里填对密码的人才用真模型，其他访客自动走演示模式，不会产生费用。

## 密钥

所有密钥只放在环境变量或 `.env` 里，`.env` 不会提交到仓库。
