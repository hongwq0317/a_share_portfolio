# A股组合投资系统

> AI 驱动的 A 股自动化投资组合管理系统，基于 LangGraph + ReAct Agent 架构，实现从市场分析、选股、组合构建、风控到模拟交易的全流程自动化。

---

## 目录

- [系统概览](#系统概览)
- [核心特性](#核心特性)
- [架构设计](#架构设计)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装部署](#安装部署)
- [配置说明](#配置说明)
- [使用方式](#使用方式)
- [分析流程详解](#分析流程详解)
- [工具模块](#工具模块)
- [数据源](#数据源)
- [报告输出](#报告输出)
- [风控体系](#风控体系)
- [回测系统](#回测系统)
- [Skills 知识库](#skills-知识库)

---

## 系统概览

本系统是一个端到端的 A 股投资组合管理平台，利用大语言模型（LLM）作为分析与决策引擎，结合多源实时行情数据和量化分析工具，实现以下目标：

1. **自动化市场分析**：实时获取 A 股行情数据，对宏观市场、板块轮动、资金流向进行全面研判
2. **智能选股**：基于多因子评分体系（基本面 + 技术面 + 资金面）筛选股票候选池
3. **组合构建**：运用均值-方差优化、Black-Litterman、风险平价等量化模型构建目标组合
4. **风险管控**：涵盖 VaR 计算、止损止盈、压力测试、熔断机制等完整风控链条
5. **模拟交易**：在模拟环境中执行买卖操作，持久化管理持仓与交易历史
6. **定时调度**：支持每日定时运行分析、风险检查、周度再平衡等自动化任务

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **LangGraph 状态图** | 9 个节点组成的有向状态图，支持节点级启用/禁用、多运行模式路由 |
| **ReAct Agent 子图** | 每个分析节点内嵌 ReAct 循环（研究 → 工具调用 → 压缩/合成），支持多轮自主推理 |
| **多模型分层** | 研究模型（research）、决策模型（decision）、压缩模型（compression）、报告模型（report）分工明确 |
| **多数据源** | ChinaLin 行情 API、新浪财经、腾讯财经、东方财富、Baostock、Tavily 搜索 |
| **量化工具集** | 技术指标计算、多因子评分、均值方差优化、Black-Litterman、风险平价、VaR、Brinson 归因 |
| **持仓持久化** | 基于 JSON 文件的状态管理，支持版本迁移、价格刷新、偏离度计算 |
| **三种投资风格** | 激进（aggressive）、均衡（balanced）、保守（conservative），配置化切换 |
| **完整风控链** | 止损止盈、追踪止损、日 VaR 限额、最大回撤、熔断机制、涨跌停检测 |
| **分节点报告** | 每次运行产出综合报告 + 各节点独立报告，便于审计与追溯 |
| **回测引擎** | 基于历史交易记录的回测分析，支持 Walk-Forward 滚动回测 |

---

## 架构设计

### 主图流程（LangGraph StateGraph）

```
┌─────────┐
│  START   │
└────┬─────┘
     │
     ▼
┌──────────────┐
│ MarketData   │ ← 获取实时行情、指数、板块、资金流等原始数据
└──────┬───────┘
       │ (有持仓时)
       ▼
┌──────────────────┐
│ PositionReview   │ ← 逐只持仓体检：基本面、技术面、资金面综合评分
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ MarketAnalysis   │ ← 宏观研判：趋势、情绪、量能、政策、板块轮动
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ StockScreening   │ ← 多因子选股，生成候选股票池及评分
└──────┬───────────┘
       │
       ▼
┌────────────────────┐
│ PortfolioStrategy  │ ← 组合构建：目标权重、优化模型、约束检查
└──────┬─────────────┘
       │
       ▼
┌──────────────────┐
│ RiskAssessment   │ ← 风险评估：VaR、回撤、压力测试、止损检查
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ TradeExecution   │ ← 模拟交易执行：计算交易指令并下单
└──────┬───────────┘
       │
       ▼
┌────────────────────┐
│ RebalanceMonitor   │ ← 再平衡监控：偏离度、调仓触发判断
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ ReportGeneration   │ ← 汇总所有节点产出，生成投资日报
└──────┬─────────────┘
       │
       ▼
┌─────────┐
│   END   │
└─────────┘
```

### ReAct 子图模式

每个分析节点内部采用 ReAct（Reasoning + Acting）循环：

```
researcher（LLM 推理） → tool_call（调用工具获取数据） → compress/synthesize（压缩摘要）
     ↑                           |
     └───────────────────────────┘  (循环直到达到最大迭代次数)
```

- **Researcher**：LLM 根据系统提示和已有信息，决定调用哪些工具
- **Tool Execution**：执行工具调用（行情查询、指标计算、搜索等）
- **Compress/Synthesize**：将工具返回的大量数据压缩为结构化摘要，控制上下文长度

---

## 项目结构

```
a_share_portfolio/
├── run_portfolio.py              # 主入口：CLI、多模式运行、调度、报告保存
├── portfolio_config.yaml         # 主配置文件
├── requirements.txt              # Python 依赖
│
├── src/                          # 核心源码
│   ├── config.py                 # 配置加载与模型管理（AppConfig、get_configured_model）
│   ├── state.py                  # 状态定义（PortfolioState、Position、TradeOrder 等）
│   ├── graph.py                  # 主图构建（StateGraph、路由逻辑）
│   ├── prompts.py                # 所有 LLM 提示词模板（系统提示、压缩提示、合成提示）
│   ├── utils.py                  # 日志工具（NodeLogger、GraphLogger）
│   ├── portfolio_persistence.py  # 持仓状态持久化（JSON 读写、版本迁移、价格刷新）
│   ├── backtest.py               # 回测引擎（收益计算、风险指标、Walk-Forward）
│   │
│   ├── nodes/                    # 图节点实现
│   │   ├── base_react_subgraph.py    # ReAct 子图工厂（通用模式封装）
│   │   ├── market_data.py            # 市场数据获取节点
│   │   ├── position_review.py        # 持仓审查节点
│   │   ├── market_analysis.py        # 市场分析节点
│   │   ├── stock_screening.py        # 选股筛选节点
│   │   ├── portfolio_strategy.py     # 组合策略节点
│   │   ├── risk_assessment.py        # 风险评估节点
│   │   ├── trade_execution.py        # 交易执行节点
│   │   ├── rebalance_monitor.py      # 再平衡监控节点
│   │   └── report_generation.py      # 报告生成节点
│   │
│   └── tools/                    # LLM 可调用的工具集
│       ├── data_provider.py          # 多源行情数据提供器（新浪、腾讯、东财等）
│       ├── chinalin_provider.py      # ChinaLin 行情 API 封装
│       ├── baostock_provider.py      # Baostock 数据接口封装
│       ├── market_tools.py           # 市场工具（行情查询、板块筛选、概览等）
│       ├── analysis_tools.py         # 分析工具（技术指标、多因子评分、事件日历等）
│       ├── portfolio_tools.py        # 组合工具（优化模型、相关性、再平衡计算）
│       ├── risk_tools.py             # 风控工具（VaR、止损、压力测试、熔断等）
│       ├── attribution_tools.py      # 归因工具（Brinson 归因分析）
│       ├── trade_tools.py            # 交易工具（模拟买卖、持仓查询）
│       ├── search_tools.py           # 搜索工具（Tavily/ChinaLin 搜索、网页摘要）
│       └── skill_tools.py            # Skill 读取与 HTTP 工具
│
├── data/                         # 运行时数据
│   ├── portfolio_state.json      # 持仓状态（现金、持仓、交易历史）
│   └── target_portfolio.json     # 目标组合权重
│
├── reports/                      # 分析报告输出
│   ├── full_YYYYMMDD_HHMMSS.md           # 综合报告
│   ├── full_YYYYMMDD_HHMMSS_nodes/       # 分节点报告目录
│   │   ├── 00_持仓审查.md
│   │   ├── 01_市场分析.md
│   │   ├── 02_选股报告.md
│   │   ├── 03_组合策略.md
│   │   ├── 04_风险评估.md
│   │   ├── 05_交易执行.md
│   │   ├── 06_再平衡.md
│   │   └── 07_综合报告.md
│   └── backtest_YYYYMMDD_HHMMSS.md       # 回测报告
│
├── logs/                         # 运行日志
│   └── portfolio_YYYYMMDD.log
│
└── skills/                       # 领域知识库（LLM 参考资料）
    ├── a-share-trading-rules/        # A 股交易规则（T+1、涨跌停等）
    ├── chinalin-market-api/          # ChinaLin API 使用手册
    ├── risk-control-rules/           # 风控规则手册
    └── stock-scoring-framework/      # 选股评分框架
```

---

## 环境要求

- **Python**: >= 3.11
- **操作系统**: macOS / Linux（Windows 未充分测试）
- **LLM API**: 需要至少一个 OpenAI 兼容的 LLM API 端点（支持 OpenAI、Anthropic、火山引擎等）

---

## 安装部署

### 1. 克隆项目

```bash
git clone <repo-url> a_share_portfolio
cd a_share_portfolio
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

| 包 | 用途 |
|---|------|
| `langgraph` | 状态图编排框架 |
| `langchain` / `langchain-openai` | LLM 集成与工具绑定 |
| `pydantic` | 数据模型验证 |
| `pandas` / `numpy` / `scipy` | 量化计算 |
| `akshare` | A 股数据获取（东方财富等） |
| `ta` | 技术指标计算库 |
| `matplotlib` | 可视化（回测图表） |
| `apscheduler` | 定时任务调度 |
| `aiohttp` | 异步 HTTP 请求 |
| `pyyaml` | YAML 配置解析 |

### 4. 配置 API 密钥

编辑 `portfolio_config.yaml`，填入你的 LLM API 信息：

```yaml
models:
  research: "你的研究模型ID"
  decision: "你的决策模型ID"
  compression: "你的压缩模型ID"
  report: "你的报告模型ID"
  base_url: "你的API地址"

api_keys:
  OPENAI_API_KEY: "你的API Key"
```

---

## 配置说明

`portfolio_config.yaml` 是系统的核心配置文件，包含以下模块：

### 模型配置 (`models`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `research` | 研究分析模型，用于市场分析、选股等深度推理任务 | - |
| `decision` | 决策模型，用于组合构建、风控等需要精确判断的任务 | - |
| `compression` | 压缩/摘要模型，用于精简 ReAct 循环中的上下文 | - |
| `report` | 报告生成模型，用于综合各节点产出生成最终日报 | - |
| `research_temperature` | 研究模型温度 | 0.4 |
| `decision_temperature` | 决策模型温度 | 0.1 |

### 节点开关 (`nodes`)

可独立启用/禁用每个分析节点：

```yaml
nodes:
  enable_market_data: true        # 市场数据获取
  enable_market_analysis: true    # 市场分析
  enable_stock_screening: true    # 选股筛选
  enable_portfolio_strategy: true # 组合策略
  enable_risk_assessment: true    # 风险评估
  enable_trade_execution: true    # 交易执行
  enable_rebalance_monitor: false # 再平衡监控
```

### 组合策略 (`portfolio`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `initial_capital` | 初始资金（元） | 1,000,000 |
| `max_positions` | 最大持仓股票数 | 10 |
| `max_single_weight` | 单股最大仓位 | 20% |
| `max_sector_weight` | 单行业最大仓位 | 45% |
| `style` | 投资风格：aggressive / balanced / conservative | aggressive |
| `benchmark` | 基准指数代码 | 000300（沪深300） |
| `min_cash_ratio` | 最低现金比例 | 5% |
| `max_turnover_rate` | 单次最大换仓率 | 30% |

### 风险控制 (`risk`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `stop_loss_pct` | 单股止损线 | -12% |
| `take_profit_pct` | 单股止盈线 | 50% |
| `max_drawdown_pct` | 组合最大回撤 | -20% |
| `var_limit_pct` | 日 VaR 限额（95% 置信度） | 3% |
| `volatility_target` | 年化波动率目标 | 22% |
| `min_market_cap_billion` | 最小总市值（亿元） | 50 |
| `exclude_st` | 排除 ST/*ST 股票 | true |

### 交易配置 (`trading`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `mode` | 交易模式：simulation / live | simulation |
| `commission_rate` | 佣金费率 | 0.03% |
| `stamp_duty_rate` | 印花税率（卖出） | 0.05% |
| `slippage_pct` | 滑点 | 0.1% |

### 调度配置 (`schedule`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `analysis_time` | 每日分析运行时间 | 08:30 |
| `summary_time` | 收盘后汇总时间 | 15:30 |
| `rebalance_frequency` | 再平衡频率：daily / weekly / monthly | weekly |

### 调仓触发条件 (`rebalance_triggers`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `pnl_threshold` | 持仓盈亏触发阈值 | 15% |
| `deviation_threshold` | 权重偏离触发阈值 | 8% |
| `max_hold_days` | 最大持仓天数（超过则强制评估） | 30天 |
| `market_shock_pct` | 市场单日涨跌幅异常触发阈值 | 3% |

---

## 使用方式

### 完整分析流程（默认）

```bash
python run_portfolio.py
```

运行完整的 9 节点流水线：市场数据 → 持仓审查 → 市场分析 → 选股 → 组合策略 → 风险评估 → 交易执行 → 再平衡 → 报告生成。

### 仅再平衡检查

```bash
python run_portfolio.py --mode rebalance
```

跳过市场分析、选股、组合策略节点，仅执行市场数据 → 持仓审查 → 风险评估 → 再平衡 → 报告。

### 仅风险检查

```bash
python run_portfolio.py --mode risk_check
```

最轻量模式，仅执行市场数据获取 + 风险评估 + 报告。

### 回测分析

```bash
python run_portfolio.py --mode backtest
```

基于 `data/portfolio_state.json` 中的历史交易记录运行回测，计算收益率、夏普比率、最大回撤等指标。

### 定时调度

```bash
python run_portfolio.py --mode schedule
```

启动 APScheduler 守护进程：
- 每个工作日 08:30 运行完整分析
- 每个工作日 15:30 执行风险检查
- 每周五 14:30 执行再平衡检查

### 命令行参数

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--config` | `-c` | 配置文件路径（默认 `portfolio_config.yaml`） |
| `--mode` | `-m` | 运行模式：full / rebalance / risk_check / schedule / backtest |
| `--output` | `-o` | 输出文件路径（支持 .md / .json） |
| `--capital` | - | 初始资金（覆盖配置文件） |
| `--style` | - | 投资风格：aggressive / balanced / conservative |
| `--log-level` | `-l` | 日志等级：full / normal / brief |
| `--verbose` | `-v` | 启用详细日志 |

### 使用示例

```bash
# 使用自定义配置文件、保守风格、50万初始资金
python run_portfolio.py -c my_config.yaml --style conservative --capital 500000

# 输出报告到指定文件
python run_portfolio.py -o reports/my_report.md

# 详细日志模式
python run_portfolio.py -v --log-level full
```

---

## 分析流程详解

### 1. 市场数据获取（MarketData）

通过多源 API 获取实时行情数据，包括：
- 主要指数报价（上证指数、深证成指、创业板指、沪深300 等）
- 板块涨跌排行
- 北向资金流向
- 个股实时行情

### 2. 持仓审查（PositionReview）

逐只审查现有持仓的健康状况：
- 基本面评分（财务指标、盈利质量）
- 技术面评分（趋势、均线、动量）
- 资金面评分（主力资金流向、北向资金动态）
- 综合评分及操作建议（继续持有 / 加仓 / 减持 / 清仓）
- 调仓紧迫度判定

### 3. 市场分析（MarketAnalysis）

宏观市场环境研判：
- 市场趋势判断（上升 / 震荡 / 下降）
- 市场情绪分析（亢奋 / 正常 / 恐慌）
- 量能变化分析
- 板块轮动方向
- 政策与事件影响评估

### 4. 选股筛选（StockScreening）

基于多因子模型的智能选股：
- 量化初筛：市值、流动性、财务健康度等硬性条件
- 多因子评分：成长性、估值、动量、质量等维度
- 行业分散化约束
- 与现有持仓的相关性检查
- 输出评分排序的候选股票池

### 5. 组合策略（PortfolioStrategy）

目标组合构建与优化：
- 均值-方差优化（Markowitz）
- Black-Litterman 模型
- 风险平价模型
- 多约束检查（单股上限、行业上限、相关性等）
- 交易成本考量
- 目标权重输出与持久化

### 6. 风险评估（RiskAssessment）

全面风险评估：
- 组合 VaR（95% / 99% 置信度）
- 最大回撤与当前回撤
- 压力测试（极端行情模拟）
- 止损止盈检查
- 追踪止损监控
- 行业集中度分析
- 涨跌停风险检测
- 熔断机制检查

### 7. 交易执行（TradeExecution）

模拟交易下单：
- 根据风控指令执行强制交易（止损/止盈/清仓）
- 根据目标权重计算调仓交易
- T+1 规则检查
- 手数取整（100 股整数倍）
- 交易成本计算（佣金 + 印花税 + 滑点）
- 持仓状态更新与持久化

### 8. 再平衡监控（RebalanceMonitor）

持仓偏离度监控与触发判断：
- 实际权重 vs 目标权重偏离度计算
- 盈亏阈值触发
- 持仓时间触发
- 市场异动触发
- 调仓建议输出

### 9. 报告生成（ReportGeneration）

汇总所有节点产出，生成结构化投资日报：
- 持仓体检总结
- 市场环境概述
- 组合持仓概况与约束合规
- 今日操作记录
- 风险状况与预警
- 后续计划建议

---

## 工具模块

系统为 LLM 提供了丰富的可调用工具，每个工具均有结构化的输入/输出定义：

### 市场工具 (`market_tools.py`)

- `get_index_quote` - 获取指数实时行情
- `get_stock_quote` - 获取个股实时行情
- `get_sector_overview` - 获取板块排行概览
- `batch_get_stock_overview` - 批量获取股票概况
- `think_tool` - LLM 思考工具（不执行操作，仅用于推理记录）

### 分析工具 (`analysis_tools.py`)

- `calculate_technical_indicators` - 计算技术指标（MACD、RSI、KDJ、布林带等）
- `score_stock_multi_factor` - 多因子综合评分
- `detect_market_regime` - 市场状态检测
- `check_event_calendar` - 事件日历查询

### 组合工具 (`portfolio_tools.py`)

- `mean_variance_optimize` - 均值-方差优化
- `min_variance_optimize` - 最小方差优化
- `black_litterman_optimize` - Black-Litterman 优化
- `risk_parity_optimize` - 风险平价优化
- `calculate_correlation_matrix` - 相关性矩阵计算
- `calculate_rebalance_trades` - 再平衡交易计算

### 风控工具 (`risk_tools.py`)

- `calculate_portfolio_var` - 组合 VaR 计算
- `check_stop_loss` - 止损检查
- `stress_test` - 压力测试
- `check_circuit_breaker` - 熔断检查
- `check_limit_up_down` - 涨跌停检测

### 交易工具 (`trade_tools.py`)

- `simulate_buy` - 模拟买入
- `simulate_sell` - 模拟卖出
- `get_current_portfolio` - 获取当前持仓

### 归因工具 (`attribution_tools.py`)

- `brinson_attribution` - Brinson 归因分析

### 搜索工具 (`search_tools.py`)

- `web_search` - 网络搜索（Tavily / ChinaLin）
- 支持 LLM 摘要结果

---

## 数据源

系统接入多个行情数据源，自动容错切换：

| 数据源 | 功能 | 说明 |
|--------|------|------|
| **ChinaLin API** | 实时行情、K线、板块、北向资金 | 主力数据源，私有部署 |
| **新浪财经** | 实时行情 | 备用数据源 |
| **腾讯财经** | 实时行情 | 备用数据源 |
| **东方财富** | 板块数据、资金流向 | 通过 akshare 接入 |
| **Baostock** | 历史K线、财务数据、行业分类 | 回测与基本面分析 |
| **Tavily** | 新闻搜索 | 事件驱动分析 |

---

## 报告输出

每次运行自动产出两类报告：

### 综合报告（Markdown）

保存在 `reports/full_YYYYMMDD_HHMMSS.md`，包含完整的投资日报：

- **持仓体检**：持仓诊断表、评分、操作建议
- **市场环境**：趋势/情绪/量能标签、关键判断
- **组合持仓**：资产概况、行业分布、约束合规
- **今日操作**：交易记录、风控指令执行确认
- **风险状况**：风险指标、预警信息
- **后续计划**：下一步操作建议

### 分节点报告

保存在 `reports/full_YYYYMMDD_HHMMSS_nodes/` 目录下：

```
00_持仓审查.md    - 持仓审查详细过程与结论
01_市场分析.md    - 市场分析完整推理过程
02_选股报告.md    - 选股筛选详细结果
03_组合策略.md    - 组合优化方案与推理
04_风险评估.md    - 风险评估详细报告
05_交易执行.md    - 交易执行记录
06_再平衡.md      - 再平衡评估结论
07_综合报告.md    - 最终汇总报告
```

---

## 风控体系

系统实现了多层次的风控机制：

### 交易前风控

- **硬性准入**：排除 ST 股票、亏损股、低市值股、低流动性股
- **仓位约束**：单股上限、行业上限、最大持仓数
- **相关性约束**：高相关股票合计权重受限
- **现金比例**：始终保留最低现金缓冲

### 交易中风控

- **T+1 规则**：严格检查当日买入股票不可卖出
- **涨跌停检测**：避免在涨停/跌停时下单
- **滑点控制**：模拟真实交易成本
- **换手率控制**：单次最大换仓比例限制

### 交易后风控

- **止损止盈**：个股硬止损（-12%）和止盈（+50%）
- **追踪止损**：从最高点回撤超阈值触发预警
- **VaR 监控**：日度 VaR 不超过总资产限额
- **最大回撤**：组合回撤超限触发防御策略
- **熔断机制**：日内亏损超阈值暂停交易

---

## 回测系统

系统内置基于实际交易记录的回测引擎：

```bash
python run_portfolio.py --mode backtest
```

### 回测指标

- **收益指标**：累计收益率、年化收益率、日均收益率
- **风险指标**：最大回撤、年化波动率、夏普比率、Sortino 比率、Calmar 比率
- **交易分析**：胜率、盈亏比、平均持仓周期、换手率
- **基准对比**：与沪深300 基准的超额收益分析
- **Walk-Forward**：滚动窗口回测，评估策略稳定性

### 回测报告

输出到 `reports/backtest_YYYYMMDD_HHMMSS.md`，包含详细的绩效分析与风险分析。

---

## Skills 知识库

`skills/` 目录下存放领域知识文件，供 LLM 在分析过程中参考：

| Skill | 说明 |
|-------|------|
| `a-share-trading-rules` | A 股交易规则：T+1、涨跌停、集合竞价、交易时段等 |
| `chinalin-market-api` | ChinaLin 行情 API 使用说明、请求配方、响应字段定义 |
| `risk-control-rules` | 风控规则体系：止损策略、仓位管理、异常处理 |
| `stock-scoring-framework` | 选股评分框架：多因子权重、评分标准、阈值设定 |

---

## 免责声明

本系统仅用于学习和研究目的。所有分析结果和交易建议仅供参考，**不构成任何投资建议**。投资有风险，入市需谨慎。当前系统仅支持**模拟交易**模式，不连接真实券商交易系统。

---

## License

Private / 未公开
