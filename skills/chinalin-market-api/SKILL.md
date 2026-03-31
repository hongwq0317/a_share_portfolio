---
name: chinalin-market-api
description: ChinaLin内部A股行情数据API接口文档。提供个股/指数/板块实时行情、K线历史数据、板块排名、资金流向、市场总览、北向资金等数据查询能力。当项目需要获取A股市场数据时使用此Skill。
---

# ChinaLin 内部行情 API

## 环境配置

- 测试环境：`https://chinalintest.wenxingonline.com`
- 请求参数中，Query参数拼接在URL中，Body参数放在请求体中
- 股票代码前缀：`sh`-沪市，`sz`-深市，`bj`-北交所，`bk`-板块，`ix`-指数通

## 核心接口

### 1. 个股/指数/板块行情（单股）

```
GET /v1/quotes/stockdetail?code=sz300750&deal=0
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| code | string | 是 | 带前缀的证券代码 |
| deal | int | 否 | 是否返回成交明细(1=是) |

返回 `data.detail` 包含完整行情字段，详见 [response-fields.md](response-fields.md)

### 2. 多股行情（批量）

```
POST /v1/quotes/list/stockdetail?code=sz300750|sh601318&ignore_minute=true
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| code | string | 是 | 多个代码用`\|`分割 |
| ignore_minute | int | 否 | 忽略分时数据 |

返回 `data.list[]`，每项含 code/name/cur_price/change_rate/turnover_rate/market_value 等

### 3. 指定字段行情（轻量批量）

```
GET /v1/quotes/fields?codes=sz300750|sh601318&fields=code|name|cur_price|change_rate
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| codes | string | 是 | 多个代码用`\|`分割 |
| fields | string | 是 | 返回字段用`\|`分割 |

支持的fields: code, symbol, name, cur_price, change, change_rate, pre_close, open, high, low, volume, volume_num, turnover, turnover_num, turnover_rate, state, float_shares, total_shares, circulation_market_value, market_value, pe, pb, eps, bid_buy, bid_sell, bid_change, dividend_ps, dividend_ps_rate, free_float_shares, free_float_market, real_turnover_rate

返回 `data.{code}` 对象，含请求的字段

### 4. 日/周/月/年 K线

```
GET /v5/quotes/fqkline?code=sz300750&ktype=day&autype=qfq&start=2024-01-01&end=2024-12-31&count=250
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| code | string | 是 | 带前缀的证券代码 |
| ktype | string | 是 | `day`/`week`/`month`/`year` |
| autype | string | 是 | `qfq`(前复权)/`nfq`(不复权)/`hfq`(后复权) |
| start | string | 否 | 开始日期 yyyy-mm-dd |
| end | string | 是 | 结束日期 yyyy-mm-dd |
| count | int | 是 | 记录数 |

返回 `data.label[]` 为字段名数组，`data.{ktype}[]` 为对应数据二维数组。label顺序：date, open_price, close_price, high_price, low_price, trade_volume, trade_value, turnover_rate, volume_ratio, dividends, change_price, change_rate, amplitude, ...

### 5. 分时/分钟K线

```
GET /v2/quotes/mkline?code=sz300750&end=2024-01-15&ktype=m1&count=241&autype=qfq
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| code | string | 是 | 带前缀的证券代码 |
| ktype | string | 是 | `m1`/`m5`/`m15`/`m30`/`m60`/`m120` |
| autype | string | 是 | 复权类型 |
| end | string | 是 | 结束日期 |
| count | int | 是 | 记录数 |

### 6. 板块排名

```
GET /v1/block/rank?type=2&sort_field_id=1&sort_direction=1&req_seq=0&req_num=50
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| type | int | - | 1概念/2行业/3区域/4风格 |
| sort_field_id | int | 1 | 1涨跌幅 |
| sort_direction | int | 1 | 0升序/1降序 |
| req_seq | int | 0 | 分页起始索引 |
| req_num | int | 50 | 每页大小 |

返回 `data.blocks[]`：code, secu_code, secu_name, last_price, change_price, change_rate, turnover_rate, float_market_value, market_value

### 7. 指数/板块成分股

```
GET /v1/block/members/rank?code=sh000001&sort_field_id=1&sort_direction=1&req_seq=0&req_num=50
```

返回 `data.block_members[]` 和 `data.leader_list[]`，字段同板块排名

### 8. 板块热点股票（主力净流入）

```
GET /v5/block/members/hotRank?code=sh000001&top=5
```

返回 `data[]`：trade_date, code, name, price, change_rate, main_net_in, net_in, volume, turnover_rate

### 9. 多股资金净流入

```
GET /v1/capital/stock/flowSummaries?codes=sz300750|sz002945&types=1|2
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| codes | string | - | 多个代码用`\|`分割 |
| types | string | 全部 | 资金类型：1超大/2中/3小/4单，1\|2为主力 |

返回 `data.{code}`：date, code, net(净流入金额)

### 10. 大单动向

```
GET /v1/capital/stock/flow?code=sz002945&ktype=minute
```

返回 `data.capital_flows[]`：[时间, 特大单, 大单, 中单, 小单]

### 11. 市场总览（含北向资金）

```
GET /v2/market/overview
```

返回字段：
- `data.distribution` - 涨跌分布（上涨/下跌数、涨停/跌停数）
  - `szxd.szzs`: 上涨股票数, `szxd.xdzs`: 下跌股票数
  - `zdfb`: {zt: 涨停, dt: 跌停, z5: 涨>5%, d5: 跌>5%}
- `data.market_volume` - 两市成交额
  - `total`: 两市总成交(字符串含"亿"), `cmp_yesterday`: 较昨日变化
- `data.main_capital_flow_summary` - 主力净流入(net_value, 分时data数组很大可忽略)
- `data.northbound_funds` - 北向资金（注意：非交易时段可能为 null）
- `data.volume_statistics` - 成交量统计
- `data.market_status` - 市场状态(status, state)

**注意**：`main_capital_flow_summary.data` 包含逐分钟数据（240条），数据量很大。如果只需要净流入总额，只取 `net_value` 即可。

### 12. 股票搜索

```
GET /v1/sim/search?query=宁德时代&market=cn
```

| 参数 | 类型 | 说明 |
|------|------|------|
| query | string | 搜索关键字 |
| market | string | cn(A股+京市)/kcb(含科创)/fund(含基金) |

返回 `data.stocks[]`：code, symbol, name, type

### 13. 涨停动态

```
GET /v1/hs/quotes/lu_trends
```

### 14. 赚钱效应

```
GET /v1/pick/limit_up/profit
```

返回涨停家数、封板率、炸板数等

### 15. 涨停/连板/炸板池

```
GET /v1/pick/limit_up/pool?pool=ascLimit&field=change_rate&order=0
```

pool取值：ascLimit(涨停)/descLimit(跌停)/series(连板)/bomb(炸板)

## 代码前缀规则

| 前缀 | 市场 | type值 |
|------|------|--------|
| sh | 沪市 | 1(指数)/2(A股) |
| sz | 深市 | 2(A股)/22(科创板) |
| bj | 北交所 | 8(股票)/81(指数) |
| bk | 板块 | 1111 |
| ix | 指数通 | 11 |

## 使用方式

1. 调用 `read_skill("chinalin-market-api")` 获取本文档，了解可用 API 端点及参数
2. 根据任务需求，调用 `chinalin_http_request(endpoint, params)` 发起请求，工具会自动拼接 base_url
3. 解析返回的 JSON 数据，提取所需字段

## 响应自动裁剪

`chinalin_http_request` 会自动裁剪响应中超过阈值的大数组（默认 >20 条），
保留前 3 条 + 摘要，避免分钟级时序数据占用过多 token。

受影响的典型字段：
- `/v2/market/overview` 的 `main_capital_flow_summary.data`（~240 条分钟数据）
- `/v2/market/overview` 的 `volume_statistics.*.data`（~240 条分钟成交量）
- 分时 K 线 `/v2/quotes/mkline` 的数据数组

汇总字段（如 `net_value`、`total`）不受影响，始终完整返回。

如需完整数组（如日 K 线做技术分析），可传 `max_array_items=-1` 禁用裁剪：
```
chinalin_http_request(endpoint="/v5/quotes/fqkline", params={...}, max_array_items=-1)
```

## 补充文档

| 文件 | 内容 | 何时阅读 |
|------|------|----------|
| [response-fields.md](response-fields.md) | 各接口详细响应字段说明 | 需要了解具体返回字段含义时 |
| [recipes.md](recipes.md) | 常用采集方案（市场总览、个股深度、板块成分股） | 需要执行典型数据采集任务时 |
