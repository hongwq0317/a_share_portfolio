# ChinaLin API 常用采集方案

本文件提供常见数据采集场景的推荐调用方式。
使用前请先阅读 SKILL.md 了解完整的接口列表和参数说明。

---

## 方案 1：市场总览数据采集

**场景**：获取 A 股市场基础面貌——指数行情、涨跌分布、成交额、北向资金、行业排名。

### 第一步：获取主要指数行情

```
GET /v1/quotes/fields
  codes = sh000001|sz399001|sh000300|sz399006
  fields = code|name|cur_price|change_rate|turnover_num
```

从响应 `data.{code}` 中提取每个指数的名称、现价、涨跌幅、成交额。
注意：`change_rate` 是字符串如 "-1.43%"，`turnover_num` 是字符串如 "1425790202336.20"，需转为数值。

### 第二步：获取市场总览

```
GET /v2/market/overview
```

从响应中提取（分钟级 .data 数组已被自动裁剪，无需关注）：
- **涨跌家数**: `data.distribution.szxd` → szzs(上涨数), xdzs(下跌数)
- **涨停跌停**: `data.distribution.zdfb` → zt(涨停), dt(跌停), z5(涨>5%), d5(跌>5%)
- **两市成交**: `data.market_volume.total`（字符串如 "31295.10亿"，提取数值）
- **成交额较昨日**: `data.market_volume.cmp_yesterday_rate`（如 "-20.13%"）
- **主力净流入**: `data.main_capital_flow_summary.net_value`（字符串如 "-600.994亿"，提取数值）
- **北向资金**: `data.northbound_funds`（非交易时段可能为 null，此时跳过）

### 第三步：获取行业板块排名

```
GET /v1/block/rank
  type = 2         (行业)
  sort_field_id = 1 (按涨跌幅)
  sort_direction = 1 (降序)
  req_num = 15
```

从 `data.blocks[]` 提取 secu_name, change_rate（注意是百分位整数如401=4.01%，需除以100）。
注意：turnover_rate 和 market_value 在当前 API 中常为 0，可忽略。

### 输出 JSON 格式

```json
{
  "indices": {
    "000001": {"name": "上证指数", "price": 4122.68, "change_pct": -1.43, "amount": 1425790202336.20}
  },
  "sentiment": {
    "total_stocks": 5458,
    "up_count": 651,
    "down_count": 4807,
    "limit_up": 85,
    "limit_down": 82,
    "up_gt5pct": 125,
    "down_gt5pct": 130,
    "total_volume_billion": 31295.10,
    "main_net_inflow": -600.99,
    "volume_vs_yesterday": "-20.13%"
  },
  "sector_data": [
    {"名称": "燃气", "今日涨跌幅": 10.26}
  ],
  "northbound": {
    "total_net": 50.5,
    "sh_net": 30.2,
    "sz_net": 20.3,
    "date": "2026-03-03",
    "source": "chinalin"
  }
}
```

**字段说明**：
- `up_gt5pct` / `down_gt5pct`: 从 `distribution.zdfb` 的 `z5` / `d5` 获取
- `main_net_inflow`: 从 `main_capital_flow_summary.net_value` 获取（字符串如"-600.994亿"，提取数值）
- `volume_vs_yesterday`: 从 `market_volume.cmp_yesterday_rate` 获取

---

## 方案 2：个股深度数据采集

**场景**：获取某只股票的行情、K线历史、资金流向。

### 实时行情

```
GET /v1/quotes/stockdetail?code=sz300750&deal=0
```

### 日 K 线（最近 120 天，前复权）

```
GET /v5/quotes/fqkline
  code = sz300750
  ktype = day
  autype = qfq
  end = {今天日期}
  count = 120
```

### 资金流向

```
GET /v1/capital/stock/flowSummaries
  codes = sz300750
  types = 1|2    (主力=超大单+大单)
```

---

## 方案 3：板块成分股筛选

**场景**：获取某个行业板块的成分股及其表现。

### 获取板块代码

先通过板块排名接口找到目标板块的 code：

```
GET /v1/block/rank?type=2&req_num=50
```

### 获取成分股

```
GET /v1/block/members/rank
  code = {板块code}
  sort_field_id = 1
  sort_direction = 1
  req_num = 50
```

### 获取热点股（按主力净流入）

```
GET /v5/block/members/hotRank?code={板块code}&top=10
```
