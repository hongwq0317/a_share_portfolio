# ChinaLin API 响应字段参考

## stockdetail / list 行情详情字段 (data.detail)

| 字段 | 说明 | 示例 |
|------|------|------|
| code | 带前缀证券代码 | sz300750 |
| symbol | 纯数字代码 | 300750 |
| name | 名称 | 宁德时代 |
| type | 证券类型 | 2(A股) |
| cur_price | 当前价 | 247.14 |
| pre_close | 昨收价 | 246.49 |
| open | 开盘价 | 245.98 |
| high | 最高价 | 249.33 |
| low | 最低价 | 244.88 |
| change | 涨跌额 | +0.65 |
| change_rate | 涨跌幅 | +0.26% |
| volume | 成交量(格式化) | 15.07万 |
| volume_num | 成交量(原始) | 15066676 |
| turnover | 成交额(格式化) | 37.33亿 |
| turnover_num | 成交额(原始) | 3733050028.84 |
| turnover_rate | 换手率 | 0.39% |
| pe | 市盈率(动态) | 26.22 |
| pe_ttm | 动态市盈率 | 25.10 |
| pe_sta | 静态市盈率 | 35.36 |
| pb | 市净率 | 6.38 |
| eps | 每股收益 | 4.712 |
| market_value | 总市值(格式化) | 10865亿 |
| mv | 总市值(原始) | 1086499835956 |
| circulation_market_value | 流通市值 | 9596亿 |
| cmv | 流通市值(原始) | 959630936847 |
| total_shares | 总股本 | 43.96亿 |
| float_shares | 流通股本 | 38.83亿 |
| volume_ratio | 量比 | 0.00 |
| amplitude | 振幅 | 1.81% |
| bid_ratio | 委比 | -13.19% |
| limit_up | 涨停价 | 295.79 |
| limit_down | 跌停价 | 197.19 |
| high_52 | 52周最高 | 249.33 |
| low_52 | 52周最低 | 244.88 |
| avg_price | 均价 | 247.77 |
| state | 交易状态 | 盘后交易 |
| time | 时间 | 2023-08-08 15:18:00 |
| bid_buy | 委买 | 5档委托买盘总数 |
| bid_sell | 委卖 | 5档委托卖盘总数 |
| dividend_ps | 股息(TTM) | 每股派息总额 |
| dividend_ps_rate | 股息率(TTM) | 股息/最新价 |
| free_float_shares | 自由流通股本 | 38.83亿 |
| free_float_market | 自由流通值 | 10865亿 |
| real_turnover_rate | 实际换手率 | 0.39% |

## K线 label 字段顺序

```
date, open_price, close_price, high_price, low_price,
trade_volume, trade_value, turnover_rate, volume_ratio,
dividends, change_price, change_rate, amplitude,
turnover_deals, bid_ratio, float_shares, total_shares,
float_market_value, market_value, eps, pe, pe_static,
pe_ttm, pb, dps, roe, limit_up_price, limit_down_price,
high_52_price, low_52_price
```

## 板块排名字段 (blocks[])

| 字段 | 说明 |
|------|------|
| code | 带前缀板块代码 |
| secu_code | 证券代码 |
| secu_name | 板块名称 |
| last_price | 最新价 |
| change_price | 涨跌额 |
| change_rate | 涨跌幅 |
| pre_close_price | 昨收 |
| float_market_value | 流通市值 |
| market_value | 总市值 |
| turnover_rate | 换手率 |

## 市场总览 V2 字段

### distribution (涨跌分布)
- szxd: {szzs: 上涨数, xdzs: 下跌数}
- zdfb: {zt: 涨停, dt: 跌停, z5: 涨>5%, d5: 跌>5%, z25: 涨2-5%, d52: 跌2-5%, z02: 涨0-2%, d20: 跌0-2%, pnum: 平盘}

### northbound_funds (北向资金)
- 非交易时段可能返回 null
- 有数据时格式: daily_kline[]: {day, hsgt(沪深股通总成交亿), hgt(沪港通), sgt(深股通), sh(上证), sz(深证), hs300}
- latest_kline: 同上，最新一条

### market_volume (成交额)
- total: 两市总成交
- cmp_yesterday: 较昨日变化

### main_capital_flow_summary (主力净流入)
- net_value: 主力净流入总额
- data[]: {time, value} 分时数据
