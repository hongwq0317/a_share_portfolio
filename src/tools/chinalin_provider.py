"""ChinaLin 内部行情 API 数据提供层

基于 ChinaLin 内部行情系统，提供个股/指数/板块实时行情、K线历史、
板块排名、资金流向、市场总览等数据。

Base URL 通过 portfolio_config.yaml 中 data_source.chinalin_base_url 配置。
"""

import logging
import re
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger("portfolio")

_BASE_URL = "https://chinalintest.wenxingonline.com"

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
})

_TIMEOUT = 10

# ── 缓存 ──────────────────────────────────────────────────────
_market_cache: Optional[pd.DataFrame] = None
_market_cache_ts: float = 0
_MARKET_CACHE_TTL = 120
_market_lock = threading.Lock()


def set_base_url(url: str):
    """设置 ChinaLin API 基础地址。"""
    global _BASE_URL
    _BASE_URL = url.rstrip("/")


def _safe_float(val: Any) -> float:
    if val is None or val == "" or val == "--":
        return 0.0
    if isinstance(val, str):
        val = val.replace("%", "").replace("亿", "").replace("万", "").replace("+", "").strip()
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_cn_amount(val: str) -> float:
    """解析中文金额字符串为浮点数（元）。如 '37.33亿' → 3733000000.0"""
    if not val or val == "--":
        return 0.0
    val = val.replace("+", "").replace("-", "", 1 if val.startswith("-") else 0).strip()
    negative = val.startswith("-") or (isinstance(val, str) and "-" in val[:2])
    val_clean = val.lstrip("-").strip()
    multiplier = 1.0
    if "亿" in val_clean:
        multiplier = 1e8
        val_clean = val_clean.replace("亿", "")
    elif "万" in val_clean:
        multiplier = 1e4
        val_clean = val_clean.replace("万", "")
    try:
        result = float(val_clean) * multiplier
        return -result if negative else result
    except (ValueError, TypeError):
        return 0.0


def _code_with_prefix(code: str) -> str:
    """纯数字代码 → 带交易所前缀（sh/sz/bj）。"""
    code = code.strip()
    if code.startswith(("sh", "sz", "bj", "bk", "ix")):
        return code
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "1", "2", "3")):
        return f"sz{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return f"sz{code}"


def _pure_code(code: str) -> str:
    """带前缀代码 → 纯数字。"""
    return re.sub(r"^(sh|sz|bj|bk|ix)", "", code)


# ====================================================================
#  指数实时行情
# ====================================================================
def fetch_index_quote(index_codes: list[str]) -> dict[str, dict]:
    """获取指数实时行情。

    Args:
        index_codes: 指数代码列表，如 ["000001", "000300"]
    Returns:
        {code: {name, price, change, change_pct, volume, amount}}
    """
    prefixed = [_code_with_prefix(c) for c in index_codes]
    codes_str = "|".join(prefixed)
    fields = "code|name|cur_price|change|change_rate|volume_num|turnover_num|pre_close"

    url = f"{_BASE_URL}/v1/quotes/fields"
    try:
        r = _session.get(url, params={"codes": codes_str, "fields": fields}, timeout=_TIMEOUT)
        data = r.json().get("data") or {}
    except Exception as e:
        logger.warning(f"[ChinaLin] 指数行情请求失败: {e}")
        return {}

    result = {}
    for prefixed_code in prefixed:
        pure = _pure_code(prefixed_code)
        item = data.get(prefixed_code) or {}
        if not item:
            continue
        price = _safe_float(item.get("cur_price"))
        pre_close = _safe_float(item.get("pre_close"))
        result[pure] = {
            "name": item.get("name", ""),
            "price": price,
            "change": _safe_float(item.get("change")),
            "change_pct": _safe_float(item.get("change_rate")),
            "volume": _safe_float(item.get("volume_num")),
            "amount": _safe_float(item.get("turnover_num")),
        }
    return result


# ====================================================================
#  个股实时行情
# ====================================================================
def fetch_stock_quote(stock_codes: list[str]) -> dict[str, dict]:
    """获取个股实时行情。

    Args:
        stock_codes: 纯数字代码列表，如 ["000001", "600519"]
    Returns:
        {code: {name, price, change_pct, volume, amount, turnover,
                pe, pb, high, low, open, prev_close}}
    """
    results: dict[str, dict] = {}
    batch_size = 50
    for i in range(0, len(stock_codes), batch_size):
        batch = stock_codes[i:i + batch_size]
        prefixed = [_code_with_prefix(c) for c in batch]
        codes_str = "|".join(prefixed)
        fields = ("code|name|cur_price|change_rate|volume_num|turnover_num|"
                  "turnover_rate|pe|pb|high|low|open|pre_close")

        url = f"{_BASE_URL}/v1/quotes/fields"
        try:
            r = _session.get(url, params={"codes": codes_str, "fields": fields}, timeout=_TIMEOUT)
            data = r.json().get("data") or {}
        except Exception as e:
            logger.warning(f"[ChinaLin] 个股行情批量请求失败: {e}")
            continue

        for code in batch:
            pcode = _code_with_prefix(code)
            item = data.get(pcode) or {}
            if not item:
                continue
            price = _safe_float(item.get("cur_price"))
            if price <= 0:
                continue
            results[code] = {
                "name": item.get("name", ""),
                "price": price,
                "prev_close": _safe_float(item.get("pre_close")),
                "open": _safe_float(item.get("open")),
                "high": _safe_float(item.get("high")),
                "low": _safe_float(item.get("low")),
                "change_pct": _safe_float(item.get("change_rate")),
                "volume": _safe_float(item.get("volume_num")),
                "amount": _safe_float(item.get("turnover_num")),
                "turnover": _safe_float(item.get("turnover_rate")),
                "pe": _safe_float(item.get("pe")),
                "pb": _safe_float(item.get("pb")),
            }
    return results


# ====================================================================
#  全市场行情（通过 list/stockdetail 批量获取）
# ====================================================================
def fetch_all_stocks(use_cache: bool = True) -> pd.DataFrame:
    """获取全市场行情概览（通过板块成分股聚合）。

    通过查询主要指数成分股和板块获取市场全貌。
    返回 DataFrame 含: 代码, 名称, 最新价, 涨跌幅, 成交量, 成交额, 换手率, 市盈率-动态, 市净率, 总市值
    """
    global _market_cache, _market_cache_ts
    now = time.time()
    if use_cache:
        with _market_lock:
            if _market_cache is not None and (now - _market_cache_ts) < _MARKET_CACHE_TTL:
                return _market_cache

    logger.info("[ChinaLin] 获取全市场行情（成分股聚合）...")

    all_stocks: dict[str, dict] = {}
    index_codes = ["sh000001", "sz399001", "sz399006", "sh000688", "bj899050"]

    for idx_code in index_codes:
        try:
            _fetch_block_members_all(idx_code, all_stocks)
        except Exception as e:
            logger.debug(f"[ChinaLin] 获取{idx_code}成分股失败: {e}")

    if not all_stocks:
        logger.warning("[ChinaLin] 全市场行情获取失败")
        return pd.DataFrame()

    rows = list(all_stocks.values())
    df = pd.DataFrame(rows)

    with _market_lock:
        _market_cache = df
        _market_cache_ts = time.time()

    logger.debug(f"[ChinaLin] 全市场行情已刷新 ({len(df)} 只)")
    return df


def _fetch_block_members_all(index_code: str, result: dict[str, dict]):
    """获取指数全部成分股并添加到 result 字典。"""
    req_seq = 0
    req_num = 200
    while True:
        url = f"{_BASE_URL}/v1/block/members/rank"
        params = {
            "code": index_code,
            "sort_field_id": 0,
            "sort_direction": 0,
            "req_seq": req_seq,
            "req_num": req_num,
        }
        try:
            r = _session.get(url, params=params, timeout=15)
            data = r.json().get("data") or {}
        except Exception as e:
            logger.debug(f"[ChinaLin] 成分股分页 {index_code} seq={req_seq} 失败: {e}")
            break

        members = data.get("block_members") or []
        if not members:
            break

        for m in members:
            code = _pure_code(m.get("code", ""))
            if code in result or not code:
                continue
            price = _safe_float(m.get("last_price"))
            if price <= 0:
                continue
            result[code] = {
                "代码": code,
                "名称": m.get("secu_name", ""),
                "最新价": price,
                "涨跌幅": _safe_float(m.get("change_rate")),
                "换手率": _safe_float(m.get("turnover_rate")),
                "总市值": _safe_float(m.get("market_value")),
                "成交额": _safe_float(m.get("float_market_value")),
            }

        total = data.get("total_members", 0)
        req_seq += len(members)
        if req_seq >= total or len(members) < req_num:
            break


# ====================================================================
#  历史 K 线（日/周/月/年）
# ====================================================================
def fetch_stock_history(
    stock_code: str,
    days: int = 120,
    adjust: str = "qfq",
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """获取个股历史日K线。

    Returns:
        DataFrame(日期, 开盘, 收盘, 最高, 最低, 成交量)
    """
    prefixed = _code_with_prefix(stock_code)
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

    url = f"{_BASE_URL}/v5/quotes/fqkline"
    params = {
        "code": prefixed,
        "ktype": "day",
        "autype": adjust or "qfq",
        "start": start_date,
        "end": end_date,
        "count": days + 50,
    }

    try:
        r = _session.get(url, params=params, timeout=15)
        resp = r.json()
    except Exception as e:
        logger.warning(f"[ChinaLin] K线请求失败 {stock_code}: {e}")
        return pd.DataFrame()

    data = resp.get("data") or {}
    label = data.get("label") or []
    klines = data.get("day") or data.get("week") or data.get("month") or data.get("year") or []

    if not klines or not label:
        return pd.DataFrame()

    idx_map = {name: i for i, name in enumerate(label)}
    rows = []
    for k in klines:
        if len(k) < 6:
            continue
        rows.append({
            "日期": k[idx_map.get("date", 0)],
            "开盘": float(k[idx_map.get("open_price", 1)]),
            "收盘": float(k[idx_map.get("close_price", 2)]),
            "最高": float(k[idx_map.get("high_price", 3)]),
            "最低": float(k[idx_map.get("low_price", 4)]),
            "成交量": float(k[idx_map.get("trade_volume", 5)]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if start_date:
        df = df[df["日期"] >= start_date]
    if end_date:
        df = df[df["日期"] <= end_date]
    if len(df) > days and not start_date:
        df = df.tail(days)
    return df.reset_index(drop=True)


# ====================================================================
#  指数历史 K 线
# ====================================================================
def fetch_index_history(
    index_code: str = "000300",
    days: int = 252,
) -> pd.DataFrame:
    """获取指数历史日K线。"""
    return fetch_stock_history(index_code, days=days, adjust="", start_date="", end_date="")


# ====================================================================
#  行业板块排名/表现
# ====================================================================
def fetch_sector_performance() -> pd.DataFrame:
    """获取行业板块排名（行业类型=2）。

    Returns:
        DataFrame(行业, 涨跌幅, 换手率, 总市值, ...)
    """
    url = f"{_BASE_URL}/v1/block/rank"
    params = {
        "type": 2,
        "sort_field_id": 1,
        "sort_direction": 1,
        "req_seq": 0,
        "req_num": 100,
    }
    try:
        r = _session.get(url, params=params, timeout=_TIMEOUT)
        data = r.json().get("data") or {}
    except Exception as e:
        logger.warning(f"[ChinaLin] 板块排名请求失败: {e}")
        return pd.DataFrame()

    blocks = data.get("blocks") or []
    if not blocks:
        return pd.DataFrame()

    rows = []
    for b in blocks:
        rows.append({
            "行业": b.get("secu_name", ""),
            "行业代码": b.get("code", ""),
            "平均涨跌幅": _safe_float(b.get("change_rate")),
            "换手率": _safe_float(b.get("turnover_rate")),
            "总市值": _safe_float(b.get("market_value")),
            "流通市值": _safe_float(b.get("float_market_value")),
        })

    return pd.DataFrame(rows)


# ====================================================================
#  市场总览（含北向资金、成交额、涨跌分布）
# ====================================================================
def fetch_market_overview() -> dict:
    """获取市场总览数据（V2接口）。

    Returns:
        dict 包含 distribution, market_volume, main_capital_flow,
        north_capital, market_status 等
    """
    url = f"{_BASE_URL}/v2/market/overview"
    try:
        r = _session.get(url, timeout=_TIMEOUT)
        data = r.json().get("data") or {}
        return data
    except Exception as e:
        logger.warning(f"[ChinaLin] 市场总览请求失败: {e}")
        return {}


def fetch_northbound_flow() -> dict:
    """获取北向资金流向。

    Returns:
        {total_net, sh_net, sz_net, date, source}，单位：亿元
    """
    overview = fetch_market_overview()
    if not overview:
        return {}

    north = overview.get("north_capital") or {}
    latest = north.get("latest_kline") or {}

    if not latest:
        return {}

    hsgt = _safe_float(latest.get("hsgt"))
    hgt = _safe_float(latest.get("hgt"))
    sgt = _safe_float(latest.get("sgt"))

    return {
        "total_net": round(hsgt, 2),
        "sh_net": round(hgt, 2),
        "sz_net": round(sgt, 2),
        "date": latest.get("day", ""),
        "source": "chinalin-market-overview",
    }


# ====================================================================
#  个股资金流向
# ====================================================================
def fetch_stock_fund_flow(stock_code: str, days: int = 5) -> pd.DataFrame:
    """获取个股资金净流入汇总。

    Returns:
        DataFrame(日期, 主力净流入_万, ...) 或单行净流入数据
    """
    prefixed = _code_with_prefix(stock_code)
    url = f"{_BASE_URL}/v1/capital/stock/flowSummaries"
    params = {"codes": prefixed}
    try:
        r = _session.get(url, params=params, timeout=_TIMEOUT)
        data = r.json().get("data") or {}
    except Exception as e:
        logger.debug(f"[ChinaLin] 资金流向请求失败 {stock_code}: {e}")
        return pd.DataFrame()

    item = data.get(prefixed) or {}
    if not item:
        return pd.DataFrame()

    net = _safe_float(item.get("net", 0))
    date = item.get("date", "")
    return pd.DataFrame([{
        "日期": date,
        "主力净流入_万": round(net / 1e4, 1),
    }])


def fetch_stock_fund_flow_detail(stock_code: str) -> pd.DataFrame:
    """获取个股大单动向分时数据。"""
    prefixed = _code_with_prefix(stock_code)
    url = f"{_BASE_URL}/v1/capital/stock/flow"
    params = {"code": prefixed, "ktype": "minute"}
    try:
        r = _session.get(url, params=params, timeout=_TIMEOUT)
        data = r.json().get("data") or {}
    except Exception as e:
        logger.debug(f"[ChinaLin] 大单动向请求失败 {stock_code}: {e}")
        return pd.DataFrame()

    flows = data.get("capital_flows") or []
    if not flows:
        return pd.DataFrame()

    rows = []
    for f in flows:
        if len(f) < 5:
            continue
        rows.append({
            "时间": f[0],
            "特大单": _safe_float(f[1]),
            "大单": _safe_float(f[2]),
            "中单": _safe_float(f[3]),
            "小单": _safe_float(f[4]),
        })
    return pd.DataFrame(rows)


# ====================================================================
#  板块热点股票（主力净流入排名）
# ====================================================================
def fetch_block_hot_rank(block_code: str = "sh000001", top: int = 10) -> pd.DataFrame:
    """获取板块/指数主力净流入前N股票。"""
    url = f"{_BASE_URL}/v5/block/members/hotRank"
    params = {"code": block_code, "top": top}
    try:
        r = _session.get(url, params=params, timeout=_TIMEOUT)
        data = r.json().get("data") or []
    except Exception as e:
        logger.debug(f"[ChinaLin] 热点股票请求失败: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        rows.append({
            "日期": item.get("trade_date", ""),
            "代码": item.get("code", ""),
            "名称": item.get("name", ""),
            "最新价": _safe_float(item.get("price")),
            "涨跌幅": _safe_float(item.get("change_rate")),
            "主力净流入": item.get("main_net_in", ""),
            "成交量": item.get("volume", ""),
            "换手率": _safe_float(item.get("turnover_rate")),
        })
    return pd.DataFrame(rows)


# ====================================================================
#  股票搜索
# ====================================================================
def search_stock(query: str, market: str = "cn") -> list[dict]:
    """搜索股票。"""
    url = f"{_BASE_URL}/v1/sim/search"
    params = {"query": query, "market": market}
    try:
        r = _session.get(url, params=params, timeout=_TIMEOUT)
        data = r.json().get("data") or {}
        return data.get("stocks") or []
    except Exception as e:
        logger.debug(f"[ChinaLin] 搜索请求失败: {e}")
        return []
