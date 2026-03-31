"""A股市场数据提供层

多源回退架构：

数据源分工（按优先级）：
  - 新浪 hq.sinajs.cn                → 指数实时行情 ⭐⭐⭐⭐
  - 腾讯 qt.gtimg.cn                  → 个股实时行情 ⭐⭐⭐⭐
  - 腾讯 web.ifzq.gtimg.cn            → 历史 K 线 ⭐⭐⭐
  - Baostock                          → K线/财务/行业 ⭐⭐⭐
  - 东方财富 datacenter-web            → 北向资金/财务报表 ⭐⭐⭐
  - 东方财富 push2/push2his            → 资金流向 ⭐⭐⭐

注意：ChinaLin 内部 API 现通过 Skill 驱动方式由 LLM 动态调用，
不再在此模块中直接集成。详见 src/tools/skill_tools.py。
"""

import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger("portfolio")

# ── HTTP Session ──────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
})

_SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": _session.headers["User-Agent"],
}

# ── 全市场行情缓存 ────────────────────────────────────────────
_market_cache: Optional[pd.DataFrame] = None
_market_cache_ts: float = 0
_MARKET_CACHE_TTL = 120
_market_lock = threading.Lock()


# ====================================================================
#  指数实时行情（新浪）
# ====================================================================
def fetch_index_quote(index_codes: list[str]) -> dict[str, dict]:
    """获取指数实时行情。

    Args:
        index_codes: 指数代码列表，如 ["000001", "000300"]
    Returns:
        {code: {name, price, change, change_pct, volume, amount}} 或空 dict。
    """
    sina_codes = []
    for code in index_codes:
        prefix = "sh" if code.startswith(("000", "880", "99")) else "sz"
        sina_codes.append(f"{prefix}{code}")

    url = f"https://hq.sinajs.cn/list={','.join(sina_codes)}"
    try:
        r = _session.get(url, headers=_SINA_HEADERS, timeout=10)
        r.encoding = "gbk"
    except Exception as e:
        logger.warning(f"[数据] 新浪指数行情请求失败: {e}")
        return {}

    result = {}
    for line in r.text.strip().split("\n"):
        m = re.search(r'hq_str_(\w+)="(.+)"', line)
        if not m:
            continue
        raw_code = m.group(1)  # e.g. sh000001
        pure_code = raw_code[2:]
        fields = m.group(2).split(",")
        if len(fields) < 10:
            continue
        result[pure_code] = {
            "name": fields[0],
            "price": _safe_float(fields[3]),
            "change": _safe_float(fields[3]) - _safe_float(fields[2]),
            "change_pct": round(
                (_safe_float(fields[3]) / _safe_float(fields[2]) - 1) * 100, 2
            ) if _safe_float(fields[2]) else 0,
            "volume": _safe_float(fields[8]),
            "amount": _safe_float(fields[9]),
        }
    return result


# ====================================================================
#  个股实时行情（腾讯）
# ====================================================================
def fetch_stock_quote(stock_codes: list[str]) -> dict[str, dict]:
    """获取个股实时行情（腾讯数据源）。

    Args:
        stock_codes: 纯数字代码列表，如 ["000001", "600519"]
    Returns:
        {code: {name, price, change_pct, volume, amount, turnover,
                pe, pb, high, low, open, prev_close}}
    """
    tencent_codes = []
    for code in stock_codes:
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        if code.startswith(("4", "8")) and not code.startswith("88"):
            prefix = "bj"
        tencent_codes.append(f"{prefix}{code}")

    results: dict[str, dict] = {}
    # 腾讯 API 每次最多约 60 只
    for i in range(0, len(tencent_codes), 60):
        batch = tencent_codes[i:i + 60]
        url = f"https://qt.gtimg.cn/q={','.join(batch)}"
        try:
            r = _session.get(url, timeout=10)
            r.encoding = "gbk"
        except Exception as e:
            logger.warning(f"[数据] 腾讯行情请求失败: {e}")
            continue

        for line in r.text.strip().split(";"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("~")
            if len(parts) < 50:
                continue
            code = parts[2]
            results[code] = {
                "name": parts[1],
                "price": _safe_float(parts[3]),
                "prev_close": _safe_float(parts[4]),
                "open": _safe_float(parts[5]),
                "volume": _safe_float(parts[6]),
                "buy1": _safe_float(parts[9]),
                "sell1": _safe_float(parts[19]),
                "high": _safe_float(parts[33]),
                "low": _safe_float(parts[34]),
                "change_pct": _safe_float(parts[32]),
                "amount": _safe_float(parts[37]) * 10000,  # 万→元
                "turnover": _safe_float(parts[38]),
                "pe": _safe_float(parts[39]),
                "pb": _safe_float(parts[46]),
            }
    return results


# ====================================================================
#  全市场行情（腾讯批量 + 并发，~1 秒获取 6000+ 只）
# ====================================================================

# A 股代码段预生成（覆盖所有交易所）
def _generate_all_a_codes() -> list[str]:
    """生成所有 A 股可能的代码列表（带交易所前缀）。"""
    codes: list[str] = []
    for i in range(1, 5000):         # SZ 主板 000001-004999
        codes.append(f"sz{i:06d}")
    for i in range(300000, 302000):   # SZ 创业板
        codes.append(f"sz{i:06d}")
    for i in range(600000, 606000):   # SH 主板
        codes.append(f"sh{i:06d}")
    for i in range(688000, 689500):   # SH 科创板
        codes.append(f"sh{i:06d}")
    for start, end in [(430000, 440000), (830000, 840000),
                       (870000, 875000), (920000, 921000)]:
        for i in range(start, end):   # BJ 北交所
            codes.append(f"bj{i:06d}")
    return codes


_ALL_CODES = _generate_all_a_codes()


def fetch_all_stocks(use_cache: bool = True) -> pd.DataFrame:
    """获取全市场 A 股实时行情（腾讯数据源），带 TTL 缓存。

    Returns:
        DataFrame 含字段: 代码, 名称, 最新价, 涨跌幅, 成交量, 成交额,
                         换手率, 市盈率-动态, 市净率, 总市值
    """
    global _market_cache, _market_cache_ts
    now = time.time()
    if use_cache:
        with _market_lock:
            if _market_cache is not None and (now - _market_cache_ts) < _MARKET_CACHE_TTL:
                return _market_cache

    batch_size = 800
    batches = [_ALL_CODES[i:i + batch_size]
               for i in range(0, len(_ALL_CODES), batch_size)]

    def _fetch_batch(codes: list[str]) -> list[dict]:
        url = f"https://qt.gtimg.cn/q={','.join(codes)}"
        resp = _session.get(url, timeout=20)
        resp.encoding = "gbk"
        items: list[dict] = []
        for line in resp.text.strip().split(";"):
            parts = line.strip().split("~")
            if len(parts) < 50:
                continue
            price = _safe_float(parts[3])
            if price <= 0:
                continue
            items.append({
                "代码": parts[2],
                "名称": parts[1],
                "最新价": price,
                "昨收": _safe_float(parts[4]),
                "今开": _safe_float(parts[5]),
                "成交量": _safe_float(parts[6]),
                "最高": _safe_float(parts[33]),
                "最低": _safe_float(parts[34]),
                "涨跌幅": _safe_float(parts[32]),
                "成交额": _safe_float(parts[37]) * 10000,
                "换手率": _safe_float(parts[38]),
                "市盈率-动态": _safe_float(parts[39]),
                "市净率": _safe_float(parts[46]),
                "总市值": _safe_float(parts[45]) * 10000,
            })
        return items

    all_stocks: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_fetch_batch, b) for b in batches]
        for f in as_completed(futures):
            try:
                all_stocks.extend(f.result(timeout=30))
            except Exception as e:
                logger.debug(f"[数据] 腾讯批量查询失败: {e}")

    if not all_stocks:
        logger.warning("[数据] 全市场行情获取失败，返回空 DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(all_stocks)

    with _market_lock:
        _market_cache = df
        _market_cache_ts = time.time()

    logger.debug(f"[缓存] 全市场行情已刷新 ({len(df)} 只股票)")
    return df


# ====================================================================
#  历史 K 线（腾讯）
# ====================================================================
def fetch_stock_history(
    stock_code: str,
    days: int = 120,
    adjust: str = "qfq",
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """获取个股历史日 K 线。

    Args:
        stock_code: 纯数字代码
        days: 获取最近 N 天（start_date/end_date 未指定时使用）
        adjust: 复权类型 qfq/hfq/空串
        start_date: 起始日期 YYYY-MM-DD（可选，指定后 days 参数仅用于接口请求量）
        end_date: 结束日期 YYYY-MM-DD（可选，默认今天）
    Returns:
        DataFrame(日期, 开盘, 收盘, 最高, 最低, 成交量)
    """
    prefix = "sh" if stock_code.startswith(("6", "9")) else "sz"
    tc_code = f"{prefix}{stock_code}"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
    kline_type = "qfqday" if adjust == "qfq" else ("hfqday" if adjust == "hfq" else "day")

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{tc_code},day,{start_date},{end_date},{days + 50},{adjust}"}

    try:
        r = _session.get(url, params=params, timeout=15)
        data = r.json()
    except Exception as e:
        logger.warning(f"[数据] 腾讯K线请求失败 {stock_code}: {e}")
        return _fallback_stock_history(stock_code, days, adjust, start_date, end_date)

    stock_data = (data.get("data") or {}).get(tc_code) or {}
    klines = stock_data.get(kline_type) or stock_data.get("day") or []
    if not klines:
        return _fallback_stock_history(stock_code, days, adjust, start_date, end_date)

    rows = []
    for k in klines:
        if len(k) >= 6:
            rows.append({
                "日期": k[0],
                "开盘": float(k[1]),
                "收盘": float(k[2]),
                "最高": float(k[3]),
                "最低": float(k[4]),
                "成交量": float(k[5]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return _fallback_stock_history(stock_code, days, adjust, start_date, end_date)
    if start_date:
        df = df[df["日期"] >= start_date]
    if end_date:
        df = df[df["日期"] <= end_date]
    if len(df) > days and not start_date:
        df = df.tail(days)
    return df.reset_index(drop=True)


def _fallback_stock_history(
    stock_code: str, days: int, adjust: str, start_date: str, end_date: str,
) -> pd.DataFrame:
    """K线回退: Baostock（直连服务器，极其稳定）。"""
    try:
        from src.tools.baostock_provider import bs_fetch_stock_history
        logger.info(f"[数据] 腾讯K线不可用, 回退 Baostock → {stock_code}")
        return bs_fetch_stock_history(stock_code, days, adjust, start_date, end_date)
    except Exception as e:
        logger.warning(f"[数据] Baostock K线也失败 {stock_code}: {e}")
        return pd.DataFrame()


# ====================================================================
#  指数历史 K 线（腾讯）
# ====================================================================
def fetch_index_history(
    index_code: str = "000300",
    days: int = 252,
) -> pd.DataFrame:
    """获取指数历史日 K 线。"""
    prefix = "sh" if index_code.startswith(("000", "880", "99")) else "sz"
    tc_code = f"{prefix}{index_code}"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{tc_code},day,{start_date},{end_date},{days + 50},"}

    try:
        r = _session.get(url, params=params, timeout=15)
        data = r.json()
    except Exception as e:
        logger.warning(f"[数据] 腾讯指数K线请求失败 {index_code}: {e}")
        return _fallback_index_history(index_code, days)

    stock_data = (data.get("data") or {}).get(tc_code) or {}
    klines = stock_data.get("day") or stock_data.get("qfqday") or []
    if not klines:
        return _fallback_index_history(index_code, days)

    rows = []
    for k in klines:
        if len(k) >= 6:
            rows.append({
                "日期": k[0],
                "开盘": float(k[1]),
                "收盘": float(k[2]),
                "最高": float(k[3]),
                "最低": float(k[4]),
                "成交量": float(k[5]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return _fallback_index_history(index_code, days)
    if len(df) > days:
        df = df.tail(days).reset_index(drop=True)
    return df


def _fallback_index_history(index_code: str, days: int) -> pd.DataFrame:
    """指数K线回退: Baostock。"""
    try:
        from src.tools.baostock_provider import bs_fetch_index_history
        logger.info(f"[数据] 腾讯指数K线不可用, 回退 Baostock → {index_code}")
        return bs_fetch_index_history(index_code, days)
    except Exception as e:
        logger.warning(f"[数据] Baostock 指数K线也失败 {index_code}: {e}")
        return pd.DataFrame()


# ====================================================================
#  个股行业分类（Chinalin 内部 API，申万二级行业）
# ====================================================================

_STOCK_DETAIL_URL = "https://htai-test.chinalin.com/ai/data/getStockDetail"

_industry_cache: dict[str, str] = {}
_industry_cache_ts: float = 0
_INDUSTRY_CACHE_TTL = 86400  # 24h


def fetch_stock_industry(stock_code: str) -> str:
    """获取单只股票的申万行业分类（Chinalin API → 关键词回退）。

    返回申万二级行业名称（如"白酒Ⅱ"、"光伏设备"），缓存 24h。
    获取失败返回空字符串，由调用方回退到关键词匹配。
    """
    global _industry_cache, _industry_cache_ts

    now = time.time()
    if now - _industry_cache_ts > _INDUSTRY_CACHE_TTL:
        _industry_cache.clear()
        _industry_cache_ts = now

    if stock_code in _industry_cache:
        return _industry_cache[stock_code]

    pure_code = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")

    # 方案1: Chinalin API（短超时避免阻塞）
    try:
        resp = _session.post(
            _STOCK_DETAIL_URL,
            json={"stockCode": pure_code},
            timeout=3,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                industry = data["data"].get("secondIndustryName", "")
                if industry:
                    _industry_cache[stock_code] = industry
                    return industry
    except Exception:
        pass

    # 方案2: 东方财富个股信息 API
    industry = _fetch_industry_from_em(pure_code)
    if industry:
        _industry_cache[stock_code] = industry
        return industry

    # 方案3: Baostock 行业分类（证监会分类，兜底）
    try:
        from src.tools.baostock_provider import bs_fetch_industry
        industry = bs_fetch_industry(stock_code)
        if industry:
            _industry_cache[stock_code] = industry
            return industry
    except Exception:
        pass

    return ""


def _fetch_industry_from_em(stock_code: str) -> str:
    """通过东方财富 F10 接口获取行业分类。"""
    market = "1" if stock_code.startswith(("6", "9")) else "0"
    secid = f"{market}.{stock_code}"
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": secid,
        "fields": "f57,f58,f127,f128,f129,f130,f131,f132",
    }
    try:
        r = _session.get(url, params=params, headers=_EM_HEADERS, timeout=5)
        data = r.json().get("data") or {}
        industry = data.get("f127", "")
        if industry and industry != "-":
            return industry
    except Exception:
        pass
    return ""


def fetch_batch_stock_industry(stock_codes: list[str]) -> dict[str, str]:
    """批量获取股票行业分类。

    缓存命中的直接返回。
    返回 {stock_code: industry_name}，获取失败的不在结果中。
    """
    result = {}
    codes_to_fetch = []

    for code in stock_codes:
        cached = _industry_cache.get(code)
        if cached:
            result[code] = cached
        else:
            codes_to_fetch.append(code)

    for code in codes_to_fetch:
        industry = fetch_stock_industry(code)
        if industry:
            result[code] = industry

    return result


# ====================================================================
#  行业板块表现（基于腾讯全市场数据计算，无外部依赖）
# ====================================================================

SW_SECTOR_KEYWORDS: dict[str, list[str]] = {
    # ---------- 金融 ----------
    "银行": ["银行"],
    "非银金融": ["证券", "保险", "期货", "信托", "中信建投", "东方财富",
                "中国人寿", "中国平安", "新华保险", "中国太保", "国泰君安"],
    # ---------- 消费 ----------
    "食品饮料": ["酒", "食品", "饮料", "乳业", "伊利", "蒙牛", "海天", "茅台", "五粮液"],
    "家用电器": ["电器", "美的", "格力", "海尔", "海信", "奥克斯", "苏泊尔", "小熊"],
    "商贸零售": ["百货", "商业", "零售", "超市", "永辉", "苏宁", "天虹", "王府井"],
    "社会服务": ["旅游", "酒店", "餐饮", "教育", "体育", "物业", "家政"],
    "纺织服饰": ["纺织", "服装", "服饰", "鞋", "皮革", "家纺", "安踏", "李宁"],
    "美容护理": ["化妆品", "护肤", "美容", "日化", "珀莱雅", "贝泰妮"],
    "轻工制造": ["造纸", "包装", "印刷", "家具", "文具", "晨光"],
    "农林牧渔": ["农业", "牧业", "渔业", "种业", "饲料", "养殖", "猪", "鸡", "牧原", "温氏"],
    # ---------- 科技/制造 ----------
    "电子": ["电子", "芯片", "半导体", "集成电路", "光电", "显示", "面板",
            "封测", "LED", "传感", "PCB"],
    "计算机": ["软件", "信息", "计算机", "数据", "科技", "云计算", "网络安全",
              "人工智能", "AI", "算力", "服务器", "GPU", "旭创", "光模块"],
    "通信": ["通信", "中兴", "光纤", "光缆", "5G", "物联网", "卫星"],
    "传媒": ["传媒", "文化", "影视", "游戏", "出版", "广告", "营销"],
    "电力设备": ["电力设备", "电气", "新能源", "光伏", "风电", "储能", "锂电",
                "电池", "逆变器", "充电", "特高压", "变压器", "宁德", "隆基", "绿能"],
    "汽车": ["汽车", "整车", "零部件", "比亚迪", "长城", "长安", "吉利"],
    "机械设备": ["机械", "设备", "自动化", "机器人", "工控", "数控", "激光",
                "工程机械", "叉车", "起重"],
    "国防军工": ["军工", "航天", "兵器", "船舶", "中航", "导弹", "雷达", "卫星导航"],
    # ---------- 周期 ----------
    "基础化工": ["化工", "化学", "材料", "石化", "氟", "磷", "钾", "氯", "纯碱",
                "钛白粉", "MDI", "聚氨酯"],
    "钢铁": ["钢铁", "钢", "冶金", "特钢", "不锈钢"],
    "有色金属": ["有色", "铜", "铝", "锌", "镍", "锡", "铅", "钴", "钨", "钼", "钛",
                "锂", "稀土", "黄金", "白银", "矿业", "冶炼", "磁材", "永磁"],
    "建筑材料": ["水泥", "玻璃", "陶瓷", "建材", "防水", "管材"],
    "建筑装饰": ["建筑", "建设", "工程", "装饰", "设计院", "中铁", "中建"],
    "煤炭": ["煤炭", "煤业", "焦煤", "焦炭", "煤气", "神华", "中煤"],
    "石油石化": ["石油", "石化", "油服", "油气", "中石油", "中石化", "中海油"],
    # ---------- 交运/公用 ----------
    "交通运输": ["航空", "航运", "物流", "快递", "港口", "铁路", "公路", "机场",
                "高速", "顺丰", "中远"],
    "公用事业": ["电投", "水务", "燃气", "热力", "环保", "水电", "核电", "火电",
                "垃圾", "污水"],
    # ---------- 地产 ----------
    "房地产": ["地产", "置业", "置地", "万科", "保利发展", "招商蛇口", "金地", "中海"],
    # ---------- 医药 ----------
    "医药生物": ["医药", "药业", "制药", "生物", "医疗", "健康", "疫苗", "器械",
                "CXO", "创新药", "中药", "恒瑞", "迈瑞"],
    # ---------- 综合 ----------
    "综合": ["综合", "控股", "集团", "实业"],
}


def fetch_sector_performance() -> pd.DataFrame:
    """从全市场行情计算行业板块表现与资金流向近似值。

    资金流向近似算法: 对每只股票，按 (涨跌幅/100) * 成交额 估算主力方向性资金，
    聚合到行业后得到 "净流入估算"。正值表示资金倾向流入，负值表示流出。

    Returns:
        DataFrame(行业, 股票数, 平均涨跌幅, 上涨比例%, 领涨股, 领涨涨幅,
                  总成交额_亿, 净流入估算_亿)
    """
    logger.info("[数据] 计算行业板块表现+资金流向（腾讯全市场数据源）...")
    df = fetch_all_stocks()
    if df.empty:
        logger.warning("[数据] 腾讯全市场数据为空，无法计算行业表现")
        return pd.DataFrame()
    logger.debug(f"[数据] 全市场共 {len(df)} 只股票，开始按行业关键词匹配...")

    results = []
    for sector, keywords in SW_SECTOR_KEYWORDS.items():
        pattern = "|".join(keywords)
        matched = df[df["名称"].str.contains(pattern, na=False)]
        if len(matched) < 3:
            continue

        avg_chg = matched["涨跌幅"].mean()
        up_ratio = (matched["涨跌幅"] > 0).sum() / len(matched) * 100
        total_amount = matched["成交额"].sum() / 1e8
        top = matched.nlargest(1, "涨跌幅").iloc[0]

        net_flow_est = (matched["涨跌幅"] / 100 * matched["成交额"]).sum() / 1e8

        results.append({
            "行业": sector,
            "股票数": len(matched),
            "平均涨跌幅": round(avg_chg, 2),
            "上涨比例%": round(up_ratio, 1),
            "领涨股": f"{top['名称']}({top['代码']})",
            "领涨涨幅": round(top["涨跌幅"], 2),
            "总成交额_亿": round(total_amount, 1),
            "净流入估算_亿": round(net_flow_est, 2),
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("净流入估算_亿", ascending=False).reset_index(drop=True)
        top3 = [f"{r['行业']}({r['净流入估算_亿']:+.1f}亿)"
                for _, r in result_df.head(3).iterrows()]
        bot3 = [f"{r['行业']}({r['净流入估算_亿']:+.1f}亿)"
                for _, r in result_df.tail(3).iterrows()]
        logger.info(f"[数据] 行业板块计算完成: {len(result_df)}个行业, "
                     f"资金流入前3: {', '.join(top3)}, "
                     f"流出前3: {', '.join(bot3)}")
    else:
        logger.warning("[数据] 行业板块计算结果为空")
    return result_df


# ====================================================================
#  北向资金（东方财富直连 + 新浪回退，不走 akshare）
# ====================================================================

_EM_HEADERS = {
    "User-Agent": _session.headers["User-Agent"],
    "Referer": "https://data.eastmoney.com/",
}


def fetch_northbound_flow() -> dict:
    """获取北向资金流向，多源回退。

    Returns:
        dict with keys: total_net, sh_net, sz_net, date, source
        金额单位：亿元。获取失败返回空 dict。
    """
    logger.info("[数据] 获取北向资金 → 尝试东方财富 datacenter...")
    result = _fetch_northbound_datacenter()
    if result:
        logger.info(f"[数据] 北向资金(datacenter): 合计{result['total_net']:+.2f}亿, "
                     f"沪{result['sh_net']:+.2f}亿, 深{result['sz_net']:+.2f}亿, "
                     f"日期={result.get('date', 'N/A')}")
        return result

    logger.info("[数据] datacenter 不可用 → 尝试 kamt.rtmin 分时...")
    result = _fetch_northbound_rtmin()
    if result:
        logger.info(f"[数据] 北向资金(rtmin): 合计{result['total_net']:+.2f}亿, "
                     f"沪{result['sh_net']:+.2f}亿, 深{result['sz_net']:+.2f}亿")
        return result

    logger.warning("[数据] 北向资金: 所有数据源均不可用")
    return {}


def _fetch_northbound_datacenter() -> dict:
    """通过 datacenter-web API 获取北向资金净买入（最可靠）。

    RPT_MUTUAL_DEAL_HISTORY 的 MUTUAL_TYPE:
      002 = 沪股通(北向-沪), 004 = 深股通(北向-深), 006 = 北向合计
    金额单位: 百万元
    """
    base = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    common = {
        "reportName": "RPT_MUTUAL_DEAL_HISTORY",
        "columns": "TRADE_DATE,NET_DEAL_AMT,BUY_AMT,SELL_AMT",
        "pageSize": 1,
        "pageNumber": 1,
        "sortTypes": -1,
        "sortColumns": "TRADE_DATE",
        "source": "WEB",
        "client": "WEB",
    }

    def _query(mutual_type: str) -> tuple[float | None, str]:
        params = {**common, "filter": f'(MUTUAL_TYPE="{mutual_type}")'}
        r = _session.get(base, params=params, headers=_EM_HEADERS, timeout=8)
        result = r.json().get("result") or {}
        data_list = result.get("data") or []
        if data_list:
            item = data_list[0]
            net = item.get("NET_DEAL_AMT")
            date = str(item.get("TRADE_DATE", ""))[:10]
            return (net, date) if net is not None else (None, date)
        return None, ""

    try:
        sh_net, sh_date = _query("002")
        sz_net, sz_date = _query("004")

        if sh_net is None and sz_net is None:
            logger.debug("[数据] datacenter北向: 沪深数据均为 null")
            return {}

        sh_val = (sh_net or 0) / 100
        sz_val = (sz_net or 0) / 100
        total = sh_val + sz_val
        date = sh_date or sz_date

        return {
            "total_net": round(total, 2),
            "sh_net": round(sh_val, 2),
            "sz_net": round(sz_val, 2),
            "date": date,
            "source": "eastmoney-datacenter",
        }
    except Exception as e:
        logger.info(f"[数据] datacenter北向失败: {e}")
        return {}


def _fetch_northbound_rtmin() -> dict:
    """kamt.rtmin 分时数据回退方案。

    s2n 格式: 时间,沪净流入(万),沪余额(万),深净流入(万),深余额(万),?
    注意: 该 API 可能仅返回额度余额而无净流入数据。
    """
    url = "https://push2.eastmoney.com/api/qt/kamt.rtmin/get"
    params = {
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55,f56",
    }
    try:
        r = _session.get(url, params=params, headers=_EM_HEADERS, timeout=8)
        data = r.json().get("data") or {}
        s2n = data.get("s2n") or []
        if not s2n:
            return {}

        logger.debug(f"[数据] rtmin: s2n {len(s2n)}条, 首={s2n[0][:60]}, 末={s2n[-1][:60]}")

        for entry in reversed(s2n):
            parts = entry.split(",")
            if len(parts) < 6 or parts[1] == "-":
                continue

            sh_net = _safe_float(parts[1])
            sh_quota = _safe_float(parts[2])
            sz_net = _safe_float(parts[3])
            sz_quota = _safe_float(parts[4])

            # 如果"净流入"等于额度（520亿=5200000万），说明字段是余额而非净流入
            if sh_quota > 0 and abs(sh_net - sh_quota) < 1:
                continue
            if sz_quota > 0 and abs(sz_net - sz_quota) < 1:
                continue

            if sh_net == 0 and sz_net == 0:
                continue

            total = sh_net + sz_net
            return {
                "total_net": round(total / 1e4, 2),
                "sh_net": round(sh_net / 1e4, 2),
                "sz_net": round(sz_net / 1e4, 2),
                "date": data.get("s2nDate", ""),
                "source": "eastmoney-rtmin",
            }

        logger.debug("[数据] rtmin: s2n全部为额度数据或零值，无有效净流入")
        return {}
    except Exception as e:
        logger.info(f"[数据] rtmin北向失败: {e}")
        return {}


def fetch_northbound_history(days: int = 10) -> pd.DataFrame:
    """获取北向资金历史数据（datacenter API）。

    Returns:
        DataFrame(日期, 沪股通净买入_亿, 深股通净买入_亿, 北向合计_亿)
    """
    base = "https://datacenter-web.eastmoney.com/api/data/v1/get"

    def _history(mutual_type: str) -> list[tuple[str, float]]:
        try:
            r = _session.get(base, params={
                "reportName": "RPT_MUTUAL_DEAL_HISTORY",
                "columns": "TRADE_DATE,NET_DEAL_AMT",
                "pageSize": days,
                "pageNumber": 1,
                "sortTypes": -1,
                "sortColumns": "TRADE_DATE",
                "source": "WEB",
                "client": "WEB",
                "filter": f'(MUTUAL_TYPE="{mutual_type}")',
            }, headers=_EM_HEADERS, timeout=8)
            result = r.json().get("result") or {}
            data_list = result.get("data") or []
            return [
                (str(item["TRADE_DATE"])[:10],
                 round((item.get("NET_DEAL_AMT") or 0) / 100, 2))
                for item in data_list
            ]
        except Exception:
            return []

    try:
        sh_data = {d: v for d, v in _history("002")}
        sz_data = {d: v for d, v in _history("004")}
        all_dates = sorted(set(sh_data) | set(sz_data))

        if not all_dates:
            return pd.DataFrame()

        rows = []
        for date in all_dates:
            sh = sh_data.get(date, 0)
            sz = sz_data.get(date, 0)
            rows.append({
                "日期": date,
                "沪股通净买入_亿": sh,
                "深股通净买入_亿": sz,
                "北向合计_亿": round(sh + sz, 2),
            })

        return pd.DataFrame(rows)
    except Exception as e:
        logger.debug(f"[数据] datacenter北向历史失败: {e}")
        return pd.DataFrame()


# ====================================================================
#  个股资金流向（东方财富直连，不走 akshare）
# ====================================================================
def fetch_stock_fund_flow(stock_code: str, days: int = 5) -> pd.DataFrame:
    """获取个股资金流向（东方财富直连 → akshare 回退）。

    Returns:
        DataFrame(日期, 主力净流入_万, 超大单净流入_万, 大单净流入_万, 中单净流入_万, 小单净流入_万)
    """
    df = _fetch_stock_fund_flow_em(stock_code, days)
    if not df.empty:
        return df

    return _fetch_stock_fund_flow_ak(stock_code, days)


def _fetch_stock_fund_flow_em(stock_code: str, days: int = 5) -> pd.DataFrame:
    """东方财富 push2his 个股资金流向（日K）。"""
    market = "1" if stock_code.startswith(("6", "9")) else "0"
    secid = f"{market}.{stock_code}"
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "lmt": days,
        "klt": 101,
    }
    try:
        r = _session.get(url, params=params, headers=_EM_HEADERS, timeout=8)
        data = r.json().get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            return pd.DataFrame()

        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 10:
                continue
            rows.append({
                "日期": parts[0],
                "主力净流入_万": round(_safe_float(parts[1]) / 1e4, 1),
                "小单净流入_万": round(_safe_float(parts[2]) / 1e4, 1),
                "中单净流入_万": round(_safe_float(parts[3]) / 1e4, 1),
                "大单净流入_万": round(_safe_float(parts[4]) / 1e4, 1),
                "超大单净流入_万": round(_safe_float(parts[5]) / 1e4, 1),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.debug(f"[数据] 东方财富个股资金流向(push2his)失败 {stock_code}: {e}")
        return pd.DataFrame()


def _fetch_stock_fund_flow_ak(stock_code: str, days: int = 5) -> pd.DataFrame:
    """akshare 个股资金流向回退。"""
    try:
        import akshare as ak
        market = "sh" if stock_code.startswith(("6", "9")) else "sz"
        df = ak.stock_individual_fund_flow(stock=stock_code, market=market)
        if df.empty:
            return pd.DataFrame()

        recent = df.tail(days)
        rows = []
        for _, row in recent.iterrows():
            rows.append({
                "日期": str(row.get("日期", ""))[:10],
                "主力净流入_万": round(_safe_float(row.get("主力净流入-净额", 0)) / 1e4, 1),
                "超大单净流入_万": round(_safe_float(row.get("超大单净流入-净额", 0)) / 1e4, 1),
                "大单净流入_万": round(_safe_float(row.get("大单净流入-净额", 0)) / 1e4, 1),
                "中单净流入_万": round(_safe_float(row.get("中单净流入-净额", 0)) / 1e4, 1),
                "小单净流入_万": round(_safe_float(row.get("小单净流入-净额", 0)) / 1e4, 1),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.debug(f"[数据] akshare个股资金流向失败 {stock_code}: {e}")
        return pd.DataFrame()


# ====================================================================
#  行业板块资金流向（东方财富直连，不走 akshare）
# ====================================================================
def fetch_sector_fund_flow_direct() -> pd.DataFrame:
    """获取行业板块资金流向排名（东方财富直连，不走 akshare）。

    Returns:
        DataFrame(行业名称, 今日涨跌幅, 主力净流入_亿, 主力净流入占比%, ...)
    """
    logger.info("[数据] 获取行业资金流向 → 尝试东方财富直连...")
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1,
        "pz": 50,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f62",
        "fs": "m:90+t:2",
        "fields": "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87",
    }
    try:
        r = _session.get(url, params=params, headers=_EM_HEADERS, timeout=8)
        resp_json = r.json()
        data = resp_json.get("data") or {}
        diff = data.get("diff") or []
        if not diff:
            logger.info(f"[数据] 东方财富行业资金流向: 无数据 (HTTP {r.status_code}, "
                         f"data keys={list(data.keys()) if data else 'null'})")
            return pd.DataFrame()

        rows = []
        for item in diff:
            rows.append({
                "行业代码": item.get("f12", ""),
                "名称": item.get("f14", ""),
                "今日涨跌幅": item.get("f3", 0),
                "主力净流入_亿": round(_safe_float(item.get("f62", 0)) / 1e8, 2),
                "主力净流入占比%": item.get("f184", 0),
                "超大单净流入_亿": round(_safe_float(item.get("f66", 0)) / 1e8, 2),
                "大单净流入_亿": round(_safe_float(item.get("f72", 0)) / 1e8, 2),
                "中单净流入_亿": round(_safe_float(item.get("f78", 0)) / 1e8, 2),
                "小单净流入_亿": round(_safe_float(item.get("f84", 0)) / 1e8, 2),
            })
        df = pd.DataFrame(rows)
        logger.info(f"[数据] 东方财富行业资金流向: 成功获取{len(df)}个行业, "
                     f"净流入前3: {', '.join(df.head(3)['名称'].tolist())}")
        return df
    except Exception as e:
        logger.info(f"[数据] 东方财富行业资金流向直连失败: {e}")
        return pd.DataFrame()


# ====================================================================
#  财务数据（东方财富 datacenter → Baostock 回退）
# ====================================================================
def fetch_financial_summary(stock_code: str, years: int = 4) -> pd.DataFrame:
    """获取股票财务摘要，多源回退。

    优先级: 东方财富 datacenter → Baostock → akshare

    Returns:
        DataFrame 含财务指标，格式统一。
    """
    df = _fetch_financial_em_datacenter(stock_code, years)
    if not df.empty:
        return df

    df = _fetch_financial_baostock(stock_code, years)
    if not df.empty:
        return df

    return _fetch_financial_akshare(stock_code)


def _fetch_financial_em_datacenter(stock_code: str, years: int = 4) -> pd.DataFrame:
    """东方财富 datacenter 财务数据（最稳定的财务数据源）。"""
    pure = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPT_F10_FINANCE_MAINFINADATA",
        "columns": "ALL",
        "filter": f'(SECURITY_CODE="{pure}")',
        "pageSize": years * 4,
        "pageNumber": 1,
        "sortTypes": -1,
        "sortColumns": "REPORT_DATE",
        "source": "WEB",
        "client": "WEB",
    }
    try:
        r = _session.get(url, params=params, headers=_EM_HEADERS, timeout=8)
        result = r.json().get("result") or {}
        items = result.get("data") or []
        if not items:
            return pd.DataFrame()

        rows = []
        for item in items:
            report_date = str(item.get("REPORT_DATE", ""))[:10]
            report_name = item.get("REPORT_DATE_NAME", report_date)

            eps = item.get("EPSJB")
            roe = item.get("ROEJQ")
            revenue = item.get("TOTALOPERATEREVE")
            net_profit = item.get("PARENTNETPROFIT")
            gp_margin = item.get("XSMLL")
            np_margin = item.get("XSJLL")
            revenue_growth = item.get("TOTALOPERATEREVETZ")
            profit_growth = item.get("PARENTNETPROFITTZ")
            debt_ratio = item.get("ZCFZL")

            rows.append({
                "报告期": report_name or report_date,
                "EPS": round(eps, 3) if eps is not None else None,
                "ROE(%)": round(roe, 2) if roe is not None else None,
                "营收(亿)": round(revenue / 1e8, 2) if revenue else None,
                "净利润(亿)": round(net_profit / 1e8, 2) if net_profit else None,
                "毛利率(%)": round(gp_margin, 2) if gp_margin is not None else None,
                "净利率(%)": round(np_margin, 2) if np_margin is not None else None,
                "营收增速(%)": round(revenue_growth, 2) if revenue_growth is not None else None,
                "利润增速(%)": round(profit_growth, 2) if profit_growth is not None else None,
                "资产负债率(%)": round(debt_ratio, 2) if debt_ratio is not None else None,
            })

        logger.debug(f"[数据] EM-datacenter 财务: {pure} 获取{len(rows)}期")
        return pd.DataFrame(rows)
    except Exception as e:
        logger.debug(f"[数据] EM-datacenter 财务失败 {pure}: {e}")
        return pd.DataFrame()


def _fetch_financial_baostock(stock_code: str, years: int = 4) -> pd.DataFrame:
    """Baostock 财务数据回退。"""
    try:
        from src.tools.baostock_provider import bs_fetch_financial
        logger.info(f"[数据] EM-datacenter 财务不可用, 回退 Baostock → {stock_code}")
        return bs_fetch_financial(stock_code, years)
    except Exception as e:
        logger.debug(f"[数据] Baostock 财务失败 {stock_code}: {e}")
        return pd.DataFrame()


def _fetch_financial_akshare(stock_code: str) -> pd.DataFrame:
    """akshare 财务数据（最后回退）。"""
    try:
        return call_akshare(
            "stock_financial_abstract_ths",
            symbol=stock_code.replace("sh", "").replace("sz", ""),
            indicator="按年度",
        )
    except Exception as e:
        logger.debug(f"[数据] akshare 财务失败 {stock_code}: {e}")
        return pd.DataFrame()


# ====================================================================
#  除权除息数据（Baostock → akshare 回退）
# ====================================================================
def fetch_dividend_data(stock_code: str, years: int = 3) -> pd.DataFrame:
    """获取除权除息数据，多源回退。"""
    try:
        from src.tools.baostock_provider import bs_fetch_dividend
        df = bs_fetch_dividend(stock_code, years)
        if not df.empty:
            return df
    except Exception:
        pass

    try:
        return call_akshare("stock_dividents_cninfo", symbol=stock_code)
    except Exception:
        return pd.DataFrame()


# ====================================================================
#  akshare 回退调用（带重试）— 仅用于无替代源的场景
# ====================================================================
def call_akshare(func_name: str, *args: Any, max_retries: int = 2, **kwargs: Any) -> Any:
    """akshare 回退调用，带轻量重试。仅在新浪/腾讯 API 不支持的功能时使用。"""
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare 未安装")

    func = getattr(ak, func_name, None)
    if func is None:
        raise AttributeError(f"akshare 没有 '{func_name}' 接口")

    import random
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            retryable = any(kw in err_msg for kw in [
                "connection", "remote", "timeout", "reset",
            ])
            if not retryable or attempt >= max_retries - 1:
                raise
            backoff = (2 ** attempt) * 3 + random.uniform(0, 2)
            logger.warning(
                f"[akshare回退] {func_name} 第{attempt+1}次失败, "
                f"{backoff:.1f}s后重试: {e}"
            )
            time.sleep(backoff)
    raise last_err  # type: ignore[misc]


# ====================================================================
#  辅助函数
# ====================================================================
def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
