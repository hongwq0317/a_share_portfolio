"""Baostock 数据提供层

直连 baostock 服务器，不走爬虫，极其稳定。
提供 K 线、财务数据、行业分类等，作为腾讯/东方财富的回退兜底。

特点：
  - 完全免费、无需注册
  - 稳定性 ⭐⭐⭐⭐⭐（独立服务器，不受反爬影响）
  - 覆盖：日 K 线、财务报表、行业分类、除权除息
"""

import logging
import threading
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger("portfolio")

_bs_lock = threading.Lock()
_bs_logged_in = False


def _ensure_login() -> bool:
    """确保 baostock 已登录（线程安全，全局复用连接）。"""
    global _bs_logged_in
    if _bs_logged_in:
        return True
    with _bs_lock:
        if _bs_logged_in:
            return True
        try:
            import baostock as bs
            lg = bs.login()
            if lg.error_code == "0":
                _bs_logged_in = True
                return True
            logger.warning(f"[baostock] 登录失败: {lg.error_msg}")
            return False
        except Exception as e:
            logger.warning(f"[baostock] 登录异常: {e}")
            return False


def _to_bs_code(stock_code: str) -> str:
    """纯数字代码 → baostock 格式 (sh.600519 / sz.000001)。"""
    pure = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")
    prefix = "sh" if pure.startswith(("6", "9")) else "sz"
    return f"{prefix}.{pure}"


def _rs_to_list(rs) -> list[list[str]]:
    """将 baostock ResultSet 转为 list[list[str]]。"""
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return rows


def bs_fetch_stock_history(
    stock_code: str,
    days: int = 120,
    adjust: str = "qfq",
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """通过 Baostock 获取个股日 K 线。

    Returns:
        DataFrame(日期, 开盘, 收盘, 最高, 最低, 成交量)  与腾讯源格式一致。
    """
    if not _ensure_login():
        return pd.DataFrame()

    import baostock as bs

    bs_code = _to_bs_code(stock_code)
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days + 60)).strftime("%Y-%m-%d")

    adjust_map = {"qfq": "2", "hfq": "1", "": "3"}
    adjustflag = adjust_map.get(adjust, "2")

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
        )
        if rs.error_code != "0":
            logger.debug(f"[baostock] K线查询失败 {stock_code}: {rs.error_msg}")
            return pd.DataFrame()

        data = _rs_to_list(rs)
        if not data:
            return pd.DataFrame()

        rows = []
        for row in data:
            if len(row) >= 6 and row[1]:
                try:
                    rows.append({
                        "日期": row[0],
                        "开盘": float(row[1]),
                        "最高": float(row[2]),
                        "最低": float(row[3]),
                        "收盘": float(row[4]),
                        "成交量": float(row[5]),
                    })
                except (ValueError, TypeError):
                    continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if len(df) > days:
            df = df.tail(days)
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug(f"[baostock] K线异常 {stock_code}: {e}")
        return pd.DataFrame()


def bs_fetch_index_history(
    index_code: str = "000300",
    days: int = 252,
) -> pd.DataFrame:
    """通过 Baostock 获取指数日 K 线。"""
    if not _ensure_login():
        return pd.DataFrame()

    import baostock as bs

    prefix = "sh" if index_code.startswith(("000", "880", "99")) else "sz"
    bs_code = f"{prefix}.{index_code}"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days + 60)).strftime("%Y-%m-%d")

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
        )
        if rs.error_code != "0":
            logger.debug(f"[baostock] 指数K线失败 {index_code}: {rs.error_msg}")
            return pd.DataFrame()

        data = _rs_to_list(rs)
        if not data:
            return pd.DataFrame()

        rows = []
        for row in data:
            if len(row) >= 6 and row[1]:
                try:
                    rows.append({
                        "日期": row[0],
                        "开盘": float(row[1]),
                        "最高": float(row[2]),
                        "最低": float(row[3]),
                        "收盘": float(row[4]),
                        "成交量": float(row[5]),
                    })
                except (ValueError, TypeError):
                    continue

        df = pd.DataFrame(rows)
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
        return df
    except Exception as e:
        logger.debug(f"[baostock] 指数K线异常 {index_code}: {e}")
        return pd.DataFrame()


def bs_fetch_financial(stock_code: str, years: int = 4) -> pd.DataFrame:
    """通过 Baostock 获取财务数据摘要（利润表+成长性+偿债能力）。

    Returns:
        DataFrame 含: 报告期, EPS, ROE(%), 净利润(亿), 营收(亿), 毛利率(%), 净利率(%)
    """
    if not _ensure_login():
        return pd.DataFrame()

    import baostock as bs

    bs_code = _to_bs_code(stock_code)
    current_year = datetime.now().year
    current_month = datetime.now().month

    all_rows = []
    for year in range(current_year, current_year - years - 1, -1):
        quarters = [4, 3, 2, 1] if year < current_year else list(range(max(1, (current_month - 1) // 3), 0, -1))
        for quarter in quarters:
            try:
                rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                if rs.error_code != "0":
                    continue
                data = _rs_to_list(rs)
                if data:
                    all_rows.extend(data)
            except Exception:
                continue
            if len(all_rows) >= years * 4:
                break
        if len(all_rows) >= years * 4:
            break

    if not all_rows:
        return pd.DataFrame()

    fields = ["code", "pubDate", "statDate", "roeAvg", "npMargin", "gpMargin",
              "netProfit", "epsTTM", "MBRevenue", "totalShare", "liqaShare"]
    result_rows = []
    for row in all_rows:
        if len(row) < len(fields):
            continue
        try:
            roe = float(row[3]) * 100 if row[3] else None
            np_margin = float(row[4]) * 100 if row[4] else None
            gp_margin = float(row[5]) * 100 if row[5] else None
            net_profit = float(row[6]) / 1e8 if row[6] else None
            eps = float(row[7]) if row[7] else None
            revenue = float(row[8]) / 1e8 if row[8] else None

            result_rows.append({
                "报告期": row[2],
                "EPS(TTM)": round(eps, 3) if eps is not None else None,
                "ROE(%)": round(roe, 2) if roe is not None else None,
                "净利润(亿)": round(net_profit, 2) if net_profit is not None else None,
                "营收(亿)": round(revenue, 2) if revenue is not None else None,
                "毛利率(%)": round(gp_margin, 2) if gp_margin is not None else None,
                "净利率(%)": round(np_margin, 2) if np_margin is not None else None,
            })
        except (ValueError, TypeError, IndexError):
            continue

    return pd.DataFrame(result_rows)


def bs_fetch_industry(stock_code: str) -> str:
    """通过 Baostock 获取证监会行业分类。"""
    if not _ensure_login():
        return ""

    import baostock as bs

    bs_code = _to_bs_code(stock_code)
    try:
        rs = bs.query_stock_industry(code=bs_code)
        if rs.error_code != "0":
            return ""
        data = _rs_to_list(rs)
        if data and len(data[0]) >= 4:
            return data[0][3]
    except Exception:
        pass
    return ""


def bs_fetch_dividend(stock_code: str, years: int = 3) -> pd.DataFrame:
    """通过 Baostock 获取分红配股数据。

    Returns:
        DataFrame 含: 除权除息日, 分红方案, 每股股利, 每股红股
    """
    if not _ensure_login():
        return pd.DataFrame()

    import baostock as bs

    bs_code = _to_bs_code(stock_code)
    current_year = datetime.now().year
    all_rows = []

    for year in range(current_year, current_year - years - 1, -1):
        try:
            rs = bs.query_dividend_data(code=bs_code, year=str(year), yearType="operate")
            if rs.error_code != "0":
                continue
            data = _rs_to_list(rs)
            if data:
                all_rows.extend(data)
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()

    result_rows = []
    for row in all_rows:
        if len(row) < 10:
            continue
        ex_date = row[5] if row[5] else ""
        plan = row[3] if row[3] else ""
        cash_div = row[7] if row[7] else "0"
        stock_div = row[8] if row[8] else "0"
        if ex_date:
            result_rows.append({
                "除权除息日": ex_date,
                "分红方案": plan,
                "每股股利": cash_div,
                "每股红股": stock_div,
            })

    return pd.DataFrame(result_rows)
