"""市场数据工具

多源数据架构：新浪/腾讯 → 东方财富 datacenter → Baostock → akshare（最后回退）。
"""

import logging
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from src.tools.data_provider import (
    call_akshare,
    fetch_all_stocks,
    fetch_financial_summary,
    fetch_index_quote,
    fetch_northbound_flow,
    fetch_northbound_history,
    fetch_sector_fund_flow_direct,
    fetch_sector_performance,
    fetch_stock_fund_flow as fetch_stock_fund_flow_direct,
    fetch_stock_history,
    fetch_stock_quote,
)

logger = logging.getLogger("portfolio")

_get_spot_data = fetch_all_stocks
_call_ak = call_akshare


@tool
def get_stock_realtime_quote(stock_codes: list[str]) -> str:
    """获取股票实时行情（价格、涨跌幅、成交量等）。

    Args:
        stock_codes: 股票代码列表，如 ["000001", "600519"]
    """
    quotes = fetch_stock_quote(stock_codes[:20])
    if not quotes:
        return "获取行情失败，请稍后重试"

    results = []
    for code in stock_codes[:20]:
        q = quotes.get(code)
        if not q:
            results.append(f"{code}: 未找到数据")
            continue
        results.append(
            f"{q['name']}({code}): "
            f"现价={q['price']}, 涨跌幅={q['change_pct']}%, "
            f"成交量={q['volume']:.0f}手, 成交额={q['amount']/1e8:.2f}亿, "
            f"换手率={q['turnover']}%, 市盈率={q['pe'] or 'N/A'}, "
            f"市净率={q['pb'] or 'N/A'}"
        )
    return "\n".join(results)


@tool
def get_stock_history(stock_code: str, period: str = "daily", days: int = 120) -> str:
    """获取股票历史行情数据。

    Args:
        stock_code: 股票代码，如 "000001"
        period: 周期 daily/weekly/monthly
        days: 获取最近多少天的数据
    """
    try:
        df = fetch_stock_history(stock_code, days=days)
        if df.empty:
            return f"{stock_code}: 无历史数据"

        latest = df.tail(5).to_string(index=False)
        stats = (
            f"统计({days}天): "
            f"最高={df['最高'].max():.2f}, 最低={df['最低'].min():.2f}, "
            f"均价={df['收盘'].mean():.2f}, "
            f"区间涨跌幅={(df['收盘'].iloc[-1]/df['收盘'].iloc[0]-1)*100:.2f}%"
        )
        return f"最近5日行情:\n{latest}\n\n{stats}"
    except Exception as e:
        return f"获取 {stock_code} 历史数据失败: {e}"


@tool
def get_index_data(index_code: str = "000300") -> str:
    """获取指数实时行情（沪深300、上证指数等）。

    Args:
        index_code: 指数代码，如 "000300"(沪深300), "000001"(上证指数), "399001"(深证成指)
    """
    quotes = fetch_index_quote([index_code])
    if not quotes or index_code not in quotes:
        return f"指数 {index_code} 数据获取失败"

    q = quotes[index_code]
    return (
        f"{q['name']}({index_code}): "
        f"现价={q['price']:.2f}, 涨跌幅={q['change_pct']}%, "
        f"成交额={q['amount']/1e8:.2f}亿"
    )


@tool
def get_sector_fund_flow() -> str:
    """获取行业板块资金流向和行业表现。多数据源自动回退：东方财富直连→腾讯计算。"""
    # 方案1: 东方财富直连
    df = fetch_sector_fund_flow_direct()
    if not df.empty:
        top_inflow = df.head(10).to_string(index=False)
        top_outflow = df.tail(10).to_string(index=False)
        return f"行业资金净流入前10:\n{top_inflow}\n\n资金净流出前10:\n{top_outflow}"

    # 方案2: 从腾讯全市场数据计算行业板块表现+资金流向近似
    df = fetch_sector_performance()
    if not df.empty:
        top = df.head(10).to_string(index=False)
        bottom = df.tail(10).to_string(index=False)
        return (
            f"行业资金流向（腾讯全市场计算，净流入为成交额×涨跌幅估算）:\n"
            f"资金流入前10:\n{top}\n\n"
            f"资金流出前10:\n{bottom}"
        )

    return "行业资金流向数据获取失败（所有数据源均不可用）"


@tool
def get_stock_fund_flow(stock_code: str) -> str:
    """获取个股资金流向（主力/散户净流入流出）。

    Args:
        stock_code: 股票代码
    """
    df = fetch_stock_fund_flow_direct(stock_code, days=5)
    if not df.empty:
        return f"{stock_code} 最近资金流向:\n{df.to_string(index=False)}"

    return f"{stock_code}: 资金流向数据暂不可用，建议通过 web_search 搜索该股票资金流向信息"


@tool
def batch_get_stock_fund_flow(stock_codes: list[str]) -> str:
    """批量获取多只股票的资金流向（主力/散户净流入流出），一次调用替代多次 get_stock_fund_flow。

    Args:
        stock_codes: 股票代码列表，最多15只
    """
    if len(stock_codes) > 15:
        stock_codes = stock_codes[:15]

    results = []
    for code in stock_codes:
        df = fetch_stock_fund_flow_direct(code, days=3)
        if not df.empty:
            results.append(f"【{code}】\n{df.to_string(index=False)}")
        else:
            results.append(f"【{code}】资金流向数据暂不可用")

    return "\n\n".join(results) if results else "所有股票资金流向数据获取失败"


@tool
def get_market_sentiment() -> str:
    """获取市场情绪指标（涨跌家数、涨停跌停等）。"""
    try:
        df = fetch_all_stocks()
        if df.empty:
            return "获取市场情绪失败"

        total = len(df)
        change_col = "涨跌幅"
        up_count = int((df[change_col] > 0).sum())
        down_count = int((df[change_col] < 0).sum())
        flat_count = total - up_count - down_count
        limit_up = int((df[change_col] >= 9.8).sum())
        limit_down = int((df[change_col] <= -9.8).sum())
        avg_change = df[change_col].mean()
        total_volume = df["成交额"].sum() / 1e8

        return (
            f"A股市场情绪:\n"
            f"  总股票数: {total}\n"
            f"  上涨: {up_count} ({up_count/total*100:.1f}%)\n"
            f"  下跌: {down_count} ({down_count/total*100:.1f}%)\n"
            f"  平盘: {flat_count}\n"
            f"  涨停: {limit_up}, 跌停: {limit_down}\n"
            f"  平均涨跌幅: {avg_change:.2f}%\n"
            f"  两市总成交额: {total_volume:.0f}亿"
        )
    except Exception as e:
        return f"获取市场情绪失败: {e}"


@tool
def get_north_bound_flow() -> str:
    """获取北向资金流向数据（今日实时+历史）。多数据源自动回退。"""
    parts = []

    realtime = fetch_northbound_flow()
    if realtime:
        date_info = realtime.get("date") or realtime.get("time", "")
        parts.append(
            f"北向资金({date_info}):\n"
            f"  合计净流入: {realtime['total_net']:+.2f}亿\n"
            f"  沪股通: {realtime['sh_net']:+.2f}亿\n"
            f"  深股通: {realtime['sz_net']:+.2f}亿\n"
            f"  (数据源: {realtime['source']})"
        )

    # 历史数据
    hist_df = fetch_northbound_history(10)
    if not hist_df.empty:
        parts.append(f"\n北向资金最近{len(hist_df)}日:\n{hist_df.to_string(index=False)}")

    if parts:
        return "\n".join(parts)

    return "北向资金数据暂不可用，建议通过 web_search 搜索'今日北向资金流向'获取"


SECTOR_ALIAS_MAP: dict[str, list[str]] = {
    "有色金属": ["铜", "铝", "锌", "镍", "锡", "铅", "钴", "钨", "钼", "钛", "有色", "矿业", "冶炼"],
    "稀土": ["稀土", "磁材", "永磁", "钕铁硼", "镧", "铈"],
    "贵金属": ["黄金", "白银", "铂", "金矿", "银矿", "贵金属"],
    "锂电": ["锂", "电池", "储能", "宁德", "比亚迪"],
    "半导体": ["芯片", "半导体", "集成电路", "晶圆", "封测", "IC设计"],
    "光伏": ["光伏", "太阳能", "硅片", "组件", "逆变器"],
    "风电": ["风电", "风能", "叶片", "风机"],
    "医药": ["医药", "制药", "生物", "医疗", "药业", "疫苗"],
    "白酒": ["酒", "茅台", "五粮液", "汾酒", "泸州"],
    "房地产": ["地产", "置业", "房产", "万科", "保利"],
    "军工": ["军工", "航天", "航空", "兵器", "船舶", "导弹", "雷达"],
    "汽车": ["汽车", "整车", "车", "特斯拉"],
    "人工智能": ["智能", "AI", "算力", "GPU", "大模型"],
}


@tool
def screen_stocks_by_sector(sector: str) -> str:
    """按行业/概念关键词筛选股票。从全市场数据中按名称关键词匹配。

    内置行业别名：有色金属、稀土、贵金属、锂电、半导体、光伏、风电、
    医药、白酒、房地产、军工、汽车、人工智能 等。
    如需精确行业成分股，建议配合 web_search 搜索"XX行业龙头股"。

    Args:
        sector: 行业/概念关键词，如 "银行", "有色金属", "光伏", "白酒", "芯片"
    """
    try:
        df = fetch_all_stocks()
        if df.empty:
            return "获取市场数据失败"

        keywords = SECTOR_ALIAS_MAP.get(sector, [sector])
        pattern = "|".join(keywords)
        matched = df[df["名称"].str.contains(pattern, na=False)]

        if matched.empty:
            return (
                f"未找到名称含'{sector}'(关键词: {', '.join(keywords)})的股票。\n"
                f"建议使用 web_search 搜索\"{sector}行业成分股\"获取更准确的结果，\n"
                f"或使用 screen_stocks_by_condition 按涨幅/成交额/PE等条件筛选。"
            )

        cols = ["代码", "名称", "最新价", "涨跌幅", "市盈率-动态", "市净率", "成交额"]
        avail = [c for c in cols if c in matched.columns]
        result = matched[avail].sort_values("成交额", ascending=False).head(30)
        if "成交额" in result.columns:
            result = result.copy()
            result["成交额(亿)"] = (result["成交额"] / 1e8).round(2)
            result = result.drop(columns=["成交额"])
        return (
            f"'{sector}'相关股票(关键词: {', '.join(keywords)}, "
            f"共匹配{len(matched)}只, 按成交额排序前30):\n{result.to_string(index=False)}"
        )
    except Exception as e:
        return f"筛选失败: {e}"


@tool
def screen_stocks_by_condition(
    sort_by: str = "changepercent",
    min_amount: float = 10000,
    min_pe: float = 0,
    max_pe: float = 100,
    min_change: float = -100,
    max_change: float = 100,
    top_n: int = 30,
) -> str:
    """按多条件从全市场筛选股票。

    Args:
        sort_by: 排序字段 changepercent(涨跌幅)/amount(成交额)/pe(市盈率)/pb(市净率)/turnover(换手率)
        min_amount: 最低成交额（万元），默认 1 万元
        min_pe: 最低 PE（0 表示不限），设为正数可排除亏损股
        max_pe: 最高 PE，默认 100
        min_change: 最低涨跌幅%
        max_change: 最高涨跌幅%
        top_n: 返回前 N 只，最多 50
    """
    try:
        df = fetch_all_stocks()
        if df.empty:
            return "获取市场数据失败"

        sort_map = {
            "changepercent": "涨跌幅",
            "amount": "成交额",
            "pe": "市盈率-动态",
            "pb": "市净率",
            "turnover": "换手率",
        }
        sort_col = sort_map.get(sort_by, "涨跌幅")

        filtered = df.copy()
        if min_amount > 0:
            filtered = filtered[filtered["成交额"] >= min_amount * 10000]
        if min_pe > 0:
            filtered = filtered[filtered["市盈率-动态"] >= min_pe]
        if max_pe < 9999:
            filtered = filtered[filtered["市盈率-动态"] <= max_pe]
        filtered = filtered[
            (filtered["涨跌幅"] >= min_change) & (filtered["涨跌幅"] <= max_change)
        ]

        if filtered.empty:
            return "无满足条件的股票"

        top_n = min(top_n, 50)
        result = filtered.nlargest(top_n, sort_col)

        cols = ["代码", "名称", "最新价", "涨跌幅", "市盈率-动态", "市净率", "成交额", "换手率"]
        avail = [c for c in cols if c in result.columns]
        display = result[avail].copy()
        if "成交额" in display.columns:
            display["成交额(亿)"] = (display["成交额"] / 1e8).round(2)
            display = display.drop(columns=["成交额"])

        return (
            f"全市场筛选(共{len(filtered)}只符合条件, 按{sort_col}排序前{top_n}):\n"
            f"条件: 成交额>={min_amount}万, PE∈[{min_pe},{max_pe}], "
            f"涨跌幅∈[{min_change}%,{max_change}%]\n"
            f"{display.to_string(index=False)}"
        )
    except Exception as e:
        return f"筛选失败: {e}"


@tool
def get_financial_summary(stock_code: str) -> str:
    """获取股票财务摘要（营收、利润、ROE等核心指标）。
    多源回退: 东方财富datacenter → Baostock → akshare。

    Args:
        stock_code: 股票代码
    """
    try:
        df = fetch_financial_summary(stock_code, years=4)
        if df.empty:
            return f"{stock_code}: 无财务数据"
        recent = df.head(8).to_string(index=False)
        return f"{stock_code} 财务摘要:\n{recent}"
    except Exception as e:
        return f"获取 {stock_code} 财务数据失败: {e}"


@tool
def batch_get_stock_overview(stock_codes: list[str]) -> str:
    """批量获取多只股票概览（行情+历史统计+财务），一次调用替代逐只查询，大幅节省轮次。

    返回每只股票的实时行情、60日价格统计、最近年度财务摘要。
    建议在选股阶段对候选池（5-15只）批量调用，而非逐一调用get_stock_history。

    Args:
        stock_codes: 股票代码列表，最多15只
    """
    codes = stock_codes[:15]
    quotes = fetch_stock_quote(codes)
    results = []

    for code in codes:
        parts = []
        q = quotes.get(code) if quotes else None
        if q:
            parts.append(
                f"  行情: {q['name']} 现价={q['price']} 涨跌幅={q['change_pct']}% "
                f"成交额={q['amount']/1e8:.2f}亿 换手率={q['turnover']}% "
                f"PE={q['pe'] or 'N/A'} PB={q['pb'] or 'N/A'}"
            )
        else:
            parts.append("  行情: 获取失败")

        try:
            df = fetch_stock_history(code, days=90)
            if not df.empty and len(df) > 5:
                close = df["收盘"]
                parts.append(
                    f"  60日统计: 最高={close.max():.2f} 最低={close.min():.2f} "
                    f"均价={close.mean():.2f} "
                    f"区间涨跌幅={(close.iloc[-1]/close.iloc[0]-1)*100:.1f}% "
                    f"波动率={close.pct_change().std()*100:.2f}%/日"
                )
        except Exception:
            parts.append("  历史数据: 获取失败")

        try:
            fin = fetch_financial_summary(code, years=2)
            if not fin.empty:
                row = fin.iloc[0]
                fin_items = []
                for col in fin.columns[:8]:
                    val = row.get(col)
                    if val is not None and str(val) != "nan":
                        fin_items.append(f"{col}={val}")
                if fin_items:
                    parts.append(f"  财务: {', '.join(fin_items)}")
        except Exception:
            pass

        results.append(f"【{code}】\n" + "\n".join(parts))

    return f"批量股票概览({len(codes)}只):\n\n" + "\n\n".join(results)


@tool
def think_tool(reflection: str) -> str:
    """思考工具：在不进行外部操作的情况下进行推理和反思。

    Args:
        reflection: 你的思考内容
    """
    return f"思考记录: {reflection}"


MARKET_TOOLS = [
    get_stock_realtime_quote,
    get_stock_history,
    get_index_data,
    get_sector_fund_flow,
    get_stock_fund_flow,
    batch_get_stock_fund_flow,
    get_market_sentiment,
    get_north_bound_flow,
    screen_stocks_by_sector,
    screen_stocks_by_condition,
    get_financial_summary,
    batch_get_stock_overview,
    think_tool,
]
