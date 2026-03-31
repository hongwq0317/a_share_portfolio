"""工具函数模块

提供日志、文本处理等通用工具。
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Optional


def setup_logging(log_dir: str = "logs", file_mode: str = "a"):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("portfolio")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件
    today = datetime.now().strftime("%Y%m%d")
    fh = logging.FileHandler(
        os.path.join(log_dir, f"portfolio_{today}.log"),
        mode=file_mode, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def extract_text_from_content(content: Any) -> str:
    """从 LLM 响应的 content 中提取纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""


def log_token_usage(
    response: Any,
    node_name: str,
    step: str,
    messages: Optional[list] = None,
    max_tokens: int = 0,
):
    """记录 token 用量"""
    logger = logging.getLogger("portfolio")
    usage = getattr(response, "usage_metadata", None)
    if usage:
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        else:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
        total = input_tokens + output_tokens
        logger.info(
            f"[{node_name}] 📊 Token用量 ({step}): "
            f"输入={input_tokens:,}, 输出={output_tokens:,}, 合计={total:,}"
        )


class NodeLogger:
    """节点级日志记录器

    detail_level:
        "full"  — 打印所有信息，包含完整工具结果和 LLM 输出
        "normal" — 打印关键过程信息，工具结果/LLM输出有截断
        "brief" — 仅打印开始/完成/错误
    """

    # 不同级别下工具结果和 LLM 输出的截断长度
    _TRUNCATE = {"full": 0, "normal": 500, "brief": 0}

    def __init__(self, node_name: str, detail_level: str = "normal"):
        self.node_name = node_name
        self.detail_level = detail_level
        self.logger = logging.getLogger("portfolio")
        self.start_time = time.time()
        self._max_len = self._TRUNCATE.get(detail_level, 500)

    def _log(self, level: str, msg: str, skip_in_brief: bool = False):
        if skip_in_brief and self.detail_level == "brief":
            return
        getattr(self.logger, level)(f"[{self.node_name}] {msg}")

    def _truncate(self, text: str, max_len: int = 0) -> str:
        limit = max_len or self._max_len
        if limit and len(text) > limit:
            return text[:limit] + f"... (共{len(text)}字符)"
        return text

    # ── 生命周期 ──

    def start(self, topic: str = ""):
        self._log("info", f"{'─'*40}")
        self._log("info", f"▶ 开始: {topic}")

    def complete(self):
        elapsed = time.time() - self.start_time
        self._log("info", f"✔ 完成 (耗时 {elapsed:.1f}s)")
        self._log("info", f"{'─'*40}")

    def error(self, msg: str):
        self._log("error", f"✘ {msg}")

    # ── 迭代跟踪 ──

    def iter_start(self, iteration: int, max_iterations: int):
        self._log("info", f"── 第 {iteration}/{max_iterations} 轮 ──", skip_in_brief=True)

    def iter_summary(self, iteration: int, tool_count: int, elapsed: float):
        """单轮迭代结束时的摘要"""
        self._log("info",
            f"   第{iteration}轮小结: 调用{tool_count}个工具, 耗时{elapsed:.1f}s",
            skip_in_brief=True,
        )

    # ── LLM 输出 ──

    def llm_reasoning(self, text: str):
        """LLM 的推理/思考文本（非工具调用部分）"""
        if not text or not text.strip():
            return
        self._log("info", f"🧠 LLM推理: {self._truncate(text.strip())}", skip_in_brief=True)

    def llm_tool_selection(self, tool_calls: list):
        """LLM 选择调用的工具列表"""
        if not tool_calls:
            self._log("info", "📝 LLM决定: 不调用工具，直接输出结论", skip_in_brief=True)
            return
        lines = []
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            args_brief = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            lines.append(f"  → {name}({args_brief})")
        self._log("info", f"🔧 LLM选择工具 ({len(tool_calls)}个):\n" + "\n".join(lines), skip_in_brief=True)

    def llm_output(self, text: str, label: str = ""):
        self._log("info", f"📝 LLM输出({label}): {self._truncate(text)}", skip_in_brief=True)

    # ── 工具执行 ──

    def tool_start(self, tool_name: str, args: dict):
        """工具开始执行"""
        args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
        self._log("info", f"  🔨 执行 {tool_name}({args_str})", skip_in_brief=True)

    def tool_result(self, tool_name: str, result: str, elapsed: float = 0):
        """工具执行结果"""
        time_str = f" ({elapsed:.1f}s)" if elapsed else ""
        self._log("info",
            f"  ✓ {tool_name} 返回{time_str}: {self._truncate(result)}",
            skip_in_brief=True,
        )

    def tool_error(self, tool_name: str, error: str):
        self._log("error", f"  ✘ {tool_name} 失败: {error}")

    def calc(self, tool_name: str, result: str):
        """向后兼容：工具计算结果"""
        self._log("info", f"  ✓ {tool_name}: {self._truncate(result)}", skip_in_brief=True)

    # ── 搜索专用 ──

    def search_call(self, queries: list):
        self._log("info", f"  🔍 发起搜索: {queries}", skip_in_brief=True)

    def search_result(self, count: int, size_kb: float = 0):
        size_str = f", {size_kb:.1f}KB" if size_kb else ""
        self._log("info", f"  🔍 搜索返回: {count}条结果{size_str}", skip_in_brief=True)

    # ── 思考工具 ──

    def think(self, reflection: str):
        self._log("info", f"  💭 思考: {self._truncate(reflection, 600)}", skip_in_brief=True)

    # ── 压缩/合成 ──

    def compress_start(self, msg_count: int, chars: int):
        self._log("info", f"📦 压缩开始: {msg_count}条消息, {chars:,}字符", skip_in_brief=True)

    def compress_complete(self, text: str):
        self._log("info", f"📦 压缩完成: {len(text):,}字符", skip_in_brief=True)
        if self.detail_level == "full":
            self._log("info", f"📦 压缩内容预览: {text[:800]}", skip_in_brief=True)

    # ── 路由决策 ──

    def route(self, from_node: str, to_node: str, reason: str = ""):
        reason_str = f" ({reason})" if reason else ""
        self._log("info", f"↪ 路由: {from_node} → {to_node}{reason_str}", skip_in_brief=True)

    # ── 输入/输出摘要 ──

    def input_summary(self, summary: str):
        """节点接收到的输入数据摘要"""
        self._log("info", f"📥 输入摘要: {summary}", skip_in_brief=True)

    def output_summary(self, summary: str):
        """节点输出的数据摘要"""
        self._log("info", f"📤 输出摘要: {summary}", skip_in_brief=True)


class GraphLogger:
    """全流程日志"""

    def __init__(self):
        self.logger = logging.getLogger("portfolio")
        self.start_time = None

    def start(self, run_mode: str = "full_analysis"):
        self.start_time = time.time()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"组合投资流程启动 | 模式: {run_mode}")
        self.logger.info(f"{'='*60}")

    def log_enabled_nodes(self, nodes: list):
        self.logger.info(f"启用节点: {' → '.join(nodes)}")

    def complete(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.logger.info(f"{'='*60}")
            self.logger.info(f"流程完成 | 总耗时: {elapsed:.1f}s")
            self.logger.info(f"{'='*60}")
