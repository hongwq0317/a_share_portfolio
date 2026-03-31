"""Skill 驱动的数据获取工具

提供 read_skill / chinalin_http_request 工具，以及 Skill 发现机制。
LLM 通过 <available_skills> 发现可用 Skill，按需读取后动态调用。
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import requests
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.config import get_app_config

logger = logging.getLogger("portfolio")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SKILLS_DIR = _PROJECT_ROOT / "skills"


# ---------------------------------------------------------------------------
# Skill 发现（OpenClaw 风格）
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> dict[str, str]:
    """从 SKILL.md 的 YAML frontmatter 中提取 name 和 description。"""
    m = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
    if not m:
        return {}
    result: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            key = key.strip()
            val = val.strip()
            if key in ('name', 'description'):
                result[key] = val
    return result


def discover_skills() -> list[dict[str, str]]:
    """扫描 skills/ 目录，返回所有 Skill 的元信息列表。"""
    skills: list[dict[str, str]] = []
    if not _SKILLS_DIR.is_dir():
        return skills
    for skill_dir in sorted(_SKILLS_DIR.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        text = skill_md.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        skills.append({
            "name": meta.get("name", skill_dir.name),
            "description": meta.get("description", ""),
            "location": skill_dir.name,
        })
    return skills


def build_available_skills_prompt() -> str:
    """生成 <available_skills> XML，供注入系统提示词。

    遵循 OpenClaw 模式：LLM 扫描描述 → 选择最相关 Skill → 用 read_skill 读取。
    """
    skills = discover_skills()
    if not skills:
        return ""

    lines = [
        "## Skills",
        "你可以通过 read_skill 工具按需读取以下技能文档。",
        "规则：先扫描 <available_skills> 中的描述，选择与当前任务最相关的 Skill 读取，不要一次读取多个。",
        "",
        "<available_skills>",
    ]
    for s in skills:
        lines.append(f'  <skill name="{s["name"]}" location="{s["location"]}">')
        lines.append(f'    <description>{s["description"]}</description>')
        # 列出补充文档
        skill_dir = _SKILLS_DIR / s["location"]
        sections = [
            f.name for f in sorted(skill_dir.iterdir())
            if f.is_file() and f.name != "SKILL.md"
        ]
        if sections:
            lines.append(f'    <sections>{", ".join(sections)}</sections>')
        lines.append('  </skill>')
    lines.append("</available_skills>")
    lines.append("")
    return "\n".join(lines)

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
})

_DEFAULT_MAX_ARRAY_ITEMS = 20


def _prune_large_arrays(obj: Any, max_items: int) -> Any:
    """递归裁剪 JSON 中过大的数组，保留前几条 + 摘要。

    典型场景：/v2/market/overview 的分钟级时序数据（~240条）、
    分时K线等，裁剪后从 150K chars 降至 ~5K chars。
    """
    if max_items <= 0:
        return obj
    if isinstance(obj, dict):
        return {k: _prune_large_arrays(v, max_items) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_items:
            has_complex = any(isinstance(item, (dict, list)) for item in obj[:3])
            if has_complex:
                kept = [_prune_large_arrays(item, max_items) for item in obj[:3]]
                kept.append({"_pruned": f"共 {len(obj)} 条，已省略 {len(obj) - 3} 条"})
                return kept
        return [_prune_large_arrays(item, max_items) for item in obj]
    return obj


@tool
def read_skill(skill_name: str, section: str = "SKILL.md") -> str:
    """读取项目 skills 目录下的技能文档，获取 API 接口规范。

    第一次获取数据前必须先调用此工具了解可用接口。

    Args:
        skill_name: 技能名称，对应 skills/ 下的目录名，如 "chinalin-market-api"
        section: 要读取的文件名，默认 "SKILL.md"。
                 可传 "response-fields.md" 获取详细响应字段说明。
    """
    skill_dir = _SKILLS_DIR / skill_name
    if not skill_dir.is_dir():
        available = [d.name for d in _SKILLS_DIR.iterdir() if d.is_dir()]
        return f"技能 '{skill_name}' 不存在。可用技能: {available}"

    target = skill_dir / section
    if not target.is_file():
        files = [f.name for f in skill_dir.iterdir() if f.is_file()]
        return f"文件 '{section}' 不存在于 {skill_name}/ 中。可用文件: {files}"

    content = target.read_text(encoding="utf-8")
    logger.info(f"[Skill] 读取 {skill_name}/{section} ({len(content)} chars)")
    return content


@tool
def chinalin_http_request(
    endpoint: str,
    params: Optional[dict] = None,
    method: str = "GET",
    max_array_items: int = 0,
    config: Optional[RunnableConfig] = None,
) -> str:
    """向 ChinaLin 内部行情 API 发送 HTTP 请求并返回 JSON 结果。

    使用前请先调用 read_skill("chinalin-market-api") 了解可用端点和参数。
    工具会自动拼接 base_url，只需传入路径部分。

    响应中超过阈值的大数组会被自动裁剪（保留前 3 条 + 摘要），
    避免分钟级时序数据等占用过多 token。

    Args:
        endpoint: API 路径，如 "/v1/quotes/fields"
        params: 请求参数字典，GET 请求作为 query string，POST 请求作为 body
        method: HTTP 方法，"GET" 或 "POST"，默认 "GET"
        max_array_items: 数组最大保留条数，超过则自动裁剪。
                         0 表示使用默认值(20)，-1 表示不裁剪。
    """
    base_url = "https://chinalintest.wenxingonline.com"
    timeout = 10
    max_chars = 50000

    if config:
        try:
            app_config = get_app_config(config)
            ds = getattr(app_config, "data_source", None)
            if ds:
                base_url = getattr(ds, "chinalin_base_url", base_url)
                timeout = getattr(ds, "http_timeout", timeout)
                max_chars = getattr(ds, "max_response_chars", max_chars)
        except Exception:
            pass

    url = f"{base_url.rstrip('/')}{endpoint}"
    params = params or {}

    logger.info(f"[ChinaLin] {method} {endpoint} params={params}")

    try:
        if method.upper() == "POST":
            resp = _session.post(url, params=params, timeout=timeout)
        else:
            resp = _session.get(url, params=params, timeout=timeout)

        resp.raise_for_status()
        data = resp.json()

        prune_threshold = max_array_items if max_array_items != 0 else _DEFAULT_MAX_ARRAY_ITEMS
        if prune_threshold > 0:
            raw_text = json.dumps(data, ensure_ascii=False)
            raw_len = len(raw_text)
            data = _prune_large_arrays(data, prune_threshold)
            pruned_text = json.dumps(data, ensure_ascii=False)
            if len(pruned_text) < raw_len:
                logger.info(
                    f"[ChinaLin] {endpoint} 响应裁剪: "
                    f"{raw_len:,} → {len(pruned_text):,} chars "
                    f"(阈值={prune_threshold})"
                )

        result_text = json.dumps(data, ensure_ascii=False, indent=2)
        if len(result_text) > max_chars:
            result_text = result_text[:max_chars] + f"\n\n... [响应被截断，原始长度 {len(result_text)} chars]"

        logger.info(f"[ChinaLin] {endpoint} 最终响应 {len(result_text)} chars")
        return result_text

    except requests.Timeout:
        msg = f"请求超时 ({timeout}s): {method} {endpoint}"
        logger.warning(f"[ChinaLin] {msg}")
        return f"错误: {msg}"
    except requests.HTTPError as e:
        msg = f"HTTP 错误 {e.response.status_code}: {method} {endpoint}"
        logger.warning(f"[ChinaLin] {msg}")
        return f"错误: {msg}"
    except requests.ConnectionError:
        msg = f"连接失败: {base_url}"
        logger.warning(f"[ChinaLin] {msg}")
        return f"错误: {msg}"
    except Exception as e:
        msg = f"请求异常: {e}"
        logger.warning(f"[ChinaLin] {msg}")
        return f"错误: {msg}"


SKILL_TOOLS = [read_skill, chinalin_http_request]
