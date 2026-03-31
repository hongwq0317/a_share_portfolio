"""通用 ReAct 子图工厂

封装所有 ReAct 子图（市场分析、选股、组合策略、风险评估、交易执行）的通用模式：
  researcher  →  researcher_tools  →  compress_research
       ↑              |                    ^
       |              |  (iter=max-1)       |
       |              +--→ summarize_research
       +________________________|  (continue loop)

通过 ReActSubgraphConfig 数据类参数化，每个具体节点仅需配置参数。
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.config import get_app_config, get_configured_model
from src.state import ResearcherSubState, ResearcherOutputState
from src.utils import NodeLogger, extract_text_from_content, log_token_usage

logger = logging.getLogger("portfolio")


@dataclass
class ReActSubgraphConfig:
    """ReAct 子图实例的完整配置。

    Attributes:
        node_name: 节点标识，用于日志和节点命名
        tools: 绑定给 LLM 的工具列表
        system_prompt_template: system prompt 模板
        compress_prompt: 压缩 prompt
        synthesize_prompt: 合成 prompt（压缩禁用时的回退）
        compression_switch_attr: NodeSwitches 中的压缩开关属性名
        max_iterations_attr: ReactConfig 中的迭代次数属性名
        model_role: 使用的模型角色 (research/decision)
        early_exit_check: 可选，提前退出检查
        tool_result_hooks: 可选，工具执行后的回调
    """
    node_name: str
    tools: list
    system_prompt_template: str
    compress_prompt: str
    synthesize_prompt: str
    compression_switch_attr: str
    max_iterations_attr: str = "max_iterations"
    model_role: str = "research"
    compress_request_msg: str = "请整理以上研究结果"
    early_exit_check: Callable | None = None
    tool_result_hooks: Dict[str, Callable] | None = None
    no_tool_continuation_check: Callable | None = None
    max_no_tool_retries: int = 2
    summarize_request_msg: str = (
        "请基于以上研究对话，输出完整的分析报告。"
        "注意：即将达到最大迭代次数，本次为最后机会，请务必汇总已有信息形成完整报告。"
        "不要调用工具，直接输出报告。"
    )


def build_react_subgraph(cfg: ReActSubgraphConfig):
    """工厂函数：根据配置构建一个完整的 ReAct 子图。"""

    prefix = cfg.node_name.lower().replace(" ", "_")
    researcher_name = f"{prefix}_researcher"
    tools_name = f"{prefix}_researcher_tools"
    summarize_name = f"summarize_{prefix}_research"
    compress_name = f"compress_{prefix}_research"

    tools_by_name = {t.name: t for t in cfg.tools}

    # ============================================================
    # Node 1: Researcher
    # ============================================================

    async def researcher(
        state: ResearcherSubState, config: RunnableConfig
    ) -> Command:
        """通用 ReAct 研究员节点"""
        app_config = get_app_config(config)
        nl = NodeLogger(cfg.node_name, app_config.logging.detail_level)

        researcher_messages = state.get("researcher_messages", [])
        iteration = state.get("tool_call_iterations", 0) + 1
        max_iterations = getattr(app_config.react, cfg.max_iterations_attr, 8)

        if iteration == 1:
            nl.start(f"{cfg.node_name}: {state.get('research_topic', '')}")
            nl._log("info",
                f"可用工具: {[t.name for t in cfg.tools]}",
                skip_in_brief=True,
            )
            nl._log("info", f"最大迭代: {max_iterations}, 模型角色: {cfg.model_role}")

        nl.iter_start(iteration, max_iterations)

        research_model = get_configured_model(
            app_config, role=cfg.model_role, bind_tools=cfg.tools,
        )

        system_prompt = state.get("system_prompt", "")
        if not system_prompt:
            try:
                system_prompt = cfg.system_prompt_template.format(
                    max_iterations=max_iterations,
                )
            except KeyError:
                system_prompt = cfg.system_prompt_template
        messages = [SystemMessage(content=system_prompt)] + list(researcher_messages)
        nl._log("info",
            f"上下文长度: {len(researcher_messages)}条消息, "
            f"约{sum(len(str(m.content)) for m in researcher_messages):,}字符",
            skip_in_brief=True,
        )

        try:
            start_time = time.time()
            response = await research_model.ainvoke(messages)
            elapsed = time.time() - start_time
            log_token_usage(response, cfg.node_name, f"researcher/iter{iteration}", messages=messages)

            response_text = extract_text_from_content(response.content) if response.content else ""
            tool_calls = response.tool_calls if response.tool_calls else []

            nl._log("info", f"LLM响应耗时: {elapsed:.2f}s", skip_in_brief=True)

            if response_text:
                nl.llm_reasoning(response_text)

            nl.llm_tool_selection(tool_calls)

        except Exception as exc:
            nl.error(f"LLM调用失败 (第{iteration}轮): {exc}")
            response = AIMessage(
                content=f"研究过程中遇到异常，尝试基于已有信息进行总结。错误: {exc}"
            )

        return Command(
            goto=tools_name,
            update={
                "researcher_messages": [response],
                "tool_call_iterations": iteration,
            },
        )

    # ============================================================
    # Node 2: Researcher Tools
    # ============================================================

    async def researcher_tools(
        state: ResearcherSubState, config: RunnableConfig
    ) -> Command:
        """通用工具执行节点"""
        app_config = get_app_config(config)
        nl = NodeLogger(cfg.node_name, app_config.logging.detail_level)
        max_iterations = getattr(app_config.react, cfg.max_iterations_attr, 8)

        researcher_messages = state.get("researcher_messages", [])
        most_recent_message = researcher_messages[-1] if researcher_messages else None

        iterations = state.get("tool_call_iterations", 0)

        has_tool_calls = bool(
            most_recent_message and getattr(most_recent_message, "tool_calls", None)
        )
        if not has_tool_calls:
            if (
                cfg.no_tool_continuation_check
                and iterations < max_iterations - 1
            ):
                CONT_MARKER = "[CONTINUATION_REMINDER]"
                continuation_count = sum(
                    1 for m in researcher_messages
                    if isinstance(m, HumanMessage)
                    and CONT_MARKER in str(m.content)
                )
                if continuation_count < cfg.max_no_tool_retries:
                    ai_text = extract_text_from_content(
                        getattr(most_recent_message, "content", "")
                    ) if most_recent_message else ""
                    reminder = cfg.no_tool_continuation_check(
                        ai_text, iterations, max_iterations
                    )
                    if reminder:
                        tagged_reminder = f"{CONT_MARKER}\n{reminder}"
                        nl.route(
                            "tools", researcher_name,
                            f"检测到未完成任务(第{continuation_count+1}次提醒)，继续执行",
                        )
                        return Command(
                            goto=researcher_name,
                            update={
                                "researcher_messages": [
                                    HumanMessage(content=tagged_reminder)
                                ],
                            },
                        )
            nl.route("tools", compress_name, "LLM未调用工具，直接进入压缩")
            return Command(goto=compress_name)

        if iterations >= max_iterations:
            nl.route("tools", compress_name, f"已达最大迭代次数({iterations})")
            return Command(goto=compress_name)

        tool_calls = most_recent_message.tool_calls
        tool_outputs = []
        early_exit = False
        early_exit_reason = ""
        iter_start_time = time.time()

        nl._log("info", f"开始执行 {len(tool_calls)} 个工具调用", skip_in_brief=True)

        for i, tc in enumerate(tool_calls, 1):
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc.get("id", f"tool_{tool_name}")

            if tool_name not in tools_by_name:
                result = f"未知工具: {tool_name}"
                nl._log("warning", f"⚠️ 未知工具: {tool_name} (可用: {list(tools_by_name.keys())})")
            else:
                nl.tool_start(tool_name, tool_args)
                try:
                    tool_start = time.time()
                    result = await tools_by_name[tool_name].ainvoke(tool_args, config)
                    tool_elapsed = time.time() - tool_start
                    result_str = str(result)

                    if tool_name == "think_tool":
                        nl.think(tool_args.get("reflection", ""))
                    elif tool_name == "web_search":
                        nl.search_call(tool_args.get("queries", []))
                        result_lines = result_str.split("\n")
                        result_count = sum(1 for l in result_lines if l.strip().startswith("["))
                        nl.search_result(result_count, len(result_str) / 1024)
                        nl.tool_result(tool_name, result_str, tool_elapsed)
                    else:
                        nl.tool_result(tool_name, result_str, tool_elapsed)

                    if cfg.early_exit_check:
                        should_exit, reason = cfg.early_exit_check(tool_name, tool_args, result)
                        if should_exit:
                            early_exit = True
                            early_exit_reason = reason

                    if cfg.tool_result_hooks and tool_name in cfg.tool_result_hooks:
                        cfg.tool_result_hooks[tool_name](tool_args, result, state)

                except Exception as e:
                    result = f"工具执行错误: {str(e)}"
                    nl.tool_error(tool_name, str(e))

            tool_outputs.append(
                ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id)
            )

        nl.iter_summary(iterations, len(tool_calls), time.time() - iter_start_time)

        if early_exit:
            nl.route("tools", compress_name, f"提前退出: {early_exit_reason}")
            return Command(
                goto=compress_name,
                update={
                    "researcher_messages": tool_outputs,
                    "early_exit_not_applicable": True,
                    "early_exit_reason": early_exit_reason,
                },
            )

        if iterations >= max_iterations:
            nl.route("tools", compress_name, f"达到最大迭代({iterations})")
            return Command(
                goto=compress_name,
                update={"researcher_messages": tool_outputs},
            )

        if iterations == max_iterations - 1:
            nl.route("tools", summarize_name, "倒数第二轮，进入汇总阶段")
            return Command(
                goto=summarize_name,
                update={"researcher_messages": tool_outputs},
            )

        nl.route("tools", researcher_name, f"继续第{iterations + 1}轮研究")
        return Command(
            goto=researcher_name,
            update={"researcher_messages": tool_outputs},
        )

    # ============================================================
    # Node 2.5: Summarize Research
    # ============================================================

    async def summarize_research(
        state: ResearcherSubState, config: RunnableConfig
    ) -> Command:
        """汇总节点：基于现有对话生成完整报告"""
        app_config = get_app_config(config)
        nl = NodeLogger(cfg.node_name, app_config.logging.detail_level)

        researcher_messages = state.get("researcher_messages", [])
        total_chars = sum(len(str(m.content)) for m in researcher_messages)
        nl._log("info",
            f"📋 汇总阶段: 基于 {len(researcher_messages)} 条消息 ({total_chars:,}字符) 生成报告")

        synth_messages = (
            [SystemMessage(content=cfg.synthesize_prompt)]
            + list(researcher_messages)
            + [HumanMessage(content=cfg.summarize_request_msg)]
        )

        summarize_model = get_configured_model(app_config, role="compression", bind_tools=None)
        try:
            start_time = time.time()
            response = await summarize_model.ainvoke(synth_messages)
            elapsed = time.time() - start_time
            log_token_usage(response, cfg.node_name, "summarize", messages=synth_messages)
            summary_text = extract_text_from_content(response.content)
            summary_msg = AIMessage(content=summary_text)
            nl._log("info", f"📋 汇总完成: {len(summary_text):,}字符, 耗时{elapsed:.1f}s")
            nl.llm_output(summary_text, "汇总报告")
            nl.route("summarize", compress_name, "汇总完成，进入压缩")
            return Command(
                goto=compress_name,
                update={"researcher_messages": [summary_msg]},
            )
        except Exception as exc:
            nl.error(f"汇总失败: {exc}")
            nl.route("summarize", compress_name, "汇总失败，尝试压缩")
            return Command(goto=compress_name)

    # ============================================================
    # Node 3: Compress Research
    # ============================================================

    async def compress_research(
        state: ResearcherSubState, config: RunnableConfig
    ) -> dict:
        """通用压缩节点"""
        app_config = get_app_config(config)
        nl = NodeLogger(cfg.node_name, app_config.logging.detail_level)

        if state.get("early_exit_not_applicable"):
            reason = state.get("early_exit_reason", "")
            compressed = f"**{cfg.node_name} 提前退出**: {reason}"
            nl._log("info", f"⏭ 提前退出: {reason}")
            nl.compress_complete(compressed)
            nl.complete()
            return {"compressed_research": compressed, "raw_notes": [compressed]}

        researcher_messages = state.get("researcher_messages", [])
        total_chars = sum(len(str(m.content)) for m in researcher_messages)
        nl.compress_start(len(researcher_messages), total_chars)

        # 统计对话中各类消息
        ai_count = sum(1 for m in researcher_messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in researcher_messages if isinstance(m, ToolMessage))
        human_count = sum(1 for m in researcher_messages if isinstance(m, HumanMessage))
        nl._log("info",
            f"对话构成: AI消息={ai_count}, 工具结果={tool_count}, Human消息={human_count}",
            skip_in_brief=True,
        )

        report_content = ""
        for m in reversed(researcher_messages):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                txt = extract_text_from_content(getattr(m, "content", ""))
                if txt.strip():
                    report_content = txt
                    break

        if not report_content:
            nl._log("info", "⚠ 无有效AI报告文本，启用 synthesize 回退")
            compress_model = get_configured_model(app_config, role="compression")
            synth_messages = (
                [SystemMessage(content=cfg.synthesize_prompt)]
                + list(researcher_messages)
                + [HumanMessage(content=f"请基于以上研究对话，输出完整的{cfg.node_name}分析报告。")]
            )
            report_content = f"{cfg.node_name} 分析完成（无有效报告）"
            for attempt in range(1, 4):
                try:
                    nl._log("info", f"Synthesize 尝试 {attempt}/3 ...", skip_in_brief=True)
                    response = await compress_model.ainvoke(synth_messages)
                    log_token_usage(response, cfg.node_name, f"synthesize/attempt{attempt}")
                    report_content = extract_text_from_content(response.content)
                    nl._log("info", f"Synthesize 成功: {len(report_content):,}字符")
                    break
                except Exception as exc:
                    nl.error(f"Synthesize 尝试 {attempt}/3 失败: {exc}")
        else:
            nl._log("info", f"找到有效AI报告: {len(report_content):,}字符", skip_in_brief=True)
            nl.llm_output(report_content, "原始报告")

        enable_compression = getattr(
            app_config.nodes, cfg.compression_switch_attr, True
        )

        if not enable_compression:
            nl._log("info", "压缩已禁用，直接使用原始报告", skip_in_brief=True)
            compressed = report_content
        else:
            compress_model = get_configured_model(app_config, role="compression")
            messages = [
                SystemMessage(content=cfg.compress_prompt),
                HumanMessage(content=report_content),
                HumanMessage(content=cfg.compress_request_msg),
            ]
            compressed = report_content
            for attempt in range(1, 4):
                try:
                    nl._log("info", f"压缩尝试 {attempt}/3 ...", skip_in_brief=True)
                    start_time = time.time()
                    response = await compress_model.ainvoke(messages)
                    elapsed = time.time() - start_time
                    log_token_usage(response, cfg.node_name, f"compress/attempt{attempt}")
                    compressed = extract_text_from_content(response.content)
                    nl._log("info",
                        f"压缩完成: {len(report_content):,} → {len(compressed):,}字符 "
                        f"(压缩率{len(compressed)/max(len(report_content),1)*100:.0f}%), "
                        f"耗时{elapsed:.1f}s",
                    )
                    break
                except Exception as exc:
                    nl.error(f"压缩尝试 {attempt}/3 失败: {exc}")
                    if attempt == 3:
                        compressed = report_content
            nl.compress_complete(compressed)
        nl.output_summary(f"最终输出 {len(compressed):,}字符")
        nl.complete()
        return {
            "compressed_research": compressed,
            "raw_notes": [report_content] if report_content else [compressed],
        }

    # ============================================================
    # 构建子图
    # ============================================================

    builder = StateGraph(ResearcherSubState, output=ResearcherOutputState)

    builder.add_node(researcher_name, researcher)
    builder.add_node(tools_name, researcher_tools)
    builder.add_node(summarize_name, summarize_research)
    builder.add_node(compress_name, compress_research)

    builder.add_edge(START, researcher_name)
    builder.add_edge(compress_name, END)

    return builder.compile()
