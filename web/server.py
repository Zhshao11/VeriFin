"""VeriFin Web 演示服务。

跑的是**确定性路径**：

    四路召回 → RRF 融合 → 证据选取 → span 硬校验 → 坐标定位 → 六元组 / 拒答

这一条链不经 LLM —— 它要证明的正是「答案里的每个数字都有确定性来源」。
Agent 编排（D3）接入后，会在最前面补上意图解析与工具选择那一环，
但数值、算术、拒答判定永远走这里的确定性代码。

启动:
    .venv/bin/python -m uvicorn web.server:app --port 8765
"""

from __future__ import annotations

import sys
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from verifin.agent import (  # noqa: E402
    Budget,
    ToolRuntime,
    TraceStore,
    VeriFinAgent,
    render_graph_text,
)
from verifin.agent.planner import LLMPlanner  # noqa: E402
from verifin.formulas import FORMULA_REGISTRY, evaluate_formula  # noqa: E402
from verifin.geometry import open_pdf  # noqa: E402
from verifin.span import verify_evidence  # noqa: E402
from verifin.runtime import build_document_runtime  # noqa: E402

# --------------------------------------------------------------------------
# 演示所用的一份真实年报
# --------------------------------------------------------------------------

PDF = ROOT / "data/pdfs/MOUTAI_2024_ANNUAL.pdf"
PRODUCT = ROOT / "data/parsed/moutai2024_fs"
INDEX_DB = ROOT / "data/index/moutai2024.db"
ASSET_DIR = ROOT / "web/assets"
PAGE_FILE = ROOT / "web/index.html"

DOC_ID = "MOUTAI_2024"
COMPANY = "贵州茅台酒股份有限公司（600519）"
PERIOD = "2024 年度（2024-01-01 ~ 2024-12-31）"

#: 公司名与期间目前由演示配置给出，尚未接入封面结构化解析。
#: 这一条在页面上如实标注，不伪装成"已经解析出来了"。
METADATA_SOURCE = "演示配置（封面结构化解析待接入）"


# --------------------------------------------------------------------------
# 启动期一次性装载
# --------------------------------------------------------------------------
#
# 装配集中在 `verifin.runtime`，与命令行演示、评测执行器共用同一份 ——
# 否则三处各写一遍，迟早各自漂移，评测量到的就不是网页上跑的那个系统。

DOC = build_document_runtime(
    doc_id=DOC_ID,
    product=PRODUCT,
    index_db=INDEX_DB,
    pdf=PDF,
    stitched=ROOT / "data/parsed/moutai2024_fs_stitched.md",
)
REPORT = DOC.report
CHUNKS = DOC.chunks
BY_LABEL = DOC.by_label
INDEX = DOC.index
DISCLOSURE_UNIT = DOC.unit
UNIT_SOURCE = DOC.unit_source
#: 检索索引是共享可变资源，读也要串行化。
#: `check_same_thread=False` 只是**允许**跨线程访问，不等于跨线程安全 ——
#: 并发的 `execute` + `commit` 会事务交错。锁由 runtime 统一提供。
INDEX_LOCK = DOC.index_lock
ASSET_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VeriFin Demo", version="0.1.0")


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------


def _to_decimal(raw: str) -> Decimal | None:
    """把单元格原文转成 Decimal。千分位与全角逗号都要容忍。"""
    cleaned = (raw or "").replace(",", "").replace("，", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _render_evidence(page: int, box, out_path: Path, *, zoom: float = 3.0,
                     pad: float = 22.0) -> Path:
    """把某一行高亮后**裁剪**成证据特写图。

    整页图里那一行太小，看不清「框住的到底是不是这个科目」，
    所以对证据行做放大裁剪 —— 这张图是给人工复核用的，不是装饰。
    渲染完立刻删掉批注，避免重复查询时高亮累积。
    """
    doc = pymupdf.open(PDF)
    try:
        pypage = doc[page - 1]
        annot = pypage.add_highlight_annot(pymupdf.Rect(*box.astuple()))
        annot.set_colors(stroke=None, fill=(1.0, 0.85, 0.2))
        annot.update(opacity=0.45)
        clip = pymupdf.Rect(
            max(0.0, box.x0 - pad), max(0.0, box.y0 - pad),
            box.x1 + pad, box.y1 + pad,
        )
        pix = pypage.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)
        pix.save(out_path)
        pypage.delete_annot(annot)
    finally:
        doc.close()
    return out_path


def _claimed_fragment(chunk_text: str, raw_value: str, lead: int = 40) -> str:
    """模拟「模型回吐的原文片段」——从原文里截一段**真实存在**的子串。

    为什么不能自己拼：报表的行文本并不总是「科目 | 本期 | 上期」。
    资产负债表里有一列缩进行名（如「所有者权益（或股东权益）:」），
    拼出来的字符串原文里根本没有，第一层校验会误杀正确的证据。
    真实流程里这一步由 LLM 完成；演示里由程序截取，
    但**校验本身是真的** —— 片段必须逐字存在于原文，数值必须能从片段里解析出来。
    """
    pos = chunk_text.find(raw_value)
    if pos < 0:
        return raw_value
    start = max(0, pos - lead)
    # 从倒数第二个分隔符起截取：这样片段里既有行名又有数值，
    # 单独一个数字做片段虽然也能过校验，但看不出「这个数字属于哪个科目」。
    seps = [i for i, ch in enumerate(chunk_text) if ch == "|" and start <= i < pos]
    if len(seps) >= 2:
        start = seps[-2] + 1
    return chunk_text[start:pos + len(raw_value)].strip().lstrip("|").strip()


def _stage(name: str, status: str, detail: str) -> dict:
    return {"stage": name, "status": status, "detail": detail}


# --------------------------------------------------------------------------
# 请求/响应模型
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200)


class GuardRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200)
    perturbation: str = Field("10000000")


# --------------------------------------------------------------------------
# 页面与静态资源
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE_FILE.read_text(encoding="utf-8"))


@app.get("/assets/{name}")
def asset(name: str) -> FileResponse:
    path = ASSET_DIR / name
    if not path.exists():
        raise LookupError(f"资源不存在：{name}")
    return FileResponse(path)


# --------------------------------------------------------------------------
# 主链路
# --------------------------------------------------------------------------


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """跑一次完整的确定性核验链路。"""
    question = req.question.strip()
    trace: list[dict] = []

    # ---- 1. 四路召回 ----
    with INDEX_LOCK:
        result = INDEX.retrieve(question, top_k=5)
    per_channel = dict(result.per_channel)
    trace.append(_stage(
        "四路召回",
        "ok" if not result.is_empty else "warn",
        " · ".join(f"{k}={v}" for k, v in per_channel.items()),
    ))

    candidates = [
        {
            "label": h.chunk.label,
            "page": h.chunk.page,
            "value": h.chunk.values[0] if h.chunk.values else None,
            "via": list(h.via),
            "ranks": dict(h.ranks),
            "score": round(h.score, 6),
        }
        for h in result.hits
    ]

    if result.is_empty:
        trace.append(_stage("RRF 融合", "fail", "四路均无命中，无候选可融合"))
        trace.append(_stage("拒答判定", "fail", "六元组缺「指标/数值/来源」"))
        return {
            "question": question,
            "decision": "refuse",
            "refuse_reason": "NO_RECALL",
            "refuse_detail": "四条召回通道均无命中。系统不猜测、不补全，直接拒答。",
            "per_channel": per_channel,
            "candidates": [],
            "trace": trace,
        }

    # ---- 2. RRF 融合 ----
    top = result.hits[0]
    trace.append(_stage(
        "RRF 融合", "ok",
        f"top-1 = 「{top.chunk.label}」，融合分 {top.score:.4f}，"
        f"命中通道 {'+'.join(top.via) or '（无）'}",
    ))

    # ---- 3. 证据选取 ----
    # 报表中存在「有科目名、本期列为空」的行（如「终止经营净利润」本期为「—」）。
    # top-1 若落在这样的行上，直接拒答会冤枉正确的候选，
    # 因此按 RRF 顺序往下找到第一条**数值可解析**的证据。
    # 这仍然是确定性行为：顺序由 RRF 名次唯一决定，不做任何语义挑选。
    picked = None
    skipped: list[str] = []
    for h in result.hits:
        raw = h.chunk.values[0] if h.chunk.values else None
        val = _to_decimal(raw or "")
        if val is not None:
            picked = (h, raw, val)
            break
        skipped.append(h.chunk.label)

    if picked is None:
        trace.append(_stage(
            "证据选取", "fail",
            f"候选 {len(result.hits)} 条全部无可用数值（跳过了：{'、'.join(skipped) or '—'}）",
        ))
        trace.append(_stage("拒答判定", "fail", "六元组缺「数值」"))
        return {
            "question": question,
            "decision": "refuse",
            "refuse_reason": "VALUE_UNPARSED",
            "refuse_detail": (
                f"命中 {len(result.hits)} 条候选，但本期列均为空或无法解析"
                f"（{'、'.join(skipped) or '无'}）。不做补全，直接拒答。"
            ),
            "per_channel": per_channel,
            "candidates": candidates,
            "trace": trace,
        }

    hit, raw_value, value = picked
    chunk = hit.chunk
    if skipped:
        trace.append(_stage(
            "证据选取", "warn",
            f"跳过 {len(skipped)} 条空值候选（{'、'.join(skipped)}），取 RRF 第 "
            f"{result.hits.index(hit) + 1} 名「{chunk.label}」",
        ))
    else:
        trace.append(_stage("证据选取", "ok", f"取 RRF top-1「{chunk.label}」"))
    trace.append(_stage("数值归一化", "ok", f"{raw_value} → Decimal（单位：{DISCLOSURE_UNIT}）"))

    # ---- 3. span 硬校验（两层关卡）----
    claimed_span = _claimed_fragment(chunk.text, raw_value or "")
    verdict = verify_evidence(
        claimed_span=claimed_span,
        chunk_text=chunk.text,
        claimed_value=value,
        claimed_unit=DISCLOSURE_UNIT,
    )
    trace.append(_stage(
        "span 硬校验",
        "ok" if verdict.accepted else "fail",
        verdict.detail,
    ))
    if not verdict.accepted:
        trace.append(_stage("拒答判定", "fail", f"证据未通过校验（{verdict.reason}）"))
        return {
            "question": question,
            "decision": "refuse",
            "refuse_reason": verdict.reason or "SPAN_REJECTED",
            "refuse_detail": verdict.detail,
            "per_channel": per_channel,
            "candidates": candidates,
            "trace": trace,
        }

    # ---- 4. 坐标定位（含同行校验）----
    geometry: dict | None = None
    with open_pdf(PDF) as geo:
        loc = geo.locate_row(chunk.page, chunk.label, raw_value or "")
        if loc.same_row_verified and loc.row_box is not None:
            name = f"ev_{uuid.uuid4().hex[:10]}.png"
            _render_evidence(chunk.page, loc.row_box, ASSET_DIR / name)
            box = loc.row_box
            geometry = {
                "verified": True,
                "page": chunk.page,
                "bbox": [round(v, 1) for v in box.astuple()],
                "image": f"/assets/{name}",
            }
            trace.append(_stage(
                "坐标定位", "ok",
                f"p{chunk.page} 同行校验通过，矩形 {tuple(round(v, 1) for v in box.astuple())}",
            ))
        else:
            geometry = {
                "verified": False,
                "page": chunk.page,
                "bbox": None,
                "image": None,
            }
            trace.append(_stage(
                "坐标定位", "warn",
                "未通过同行校验 —— 数值可能不属于该科目，坐标不采纳（答案仍可给出，但标注未定位）",
            ))

    # ---- 5. 六元组完整性 ----
    six = {
        "公司": COMPANY,
        "期间": PERIOD,
        "指标": chunk.label,
        "数值": raw_value,
        "单位": DISCLOSURE_UNIT,
        "来源": f"第 {chunk.page} 页" + (f" · 坐标 {tuple(geometry['bbox'])}" if geometry and geometry["bbox"] else ""),
    }
    missing = [k for k, v in six.items() if not v]
    if missing:
        trace.append(_stage("拒答判定", "fail", f"六元组缺：{'、'.join(missing)}"))
        return {
            "question": question,
            "decision": "refuse",
            "refuse_reason": "SIX_TUPLE_INCOMPLETE",
            "refuse_detail": f"六元组缺字段：{'、'.join(missing)}。程序级拒答，模型无否决权。",
            "per_channel": per_channel,
            "candidates": candidates,
            "trace": trace,
        }

    trace.append(_stage("六元组完整性", "ok", "六个字段齐全 → 输出答案"))

    return {
        "question": question,
        "decision": "answer",
        "six_tuple": six,
        "metadata_source": METADATA_SOURCE,
        "fragment": verdict.span_hit.original_text if verdict.span_hit else chunk.text,
        "span_mode": verdict.span_hit.mode.value if verdict.span_hit else None,
        "per_channel": per_channel,
        "candidates": candidates,
        "geometry": geometry,
        "trace": trace,
    }


# --------------------------------------------------------------------------
# 防幻觉演示台
# --------------------------------------------------------------------------


@app.post("/api/guard")
def guard(req: GuardRequest) -> dict:
    """把同一条证据喂给 span 硬校验三次，看护栏怎么拦截。

    三例分别是：真实值 / 篡改值 / 伪造片段。
    第二例最危险 —— 片段是真的，数字是假的，只读片段那一层会放行。
    """
    with INDEX_LOCK:
        result = INDEX.retrieve(req.question, top_k=1)
    if result.is_empty:
        return {"question": req.question, "cases": [], "note": "无命中，无法演示"}

    chunk = result.hits[0].chunk
    raw = chunk.values[0] if chunk.values else ""
    value = _to_decimal(raw)
    if value is None:
        return {"question": req.question, "cases": [], "note": "数值无法解析"}

    base_span = _claimed_fragment(chunk.text, raw)

    def run(span: str, val: Decimal, title: str, note: str) -> dict:
        v = verify_evidence(
            claimed_span=span,
            chunk_text=chunk.text,
            claimed_value=val,
            claimed_unit=DISCLOSURE_UNIT,
        )
        return {
            "title": title,
            "note": note,
            "claimed_span": span,
            "claimed_value": str(val),
            "accepted": v.accepted,
            "reason": v.reason,
            "mode": v.match_mode.value if v.match_mode else None,
            "detail": v.detail,
        }

    try:
        delta = Decimal(req.perturbation)
    except InvalidOperation:
        delta = Decimal("10000000")

    cases = [
        run(base_span, value, "① 真实值", "模型如实回吐原文片段与数值"),
        run(base_span, value + delta, "② 篡改值",
            "片段是真的，数值被改了 —— 只读片段那一层拦不住"),
        run("经管理层确认，该科目金额无误", value, "③ 伪造片段",
            "模型编了一段原文里不存在的片段"),
    ]
    return {
        "question": req.question,
        "label": chunk.label,
        "page": chunk.page,
        "true_value": raw,
        "cases": cases,
    }


# --------------------------------------------------------------------------
# 勾稽核验
# --------------------------------------------------------------------------


@app.get("/api/formulas")
def formulas() -> dict:
    return {
        "formulas": [
            {
                "id": f.id,
                "name": f.name,
                "kind": f.kind.value,
                "description": f.description,
                "operands": list(f.operand_names),
            }
            for f in FORMULA_REGISTRY.values()
        ]
    }


F1_TERMS: tuple[tuple[str, int], ...] = (
    ("资产总计", +1),
    ("负债合计", -1),
    ("所有者权益（或股东权益）合计", -1),
)

#: 公式注册表用**规范科目名**，报表用的是**披露科目名**，两者不一致。
#: 例：F1 的操作数写作「所有者权益合计」，茅台报表里叫「所有者权益（或股东权益）合计」。
#: 不做这张对齐表，缺操作数会被判成 WARN（无法判定），
#: 于是「没取到科目」和「取到了但算不平」混成同一种结论 —— 这是口径缺陷，不是工程 bug。
#: 见 docs/技术问题留档.md P-012。
LABEL_ALIAS: dict[str, str] = {
    "所有者权益合计": "所有者权益（或股东权益）合计",
    "归属于母公司股东的净利润": "归属于母公司所有者的净利润",
    "归属于母公司所有者权益合计": "归属于母公司所有者权益合计",
}


def _resolve_label(name: str) -> str | None:
    """把公式里的规范科目名对齐到报表实际行名。

    只做**精确匹配 + 显式别名**，不做模糊包含 ——
    模糊匹配会让「负债合计」落到「非流动负债合计」上，
    那正是本项目要防的张冠李戴。
    """
    if name in BY_LABEL:
        return name
    alias = LABEL_ALIAS.get(name)
    if alias and alias in BY_LABEL:
        return alias
    return None


@app.post("/api/identity")
def identity(period: str = "current") -> dict:
    """核验 资产 = 负债 + 所有者权益（本期 / 上期双列）。"""
    column = 0 if period == "current" else 1
    operands: dict[str, Decimal] = {}
    missing: list[str] = []
    sources: list[dict] = []

    formula = FORMULA_REGISTRY["F1"]
    for name in formula.operand_names:
        label = _resolve_label(name)
        chunk = BY_LABEL.get(label) if label else None
        if chunk is None or len(chunk.values) <= column:
            missing.append(name)
            continue
        raw = chunk.values[column]
        val = _to_decimal(raw)
        if val is None:
            missing.append(name)
            continue
        operands[name] = val
        sources.append({
            "科目": name,
            "报表行名": label,
            "值": raw,
            "页码": chunk.page,
        })

    if missing:
        return {
            "period": period,
            "available": False,
            "verdict": "WARN",
            "detail": f"缺科目：{'、'.join(missing)}，无法核验。",
            "sources": sources,
        }

    outcome = evaluate_formula("F1", operands, disclosure_unit=DISCLOSURE_UNIT)
    return {
        "period": period,
        "available": True,
        "formula_id": outcome.formula_id,
        "formula_name": outcome.formula_name,
        "verdict": outcome.verdict.value,
        "lhs": str(outcome.lhs),
        "rhs": str(outcome.rhs),
        "diff": str(outcome.diff),
        "tolerance": str(outcome.tolerance),
        "detail": outcome.detail,
        "sources": sources,
        "disclosure_unit": DISCLOSURE_UNIT,
    }


@app.get("/api/meta")
def meta() -> dict:
    with INDEX_LOCK:
        embedder = INDEX.embedder_name
    return {
        "company": COMPANY,
        "period": PERIOD,
        "metadata_source": METADATA_SOURCE,
        "disclosure_unit": DISCLOSURE_UNIT,
        "disclosure_unit_source": UNIT_SOURCE,
        "doc_id": DOC_ID,
        "chunk_count": len(CHUNKS),
        "table_count": REPORT.table_count,
        "embedder": embedder,
        "pdf": PDF.name,
        "agent_graph": render_graph_text(),
    }


# --------------------------------------------------------------------------
# Agent 编排（D3）
# --------------------------------------------------------------------------

AGENT_TRACE = TraceStore(ROOT / "data/index/agent_trace_web.db")

AGENT_RUNTIME = DOC.tool_runtime(
    company=COMPANY,
    period=PERIOD,
    pdf_open=lambda: open_pdf(PDF),
)

# LLM 客户端可以共享（无状态、贵在建连），但**调度器必须每次请求新建**：
# LLMPlanner 自己持有 LLM 调用计数器，共享它的话并发请求会互相偷预算，
# 表现是"某个请求莫名其妙走了兜底策略"——不报错、只降级（留档 P-015）。
_CLIENT_STATE: dict[str, object] = {"client": None, "error": None}


def _llm_client():
    """惰性构造 LLM 客户端。端点不可用时把原因留下，不静默降级。"""
    if _CLIENT_STATE["client"] is not None or _CLIENT_STATE["error"] is not None:
        return _CLIENT_STATE["client"], _CLIENT_STATE["error"]
    try:
        from verifin.llm import LLMClient

        _CLIENT_STATE["client"] = LLMClient.from_env()
    except Exception as exc:  # noqa: BLE001
        _CLIENT_STATE["error"] = f"{type(exc).__name__}: {exc}"
    return _CLIENT_STATE["client"], _CLIENT_STATE["error"]


def _fresh_planner(max_calls: int):
    """每次请求一个**新的**调度器 —— 预算按请求隔离。"""
    client, err = _llm_client()
    if client is None:
        return None, err
    return LLMPlanner(client, max_calls=max_calls), None


@app.post("/api/agent")
def agent_run(req: AskRequest, planner: str = "policy") -> dict:
    """跑一次完整编排：调度器在图里选节点，每一步工具调用都入轨迹库。"""
    use_llm = planner == "llm"
    impl = None
    note = None
    budget = Budget()
    if use_llm:
        impl, err = _fresh_planner(budget.max_llm_calls)
        if impl is None:
            note = f"LLM 调度不可用（{err}），本次改用确定性策略"

    agent = VeriFinAgent(
        AGENT_RUNTIME,
        budget=budget,
        planner=impl,
        trace=AGENT_TRACE,
        use_llm=impl is not None,
    )
    result = agent.run(req.question.strip())

    steps = [
        {
            "seq": s.seq,
            "node": s.node,
            "tool": s.tool,
            "args": s.args,
            "ok": s.ok,
            "detail": s.detail,
            "source": s.source,
        }
        for s in result.steps
    ]
    payload = {
        "question": result.question,
        "run_id": result.run_id,
        "route": result.route,
        "decision": result.decision,
        "answer": result.answer,
        "refusal": result.refusal,
        "steps": steps,
        "tool_calls": result.tool_calls,
        "llm_calls": result.llm_calls,
        "budget_exceeded": result.budget_exceeded,
        "note": note,
        "replay": AGENT_TRACE.abbreviated(result.run_id),
        "graph": render_graph_text(),
    }
    return payload


@app.get("/api/agent/runs")
def agent_runs(limit: int = 10) -> dict:
    """最近的运行记录（轨迹可复盘）。"""
    rows = AGENT_TRACE.runs()[: max(1, min(limit, 50))]
    return {
        "runs": [
            {
                "run_id": r["run_id"],
                "question": r["question"],
                "decision": r["decision"],
                "reason": r["reason"],
                "route": r["route"],
                "steps": r["steps"],
                "tool_calls": r["tool_calls"],
                "llm_calls": r["llm_calls"],
                "budget_exceeded": bool(r["budget_exceeded"]),
                "planner_source": r["planner_source"],
                "started_at": r["started_at"],
            }
            for r in rows
        ]
    }
