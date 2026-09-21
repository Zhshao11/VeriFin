"""生成 Track B 评测题库（贵州茅台 2024 年年度报告）。

用法::

    .venv/bin/python scripts/build_eval_bank.py            # 写入题库
    .venv/bin/python scripts/build_eval_bank.py --dry-run  # 只校验不写盘

**为什么用脚本生成而不是手写 JSONL。**
题库里最脆的一环是 `gold_evidence`（标注的原文片段）—— 手抄一次就可能抄错，
而抄错的片段会让「三道闸门」把一条好题判死，排查时又很难看出是抄错还是解析错。
这里改成从检索索引里**原样取行文本**，手抄错的可能性被彻底消除；
金标准数值仍然逐条写在下面的表里，并与索引里的值做断言比对
（两处独立填写、互相校验，任一填错都会在这里就炸掉）。

题库的分层、指标口径与拒答设计见 `docs/评测方案-v1.0.md`。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifin.eval import (  # noqa: E402
    DOC_REGISTRY,
    EvalItem,
    bank_stats,
    render_stats,
    report_problems,
    save_bank,
    validate_bank,
)

INDEX_DB = DOC_REGISTRY["MOUTAI_2024"].index_db
OUT = ROOT / "data" / "eval" / "b_moutai_2024.jsonl"

DOC = "MOUTAI_2024"
TRACK = "B"
COMPANY = "贵州茅台酒股份有限公司（600519）"
PERIOD = "2024 年度"
SOURCE = "自建（Track B，人工核验自年报原文）"


# ---------------------------------------------------------------------------
# 题表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """一条 L1 单值抽取题：一个科目 → 一个值。"""

    seq: int
    page: int
    label: str
    value: str
    question: str
    difficulty: str
    scope: str = "合并"
    unit: str = "元"
    notes: str = ""


@dataclass(frozen=True)
class Verify:
    """一条 L2 勾稽 / 计算题。

    `operands` 每项是 `(页码, 报表行名, 公式操作数名, 数值)`：
    报表行名用来从索引里取证据片段，公式操作数名用来和系统返回的操作数对齐。

    两者**必须分开**：报表行名是「所有者权益（或股东权益）合计」，
    而注册表里的公式操作数名是「所有者权益合计」——用报表行名去比操作数，
    会得到「实得 None」这种假失败（D4 实测踩过）。

    `verdict` 为 None 时表示这题的答案是数值（派生量 / 综合题），用 `value` 填。
    """

    seq: int
    question: str
    formula: str
    operands: tuple[tuple[int, str, str, str], ...]
    difficulty: str
    scope: str = "合并"
    qtype: str = "L2"
    verdict: str | None = None
    value: str | None = None
    unit: str = "元"
    notes: str = ""


@dataclass(frozen=True)
class Refusal:
    """一条 R 应拒答题。

    `reason` 的前缀是机器可读的分类，报告里按它分组统计 ——
    「问题本身不可答」与「本系统证据不足」混在一张表里，拒答率就没法解释。

    `expected_code` 是可选的**系统侧机器原因**：填了就要求系统以这个原因拒答。
    见 `EvalItem.expected_refusal_code` —— 只在"该拒的原因"本身是被测修复点时才填。
    """

    seq: int
    question: str
    reason: str
    detail: str
    difficulty: str
    notes: str = ""
    expected_code: str | None = None


# --- L1：单值抽取（合并口径 14 题 + 母公司口径 1 题 + 口语简称 3 题）---------

L1_ROWS: tuple[Row, ...] = (
    Row(1, 58, "货币资金", "59,295,822,956.89",
        "2024年合并资产负债表的货币资金是多少？", "easy"),
    Row(2, 59, "存货", "54,343,285,157.47",
        "2024年合并资产负债表的存货是多少？", "easy"),
    Row(3, 59, "流动资产合计", "251,726,674,636.66",
        "2024年合并资产负债表的流动资产合计是多少？", "easy"),
    Row(4, 59, "资产总计", "298,944,579,918.70",
        "2024年合并资产负债表的资产总计是多少？", "medium",
        notes="同一报告母公司口径为 180,236,524,477.01，取错口径即失分"),
    Row(5, 60, "流动负债合计", "56,515,990,618.96",
        "2024年合并资产负债表的流动负债合计是多少？", "easy",
        notes="与「非流动负债合计」易混"),
    Row(6, 60, "负债合计", "56,933,264,798.10",
        "2024年合并资产负债表的负债合计是多少？", "medium",
        notes="与「非流动负债合计 417,274,179.14」易混"),
    Row(7, 61, "未分配利润", "182,787,415,205.05",
        "2024年合并资产负债表的未分配利润是多少？", "hard",
        notes="母公司口径为 115,219,987,975.81，是口径消歧的关键题"),
    Row(8, 61, "所有者权益（或股东权益）合计", "242,011,315,120.60",
        "2024年合并资产负债表的所有者权益合计是多少？", "medium",
        notes="不可与「归属于母公司所有者权益合计 233,105,984,399.47」混同"),
    Row(9, 63, "营业总收入", "174,144,069,958.25",
        "2024年合并利润表的营业总收入是多少？", "medium",
        notes="与「营业收入 170,899,152,276.34」易混"),
    Row(10, 63, "营业收入", "170,899,152,276.34",
        "2024年合并利润表的营业收入是多少？", "easy"),
    Row(11, 63, "营业成本", "13,789,482,367.98",
        "2024年合并利润表的营业成本是多少？", "medium",
        notes="与「营业总成本 54,523,971,452.57」易混，后者含税金及附加与四项费用"),
    Row(12, 64, "利润总额", "119,638,578,194.46",
        "2024年合并利润表的利润总额是多少？", "medium",
        notes="与「营业利润 119,688,579,453.23」易混，差在营业外收支"),
    Row(13, 64, "净利润", "89,334,728,025.90",
        "2024年合并利润表的净利润是多少？", "medium",
        notes="与「归属于母公司股东的净利润 86,228,146,421.62」易混"),
    Row(14, 65, "基本每股收益(元/股)", "68.64",
        "2024年合并利润表的基本每股收益是多少？", "easy",
        unit="元/股"),
    # 口径消歧探针：问句点名母公司口径，正确值在母公司报表上。
    Row(15, 62, "资产总计", "180,236,524,477.01",
        "2024年母公司资产负债表的资产总计是多少？", "hard",
        scope="母公司",
        notes="口径消歧探针：同一科目名在合并（298.9B）与母公司（180.2B）两套报表下数值不同，"
              "问句点名母公司就必须取母公司那一行。"
              "（P-023 修复前此题失分，注释曾写「预期失分」；现已通过，注释同步更正 —— "
              "过时的注释比没有注释更糟，它会让人以为已知问题未修）"),
)


# --- L2：勾稽 / 计算 --------------------------------------------------------

L2_ROWS: tuple[Verify, ...] = (
    Verify(1,
           "2024年合并资产负债表的资产总计是否等于负债合计加所有者权益合计？",
           "F1",
           ((59, "资产总计", "资产总计", "298,944,579,918.70"),
            (60, "负债合计", "负债合计", "56,933,264,798.10"),
            (61, "所有者权益（或股东权益）合计", "所有者权益合计", "242,011,315,120.60")),
           "easy", verdict="PASS"),
    Verify(2,
           "2024年母公司资产负债表的资产总计是否等于负债合计加所有者权益合计？",
           "F1",
           ((62, "资产总计", "资产总计", "180,236,524,477.01"),
            (62, "负债合计", "负债合计", "15,750,242,139.09"),
            (63, "所有者权益（或股东权益）合计", "所有者权益合计", "164,486,282,337.92")),
           "hard", scope="母公司", verdict="PASS",
           notes="两套口径的资产恒等式**都能平衡**（差额都是 0.00）。"
                 "只比判定必然判不出取错口径，故判分必须先比操作数"),
    Verify(3,
           "2024年合并利润表的净利润是否等于利润总额减去所得税费用？",
           "F2a",
           ((64, "净利润", "净利润", "89,334,728,025.90"),
            (64, "利润总额", "利润总额", "119,638,578,194.46"),
            (64, "所得税费用", "所得税费用", "30,303,850,168.56")),
           "medium", verdict="PASS"),
    Verify(4,
           "2024年合并利润表中，归属于母公司股东的净利润是否等于净利润减去少数股东损益？",
           "F2b",
           ((64, "归属于母公司股东的净利润", "归属于母公司股东的净利润", "86,228,146,421.62"),
            (64, "净利润", "净利润", "89,334,728,025.90"),
            (64, "少数股东损益", "少数股东损益", "3,106,581,604.28")),
           "hard", verdict="PASS",
           notes="少数股东损益易被漏取；该项目为负时取绝对值会直接导致不平衡"),
    Verify(5,
           "2024年合并口径的毛利率是多少？",
           "F4",
           ((63, "营业收入", "营业收入", "170,899,152,276.34"),
            (63, "营业成本", "营业成本", "13,789,482,367.98")),
           "hard", value="91.93", unit="%",
           notes="派生量题：答案是算出来的（91.93 个百分点），"
                 "原文里没有这个串，故闸门校验的是两个操作数"),
)


# --- L3：综合判断（跨公式 / 跨口径的多跳）-----------------------------------

L3_ROWS: tuple[Verify, ...] = (
    Verify(1,
           "2024年合并利润表中，归属于母公司股东的净利润与净利润的差额是多少？",
           "",
           ((64, "净利润", "净利润", "89,334,728,025.90"),
            (64, "归属于母公司股东的净利润", "归属于母公司股东的净利润", "86,228,146,421.62")),
           "hard", qtype="L3", value="3,106,581,604.28",
           notes="需要两次数值抽取再做减法，超出单公式注册表。"
                 "gold 取自「一个称职的分析师会给出的答案」，不迁就系统能力"),
    Verify(2,
           "2024年母公司口径的资产总计与合并口径的资产总计相差多少？",
           "",
           ((59, "资产总计", "合并资产总计", "298,944,579,918.70"),
            (62, "资产总计", "母公司资产总计", "180,236,524,477.01")),
           "hard", qtype="L3", scope="跨口径", value="118,708,055,441.69",
           notes="跨口径题：同时点名两套口径，故 scope 记「跨口径」而非「不适用」"),
)


# --- R：应拒答 --------------------------------------------------------------

R_ROWS: tuple[Refusal, ...] = (
    Refusal(1, "火星基地2024年的折旧年限是多少年？",
            "no_recall", "文档中不存在该主体与科目，四路召回均无命中",
            "easy",
            "护栏基线：无召回时必须拒答，不得给出任何数字"),
    Refusal(2, "贵州茅台2025年第一季度的营业收入是多少？",
            "out_of_document_period", "2024 年年度报告不含 2025 年数据",
            "medium",
            "期间越界。注意问句含「营业收入」，检索层会命中真实科目 —— "
            "这题测的是「期间是否被校验」，不是「召回是否成功」"),
    Refusal(3, "宁德时代2024年的营业收入是多少？",
            "wrong_document", "本文档是贵州茅台年报，不含宁德时代任何数据",
            "medium",
            "文档主体不符。同样会命中「营业收入」科目 —— "
            "测的是「公司身份是否被校验」"),
    Refusal(4, "公司2024年审计报告的签字注册会计师是谁？",
            "out_of_parse_window", "该信息在审计报告正文，不在已解析的财务报表页（58–67 页）",
            "medium"),
    Refusal(5, "公司2024年的研发人员数量是多少？",
            "out_of_parse_window", "该信息在「员工情况」章节，不在已解析的财务报表页",
            "medium"),
    Refusal(6, "2024年合并现金流量表的现金及现金等价物净增加额是多少？",
            "out_of_parse_window", "现金流量表期末结存行在解析窗口之外，索引中不存在该科目",
            "hard",
            "与 F3a 同源：该行未入索引，核验必然缺操作数"),
    Refusal(7, "资产总计是多少？",
            "ambiguous_scope", "该科目在合并（298,944,579,918.70）与母公司"
                               "（180,236,524,477.01）两套报表下数值不同，问句未限定口径",
            "hard",
            "口径不明。正确行为是说明所用口径或澄清，而不是静默给一个数。"
            "（D5 修复前此题会静默取合并口径而失分；现以 AMBIGUOUS_SCOPE 拒答，已通过）"),
    Refusal(8, "2024年现金流是多少？",
            "ambiguous_abbreviation", "口语简称「现金流」在现金流量表里对应三张活动表的净额行，"
                                      "数值各不相同（经营 92.5B / 投资 -1.8B / 筹资 -71.1B）",
            "hard",
            "口语简称歧义（P-030）。注意它**必须**以 `AMBIGUOUS_ABBREVIATION` 拒答，"
            "而不是笼统的 `LABEL_MISMATCH` 或「没召回」——"
            "前者告诉用户\"该指明哪一行\"，后者只让人以为\"报告里没有\"。"
            "报错指错方向与拒答本身一样糟。故此题填了 expected_refusal_code",
            expected_code="AMBIGUOUS_ABBREVIATION"),
)


# --- 口语简称探针（P-030）：追加在题库末尾，不改既有题号----------------------
#
# 现金流量表里 9 行都含「现金流」，口语简称在词法上同时沾到多行。修复前
# RRF 取 top-1 会落到「支付其他与经营活动有关的现金」（9.0B）——
# 数值真、页码真、span 通过，**全部护栏放行**，是典型的静默错误（比拒答更危险，
# 因为用户无从分辨）。这三条把"简称必须归一到唯一规范行"钉死。
#
# gold 取自索引里那一行原文（与其它 L1 题同一纪律），不迁就系统输出。

P030_ROWS: tuple[Row, ...] = (
    Row(1, 67, "经营活动产生的现金流量净额", "92,463,692,168.43",
        "2024年合并现金流量表的经营现金流是多少？", "hard",
        notes="P-030 探针。简称「经营现金流」必须归一到「经营活动产生的现金流量净额」，"
              "而不是被词法检索带到同族的「支付其他与经营活动有关的现金」（9.0B）。"
              "修复前此题会「答错但看起来对」，所以只能靠评测把正确值钉死"),
    Row(2, 67, "投资活动产生的现金流量净额", "-1,785,202,630.71",
        "2024年合并现金流量表的投资现金流净额是多少？", "hard",
        notes="P-030 探针（含「净额」二字的变体，且答案为负）。"
              "负值同时守住「口径为负时不得取绝对值」——同类问题记过一次（P-016）。"
              "另一层：别名「投资活动现金流」恰好是「投资活动现金流入小计」的前缀，"
              "不做词尾保护会拼出不存在的科目名，这条同时守住那个回归"),
    Row(3, 64, "归属于母公司股东的净利润", "86,228,146,421.62",
        "2024年归母净利润是多少？", "medium",
        notes="P-030 探针。「归母」是行业通用缩写，语义唯一。"
              "「归母净利润」与「净利润」（89.3B）数值不同，"
              "缩写展开必须精确落到归母那一行，而不是退化成「净利润」"),
)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


def _row_fragment(con: sqlite3.Connection, page: int, label: str, value: str) -> str:
    """从索引里原样取出「科目 | 口径 | 数值」这一段，作为 gold 证据片段。

    取到该数值为止（不含上期列），保证片段是页文本的**真子串**。
    """
    row = con.execute(
        "SELECT text FROM chunks WHERE page = ? AND label = ? LIMIT 1", (page, label)
    ).fetchone()
    if row is None:
        raise SystemExit(f"索引里找不到 p{page} 的科目「{label}」——题表与索引不一致")
    text = row[0]
    idx = text.find(value)
    if idx < 0:
        raise SystemExit(
            f"p{page}「{label}」的行文本里没有 {value!r}（行文本：{text!r}）"
            "——金标准数值与索引里的值不一致"
        )
    return text[: idx + len(value)]


def build(con: sqlite3.Connection) -> list[EvalItem]:
    items: list[EvalItem] = []

    for r in L1_ROWS:
        items.append(
            EvalItem(
                item_id=f"{TRACK}-{DOC}-{r.seq:03d}",
                track=TRACK, doc=DOC, company=COMPANY, period=PERIOD,
                question=r.question, question_type="L1",
                expected_outcome="ANSWER", scope=r.scope,
                gold_value=r.value, gold_unit=r.unit, gold_pages=(r.page,),
                gold_evidence=(_row_fragment(con, r.page, r.label, r.value),),
                difficulty=r.difficulty, notes=r.notes, source=SOURCE,
            )
        )

    # 题号按「L1 → L2 → L3 → R」连续编号。各组的基准要分别累加，
    # 否则 L3 的 seq 从 1 重新开始，会和 L2 的题号撞上（validator 会拦下来）。
    n_l1 = len(L1_ROWS)
    n_l2 = len(L2_ROWS)

    for v in L2_ROWS + L3_ROWS:
        seq = (n_l1 if v.qtype == "L2" else n_l1 + n_l2) + v.seq
        pages = tuple(sorted({p for p, _row, _op, _val in v.operands}))
        # 操作数按**公式/题目口径的操作数名**归键，证据片段按**报表行名**取。
        # 两者在「所有者权益（或股东权益）合计」这类科目上并不相同。
        operands = {op or row: val for _p, row, op, val in v.operands}
        evidence = tuple(
            _row_fragment(con, p, row, val) for p, row, _op, val in v.operands
        )
        items.append(
            EvalItem(
                item_id=f"{TRACK}-{DOC}-{seq:03d}",
                track=TRACK, doc=DOC, company=COMPANY, period=PERIOD,
                question=v.question, question_type=v.qtype,
                expected_outcome="ANSWER", scope=v.scope,
                gold_value=v.value, gold_unit=(v.unit if v.value else None),
                gold_pages=pages, gold_evidence=evidence,
                gold_verdict=v.verdict, gold_operands=operands,
                formula=(v.formula or None),
                difficulty=v.difficulty, notes=v.notes, source=SOURCE,
            )
        )

    base = n_l1 + n_l2 + len(L3_ROWS)
    for r in R_ROWS:
        items.append(
            EvalItem(
                item_id=f"{TRACK}-{DOC}-{base + r.seq:03d}",
                track=TRACK, doc=DOC, company=COMPANY, period=PERIOD,
                question=r.question, question_type="R",
                expected_outcome="REFUSE", scope="不适用",
                refusal_reason=f"{r.reason}：{r.detail}",
                expected_refusal_code=r.expected_code,
                difficulty=r.difficulty, notes=r.notes, source=SOURCE,
            )
        )

    # --- 末尾追加：口语简称探针（P-030）---
    #
    # 为什么**追加在末尾**而不是插进 L1_ROWS：题号是外部文档的引用锚点
    # （`docs/技术问题留档.md` 里写了 B-MOUTAI_2024-016/017/028），
    # 往中间插一条会让后面所有题号顺移，那些引用就全部指向错误的题目 ——
    # 而且**不会报错**，只是安静地指错。追加不改任何既有题号。
    tail = base + len(R_ROWS)
    for r in P030_ROWS:
        items.append(
            EvalItem(
                item_id=f"{TRACK}-{DOC}-{tail + r.seq:03d}",
                track=TRACK, doc=DOC, company=COMPANY, period=PERIOD,
                question=r.question, question_type="L1",
                expected_outcome="ANSWER", scope=r.scope,
                gold_value=r.value, gold_unit=r.unit, gold_pages=(r.page,),
                gold_evidence=(_row_fragment(con, r.page, r.label, r.value),),
                difficulty=r.difficulty, notes=r.notes, source=SOURCE,
            )
        )

    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Track B 评测题库")
    parser.add_argument("--dry-run", action="store_true", help="只校验，不写盘")
    args = parser.parse_args()

    if INDEX_DB is None or not Path(INDEX_DB).exists():
        raise SystemExit(f"检索索引不存在：{INDEX_DB}（先跑解析与建索引）")

    con = sqlite3.connect(f"file:{INDEX_DB}?mode=ro", uri=True)
    try:
        items = build(con)
    finally:
        con.close()

    report = validate_bank(items)
    print(report_problems(report))
    if report:
        return 1

    print()
    print(render_stats(bank_stats(items)))

    if args.dry_run:
        print("\n[dry-run] 未写盘")
        return 0

    save_bank(items, OUT)
    print(f"\n已写入 {OUT.relative_to(ROOT)}（{len(items)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
