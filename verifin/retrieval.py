"""检索层：把「找到一个可核验的数字」这件事做成多路召回 + RRF 融合。

为什么是这个结构
----------------
本项目对检索的要求和通用 RAG 不一样，有一条硬红线：

    **embedding 不参与数字精确匹配** —— 数值必须先做归一化才能比较。

原因是 embedding 会把 `12,345.67`、`12345.67`、`1.23亿元` 当成互不相关的词串，
用它找财务数字必然失败。所以本项目刻意**不把唯一希望寄托在向量上**，而是三路并行：

| 通道 | 解决什么 | 为什么必须有它单独一路 |
|---|---|---|
| `lexical` | 中文字面召回（`资产总计`） | BM25 对缺失任一分词的结果直接判 0，召回高但脆 |
| `numeric` | 数值精确匹配（`242,011,315,120.60`） | 必须走归一化 + Decimal 比较，其他通道做不到；纯数字查询会被 lexical 通道彻底排除 |
| `vector` | 软匹配 / 多词部分命中 | 给"命中了其中几个词"的结果部分分数，补齐 lexical 的脆 |

三路结果用 **RRF（Reciprocal Rank Fusion，k=60）** 融合。RRF 只看排名不看原始分数，
因此不需要对各通道的分数做归一化——这正是它适合异构召回的原因。

★ 查询侧必须 jieba 切词（P-003）
--------------------------------
这不是优化，是正确性要求。`fts_chunks` 的 text 在入库时已用 jieba 分词并以空格连接，
查询必须用同一个 `tokenize()` 切词后再交给 FTS5，否则多字中文词会**静默 0 命中**。
已知的灾难性后果：上层拿到空结果 → 判定"报告里没有这个数" → 假拒答。

★ 关于 vector 通道：当前没有 embedding 端点，怎么办
--------------------------------------------------
已实测：本项目使用的中转端点 **不支持 `/v1/embeddings`**（18 个模型中 embedding 类为空，
调用全部返回 404）。原始选型里的"numpy 向量"因此不能依赖外部 embedding 服务落地。

处置原则：**不假装存在**，而是把降级做成显式、可插拔、可验证的一层
（详见 `docs/技术问题留档.md` P-007）：

- 默认用 `TfidfEmbedder`：纯 numpy 计算 TF-IDF 余弦，**零外部依赖、确定性、可单测**；
- `RetrievalIndex` 接受任意符合 `Embedder` 协议的对象。将来拿到 embedding 端点，
  写一个 `embed(texts) -> ndarray` 的类塞进来即可，**其余代码一行不用改**。

这样做保证了架构与选型文档一致（仍是 FTS5 + numpy 向量 + RRF），
同时把"当前用的是哪种向量"这件事显式记录在 `index_meta` 表里而不是藏在代码里。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from .lexicon import apply_lexicon, clean_label
from .normalize import normalize_text, parse_amounts

__all__ = [
    "DEFAULT_RRF_K",
    "Chunk",
    "Hit",
    "RetrievalResult",
    "Embedder",
    "TfidfEmbedder",
    "build_chunks",
    "tokenize",
    "create_index",
    "load_index",
    "rrf_fuse",
]


#: RRF 的 k。60 是 Robert Cormack 原始论文里常用的取值，
#: 作用是压低高名次与低名次之间的分差，让各通道的影响力趋于均衡。
DEFAULT_RRF_K = 60

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_NUMERIC_TOKEN = re.compile(r"^[\d,，.]+$")

#: 疑问句里的功能词。它们不构成任何财务语义，
#: 放进查询会让 AND 查询必然 0 命中、OR 查询制造噪声。
#: 例：「2024年营业收入是多少」→ 去掉这些词后才是真正的检索意图。
STOPWORDS: frozenset[str] = frozenset({
    "多少", "是多少", "为", "请问", "请", "的", "了", "和", "与", "在", "是多少?",
    "什么", "怎么", "如何", "是否", "帮我", "查", "查询", "告诉", "找出", "给出",
})


def tokenize(text: str) -> list[str]:
    """jieba 切词，只保留有检索意义的词元。

    三条约定，每条都对应一处实测教训：

    1. **索引侧与查询侧必须用同一个函数**（P-003 的直接结论）。
       任一侧漏切都会导致分词粒度不一致，进而**静默 0 命中**。
    2. **必须先注入财务词典**。jieba 默认词典不认识会计科目，
       会把「营业收入」切成「营业 + 收入」——这两个词在财报里到处都是，
       于是 `营业外收入`、`手续费及佣金收入` 会抢走「营业收入」的排名。
       这在实际效果上等同于取错科目，而它同样不会报错。
    3. 只保留汉字词元与字母数字词元，标点与空白直接丢弃。
    """
    if not text:
        return []
    import jieba  # 延迟导入：首次加载词典约 0.2s，不必在 import 本模块时就付出

    apply_lexicon()

    out: list[str] = []
    for word in jieba.lcut(text):
        word = word.strip()
        if not word:
            continue
        if not (_CJK.match(word) or _TOKEN.fullmatch(word)):
            continue  # 标点、空白。
        out.append(word)
    return out


def _to_decimal(text: str) -> Decimal | None:
    cleaned = text.replace(",", "").replace("，", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """一个可检索的最小单元。

    本项目刻意**以表格行作为 chunk**，而不是按固定长度切窗口：
    六元组的「指标 / 数值 / 页码 / 原文片段」天然是一行的信息，
    按窗口切会把一行数字切到两个 chunk 里，证据就碎了。
    """

    chunk_id: str
    doc_id: str
    table_index: int
    page: int
    label: str
    section: str | None
    values: tuple[str, ...]
    text: str

    def to_row(self) -> tuple:
        return (
            self.chunk_id, self.doc_id, self.table_index, self.page,
            self.label, self.section, json.dumps(list(self.values),
                                                 ensure_ascii=False), self.text,
        )


@dataclass(frozen=True)
class Hit:
    """一条召回结果。

    Attributes:
        chunk: 命中的 chunk。
        score: RRF 融合分（**不是**相似度，值域与 k 有关，只用于排序）。
        via: 命中了哪些通道——保留这个信息是为了溯源：
            融合后黑盒化会让"为什么这条排在前面"无法解释，
            而这个项目恰恰要求证据链路可复盘。
        ranks: 各通道上的名次。
    """

    chunk: Chunk
    score: float
    via: tuple[str, ...]
    ranks: dict[str, int]


@dataclass(frozen=True)
class RetrievalResult:
    """一次检索的完整结果。"""

    query: str
    hits: tuple[Hit, ...]
    per_channel: dict[str, int]
    warnings: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.hits


# --------------------------------------------------------------------------
# 向量化：Embedder 协议 + 本地 TF-IDF 实现
# --------------------------------------------------------------------------


class Embedder(Protocol):
    """向量化协议。

    任何满足这个接口的对象都能插进 `RetrievalIndex`。把「原本依赖外部 embedding
    服务」的那一路降级成接口 + 本地实现，是本模块应对「端点无 embedding」的核心手段：
    将来有了端点，替换实现即可，不改结构。
    """

    dim: int
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """把若干文本转成 `(n, dim)` 的浮点矩阵。**返回值必须做 L2 归一化**，
        这样余弦相似度就等于向量点积，`numpy` 一次矩阵乘即可完成全量检索。"""
        ...


class TfidfEmbedder:
    """本地 TF-IDF 向量化，纯 numpy，零外部依赖。

    **为什么会需要它**：中转端点不支持 `/v1/embeddings`（已实测）。
    但融合架构本身就依赖一个"软匹配"通道，用它来兜底比砍掉这一路要好——
    TF-IDF 余弦虽然不等于语义相似度，但它确实能做 lexical 通道做不到的
    「部分命中也给分」，这正是 RRF 融合想要的那一路信号。

    局限必须说清：它只能捕捉字面/词面重合，**不理解同义词**。
    "营收"和"营业收入"在向量空间里是无关的。这是它相对真实 embedding 的差距，
    也是为什么不能把它当成 embedding 的等价物。
    """

    def __init__(self) -> None:
        self.name = "tfidf-local"
        self._idf: dict[str, float] = {}
        self._vocab: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return len(self._vocab) or 1

    def fit(self, token_lists: Sequence[Sequence[str]]) -> "TfidfEmbedder":
        n = len(token_lists)
        if n == 0:
            return self
        df: dict[str, int] = {}
        for tokens in token_lists:
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1
        self._vocab = {tok: i for i, tok in enumerate(sorted(df))}
        self._idf = {
            tok: math.log((n + 1) / (count + 1)) + 1.0 for tok, count in df.items()
        }
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """把若干文本转成 L2 归一化的 TF-IDF 矩阵。"""
        n = len(texts)
        if not self._vocab:
            return np.zeros((n, 1), dtype=np.float32)
        mat = np.zeros((n, len(self._vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            counts: dict[str, int] = {}
            for tok in tokens:
                if tok in self._vocab:
                    counts[tok] = counts.get(tok, 0) + 1
            for tok, count in counts.items():
                mat[i, self._vocab[tok]] = float(count) * self._idf.get(tok, 1.0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    # ---- 持久化（fit 的结果不落盘会导致每次查询都要重算 IDF）----
    def to_json(self) -> str:
        return json.dumps({"vocab": self._vocab, "idf": self._idf},
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "TfidfEmbedder":
        obj = cls()
        data = json.loads(payload)
        obj._vocab = data["vocab"]
        obj._idf = data["idf"]
        return obj


# --------------------------------------------------------------------------
# 索引构建
# --------------------------------------------------------------------------


#: 表头里「不是数值列」的名字。第 0 列是科目名，其后的「附注」是编号不是金额。
_NON_VALUE_HEADERS = frozenset({"项目", "附注", "科目", "项目/附注", "行次", "序号"})


def _value_column_indices(
    header: tuple[str, ...] | None, width: int
) -> tuple[int, ...]:
    """由表头推导出哪些列是数值列。

    为什么不能一律 `cells[1:]`：A 股报表普遍有一列「附注」编号，
    落在第 1 列。把它当成数值会得到「未分配利润 = 39」这种荒谬结果，
    而且**不会有任何报错**——它会一路通过 span 校验（39 确实在原文里）
    直达六元组。这类错误只能靠结构性地读表头来防。

    Returns:
        数值列的索引元组。表头缺失时退化为 `(1..width-1)`，
        这是保守假设（假设第 0 列是科目名），并由调用方按情况告警。
    """
    if not header:
        return tuple(range(1, width))
    picked = tuple(
        i for i, name in enumerate(header)
        if normalize_text(name) not in _NON_VALUE_HEADERS
    )
    # 至少要有第 0 列以外的一列，全被判成非数值列说明表头识别失败
    return picked if picked else tuple(range(1, width))


def build_chunks(tables: Iterable["object"], doc_id: str) -> list[Chunk]:
    """把拼接后的逻辑表转成 chunk 列表。**每行一个 chunk**，带所属分区与页码。"""
    from .tables import LogicalTable  # 延迟导入避免循环依赖

    chunks: list[Chunk] = []
    for ti, table in enumerate(tables):
        if not isinstance(table, LogicalTable):
            raise TypeError(f"期望 LogicalTable，收到 {type(table).__name__}")
        # 数值列：**由表头推导**，不能想当然地取 cells[1:]。
        # 合并资产负债表的表头是 `项目 | 附注 | 2024年12月31日 | 2023年12月31日`，
        # 第 1 列是「附注」编号（如未分配利润那行的 `39`），把它当成数值会得到
        # 「未分配利润 = 39」这种荒谬结果，且不会报错。
        value_cols = _value_column_indices(table.header, table.width)
        section: str | None = None
        for ri, row in enumerate(table.rows):
            if row.is_section:
                section = row.label
                # 分区行本身没有数值语义，不作为 chunk
                continue
            values = tuple(row.cell(i).strip() for i in value_cols)
            values = tuple(v for v in values if v)
            # 剥掉 `一、` `其中:` 这类前缀：它们不携带财务语义，
            # 却会让「科目名精确匹配」通道无故失配。
            label = clean_label(row.label)
            text_parts = [label, section or "", *values]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#t{ti}r{ri}",
                    doc_id=doc_id,
                    table_index=ti,
                    page=row.page,
                    label=label,
                    section=section,
                    values=values,
                    text=" | ".join(p for p in text_parts if p).strip(),
                )
            )
    return chunks


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    table_index  INTEGER NOT NULL,
    page         INTEGER NOT NULL,
    label        TEXT NOT NULL,
    section      TEXT,
    values_json  TEXT NOT NULL,
    text         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS numbers(
    chunk_id         TEXT NOT NULL,
    normalized_value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_meta(
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_numbers ON numbers(normalized_value);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_id UNINDEXED,
    body,
    tokenize='unicode61'
);
"""


def create_index(
    db_path: str | Path,
    chunks: Sequence[Chunk],
    *,
    embedder: Embedder | None = None,
) -> sqlite3.Connection:
    """建库 + 灌入 chunk + 构建三路索引。返回已提交的连接。"""
    path = Path(db_path)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(_SCHEMA)

    # 1) 主表
    con.executemany(
        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
        [c.to_row() for c in chunks],
    )

    # 2) 数值通道：归一化后的 Decimal 字符串。
    #    存字符串而不是 REAL，是为了保留 Decimal 的精确性（REAL 会引入浮点误差）。
    number_rows = []
    for c in chunks:
        for v in c.values:
            d = _to_decimal(v)
            if d is not None:
                number_rows.append((c.chunk_id, str(d)))
    con.executemany("INSERT INTO numbers VALUES (?,?)", number_rows)

    # 3) 词法通道：入库前用 jieba 分词并以空格连接。
    #    查询侧必须用同一个 tokenize()，否则多字中文词静默 0 命中（P-003）。
    con.executemany(
        "INSERT INTO fts_chunks(chunk_id, body) VALUES (?,?)",
        [(c.chunk_id, " ".join(tokenize(c.text))) for c in chunks],
    )

    # 4) 向量通道
    emb = embedder or TfidfEmbedder().fit([tokenize(c.text) for c in chunks])
    con.execute(
        "CREATE TABLE IF NOT EXISTS vectors(chunk_id TEXT PRIMARY KEY, vec BLOB NOT NULL)"
    )
    matrix = emb.embed([c.text for c in chunks])
    con.executemany(
        "INSERT INTO vectors VALUES (?,?)",
        [(c.chunk_id, matrix[i].astype(np.float32).tobytes())
         for i, c in enumerate(chunks)],
    )
    con.execute(
        "INSERT INTO index_meta(key, value) VALUES ('embedder', ?), ('embedder_dim', ?)",
        (emb.name, str(matrix.shape[1] if matrix.ndim == 2 else 1)),
    )
    if isinstance(emb, TfidfEmbedder):
        con.execute(
            "INSERT OR REPLACE INTO index_meta(key, value) VALUES ('tfidf_model', ?)",
            (emb.to_json(),),
        )
    con.commit()
    return con


def load_index(db_path: str | Path) -> "RetrievalIndex":
    return RetrievalIndex(sqlite3.connect(str(db_path)))


def rrf_fuse(
    ranked: dict[str, Sequence[str]], k: int = DEFAULT_RRF_K
) -> list[tuple[str, float, tuple[str, ...], dict[str, int]]]:
    """RRF 融合。`ranked` 的各列表须已按该通道的相关度从高到低排好。

    Returns:
        `[(chunk_id, score, via, ranks), ...]`，按 score 降序；
        同分时按 chunk_id 升序，保证结果确定性可比。

    为什么用 RRF 而不是加权求和：三路通道的分数量纲完全不同
    （BM25 是无界整数、余弦是 [-1,1]、数值匹配是非 0 即 1），
    直接相加等于让某一路主导。RRF 只用名次，天然规避了归一化问题。
    """
    scores: dict[str, float] = {}
    via: dict[str, set[str]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for channel, ids in ranked.items():
        for position, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + position)
            via.setdefault(cid, set()).add(channel)
            ranks.setdefault(cid, {})[channel] = position
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        (cid, score, tuple(sorted(via[cid])), ranks[cid]) for cid, score in ordered
    ]


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------


class RetrievalIndex:
    """支持三路召回 + RRF 融合的检索入口。"""

    def __init__(self, con: sqlite3.Connection, *, embedder: Embedder | None = None):
        self._con = con
        self._con.row_factory = sqlite3.Row
        self._embedder = embedder or self._load_embedder()

    def close(self) -> None:
        self._con.close()

    def _load_embedder(self) -> Embedder:
        row = self._con.execute(
            "SELECT value FROM index_meta WHERE key='tfidf_model'"
        ).fetchone()
        if row is None:
            return TfidfEmbedder()
        return TfidfEmbedder.from_json(row["value"])

    @property
    def embedder_name(self) -> str:
        row = self._con.execute(
            "SELECT value FROM index_meta WHERE key='embedder'"
        ).fetchone()
        return row["value"] if row else "unknown"

    def _get_chunk(self, chunk_id: str) -> Chunk:
        r = self._con.execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        return Chunk(
            chunk_id=r["chunk_id"], doc_id=r["doc_id"],
            table_index=r["table_index"], page=r["page"], label=r["label"],
            section=r["section"], values=tuple(json.loads(r["values_json"])),
            text=r["text"],
        )

    # ---- 通道 A：词法 ----
    def lexical_search(self, query: str, limit: int = 20) -> list[str]:
        """FTS5 词法召回。

        三条约定，每条都对应一个实测教训：

        1. **必须 jieba 切词**（P-003 强制约定）。索引侧已用同一 `tokenize()` 分词，
           查询侧不切词会静默 0 命中，进而被上层当成"报告里没有这个数"。
        2. **纯数字词元直接剔除**。`298,944,579,918.70` 进 BM25 会被切成
           `298 AND 944 AND 579 AND 918.70`，制造大量垃圾命中。
           数字有专门的 `numeric` 通道，不该混进来。
        3. **用 OR 而不是 AND**。AND 要求一条记录同时包含所有分词，
           这对自然语言提问几乎必然失败——「2024年营业收入是多少」里
           没有任何一行会同时出现「营业收入」和「多少」。
           改 OR 后靠 **BM25 自身的排序**天然表达"命中词越多越靠前"，
           召回与排序同时得到改善，且不引入任何需要调参的加权逻辑。
        """
        tokens = [
            t for t in tokenize(query)
            if not _NUMERIC_TOKEN.match(t) and t not in STOPWORDS
        ]
        if not tokens:
            return []
        match_expr = " OR ".join(tokens)
        try:
            rows = self._con.execute(
                "SELECT chunk_id FROM fts_chunks "
                "WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
                (match_expr, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # 查询串含 FTS5 无法解析的语法（如孤立的分隔符）。
            # 静默吞掉会被误判为"报告里没有"，因此这里依赖上层检查 empty 并告警。
            return []
        return [r["chunk_id"] for r in rows]

    # ---- 通道 B：精确科目名 ----
    def label_search(self, query: str, limit: int = 20) -> list[str]:
        """科目名精确匹配。

        为什么需要单独一路：BM25 只看词的重合，**不区分完整匹配与子串匹配**。
        实测查询「负债合计」时，`非流动负债合计` 会因为同时命中「负债」「合计」
        而排在真正的 `负债合计` 之前——这在财务场景是致命的，
        因为取错一个科目就等于取错一个数。

        这一路用「科目名完全相等」把目标顶到第一名。
        返回时按页码升序，保证同名科目（合并报表 vs 母公司报表）顺序稳定。
        """
        key = normalize_text(query)
        if not key:
            return []
        rows = self._con.execute(
            "SELECT chunk_id, label, page FROM chunks"
        ).fetchall()
        hits = [
            (r["page"], r["chunk_id"])
            for r in rows
            if normalize_text(r["label"]) == key
        ]
        hits.sort(key=lambda kv: (kv[0], kv[1]))
        return [cid for _, cid in hits[:limit]]

    # ---- 通道 B：数值精确 ----
    def numeric_search(self, query: str, limit: int = 20) -> list[str]:
        """数值精确匹配。

        查询串里的每个数字先经 `parse_amounts` 归一化（统一到「元」），
        再与本库存储的 Decimal 字符串做**精确相等**比较。

        刻意**不做数值近似**（如"相差 1% 以内"）：近似会让 298,944,579,918.70
        和另一个相近的金额互相召回，进而导致"我用这一行的数去核验另一行"的错配。
        这个项目里错配比漏召回严重得多。
        """
        targets = {
            str(amount.normalized) for amount in parse_amounts(query)
        }
        if not targets:
            return []
        placeholders = ",".join("?" * len(targets))
        rows = self._con.execute(
            f"SELECT DISTINCT chunk_id FROM numbers "
            f"WHERE normalized_value IN ({placeholders}) LIMIT ?",
            (*targets, limit),
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    # ---- 通道 C：向量 ----
    def vector_search(self, query: str, limit: int = 20) -> list[str]:
        """numpy 向量余弦召回。一次矩阵乘完成全量比较，无需向量库。"""
        rows = self._con.execute(
            "SELECT chunk_id, vec FROM vectors"
        ).fetchall()
        if not rows:
            return []
        matrix = np.vstack(
            [np.frombuffer(r["vec"], dtype=np.float32) for r in rows]
        )
        if matrix.shape[1] != self._embedder.dim and hasattr(
            self._embedder, "dim"
        ):
            # embedder 与库不对齐（例如换了实现却没重建索引）→ 明说，不猜
            return []
        qvec = self._embedder.embed([query])
        if qvec.shape[1] != matrix.shape[1]:
            return []
        scores = matrix @ qvec[0]
        top = np.argsort(-scores)[:limit]
        return [rows[i]["chunk_id"] for i in top if scores[i] > 0]

    # ---- 融合 ----
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        channels: tuple[str, ...] = ("label", "lexical", "numeric", "vector"),
        rrf_k: int = DEFAULT_RRF_K,
    ) -> RetrievalResult:
        """四路召回 → RRF 融合 → Top-K。"""
        warnings: list[str] = []
        ranked: dict[str, list[str]] = {}
        searchers = {
            "label": self.label_search,
            "lexical": self.lexical_search,
            "numeric": self.numeric_search,
            "vector": self.vector_search,
        }
        for channel in channels:
            try:
                ranked[channel] = searchers[channel](query, top_k * 3)
            except Exception as exc:  # 单通道失败不应拖垮整体
                warnings.append(f"{channel} 通道失败：{type(exc).__name__}: {exc}")
                ranked[channel] = []

        fused = rrf_fuse(ranked, k=rrf_k)[:top_k]
        hits = tuple(
            Hit(chunk=self._get_chunk(cid), score=score, via=via_, ranks=ranks_)
            for cid, score, via_, ranks_ in fused
        )
        return RetrievalResult(
            query=query,
            hits=hits,
            per_channel={k: len(v) for k, v in ranked.items()},
            warnings=tuple(warnings),
        )
