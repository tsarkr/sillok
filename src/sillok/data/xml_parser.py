"""XML parser for the Annals of the Joseon Dynasty (국사편찬위원회 한국사DB XML)."""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from sillok.constants import DATASET_INFO, KING_CODE_MAP

# Match consecutive UTF-16 high + low surrogate numeric character references (decimal or hex).
_SURROGATE_NCR_RE = re.compile(
    r"&#(?:(?P<hdec>5[56]\d{3})|x(?P<hhex>[dD][89aAbB][0-9a-fA-F]{2}));"
    r"&#(?:(?P<ldec>5[67]\d{3})|x(?P<lhex>[dD][cCdDeEfF][0-9a-fA-F]{2}));"
)
_ANY_SURROGATE_NCR_RE = re.compile(r"&#(?:(?P<dec>5[5-7]\d{3})|x(?P<hex>[dD][89a-fA-F][0-9a-fA-F]{2}));")
_CTRL_NCR_RE = re.compile(r"&#(?:0*[0-8]|0*11|0*12|0*1[4-9]|0*2\d|0*3[01]|x0*[0-8bBcCeEfF]|x1[0-9a-fA-F]);")
_RAW_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_INDENT_NEWLINE_RE = re.compile(r"[ \t\r]*\n[ \t\r]*")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_FILENAME_KING_RE = re.compile(r"^(?:\d+(?:st|nd|rd|th)_)?([a-z]{3})_", re.IGNORECASE)
_LEVEL_TAG_RE = re.compile(r"^level([1-6])$")


@dataclass
class SillokArticle:
    """Structured representation of a single Sillok article (`biblioData[@type="T"]`)."""

    article_id: str
    king_code: str
    king_name_ko: str
    king_name_hanja: str
    date_iso: str = ""
    date_ganji: str = ""
    date_reign: str = ""
    date_china: str = ""
    subjects: list[str] = field(default_factory=list)
    title_ko: str = ""
    body_hanja: str = ""
    postscripts: list[str] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)
    source_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SillokArticle":
        return cls(**data)


def sanitize_xml_surrogates(xml_text: str) -> str:
    """Repair invalid UTF-16 surrogate NCRs (e.g. `&#55381;&#56892;`) and strip XML 1.0 control chars."""

    def _replace_pair(match: re.Match[str]) -> str:
        hi = int(match.group("hdec")) if match.group("hdec") else int(match.group("hhex"), 16)
        lo = int(match.group("ldec")) if match.group("ldec") else int(match.group("lhex"), 16)
        if 0xD800 <= hi <= 0xDBFF and 0xDC00 <= lo <= 0xDFFF:
            codepoint = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
            return chr(codepoint)
        return match.group(0)

    def _replace_lone(match: re.Match[str]) -> str:
        val = int(match.group("dec")) if match.group("dec") else int(match.group("hex"), 16)
        if 0xD800 <= val <= 0xDFFF:
            return "\uFFFD"
        return match.group(0)

    cleaned = _SURROGATE_NCR_RE.sub(_replace_pair, xml_text)
    cleaned = _ANY_SURROGATE_NCR_RE.sub(_replace_lone, cleaned)
    cleaned = _CTRL_NCR_RE.sub("", cleaned)
    cleaned = _RAW_CTRL_RE.sub("", cleaned)
    return cleaned


def _normalize_text(raw: str) -> str:
    # XML pretty-printing inserts `\n` + indentation between adjacent `<index>` tags;
    # stripping line-break indentation prevents splitting Hanja compounds.
    collapsed = _INDENT_NEWLINE_RE.sub("", raw)
    collapsed = _MULTI_SPACE_RE.sub(" ", collapsed)
    return collapsed.strip()


def _iter_clean_text(
    elem: ET.Element,
    *,
    exclude_tags: set[str],
    skip_annotation_types: set[str],
) -> Iterable[str]:
    if elem.tag in exclude_tags:
        return
    if elem.tag == "annotation" and elem.get("type", "") in skip_annotation_types:
        return
    if elem.text:
        yield elem.text
    for child in elem:
        yield from _iter_clean_text(
            child,
            exclude_tags=exclude_tags,
            skip_annotation_types=skip_annotation_types,
        )
        if child.tail:
            yield child.tail


def _extract_element_text(
    elem: ET.Element,
    *,
    exclude_tags: set[str] | None = None,
    skip_annotation_types: set[str] | None = None,
) -> str:
    chunks = _iter_clean_text(
        elem,
        exclude_tags=exclude_tags or set(),
        skip_annotation_types=skip_annotation_types or set(),
    )
    return _normalize_text("".join(chunks))


def _infer_king_code(xml_path: Path, root: ET.Element) -> str:
    match = _FILENAME_KING_RE.match(xml_path.name)
    if match:
        return match.group(1).lower()
    root_id = root.get("id", "")
    if "_" in root_id:
        return root_id.split("_", 1)[0].lower()
    return root_id[:3].lower()


def _extract_date_context(biblio: ET.Element | None, parent_ctx: dict[str, str]) -> dict[str, str]:
    ctx = dict(parent_ctx)
    if biblio is None:
        return ctx
    for d in biblio.findall("./date/dateOccured"):
        dtype = d.get("type", "").strip()
        if dtype == "서기":
            val = (d.get("date") or d.text or "").strip()
            if val:
                ctx["date_iso"] = val
        elif dtype == "간지":
            val = (d.text or "").strip()
            if val:
                ctx["date_ganji"] = val
        elif dtype == "재위연도":
            val = (d.text or "").strip()
            if val:
                ctx["date_reign"] = val
        elif dtype == "중국연호":
            val = (d.text or "").strip()
            if val:
                ctx["date_china"] = val
    return ctx


def _extract_entities(node: ET.Element) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    text_root = node.find("./text")
    if text_root is None:
        return entities
    for idx in text_root.findall(".//index"):
        surface = _normalize_text("".join(idx.itertext()))
        if not surface:
            continue
        etype = idx.get("type", "")
        ref = idx.get("ref", "")
        key = (surface, etype, ref)
        if key in seen:
            continue
        seen.add(key)
        entry = {"text": surface, "type": etype}
        if ref:
            entry["ref"] = ref
        entities.append(entry)
    return entities


def _extract_body_and_postscripts(
    node: ET.Element,
    *,
    include_annotations: bool,
    include_collation_notes: bool,
) -> tuple[str, list[str]]:
    text_el = node.find("./text")
    if text_el is None:
        return "", []

    skip_ann: set[str] = set()
    if not include_annotations:
        skip_ann.update({"원주", "교감주", "번역주", "악보", "소자쌍행"})
    elif not include_collation_notes:
        skip_ann.add("교감주")

    paragraphs: list[str] = []
    content_els = text_el.findall("./content")
    if content_els:
        for content in content_els:
            para_nodes = content.findall("./paragraph")
            if para_nodes:
                for p in para_nodes:
                    txt = _extract_element_text(
                        p,
                        exclude_tags={"postScript"},
                        skip_annotation_types=skip_ann,
                    )
                    if txt:
                        paragraphs.append(txt)
            else:
                txt = _extract_element_text(
                    content,
                    exclude_tags={"postScript"},
                    skip_annotation_types=skip_ann,
                )
                if txt:
                    paragraphs.append(txt)
    else:
        txt = _extract_element_text(
            text_el,
            exclude_tags={"postScript"},
            skip_annotation_types=skip_ann,
        )
        if txt:
            paragraphs.append(txt)

    postscripts: list[str] = []
    for ps in text_el.findall(".//postScript"):
        ps_paras = ps.findall("./paragraph")
        if ps_paras:
            parts = [
                _extract_element_text(p, skip_annotation_types=skip_ann)
                for p in ps_paras
            ]
            ps_text = "\n".join(part for part in parts if part)
        else:
            ps_text = _extract_element_text(ps, skip_annotation_types=skip_ann)
        if ps_text:
            postscripts.append(ps_text)

    return "\n".join(paragraphs), postscripts


def parse_sillok_xml(
    xml_path: str | Path,
    *,
    include_annotations: bool = True,
    include_collation_notes: bool = False,
    extract_entities: bool = False,
) -> list[SillokArticle]:
    """Parse one NHIC Sillok XML file into a list of `SillokArticle` records."""
    path = Path(xml_path)
    raw_xml = path.read_text(encoding="utf-8")
    sanitized = sanitize_xml_surrogates(raw_xml)
    root = ET.fromstring(sanitized)

    king_code = _infer_king_code(path, root)
    king_name_ko, king_name_hanja = KING_CODE_MAP.get(king_code, (king_code, king_code))

    articles: list[SillokArticle] = []

    def _walk(node: ET.Element, parent_date_ctx: dict[str, str]) -> None:
        biblio = node.find("./front/biblioData")
        date_ctx = _extract_date_context(biblio, parent_date_ctx)

        if biblio is not None and biblio.get("type") == "T":
            title_ko = _normalize_text(biblio.findtext("./title/mainTitle") or "")
            subjects = [
                _normalize_text(s.text or "")
                for s in biblio.findall("./subjectClass")
                if _normalize_text(s.text or "")
            ]
            body_hanja, postscripts = _extract_body_and_postscripts(
                node,
                include_annotations=include_annotations,
                include_collation_notes=include_collation_notes,
            )
            if body_hanja or title_ko:
                entities = _extract_entities(node) if extract_entities else []
                article_id = node.get("id") or f"{path.stem}_{len(articles)}"
                articles.append(
                    SillokArticle(
                        article_id=article_id,
                        king_code=king_code,
                        king_name_ko=king_name_ko,
                        king_name_hanja=king_name_hanja,
                        date_iso=date_ctx.get("date_iso", ""),
                        date_ganji=date_ctx.get("date_ganji", ""),
                        date_reign=date_ctx.get("date_reign", ""),
                        date_china=date_ctx.get("date_china", ""),
                        subjects=subjects,
                        title_ko=title_ko,
                        body_hanja=body_hanja,
                        postscripts=postscripts,
                        entities=entities,
                        source_file=path.name,
                    )
                )

        for child in node:
            if _LEVEL_TAG_RE.match(child.tag):
                _walk(child, date_ctx)

    _walk(root, {})
    return articles


def iter_xml_files(data_dir: str | Path, pattern: str = "2nd_*.xml") -> list[Path]:
    """Return sorted list of Sillok XML files in `data_dir`."""
    return sorted(Path(data_dir).glob(pattern))


def _parse_worker(task: tuple[Path, bool, bool, bool]) -> tuple[str, list[SillokArticle]]:
    path, include_annotations, include_collation_notes, extract_entities = task
    records = parse_sillok_xml(
        path,
        include_annotations=include_annotations,
        include_collation_notes=include_collation_notes,
        extract_entities=extract_entities,
    )
    return path.name, records


def _default_worker_count() -> int:
    cpus = mp.cpu_count() or 2
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
                if total_kb < 6 * 1024 * 1024:  # < 6 GiB RAM (e.g. 3.7 GiB tsarkr host)
                    return max(1, min(cpus, 2))
                break
    return max(1, min(cpus, 8))


def parse_sillok_corpus(
    data_dir: str | Path,
    output_jsonl: str | Path,
    *,
    pattern: str = "2nd_*.xml",
    limit_files: int | None = None,
    workers: int | None = None,
    include_annotations: bool = True,
    include_collation_notes: bool = False,
    extract_entities: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Parse Sillok XML files in parallel and stream records to a JSONL file."""
    xml_files = iter_xml_files(data_dir, pattern=pattern)
    if limit_files is not None and limit_files > 0:
        xml_files = xml_files[:limit_files]

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (p, include_annotations, include_collation_notes, extract_entities)
        for p in xml_files
    ]
    n_workers = workers if workers is not None else _default_worker_count()

    total_files = len(tasks)
    total_articles = 0
    total_hanja_chars = 0
    total_title_chars = 0
    total_postscripts = 0
    king_counts: dict[str, int] = {}
    king_stats: dict[str, dict[str, Any]] = {}

    def _consume_stream(stream: Iterable[tuple[str, list[SillokArticle]]], fout) -> None:
        nonlocal total_articles, total_hanja_chars, total_title_chars, total_postscripts
        for idx, (fname, articles) in enumerate(stream, start=1):
            seen_kings_in_file: set[str] = set()
            for art in articles:
                fout.write(json.dumps(art.to_dict(), ensure_ascii=False) + "\n")
                hanja_len = len(art.body_hanja)
                title_len = len(art.title_ko)
                ps_len = len(art.postscripts)

                total_hanja_chars += hanja_len
                total_title_chars += title_len
                total_postscripts += ps_len
                king_counts[art.king_code] = king_counts.get(art.king_code, 0) + 1

                entry = king_stats.setdefault(
                    art.king_code,
                    {
                        "king_name_ko": art.king_name_ko,
                        "king_name_hanja": art.king_name_hanja,
                        "files": 0,
                        "articles": 0,
                        "hanja_chars": 0,
                        "title_chars": 0,
                        "postscripts": 0,
                    },
                )
                if art.king_code not in seen_kings_in_file:
                    entry["files"] += 1
                    seen_kings_in_file.add(art.king_code)
                entry["articles"] += 1
                entry["hanja_chars"] += hanja_len
                entry["title_chars"] += title_len
                entry["postscripts"] += ps_len

            total_articles += len(articles)
            if progress_callback is not None:
                progress_callback(idx, total_files, fname)

    with out_path.open("w", encoding="utf-8") as fout:
        if n_workers <= 1 or total_files <= 1:
            _consume_stream((_parse_worker(t) for t in tasks), fout)
        else:
            with mp.Pool(processes=n_workers) as pool:
                _consume_stream(pool.imap(_parse_worker, tasks, chunksize=4), fout)

    return {
        "dataset_info": DATASET_INFO,
        "files_processed": total_files,
        "articles_extracted": total_articles,
        "total_hanja_chars": total_hanja_chars,
        "total_title_chars": total_title_chars,
        "total_postscripts": total_postscripts,
        "output_jsonl": str(out_path),
        "king_counts": dict(sorted(king_counts.items())),
        "king_stats": dict(sorted(king_stats.items())),
    }
