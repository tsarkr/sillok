import os, sys, time, json, re, sqlite3, urllib.parse, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hanja

MODEL_ID = "darkstar-11e/sillok-100m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEARCH_DB_PATH = os.environ.get("SILLOK_SEARCH_DB", "/content/sillok_search.db")
if not os.path.exists(SEARCH_DB_PATH) and os.path.exists("data/processed/sillok_search.db"):
    SEARCH_DB_PATH = "data/processed/sillok_search.db"

print(f"Loading {MODEL_ID} on {DEVICE}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
model.eval()
print("Model loaded successfully!", flush=True)

# ---------------------------------------------------------------------------
# Historical Glossary & Natural-Language Query Processing
# ---------------------------------------------------------------------------
GLOSSARY_DB = {
    "議政府": "영의정·좌의정·우의정이 국정을 총괄하던 조선 최고 행정·의결 관청",
    "承政院": "왕명 출납과 비서 업무를 전담하던 핵심 관서 (도승지 등 6승지)",
    "弘文館": "궁중의 경서·문한을 관장하고 왕의 자문에 응하던 옥당(玉堂)",
    "司憲府": "백관의 비리를 감찰하고 풍속을 바로잡던 대간(臺諫) 관청",
    "司諫院": "국왕에 대한 간쟁과 논박을 담당하던 언론 관서",
    "吏曹": "문관의 인사, 서열, 포상을 담당하던 6조의 으뜸 관서",
    "戶曹": "호구 조사, 조세 징수, 국가 재정을 총괄하던 관서",
    "禮曹": "국가의 전례, 외교 의례, 과거 시험, 교육을 관장하던 관서",
    "兵曹": "무관 인사와 군사 행정, 국방 업무를 담당하던 관서",
    "刑曹": "법률 집행, 사법 재판, 형벌을 관장하던 관서",
    "工曹": "국가 토목, 영선, 수공업 장인을 관리하던 관서",
    "備邊司": "임진왜란 이후 군사와 국정을 총괄한 최고 합의 기구",
    "義禁府": "왕명을 받들어 반역죄, 옥사 등 중죄인을 다스리던 특별 사법 기관",
    "集賢殿": "세종대왕이 훈민정음 창제와 학문 연구를 위해 설치한 학술 기관",
    "奎章閣": "정조가 설립한 왕실 도서관이자 정책 연구의 요람",
    "啓曰": "신하가 임금에게 정식으로 말씀을 올리다",
    "從之": "임금이 신하의 건의와 품의를 그대로 허락하여 따르다",
    "呈啓": "관청의 공문서를 받아서 임금께 정식으로 아뢰어 올림",
    "史臣曰": "사관이 춘추필법에 따라 인물과 역사적 사건을 준엄하게 비평함",
    "下敎": "임금이 신하들에게 명령이나 훈계를 내림",
    "親鞫": "임금이 의금부 등에서 중죄인을 직접 심문함",
    "安與": "임금이 타던 가마(어연·안여)",
    "諺文": "조선 시대 훈민정음(한글)을 가리키던 명칭",
    "倭賊": "조선 연해나 국토를 침략한 일본군(왜구·왜군)",
    "上疏": "신하나 유생이 국정에 대한 의견을 적어 임금께 올리는 글",
    "賜死": "임금이 죄인에게 사약을 내려 스스로 목숨을 끊게 하던 형벌",
}

KING_NAMES = [
    "태조", "정종", "태종", "세종", "문종", "단종", "세조", "예종", "성종",
    "연산군", "중종", "인종", "명종", "선조", "광해군", "인조", "효종",
    "현종", "숙종", "경종", "영조", "정조", "순조", "헌종", "철종", "고종", "순종",
]

KOREAN_STOPWORDS = {
    "알려줘", "알려주세요", "알려", "찾아줘", "찾아주세요", "찾아", "보여줘", "보여주세요",
    "어떻게", "어디서", "언제", "누가", "무엇", "뭐야", "뭔가요", "뭔지", "무슨",
    "대해", "대해서", "대한", "관련", "관련된", "기록", "실록", "사건", "내용", "역사",
    "과거", "조선", "조선시대", "조선왕조", "왕조", "당시", "어떤", "있어", "있나",
    "있나요", "있는", "있었", "하는", "했던", "한", "된", "만든", "받은", "죽은",
    "간", "온", "왜", "좀", "그", "이", "저", "것", "수", "등", "때", "반응",
    "이야기", "얘기", "설명", "설명해줘", "궁금", "궁금해", "어땠어", "어땠는지",
}

PARTICLE_SUFFIXES = (
    "에서는", "에서도", "으로는", "으로도", "에게서", "한테서", "이라고", "이라는",
    "에서", "으로", "에게", "한테", "까지", "부터", "처럼", "보다", "하고", "이나",
    "이라", "라는", "이나마", "은", "는", "이", "가", "을", "를", "의", "에", "로",
    "와", "과", "도", "만", "께",
)

HISTORICAL_SYNONYMS = {
    "가마": ["안여", "가마", "어연"],
    "뒤주": ["세자", "폐하여", "서인", "임오"],
    "사도세자": ["세자", "사도", "장헌"],
    "훈민정음": ["훈민정음", "언문", "정음"],
    "한글": ["훈민정음", "언문", "정음"],
    "노량해전": ["노량", "이순신"],
    "명량해전": ["명량", "이순신"],
    "한산도": ["한산", "이순신"],
    "임진왜란": ["왜적", "왜변", "임진", "왜군"],
    "몽진": ["파천", "의주", "거가", "피난"],
    "피난": ["파천", "의주", "피난", "거가"],
    "병자호란": ["남한산성", "청병", "호병", "병자"],
    "수원화성": ["화성", "수원", "성역"],
    "동의보감": ["동의보감", "허준", "의서"],
    "측우기": ["측우기", "우량"],
    "갑자사화": ["갑자", "폐비", "윤씨"],
    "무오사화": ["무오", "김일손", "조의제문"],
    "기묘사화": ["조광조", "기묘", "위훈"],
    "을사사화": ["을사", "윤임", "윤원형"],
    "장희빈": ["장씨", "희빈"],
    "홍경래": ["홍경래", "정주", "평안도"],
    "전사": ["전사", "전몰", "순절", "죽다"],
    "파직": ["파직", "삭직"],
    "상소": ["상소", "상서"],
    "상소문": ["상소", "상서"],
    "창제": ["창제", "제작"],
    "축조": ["축성", "화성", "성역"],
    "부서진": ["부서", "파손", "훼손", "절손"],
    "부서져": ["부서", "파손", "훼손"],
}


def strip_particle(word: str) -> str:
    if len(word) <= 2:
        return word
    for suf in PARTICLE_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


def parse_historical_query(raw_query: str):
    cleaned = re.sub(r"[?!.,\"'()\[\]{}<>~·…]", " ", raw_query).strip()
    raw_tokens = [t for t in cleaned.split() if t]

    detected_king = None
    keywords = []
    expanded_groups = []

    for tok in raw_tokens:
        base = strip_particle(tok)
        # Strip "대왕", "장군", "실록" suffixes for better matching
        for honorific in ("대왕", "장군", "실록", "선생"):
            if base.endswith(honorific) and len(base) > len(honorific) + 1:
                base = base[: -len(honorific)]

        if base in KING_NAMES and not detected_king:
            detected_king = base
            continue

        if base in KOREAN_STOPWORDS or tok in KOREAN_STOPWORDS or len(base) < 2:
            continue

        if base not in keywords:
            keywords.append(base)
            syns = HISTORICAL_SYNONYMS.get(base)
            if syns:
                expanded_groups.append(syns)
            else:
                expanded_groups.append([base])

    if not keywords and detected_king:
        keywords.append(detected_king)
        expanded_groups.append([detected_king])
    elif not keywords and raw_tokens:
        fallback = strip_particle(raw_tokens[0])
        keywords.append(fallback)
        expanded_groups.append([fallback])

    return keywords, expanded_groups, detected_king


def _fts_token(term: str) -> str:
    safe = re.sub(r'["*\'^:]', "", term).strip()
    if not safe:
        return ""
    # Prefix wildcard works great for Korean agglutinative forms in unicode61
    return f'"{safe}"*'


def build_fts_query(expanded_groups, mode="and") -> str:
    parts = []
    for grp in expanded_groups:
        sub = [_fts_token(t) for t in grp if _fts_token(t)]
        if len(sub) == 1:
            parts.append(sub[0])
        elif len(sub) > 1:
            parts.append("(" + " OR ".join(sub) + ")")
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(parts)


def make_ruby(text: str, max_chars: int = 1800) -> str:
    truncated = text[:max_chars]
    out = []
    for ch in truncated:
        if hanja.is_hanja(ch):
            reading = hanja.translate(ch, "substitution")
            out.append(f"<ruby>{ch}<rt>{reading}</rt></ruby>")
        else:
            out.append(ch)
    if len(text) > max_chars:
        out.append(" …")
    return "".join(out)


def make_reading(text: str) -> str:
    return hanja.translate(text, "substitution")


def extract_glossary(text: str, limit: int = 5):
    found = []
    for term, desc in GLOSSARY_DB.items():
        if term in text:
            found.append({"term": f"{term} ({hanja.translate(term, 'substitution')})", "desc": desc})
        elif hanja.translate(term, "substitution") in text:
            found.append({"term": f"{term} ({hanja.translate(term, 'substitution')})", "desc": desc})
        if len(found) >= limit:
            break
    return found


def build_scribe_briefing(raw_query: str, keywords: list[str], total_hits: int, rows: list[dict]) -> dict:
    if not rows:
        return {
            "headline": f"'{raw_query}'에 해당하는 실록 원문 기사를 찾지 못했습니다.",
            "summary_lines": [
                "검색어를 인물명·사건명 중심의 핵심 단어(예: '장영실 안여', '최만리 훈민정음', '이순신 노량')로 줄여서 다시 질문해 보세요."
            ],
            "timeline": "",
            "king_distribution": [],
            "saron_hits": 0,
        }

    king_counts = Counter(r["king_name_ko"] for r in rows)
    top_kings = [f"{k}({v}건)" for k, v in king_counts.most_common(4)]
    years = sorted(
        int(r["date_iso"][:4])
        for r in rows
        if r.get("date_iso") and len(r["date_iso"]) >= 4 and r["date_iso"][:4].isdigit()
    )
    timeline = f"{years[0]}년 ~ {years[-1]}년" if len(years) >= 2 and years[0] != years[-1] else (f"{years[0]}년" if years else "연대 미상")
    saron_hits = sum(1 for r in rows if r["has_saron"])

    top_record = rows[0]
    date_label = top_record["date_reign"] or top_record["king_name_ko"]
    iso_label = f" ({top_record['date_iso']})" if top_record["date_iso"] else ""

    summary_lines = [
        f"조선왕조실록 전체 381,701건 원문 중 핵심어 [{', '.join(keywords)}]와 관련된 실제 사료 {total_hits:,}건을 찾아 가장 연관도가 높은 상위 {len(rows)}건을 추렸습니다.",
        f"가장 핵심적인 기록은 [{top_record['king_name_ko']} · {date_label}{iso_label}] 기사 — \"{top_record['title_ko']}\" 입니다.",
    ]
    if len(rows) >= 2:
        second = rows[1]
        summary_lines.append(
            f"함께 살펴볼 관련 사료로 [{second['king_name_ko']} · {second['date_reign'] or '연대기'}] \"{second['title_ko']}\" 기록이 확인됩니다."
        )
    if saron_hits > 0:
        summary_lines.append(
            f"검색된 상위 기사 중 {saron_hits}건에는 당대 사관이 직접 남긴 춘추필법 논평인 「사론(史論)」이 수록되어 있습니다."
        )

    return {
        "headline": f"실록 원문 {total_hits:,}건 탐색 완료 — 핵심 시기: {timeline}",
        "summary_lines": summary_lines,
        "timeline": timeline,
        "king_distribution": top_kings,
        "saron_hits": saron_hits,
    }


def get_db_stats() -> dict:
    if not os.path.exists(SEARCH_DB_PATH):
        return {"search_db_ready": False, "total_articles": 0, "saron_articles": 0}
    try:
        conn = sqlite3.connect(f"file:{SEARCH_DB_PATH}?mode=ro", uri=True)
        total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        saron = conn.execute("SELECT COUNT(*) FROM articles WHERE has_saron = 1").fetchone()[0]
        conn.close()
        return {"search_db_ready": True, "total_articles": total, "saron_articles": saron}
    except Exception:
        return {"search_db_ready": False, "total_articles": 0, "saron_articles": 0}


DB_STATS_CACHE = get_db_stats()
print(f"Search DB status ({SEARCH_DB_PATH}): {DB_STATS_CACHE}", flush=True)


def search_sillok_archive(raw_query: str, king_filter: str = "", saron_only: bool = False, limit: int = 12) -> dict:
    t0 = time.perf_counter()
    if not os.path.exists(SEARCH_DB_PATH):
        return {"error": "실록 검색 데이터베이스(sillok_search.db)가 아직 준비되지 않았습니다.", "results": []}

    limit = max(1, min(int(limit), 30))
    keywords, expanded_groups, detected_king = parse_historical_query(raw_query)
    active_king = king_filter.strip() or (detected_king or "")

    conn = sqlite3.connect(f"file:{SEARCH_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    def run_query(fts_expr: str, use_king: str):
        where_clauses = ["articles_fts MATCH ?"]
        params: list = [fts_expr]
        if use_king:
            where_clauses.append("a.king_name_ko LIKE ?")
            params.append(f"%{use_king}%")
        if saron_only:
            where_clauses.append("a.has_saron = 1")
        where_sql = " AND ".join(where_clauses)

        count_sql = f"""
            SELECT COUNT(*) FROM articles_fts
            JOIN articles a ON a.id = articles_fts.rowid
            WHERE {where_sql}
        """
        total_hits = conn.execute(count_sql, params).fetchone()[0]

        select_sql = f"""
            SELECT
                a.article_id, a.king_code, a.king_name_ko, a.king_name_hanja,
                a.date_iso, a.date_ganji, a.date_reign, a.subjects,
                a.title_ko, a.body_hanja, a.postscripts, a.has_saron,
                bm25(articles_fts, 2.0, 1.0, 1.5, 10.0, 3.0, 4.0) AS rank_score
            FROM articles_fts
            JOIN articles a ON a.id = articles_fts.rowid
            WHERE {where_sql}
            ORDER BY rank_score ASC
            LIMIT ?
        """
        rows = conn.execute(select_sql, params + [limit]).fetchall()
        return total_hits, rows

    fts_and = build_fts_query(expanded_groups, mode="and")
    total_hits, rows = (0, [])
    match_mode = "strict"

    try:
        if fts_and:
            total_hits, rows = run_query(fts_and, active_king)
            # If king filter was auto-detected (not explicit) and yielded 0, retry without king filter
            if total_hits == 0 and not king_filter and active_king:
                total_hits, rows = run_query(fts_and, "")
                if total_hits > 0:
                    active_king = ""
            # If strict AND returned < 3 hits and multiple keywords exist, try relaxed OR
            if len(rows) < 3 and len(expanded_groups) > 1:
                fts_or = build_fts_query(expanded_groups, mode="or")
                or_hits, or_rows = run_query(fts_or, active_king)
                if or_hits == 0 and not king_filter and active_king:
                    or_hits, or_rows = run_query(fts_or, "")
                if len(or_rows) > len(rows):
                    seen = {r["article_id"] for r in rows}
                    merged = list(rows)
                    for r in or_rows:
                        if r["article_id"] not in seen:
                            merged.append(r)
                            seen.add(r["article_id"])
                        if len(merged) >= limit:
                            break
                    rows = merged
                    total_hits = max(total_hits, or_hits)
                    match_mode = "relaxed"
    except sqlite3.OperationalError as exc:
        conn.close()
        return {"error": f"검색 쿼리 구문 오류: {exc}", "results": []}

    conn.close()

    results = []
    for r in rows:
        body_hanja = r["body_hanja"] or ""
        postscripts_raw = r["postscripts"] or ""
        postscripts_list = [p for p in postscripts_raw.split("\n") if p.strip()]
        title_ko = r["title_ko"] or ""
        gloss = extract_glossary(title_ko + " " + body_hanja + " " + postscripts_raw)
        nikh_kw = urllib.parse.quote(title_ko[:40])

        results.append(
            {
                "article_id": r["article_id"],
                "king_code": r["king_code"],
                "king_name_ko": r["king_name_ko"],
                "king_name_hanja": r["king_name_hanja"],
                "date_iso": r["date_iso"],
                "date_ganji": r["date_ganji"],
                "date_reign": r["date_reign"],
                "subjects": [s for s in (r["subjects"] or "").split(" ") if s],
                "title_ko": title_ko,
                "body_hanja": body_hanja,
                "body_ruby": make_ruby(body_hanja, max_chars=1600),
                "postscripts": postscripts_list,
                "postscripts_ruby": [make_ruby(p, max_chars=800) for p in postscripts_list],
                "has_saron": bool(r["has_saron"]),
                "glossary": gloss,
                "nikh_url": f"https://sillok.history.go.kr/search/inspectionList.do?searchKeyword={nikh_kw}",
            }
        )

    briefing = build_scribe_briefing(raw_query, keywords, total_hits, results)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "query": raw_query,
        "extracted_keywords": keywords,
        "detected_king": active_king,
        "match_mode": match_mode,
        "total_hits": total_hits,
        "returned": len(results),
        "elapsed_ms": elapsed_ms,
        "briefing": briefing,
        "results": results,
    }


def interpret_story(king, date, subject, title, generated):
    story = [
        f"1. [역사 배경] {king} 재위 시기, '{title}'에 관한 조정의 중대한 보고와 건의가 제기되었습니다.",
        f"2. [주요 논의] 해당 분야({subject})의 관원들이 국가의 선례와 전례를 살펴 계책을 아뢰었습니다.",
        "3. [조정 결말] 조정의 신하들과 유학자들이 당시의 국정 과제를 논의하여 역사에 기록하였습니다.",
    ]
    sagwan = (
        f"사관(史臣)은 논한다: 예로부터 성군(聖君)의 치세는 신하의 직언을 귀담아듣고 "
        f"법도를 바로 세우는 데서 시작되었다. '{title}' 일은 오늘날에도 거울로 삼을 만한 역사적 장면이다."
    )
    return story, sagwan, extract_glossary(title + " " + generated, limit=4)


def get_train_350m_status() -> dict:
    running = False
    my_pid = str(os.getpid())
    for p in Path("/proc").glob("[0-9]*"):
        if p.name == my_pid:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().decode("utf-8", "ignore")
            if ("14_train_with_hf_sync" in cmd or "04_pretrain.py" in cmd) and "python" in cmd:
                running = True
                break
        except Exception:
            pass
    last_line = ""
    if os.path.exists("/content/train_350m.log"):
        try:
            with open("/content/train_350m.log", "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if lines:
                    last_line = lines[-1]
        except Exception:
            pass
    return {"train_350m_running": running, "train_350m_last_log": last_line}


class StudioHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _send_json(self, code: int, payload: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            global DB_STATS_CACHE
            if not DB_STATS_CACHE.get("search_db_ready"):
                DB_STATS_CACHE = get_db_stats()
            data = {
                "checkpoint": MODEL_ID,
                "global_step": 4400,
                "best_val_loss": 3.84,
                "parameters": 100700000,
                "device": DEVICE,
                "supported_kings": 30,
                "status": "ready",
                **DB_STATS_CACHE,
                **get_train_350m_status(),
            }
            self._send_json(200, data)
        elif parsed.path == "/api/search":
            qs = urllib.parse.parse_qs(parsed.query)
            q = (qs.get("q") or [""])[0].strip()
            king = (qs.get("king") or [""])[0].strip()
            saron_only = (qs.get("saron_only") or ["false"])[0].lower() in ("1", "true", "yes")
            limit = int((qs.get("limit") or ["12"])[0])
            if not q:
                self._send_json(400, {"error": "질문 또는 검색어(q)를 입력해 주세요."})
                return
            self._send_json(200, search_sillok_archive(q, king_filter=king, saron_only=saron_only, limit=limit))
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": str(e)})
            return

        if parsed.path == "/api/search":
            q = str(payload.get("query") or payload.get("q") or "").strip()
            king = str(payload.get("king") or "").strip()
            saron_only = bool(payload.get("saron_only", False))
            limit = int(payload.get("limit", 12))
            if not q:
                self._send_json(400, {"error": "질문 또는 검색어를 입력해 주세요."})
                return
            self._send_json(200, search_sillok_archive(q, king_filter=king, saron_only=saron_only, limit=limit))
            return

        if parsed.path == "/api/generate":
            king = payload.get("king", "세종실록(世宗實錄)")
            date = payload.get("date", "세종 28년 9월 29일 | 庚戌")
            subject = payload.get("subject", "문화-출판/인쇄")
            title = payload.get("title_ko", "")
            hanja_body = payload.get("body_hanja", "")
            target_mode = payload.get("target_mode", "hanja")
            max_new_tokens = int(payload.get("max_new_tokens", 65))
            temperature = float(payload.get("temperature", 0.6))
            rep_pen = float(payload.get("repetition_penalty", 1.15))

            if target_mode == "title":
                prompt = f"<|bos|><|king|>{king}<|date|>{date}<|subject|>{subject}<|hanja|>{hanja_body}<|title|>"
            elif target_mode == "saron":
                prompt = f"<|bos|><|king|>{king}<|date|>{date}<|subject|>{subject}<|title|>{title}<|hanja|>{hanja_body}<|postscript|>"
            else:
                prompt = f"<|bos|><|king|>{king}<|date|>{date}<|subject|>{subject}<|title|>{title}<|hanja|>"

            t0 = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.1),
                    repetition_penalty=rep_pen,
                )
            elapsed = time.time() - t0
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            new_text = full_text[len(prompt) :].replace("<|eos|>", "").strip()

            p_toks = inputs["input_ids"].shape[1]
            g_toks = outputs.shape[1] - p_toks
            tok_per_sec = round(g_toks / max(elapsed, 0.001), 1)

            ruby_html = make_ruby(new_text)
            reading = make_reading(new_text)
            story_sum, saron_c, gloss = interpret_story(king, date, subject, title, new_text)

            res = {
                "prompt": prompt,
                "generated": new_text,
                "full_text": full_text,
                "hangul_reading": reading,
                "ruby_html": ruby_html,
                "story_summary": story_sum,
                "sagwan_comment": saron_c,
                "glossary": gloss,
                "prompt_tokens": p_toks,
                "generated_tokens": g_toks,
                "elapsed_sec": round(elapsed, 3),
                "tokens_per_sec": tok_per_sec,
                "global_step": 4400,
                "device": DEVICE,
            }
            self._send_json(200, res)
            return

        self._send_json(404, {"error": "Not found"})


def run():
    server = ThreadingHTTPServer(("0.0.0.0", 8020), StudioHandler)
    print("Serving Sillok Archive Search & Studio API on 0.0.0.0:8020...", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run()
