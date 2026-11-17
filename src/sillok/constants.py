"""Domain constants, dataset provenance, king code mappings, and control tokens for Sillok-LM."""

from __future__ import annotations

# Official dataset provenance metadata
DATASET_INFO: dict[str, object] = {
    "dataset_name": "교육부 국사편찬위원회_조선왕조실록 정보_실록원문_20260821",
    "provider": "교육부 국사편찬위원회 (National Institute of Korean History)",
    "dtd_version": "history.dtd ver 1.4 (2020-07-30)",
    "license": "공공누리 제1유형(출처표시) 및 공공누리 AI유형",
    "total_xml_files": 673,
    "total_articles": 381701,
    "total_annals": 30,
}

# Mapping from 3-letter NHIC Sillok dataset code to (Korean title, Hanja title)
KING_CODE_MAP: dict[str, tuple[str, str]] = {
    "waa": ("태조실록", "太祖實錄"),
    "wba": ("정종실록", "定宗實錄"),
    "wca": ("태종실록", "太宗實錄"),
    "wda": ("세종실록", "世宗實錄"),
    "wea": ("문종실록", "文宗實錄"),
    "wfa": ("단종실록", "端宗實錄"),
    "wga": ("세조실록", "世祖實錄"),
    "wha": ("예종실록", "睿宗實錄"),
    "wia": ("성종실록", "成宗實錄"),
    "wja": ("연산군일기", "燕山君日記"),
    "wka": ("중종실록", "中宗實錄"),
    "wla": ("인종실록", "仁宗實錄"),
    "wma": ("명종실록", "明宗實錄"),
    "wna": ("선조실록", "宣祖實錄"),
    "wnb": ("선조수정실록", "宣祖修正實錄"),
    "woa": ("광해군일기(중초본)", "光海君日記(中草本)"),
    "wob": ("광해군일기(정초본)", "光海君日記(正草本)"),
    "wpa": ("인조실록", "仁祖實錄"),
    "wqa": ("효종실록", "孝宗實錄"),
    "wra": ("현종실록", "顯宗實錄"),
    "wrb": ("현종개수실록", "顯宗改修實錄"),
    "wsa": ("숙종실록", "肅宗實錄"),
    "wsb": ("숙종실록보궐정오", "肅宗實錄補闕正誤"),
    "wta": ("경종실록", "景宗實錄"),
    "wtb": ("경종수정실록", "景宗修正實錄"),
    "wua": ("영조실록", "英祖實錄"),
    "wva": ("정조실록", "正祖實錄"),
    "wwa": ("순조실록", "純祖實錄"),
    "wxa": ("헌종실록", "憲宗實錄"),
    "wya": ("철종실록", "哲宗實錄"),
}

# Ordered special tokens so IDs remain stable across tokenizer training and modeling.
SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|king|>",
    "<|date|>",
    "<|subject|>",
    "<|title|>",
    "<|hanja|>",
    "<|ko|>",
    "<|postscript|>",
]

SPECIAL_TOKEN_IDS: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}

DEFAULT_SPECIAL_TOKENS: dict[str, str] = {
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "unk": "<|unk|>",
    "king": "<|king|>",
    "date": "<|date|>",
    "subject": "<|subject|>",
    "title": "<|title|>",
    "hanja": "<|hanja|>",
    "ko": "<|ko|>",
    "postscript": "<|postscript|>",
}
