#!/usr/bin/env python3
"""Digital Chunchugwan (디지털 춘추관) Web Studio & REST API Server for Sillok-LM.

Provides:
- Interactive Web UI at GET / (http://127.0.0.1:8020) with full dropdown menus for 30 Kings, 60 Ganji, and 12 Historical Subjects
- Model status telemetry at GET /api/status
- Conditional generation REST API at POST /api/generate
"""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sillok.constants import KING_CODE_MAP
from sillok.training.generate import SillokInferenceEngine

# 60 Ganji list with Korean pronunciation
GANJI_60: list[tuple[str, str]] = [
    ("甲子", "갑자"), ("乙丑", "을축"), ("丙寅", "병인"), ("丁卯", "정묘"), ("戊辰", "무진"),
    ("己巳", "기사"), ("庚午", "경오"), ("辛未", "신미"), ("壬申", "임신"), ("癸酉", "계유"),
    ("甲戌", "갑술"), ("乙亥", "을해"), ("丙子", "병자"), ("丁丑", "정축"), ("戊寅", "무인"),
    ("己卯", "기묘"), ("庚辰", "경진"), ("辛巳", "신사"), ("壬午", "임오"), ("癸未", "계미"),
    ("甲申", "갑신"), ("乙酉", "을유"), ("丙戌", "병술"), ("丁亥", "정해"), ("戊子", "무자"),
    ("己丑", "기축"), ("庚寅", "경인"), ("辛卯", "신묘"), ("壬辰", "임진"), ("癸巳", "계사"),
    ("甲午", "갑오"), ("乙未", "을미"), ("丙申", "병신"), ("丁酉", "정유"), ("戊戌", "무술"),
    ("己亥", "기해"), ("庚子", "경자"), ("辛丑", "신축"), ("壬寅", "임인"), ("癸卯", "계묘"),
    ("甲辰", "갑진"), ("乙巳", "을사"), ("丙午", "병오"), ("丁未", "정미"), ("戊申", "무신"),
    ("己酉", "기유"), ("庚戌", "경술"), ("辛亥", "신해"), ("壬子", "임자"), ("癸丑", "계축"),
    ("甲寅", "갑인"), ("乙卯", "을묘"), ("丙辰", "병진"), ("丁巳", "정사"), ("戊午", "무오"),
    ("己未", "기미"), ("庚申", "경신"), ("辛酉", "신유"), ("壬戌", "임술"), ("癸亥", "계해"),
]

# 30 Kings in chronological order
KINGS_30: list[tuple[str, str]] = [
    ("태조실록", "太祖實錄"),
    ("정종실록", "定宗實錄"),
    ("태종실록", "太宗實錄"),
    ("세종실록", "世宗實錄"),
    ("문종실록", "文宗實錄"),
    ("단종실록", "端宗實錄"),
    ("세조실록", "世祖實錄"),
    ("예종실록", "睿宗實錄"),
    ("성종실록", "成宗實錄"),
    ("연산군일기", "燕山君日記"),
    ("중종실록", "中宗實錄"),
    ("인종실록", "仁宗實錄"),
    ("명종실록", "明宗實錄"),
    ("선조실록", "宣祖實錄"),
    ("선조수정실록", "宣祖修正實錄"),
    ("광해군일기", "光海君日記"),
    ("인조실록", "仁祖實錄"),
    ("효종실록", "孝宗實錄"),
    ("현종실록", "顯宗實錄"),
    ("현종개수실록", "顯宗改修實錄"),
    ("숙종실록", "肅宗實錄"),
    ("숙종실록보궐정오", "肅宗實錄補闕正誤"),
    ("경종실록", "景宗實錄"),
    ("경종수정실록", "景宗修正實錄"),
    ("영조실록", "英祖實錄"),
    ("정조실록", "正祖實錄"),
    ("순조실록", "純祖實錄"),
    ("헌종실록", "憲宗實錄"),
    ("철종실록", "哲宗實錄"),
]

# 12 Major historical subjects in the Annals
SUBJECTS_12: list[tuple[str, str]] = [
    ("왕실-국왕/임면", "왕실 의전, 국왕 즉위, 양위, 책봉, 승하"),
    ("정치-정관/임면", "의정부, 육조, 승정원 등 조정 관원 임명·해임"),
    ("정치-정관/탄핵", "사헌부·사간원 대간의 탄핵, 상소, 당쟁"),
    ("문화-출판/인쇄", "훈민정음, 서책 편찬, 주자소 활자 인쇄"),
    ("군사-전쟁/방비", "왜란·호란 외침, 국방, 변방 경비, 군제 개편"),
    ("사법-재판/형벌", "의금부 옥사, 추국, 역모 심문, 형벌"),
    ("외교-명/청/왜", "명·청 황제 사신 영접, 조천사 파견, 왜관 교섭"),
    ("천문-관측/재해", "일식, 월식, 혜성, 지진, 가뭄 구휼, 제사"),
    ("재정-조세/공물", "전세, 호포법, 대동법, 공납, 국가 재정"),
    ("교육-성균관/과거", "성균관 유생, 과거 시험, 문무과 급제"),
    ("사회-신분/노비", "양민 보호, 노비 환천, 서얼 차별, 민란"),
    ("직접입력", "사용자 정의 직접 입력"),
]

# 10 Major Historical Landmark Presets
LANDMARK_PRESETS: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "① 태조 1년 수창궁 즉위",
        "king": "태조실록(太祖實錄)",
        "year": "1", "month": "7", "day": "17", "ganji": "丙申",
        "subject": "왕실-국왕/임면",
        "title": "태조가 수창궁에서 왕위에 오르다",
        "hanja": "",
        "mode": "hanja",
        "tokens": 60, "temp": 0.5,
    },
    {
        "id": 2,
        "name": "② 세종 28년 훈민정음 반포",
        "king": "세종실록(世宗實錄)",
        "year": "28", "month": "9", "day": "29", "ganji": "庚戌",
        "subject": "문화-출판/인쇄",
        "title": "어제 훈민정음 서책을 완성하여 인쇄하게 하다",
        "hanja": "",
        "mode": "hanja",
        "tokens": 65, "temp": 0.6,
    },
    {
        "id": 3,
        "name": "③ 성종 16년 경국대전 완성",
        "king": "성종실록(成宗實錄)",
        "year": "16", "month": "1", "day": "1", "ganji": "乙巳",
        "subject": "문화-출판/인쇄",
        "title": "경국대전을 인쇄하여 팔도에 반포하다",
        "hanja": "",
        "mode": "hanja",
        "tokens": 60, "temp": 0.6,
    },
    {
        "id": 4,
        "name": "④ 연산군 4년 무오사화 사론",
        "king": "연산군일기(燕山君日記)",
        "year": "4", "month": "7", "day": "12", "ganji": "戊午",
        "subject": "정치-정관/탄핵",
        "title": "김일손 등을 국문하여 조의제문 작성 죄를 묻다",
        "hanja": "○戊午/親鞫金馹孫等, 下義禁府獄。",
        "mode": "saron",
        "tokens": 70, "temp": 0.7,
    },
    {
        "id": 5,
        "name": "⑤ 중종 14년 기묘사화 조광조 하옥",
        "king": "중종실록(中宗實錄)",
        "year": "14", "month": "11", "day": "15", "ganji": "己卯",
        "subject": "정치-정관/탄핵",
        "title": "홍경주 등이 조광조 일파를 탄핵하여 하옥하다",
        "hanja": "○己卯/夜二鼓, 洪景舟、南袞、沈貞等詣闕請對, 下趙光祖、金淨、金絿等于義禁府獄。",
        "mode": "saron",
        "tokens": 75, "temp": 0.7,
    },
    {
        "id": 6,
        "name": "⑥ 선조 25년 임진왜란 발발",
        "king": "선조실록(宣祖實錄)",
        "year": "25", "month": "4", "day": "13", "ganji": "壬辰",
        "subject": "군사-전쟁/방비",
        "title": "왜적이 부산포에 침입하여 전란이 시작되다",
        "hanja": "",
        "mode": "hanja",
        "tokens": 65, "temp": 0.65,
    },
    {
        "id": 7,
        "name": "⑦ 광해군 2년 동의보감 간행",
        "king": "광해군일기(光海君日記)",
        "year": "2", "month": "8", "day": "6", "ganji": "庚戌",
        "subject": "문화-출판/인쇄",
        "title": "허준이 동의보감을 완성하여 바치니 출판을 명하다",
        "hanja": "",
        "mode": "hanja",
        "tokens": 60, "temp": 0.6,
    },
    {
        "id": 8,
        "name": "⑧ 인조 14년 병자호란 남한산성",
        "king": "인조실록(仁祖實錄)",
        "year": "14", "month": "12", "day": "14", "ganji": "丙子",
        "subject": "군사-전쟁/방비",
        "title": "청나라 군대가 침입하매 어가를 남한산성으로 옮기다",
        "hanja": "",
        "mode": "hanja",
        "tokens": 65, "temp": 0.65,
    },
    {
        "id": 9,
        "name": "⑨ 정조 20년 수원 화성 준공",
        "king": "정조실록(正祖實錄)",
        "year": "20", "month": "10", "day": "16", "ganji": "丙辰",
        "subject": "왕실-의식/행차",
        "title": "화성 성역이 완성되니 낙성연을 베풀고 장인들을 포상하다",
        "hanja": "",
        "mode": "hanja",
        "tokens": 60, "temp": 0.6,
    },
    {
        "id": 10,
        "name": "⑩ 역방향 한문→한글 제목 요약",
        "king": "태조실록(太祖實錄)",
        "year": "1", "month": "7", "day": "17", "ganji": "丙申",
        "subject": "왕실-국왕/임면",
        "title": "",
        "hanja": "○丙申/太祖卽位于壽昌宮, 百官奉國寶陳賀。",
        "mode": "title",
        "tokens": 40, "temp": 0.4,
    },
]

STUDIO_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>디지털 춘추관 (Digital Chunchugwan) — 조선왕조실록 AI 사관 스튜디오</title>
<style>
  :root {
    --bg: #f5f0e6;
    --card: #fffdfa;
    --ink: #1a1714;
    --accent: #7c2d12;
    --gold: #b45309;
    --border: #d4c8b5;
    --muted: #645a4f;
    --highlight: #fef3c7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px;
    background: var(--bg); color: var(--ink);
    font-family: "Noto Serif KR", "Nanum Myeongjo", Georgia, serif;
    line-height: 1.6;
  }
  .container { max-width: 1180px; margin: 0 auto; }
  header {
    background: linear-gradient(135deg, #38190d 0%, #5a2612 100%);
    color: #fdf8f0; padding: 22px 28px; border-radius: 12px;
    border-bottom: 4px solid var(--gold); margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
  }
  header h1 { margin: 0 0 6px 0; font-size: 1.65rem; letter-spacing: 0.5px; }
  header p { margin: 0; color: #e7d8c5; font-size: 0.93rem; }
  .badge-bar {
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; font-family: monospace; font-size: 0.85rem;
  }
  .badge {
    background: rgba(255,255,255,0.14); padding: 4px 10px; border-radius: 6px;
  }
  .grid { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 20px; }
  @media (max-width: 920px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  }
  h2 { margin-top: 0; font-size: 1.22rem; color: var(--accent); border-bottom: 2px solid var(--border); padding-bottom: 8px; }
  label { display: block; font-weight: bold; font-size: 0.88rem; margin: 12px 0 4px; color: #3b2e25; }
  select, input[type="text"], input[type="number"], textarea {
    width: 100%; padding: 9px 11px; border: 1px solid var(--border);
    border-radius: 6px; background: #fff; font-size: 0.95rem; font-family: inherit; color: var(--ink);
  }
  select:focus, input:focus, textarea:focus {
    outline: 2px solid var(--gold); border-color: transparent;
  }
  textarea { min-height: 85px; resize: vertical; }
  .presets-wrap { margin-bottom: 14px; }
  .presets-wrap p { margin: 0 0 6px 0; font-size: 0.85rem; font-weight: bold; color: var(--muted); }
  .presets { display: flex; flex-wrap: wrap; gap: 6px; }
  .preset-btn {
    background: #ebe1cf; border: 1px solid #c9b99f; color: var(--ink);
    padding: 5px 9px; border-radius: 6px; cursor: pointer; font-size: 0.8rem;
    transition: all 0.15s ease;
  }
  .preset-btn:hover { background: #dfd0b7; border-color: var(--gold); }
  
  /* Date builder grid */
  .date-builder {
    display: grid; grid-template-columns: 1fr 1fr 1fr 1.2fr; gap: 8px;
    background: #fbf7ee; padding: 10px; border-radius: 8px; border: 1px dashed var(--border);
  }
  .date-builder > div > label { margin: 0 0 3px 0; font-size: 0.78rem; color: var(--muted); }
  .date-preview {
    grid-column: 1 / -1; margin-top: 6px; font-family: monospace; font-size: 0.85rem;
    color: var(--accent); background: #f3ece0; padding: 6px 10px; border-radius: 4px;
  }

  .controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }
  .submit-btn {
    width: 100%; margin-top: 18px; padding: 13px;
    background: linear-gradient(135deg, #7c2d12 0%, #9a3412 100%);
    color: #fff; border: none; border-radius: 8px; font-size: 1.08rem; font-weight: bold;
    cursor: pointer; box-shadow: 0 2px 6px rgba(124, 45, 18, 0.3);
    transition: all 0.15s ease;
  }
  .submit-btn:hover { background: linear-gradient(135deg, #60210c 0%, #7c2d12 100%); }
  .submit-btn:disabled { opacity: 0.6; cursor: wait; }

  .output-box {
    background: #faf6ec; border: 1px solid var(--border); border-radius: 8px;
    padding: 18px; min-height: 200px; white-space: pre-wrap; font-size: 1.18rem;
    line-height: 1.85; border-left: 5px solid var(--accent);
  }
  .prompt-box {
    background: #efe8dc; border-radius: 6px; padding: 10px;
    font-family: monospace; font-size: 0.82rem; color: var(--muted);
    white-space: pre-wrap; word-break: break-all; margin-top: 12px;
  }
  .meta-line { font-family: monospace; font-size: 0.85rem; color: var(--muted); margin-top: 10px; }
  .guide-tip {
    margin-top: 16px; padding: 12px; background: #fdf8ee; border-radius: 8px;
    border: 1px solid #ebdcc5; font-size: 0.85rem; color: #5a4b3d;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🏛️ 디지털 춘추관 (Digital Chunchugwan) — 조선왕조실록 AI 사관 스튜디오</h1>
    <p>30대 국왕 실록 · 60간지(干支) · 12대 역사 주제 드롭다운 선택형 AI 사관 문장 집필기</p>
    <div class="badge-bar" id="status-bar">
      <span class="badge">상태 불러오는 중...</span>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <h2>1. 사초(史草) 조건 선택 (드롭다운 목록)</h2>
      
      <div class="presets-wrap">
        <p>📜 주요 역사 사건 원클릭 프리셋 (클릭 시 전체 항목 자동 세팅):</p>
        <div class="presets" id="presets-container"></div>
      </div>

      <label>집필 모드 (Target Mode)</label>
      <select id="mode" onchange="onModeChange()">
        <option value="hanja">정방향 집필: 한글 제목 → 한문 실록 원문 생성 (&lt;|hanja|&gt;)</option>
        <option value="title">역방향 요약: 한문 사초 원문 → 한글 기사 제목 생성 (&lt;|title|&gt;)</option>
        <option value="saron">춘추필법 비평: 본문 뒤에 사신왈(史臣曰) 사론 논평 생성 (&lt;|postscript|&gt;)</option>
      </select>

      <label>왕조 실록 선택 (30종 실록 목록)</label>
      <select id="king-select" onchange="updateDateString()">
        <!-- Kings populated by JS -->
      </select>

      <label>재위 연월일 및 60간지(干支) 선택</label>
      <div class="date-builder">
        <div>
          <label>재위 연차</label>
          <select id="reign-year" onchange="updateDateString()">
            <!-- 1 ~ 50 populated by JS -->
          </select>
        </div>
        <div>
          <label>월 (月)</label>
          <select id="month" onchange="updateDateString()">
            <!-- 1 ~ 12 populated by JS -->
          </select>
        </div>
        <div>
          <label>일 (日)</label>
          <select id="day" onchange="updateDateString()">
            <!-- 1 ~ 30 populated by JS -->
          </select>
        </div>
        <div>
          <label>60간지 (干支)</label>
          <select id="ganji-select" onchange="updateDateString()">
            <!-- 60 Ganji populated by JS -->
          </select>
        </div>
        <div class="date-preview" id="date-preview">날짜 미리보기: 태조 1년 7월 17일 | 丙申</div>
      </div>

      <label>역사 주제 분류 (12대 분류 목록)</label>
      <select id="subject-select" onchange="onSubjectChange()">
        <!-- Subjects populated by JS -->
      </select>
      <input type="text" id="subject-custom" placeholder="직접 입력 시 작성" style="display:none; margin-top:6px;">

      <label id="title-label">한글 기사 제목</label>
      <input type="text" id="title_ko" value="태조가 수창궁에서 왕위에 오르다">

      <label id="hanja-label">한문 원문 사초 (&lt;|hanja|&gt; — 역방향 요약 또는 사론 생성 시 사용)</label>
      <textarea id="body_hanja" placeholder="정방향 한문 생성 시에는 비워두셔도 됩니다."></textarea>

      <div class="controls">
        <div>
          <label>최대 생성 글자(토큰)</label>
          <input type="number" id="max_new_tokens" value="60" min="10" max="256">
        </div>
        <div>
          <label>창의성 온도 (Temp)</label>
          <input type="number" id="temperature" value="0.6" step="0.05" min="0.1" max="1.5">
        </div>
        <div>
          <label>반복 억제 페널티</label>
          <input type="number" id="repetition_penalty" value="1.15" step="0.05" min="1.0" max="2.0">
        </div>
      </div>

      <button class="submit-btn" id="gen-btn" onclick="runGenerate()">🖌️ 인공지능 사관에게 집필 명하기</button>
    </div>

    <div class="card">
      <h2>2. 인공지능 사관 집필 결과 (實錄 生成文)</h2>
      <div class="output-box" id="output-box">좌측에서 왕조와 간지, 주제를 선택하신 뒤 [🖌️ 집필 명하기] 버튼을 누르십시오.</div>
      <div class="meta-line" id="telemetry-line"></div>
      
      <label style="margin-top:16px;">조립된 편년체 제어 프롬프트 (Formatted Prompt)</label>
      <div class="prompt-box" id="prompt-box">-</div>

      <div class="guide-tip">
        💡 <strong>인문학도를 위한 실록 집필 팁</strong>:
        <ul style="margin:6px 0 0 18px; padding:0;">
          <li><strong>정방향 집필</strong>: 국왕과 간지를 정하고 한글 제목만 넣으면 당대 편년체 한문 기사(<code>○간지/...</code>)가 지어집니다.</li>
          <li><strong>사론(史論) 비평</strong>: 기사 끝에 <code>【史臣曰: "..."】</code> 형태의 성리학적 도덕률과 인물 비평이 작성됩니다.</li>
          <li><strong>역방향 요약</strong>: 한문 사초를 복사해 넣으면 국사편찬위원회 연구원 스타일의 한글 제목이 자동 추출됩니다.</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<script>
const KINGS = JSON.parse('__KINGS_JSON__');
const GANJI = JSON.parse('__GANJI_JSON__');
const SUBJECTS = JSON.parse('__SUBJECTS_JSON__');
const PRESETS = JSON.parse('__PRESETS_JSON__');

// Populate Kings dropdown
const kingSel = document.getElementById("king-select");
KINGS.forEach(k => {
  const opt = document.createElement("option");
  opt.value = `${k[0]}(${k[1]})`;
  opt.textContent = `${k[0]} (${k[1]})`;
  kingSel.appendChild(opt);
});

// Populate Reign Year (1~50)
const yearSel = document.getElementById("reign-year");
const opt0 = document.createElement("option");
opt0.value = "즉위년"; opt0.textContent = "즉위년 (卽位年)";
yearSel.appendChild(opt0);
for (let y = 1; y <= 50; y++) {
  const opt = document.createElement("option");
  opt.value = `${y}년`; opt.textContent = `${y}년`;
  if (y === 1) opt.selected = true;
  yearSel.appendChild(opt);
}

// Populate Month (1~12)
const monthSel = document.getElementById("month");
for (let m = 1; m <= 12; m++) {
  const opt = document.createElement("option");
  opt.value = `${m}월`; opt.textContent = `${m}월`;
  if (m === 7) opt.selected = true;
  monthSel.appendChild(opt);
}

// Populate Day (1~30)
const daySel = document.getElementById("day");
for (let d = 1; d <= 30; d++) {
  const opt = document.createElement("option");
  opt.value = `${d}일`; opt.textContent = `${d}일`;
  if (d === 17) opt.selected = true;
  daySel.appendChild(opt);
}

// Populate 60 Ganji dropdown
const ganjiSel = document.getElementById("ganji-select");
GANJI.forEach((g, idx) => {
  const opt = document.createElement("option");
  opt.value = g[0];
  opt.textContent = `${idx + 1}. ${g[0]} (${g[1]})`;
  if (g[0] === "丙申") opt.selected = true;
  ganjiSel.appendChild(opt);
});

// Populate Subjects dropdown
const subjSel = document.getElementById("subject-select");
SUBJECTS.forEach(s => {
  const opt = document.createElement("option");
  opt.value = s[0];
  opt.textContent = `${s[0]} — ${s[1]}`;
  subjSel.appendChild(opt);
});

// Populate Preset Buttons
const presetCont = document.getElementById("presets-container");
PRESETS.forEach(p => {
  const btn = document.createElement("button");
  btn.className = "preset-btn";
  btn.textContent = p.name;
  btn.onclick = () => applyPreset(p.id);
  presetCont.appendChild(btn);
});

function getKingShortName() {
  const val = kingSel.value;
  return val.replace(/\(.*?\)/g, "").replace(/실록.*|일기.*/, "").trim();
}

function updateDateString() {
  const kingShort = getKingShortName();
  const yr = yearSel.value;
  const m = monthSel.value;
  const d = daySel.value;
  const g = ganjiSel.value;
  const preview = `${kingShort} ${yr} ${m} ${d} | ${g}`;
  document.getElementById("date-preview").textContent = `날짜 미리보기: ${preview}`;
  return preview;
}

function onModeChange() {
  const mode = document.getElementById("mode").value;
  const titleBox = document.getElementById("title_ko");
  const hanjaBox = document.getElementById("body_hanja");
  if (mode === "title") {
    titleBox.placeholder = "(역방향 요약 모드: 제목이 자동 생성되므로 비워두셔도 됩니다)";
    hanjaBox.placeholder = "요약할 한문 사초 원문을 입력하십시오. (예: ○丙申/太祖卽位于壽昌宮...)";
  } else if (mode === "saron") {
    titleBox.placeholder = "사론을 지을 사건 한글 제목을 입력하십시오.";
    hanjaBox.placeholder = "사건 한문 본문 기사를 입력하십시오. (기사 뒤에 【史臣曰: ...】이 이어집니다)";
  } else {
    titleBox.placeholder = "집필할 기사의 한글 제목을 입력하십시오.";
    hanjaBox.placeholder = "정방향 한문 생성 시에는 비워두셔도 됩니다.";
  }
}

function onSubjectChange() {
  const val = subjSel.value;
  const custom = document.getElementById("subject-custom");
  custom.style.display = (val === "직접입력") ? "block" : "none";
}

function getFinalSubject() {
  return (subjSel.value === "직접입력") ? document.getElementById("subject-custom").value : subjSel.value;
}

function applyPreset(id) {
  const p = PRESETS.find(x => x.id === id);
  if (!p) return;
  document.getElementById("mode").value = p.mode;
  kingSel.value = p.king;
  yearSel.value = (p.year === "즉위년") ? "즉위년" : `${p.year}년`;
  monthSel.value = `${p.month}월`;
  daySel.value = `${p.day}일`;
  ganjiSel.value = p.ganji;
  subjSel.value = p.subject;
  onSubjectChange();
  document.getElementById("title_ko").value = p.title;
  document.getElementById("body_hanja").value = p.hanja;
  document.getElementById("max_new_tokens").value = p.tokens;
  document.getElementById("temperature").value = p.temp;
  onModeChange();
  updateDateString();
}

async function loadStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    document.getElementById("status-bar").innerHTML = `
      <span class="badge">가중치: ${d.checkpoint}</span>
      <span class="badge">학습 스텝: Step ${d.global_step}</span>
      <span class="badge">검증 손실: ${Number(d.best_val_loss).toFixed(4)}</span>
      <span class="badge">파라미터: ${(d.parameters/1e6).toFixed(1)}M</span>
      <span class="badge">연산 장치: ${d.device.toUpperCase()}</span>
      <span class="badge">지원 국왕: ${d.supported_kings}종</span>
    `;
  } catch (e) {}
}

async function runGenerate() {
  const btn = document.getElementById("gen-btn");
  const out = document.getElementById("output-box");
  btn.disabled = true;
  btn.textContent = "⏳ 사관이 붓을 들어 실록을 집필 중입니다...";
  out.textContent = "사관이 사초를 살피며 글을 짓고 있습니다...";
  
  try {
    const dateStr = updateDateString();
    const payload = {
      target_mode: document.getElementById("mode").value,
      king: kingSel.value,
      date: dateStr,
      subject: getFinalSubject(),
      title_ko: document.getElementById("title_ko").value,
      body_hanja: document.getElementById("body_hanja").value,
      max_new_tokens: parseInt(document.getElementById("max_new_tokens").value || "60", 10),
      temperature: parseFloat(document.getElementById("temperature").value || "0.6"),
      repetition_penalty: parseFloat(document.getElementById("repetition_penalty").value || "1.15")
    };
    const r = await fetch("/api/generate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const res = await r.json();
    out.textContent = res.generated || "(생성된 텍스트 없음)";
    document.getElementById("prompt-box").textContent = res.prompt;
    document.getElementById("telemetry-line").textContent =
      `[실행 완료] 스텝: ${res.global_step} | 연산 장치: ${res.device} | 생성 글자: ${res.generated_tokens}개 | 소요 시간: ${res.elapsed_sec}초 (${res.tokens_per_sec} tok/s)`;
  } catch (err) {
    out.textContent = "오류 발생: " + err;
  } finally {
    btn.disabled = false;
    btn.textContent = "🖌️ 인공지능 사관에게 집필 명하기";
  }
}

updateDateString();
loadStatus();
</script>
</body>
</html>
"""

# Embed JSON data directly into the HTML template for zero-latency client rendering
_HTML_RENDERED = (
    STUDIO_HTML.replace("__KINGS_JSON__", json.dumps(KINGS_30, ensure_ascii=False))
    .replace("__GANJI_JSON__", json.dumps(GANJI_60, ensure_ascii=False))
    .replace("__SUBJECTS_JSON__", json.dumps(SUBJECTS_12, ensure_ascii=False))
    .replace("__PRESETS_JSON__", json.dumps(LANDMARK_PRESETS, ensure_ascii=False))
)


class StudioState:
    def __init__(self, checkpoint_path: Path, tokenizer_path: Path, device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.lock = threading.Lock()
        self.mtime = checkpoint_path.stat().st_mtime if checkpoint_path.exists() else 0.0
        self.engine = SillokInferenceEngine(checkpoint_path, tokenizer_path, device=device)

    def get_engine(self) -> SillokInferenceEngine:
        with self.lock:
            if self.checkpoint_path.exists():
                cur_mtime = self.checkpoint_path.stat().st_mtime
                if cur_mtime > self.mtime:
                    print(f"[studio] Reloading updated checkpoint {self.checkpoint_path}...")
                    self.engine = SillokInferenceEngine(
                        self.checkpoint_path, self.tokenizer_path, device=self.device
                    )
                    self.mtime = cur_mtime
            return self.engine


def make_handler(state: StudioState) -> type[BaseHTTPRequestHandler]:
    class StudioHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, obj: Any, status: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html", "/studio", "/studio/"):
                raw = _HTML_RENDERED.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if self.path == "/api/status":
                eng = state.get_engine()
                info = eng.status_dict()
                info["supported_kings"] = len(KING_CODE_MAP)
                self._send_json(info)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/generate":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, status=400)
                return

            eng = state.get_engine()
            with state.lock:
                res = eng.generate(
                    king=str(req.get("king", "태조실록(太祖實錄)")),
                    date=str(req.get("date", "")),
                    subject=str(req.get("subject", "")),
                    title_ko=str(req.get("title_ko", "")),
                    body_hanja=str(req.get("body_hanja", "")),
                    raw_prompt=str(req.get("raw_prompt", "")),
                    target_mode=str(req.get("target_mode", "hanja")),
                    max_new_tokens=min(256, max(1, int(req.get("max_new_tokens", 60)))),
                    temperature=float(req.get("temperature", 0.65)),
                    top_k=int(req.get("top_k", 50)),
                    top_p=float(req.get("top_p", 0.92)),
                    repetition_penalty=float(req.get("repetition_penalty", 1.15)),
                )
            self._send_json(res)

    return StudioHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Digital Chunchugwan Web Studio & REST API.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sillok_100m_t4/infer.pt"),
        help="Path to inference checkpoint (.pt)",
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print(f"[studio] Loading SillokInferenceEngine from {args.checkpoint}...")
    state = StudioState(args.checkpoint, args.tokenizer, device=args.device)
    st = state.engine.status_dict()
    print(
        f"[studio] Ready! step={st['global_step']} | device={st['device']} | "
        f"load_time={st['load_time_sec']}s | listening on http://{args.host}:{args.port}"
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[studio] Shutting down.")


if __name__ == "__main__":
    main()
