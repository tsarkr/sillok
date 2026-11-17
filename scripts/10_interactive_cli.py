#!/usr/bin/env python3
"""Interactive CLI studio (REPL) and numbered-list selector for Sillok-LM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sillok.training.generate import SillokInferenceEngine

# Import domain lists from web studio
import importlib
import re
import sys
_studio_mod = importlib.import_module("scripts.11_serve_web_studio")
GANJI_60 = _studio_mod.GANJI_60
KINGS_30 = _studio_mod.KINGS_30
LANDMARK_PRESETS = _studio_mod.LANDMARK_PRESETS
SUBJECTS_12 = _studio_mod.SUBJECTS_12


def _clean_king_title(king_str: str) -> str:
    clean = re.sub(r"\(.*?\)", "", king_str).strip()
    return clean.replace("실록", "").replace("일기", "")


def _print_result(res: dict[str, Any], header: str = "생성 결과") -> None:
    print("\n" + "─" * 70)
    print(f" 📜 [{header}]")
    print("─" * 70)
    print(f" • 조립된 프롬프트 : {res['prompt']}")
    print(f" • 사관 집필문     : {res['generated']}")
    print(
        f" • 실행 텔레메트리  : step={res['global_step']} | device={res['device']} | "
        f"{res['generated_tokens']} tokens | {res['elapsed_sec']}s ({res['tokens_per_sec']} tok/s)"
    )
    print("─" * 70)


def _select_from_list(prompt_text: str, items: list[str], default_idx: int = 0) -> str:
    print(f"\n[{prompt_text}]")
    # Display items in 3 columns for readability
    cols = 3
    for i in range(0, len(items), cols):
        row = items[i : i + cols]
        line = "  ".join(f"{i + j + 1:2d}) {row[j]:<22}" for j in range(len(row)))
        print(line)
    choice = input(f"\n번호를 선택하십시오 (1~{len(items)}) [기본값: {default_idx + 1} ({items[default_idx]})]: ").strip()
    if not choice:
        return items[default_idx]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            return items[idx]
    except ValueError:
        pass
    print(f"올바르지 않은 입력입니다. 기본값({items[default_idx]})을 적용합니다.")
    return items[default_idx]


def run_interactive_repl(engine: SillokInferenceEngine) -> None:
    st = engine.status_dict()
    print("=" * 70)
    print("   🏛️  디지털 춘추관 (Digital Chunchugwan) 대화형 AI 사관 터미널   ")
    print("=" * 70)
    print(
        f" 모델 가중치 : {st['checkpoint']} (Step {st['global_step']:,}, Best Loss {st['best_val_loss']:.4f})\n"
        f" 연산 장치   : {st['device']} | 파라미터: {st['parameters']:,} | 로드 시간: {st['load_time_sec']}s"
    )
    print("=" * 70)

    king_names = [f"{k[0]}({k[1]})" for k in KINGS_30]
    ganji_names = [f"{g[0]} ({g[1]})" for g in GANJI_60]
    subj_names = [s[0] for s in SUBJECTS_12 if s[0] != "직접입력"]

    while True:
        print("\n[실행 메뉴 선택]")
        print("  1) 정방향 집필 : 번호 목록 선택(왕조/간지/주제) → 한문 실록 본문 생성 (hanja)")
        print("  2) 춘추필법 사론 : 본문 뒤에 사신왈(史臣曰) 사론 논평 생성 (saron)")
        print("  3) 역방향 요약 : 한문 사초 원문 → 한글 기사 제목 생성 (title)")
        print("  4) 역사 프리셋 : 10대 주요 역사적 사건 프리셋 번호 선택 시연")
        print("  q) 종료 (Quit)")
        choice = input("\n메뉴 선택 (1/2/3/4/q) [기본값 1]: ").strip().lower() or "1"

        if choice in ("q", "quit", "exit"):
            print("디지털 춘추관 터미널을 종료합니다.")
            break

        if choice == "4":
            print("\n[10대 주요 역사적 사건 프리셋]")
            for p in LANDMARK_PRESETS:
                print(f"  {p['id']:2d}) {p['name']} ({p['king']}, {p['ganji']}, {p['mode']})")
            p_choice = input(f"\n프리셋 번호를 선택하십시오 (1~{len(LANDMARK_PRESETS)}, all: 전체 실행) [기본값 2]: ").strip() or "2"
            if p_choice.lower() == "all":
                for p in LANDMARK_PRESETS:
                    date_str = f"{_clean_king_title(p['king'])} {p['year']}년 {p['month']}월 {p['day']}일 | {p['ganji']}"
                    res = engine.generate(
                        king=p["king"],
                        date=date_str,
                        subject=p["subject"],
                        title_ko=p["title"],
                        body_hanja=p["hanja"],
                        target_mode=p["mode"],
                        max_new_tokens=p["tokens"],
                        temperature=p["temp"],
                    )
                    _print_result(res, header=p["name"])
                continue
            try:
                pid = int(p_choice)
                p = next((x for x in LANDMARK_PRESETS if x["id"] == pid), LANDMARK_PRESETS[1])
            except ValueError:
                p = LANDMARK_PRESETS[1]

            date_str = f"{_clean_king_title(p['king'])} {p['year']}년 {p['month']}월 {p['day']}일 | {p['ganji']}"
            res = engine.generate(
                king=p["king"],
                date=date_str,
                subject=p["subject"],
                title_ko=p["title"],
                body_hanja=p["hanja"],
                target_mode=p["mode"],
                max_new_tokens=p["tokens"],
                temperature=p["temp"],
            )
            _print_result(res, header=p["name"])
            continue

        # Step 1: Select King from List
        king = _select_from_list("1. 왕조 실록 선택 (30종 실록)", king_names, default_idx=3)  # default 세종

        # Step 2: Select Reign Year, Month, Day
        reign_yr = input("\n2. 재위 연차 입력 [기본값: 1년]: ").strip() or "1년"
        if not reign_yr.endswith("년"):
            reign_yr += "년"
        month_in = input("3. 월(月) 입력 (1~12) [기본값: 7월]: ").strip() or "7월"
        if not month_in.endswith("월"):
            month_in += "월"
        day_in = input("4. 일(日) 입력 (1~30) [기본값: 15일]: ").strip() or "15일"
        if not day_in.endswith("일"):
            day_in += "일"

        # Step 3: Select Ganji from 60 Ganji List
        ganji_full = _select_from_list("5. 60간지(干支) 선택", ganji_names, default_idx=32)  # default 丙申
        ganji_code = ganji_full.split()[0]

        king_short = _clean_king_title(king)
        date_combined = f"{king_short} {reign_yr} {month_in} {day_in} | {ganji_code}"
        print(f"\n=> 자동 조립된 날짜 규격: {date_combined}")

        # Step 4: Select Subject from List
        subject = _select_from_list("6. 역사 주제 분류 선택", subj_names, default_idx=0)

        if choice == "2":  # Saron mode
            title_ko = input("\n7. 사론을 지을 사건 한글 제목: ").strip() or "대간이 조광조 등을 탄핵하다"
            body_hanja = input("8. 사건 한문 본문 기사 입력: ").strip() or f"○{ganji_code}/下趙光祖、金淨等于義禁府獄。"
            res = engine.generate(
                king=king,
                date=date_combined,
                subject=subject,
                title_ko=title_ko,
                body_hanja=body_hanja,
                target_mode="saron",
                max_new_tokens=75,
                temperature=0.7,
            )
            _print_result(res, header="사관의 춘추필법 사론(史論) 비평 집필")
        elif choice == "3":  # Title mode
            body_hanja = input("\n7. 제목을 요약할 한문 사초 원문 입력: ").strip() or f"○{ganji_code}/太祖卽位于壽昌宮, 百官奉國寶陳賀。"
            res = engine.generate(
                king=king,
                date=date_combined,
                subject=subject,
                body_hanja=body_hanja,
                target_mode="title",
                max_new_tokens=40,
                temperature=0.4,
            )
            _print_result(res, header="역방향 한문→한글 제목 요약")
        else:  # Hanja mode
            title_ko = input("\n7. 집필할 기사의 한글 제목 입력 [기본값: 서책을 완성하여 인쇄를 명하다]: ").strip() or "서책을 완성하여 인쇄를 명하다"
            res = engine.generate(
                king=king,
                date=date_combined,
                subject=subject,
                title_ko=title_ko,
                target_mode="hanja",
                max_new_tokens=65,
                temperature=0.6,
            )
            _print_result(res, header="정방향 한문 실록 기사 집필")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sillok-LM Interactive Numbered-List CLI Studio.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sillok_100m_t4/infer.pt"),
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--preset", type=int, choices=list(range(1, 11)), default=None, help="Run a specific preset (1-10)")
    parser.add_argument("--run-all-presets", action="store_true", help="Run all 10 presets and exit")
    args = parser.parse_args()

    engine = SillokInferenceEngine(args.checkpoint, args.tokenizer, device=args.device)

    if args.run_all_presets:
        for p in LANDMARK_PRESETS:
            date_str = f"{_clean_king_title(p['king'])} {p['year']}년 {p['month']}월 {p['day']}일 | {p['ganji']}"
            res = engine.generate(
                king=p["king"],
                date=date_str,
                subject=p["subject"],
                title_ko=p["title"],
                body_hanja=p["hanja"],
                target_mode=p["mode"],
                max_new_tokens=p["tokens"],
                temperature=p["temp"],
            )
            _print_result(res, header=p["name"])
        return

    if args.preset is not None:
        p = next((x for x in LANDMARK_PRESETS if x["id"] == args.preset), LANDMARK_PRESETS[0])
        date_str = f"{p['king'].replace('실록', '').replace('일기', '')} {p['year']}년 {p['month']}월 {p['day']}일 | {p['ganji']}"
        res = engine.generate(
            king=p["king"],
            date=date_str,
            subject=p["subject"],
            title_ko=p["title"],
            body_hanja=p["hanja"],
            target_mode=p["mode"],
            max_new_tokens=p["tokens"],
            temperature=p["temp"],
        )
        _print_result(res, header=p["name"])
        return

    run_interactive_repl(engine)


if __name__ == "__main__":
    main()
