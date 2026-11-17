# 『조선왕조실록으로 배우는 언어모델 만들기』 문서 안내

이 문서 모음은 **교육부 국사편찬위원회_조선왕조실록 정보_실록원문_20260821** 원천 XML 데이터를 이용해 **Decoder-only Transformer (`SillokForCausalLM`)**를 처음부터(From Scratch) 직접 만들고 학습하는 과정을 인문학 전공 학생도 따라올 수 있도록 정리한 교육용 기록입니다.

---

## 목차 (Table of Contents)

| 구분 | 문서 | 내용 안내 |
| :---: | :--- | :--- |
| **교재 통합본** | [BOOK_SILLOK_LM_COMPLETE.md](file:///home/nsua/Dev/sillok/docs/BOOK_SILLOK_LM_COMPLETE.md) | **실록 모델의 목적·쓰임·방향성, 소스 코드 줄 단위 심층 해설, 정량·정성 평가 방법론, 모델 실행환경(디지털 춘추관) 및 무인 자동화 전 과정을 집대성한 단행본 원고 (6.1만 자, 1,358행)** |
| **서문 / 색인** | [00_BOOK_INDEX.md](file:///home/nsua/Dev/sillok/docs/00_BOOK_INDEX.md) | 프로젝트 개요, 실측 통계 요약 및 전체 파일 목록 |
| **제1장** | [01_DATA_AND_XML_SPEC.md](file:///home/nsua/Dev/sillok/docs/01_DATA_AND_XML_SPEC.md) | 원천 XML(`history.dtd ver 1.4`) 스키마, 673개 파일·381,701개 기사 전수 통계, UTF-16 서로게이트 복구 및 계층 날짜 상속 규격 |
| **제2장** | [02_SYSTEM_AND_REMOTE_ARCHITECTURE.md](file:///home/nsua/Dev/sillok/docs/02_SYSTEM_AND_REMOTE_ARCHITECTURE.md) | 에지 호스트 `tsarkr`(`Celeron J3160`, `3.7 GiB RAM`), `sshd`/`.vscode-server`/`cloudflared` 프로세스 토폴로지, 저메모리 스트리밍 및 원격 Colab 연동 구조 |
| **제3장** | [03_TOKENIZER_AND_MODEL_ARCHITECTURE.md](file:///home/nsua/Dev/sillok/docs/03_TOKENIZER_AND_MODEL_ARCHITECTURE.md) | 11종 제어 토큰 스키마, 양방향 학습 포맷, Byte-Level BPE 토크나이저, `SillokForCausalLM` 수식 및 `Tiny / 350M / 1B` 파라미터 완전 분해표 |
| **제4장** | [04_TRAINING_AND_OPERATIONS_MANUAL.md](file:///home/nsua/Dev/sillok/docs/04_TRAINING_AND_OPERATIONS_MANUAL.md) | XML 파싱 → 터널 전송 → 토크나이저 학습 → `uint16` Memmap 패킹 → Colab A100 사전학습 및 조건부 생성 실전 매뉴얼 |

---

## 프로젝트 핵심 수치 요약 (실측 기준)

- **공식 원천 데이터**: `교육부 국사편찬위원회_조선왕조실록 정보_실록원문_20260821` ([constants.py](file:///home/nsua/Dev/sillok/src/sillok/constants.py))
- **저작권/이용허락**: 공공누리 제1유형(출처표시) 및 공공누리 AI유형 (`data/공공누리 1유형 마크.jpg`, `data/공공누리 AI유형 마크.png`)
- **원천 XML 규모**: `673`개 파일 (`data/2nd_*.xml`, 약 `815 MiB`) · `30`개 실록(개수·수정실록 포함)
- **전수 파싱 결과 ([data/processed/parse_summary.json](file:///home/nsua/Dev/sillok/data/processed/parse_summary.json))**:
  - 추출 기사 수: **`381,701`** 건 (`articles.jsonl`, 약 `352 MiB`)
  - 한문 본문 총 글자 수: **`67,707,578`** 자
  - 한글 기사 제목 총 글자 수: **`9,859,443`** 자 (`381,701`건 전체 100% 제목 보유)
  - 사론(`postScript`) 총 건수: **`5,873`** 건 (`5,487`개 기사에 분포)
  - 서기/간지/재위연도 보유 기사: **`380,010`** 건 (`99.56%`)
- **모델 라인업 ([configs/](file:///home/nsua/Dev/sillok/configs/))**:
  - `sillok_tiny_debug.yaml`: `4,981,888` 파라미터 (~5.0M)
  - `sillok_120m_t4.yaml`: `100,682,496` 파라미터 (~100.7M, Colab T4용)
  - `sillok_350m_a100.yaml`: `347,132,928` 파라미터 (~347.1M, Colab A100용)
  - `sillok_1b_a100.yaml`: `1,014,061,056` 파라미터 (~1.01B, Colab A100용)

---

## 전체 코드베이스 및 산출물 색인

| 구분 | 경로 | 역할 |
| :--- | :--- | :--- |
| **패키지 메타** | [pyproject.toml](file:///home/nsua/Dev/sillok/pyproject.toml), [requirements.txt](file:///home/nsua/Dev/sillok/requirements.txt) | `sillok-lm` 패키지 및 원격 GPU 의존성 정의 |
| **상수/출처** | [src/sillok/constants.py](file:///home/nsua/Dev/sillok/src/sillok/constants.py) | `DATASET_INFO`, `KING_CODE_MAP`(30개 실록), `SPECIAL_TOKENS` |
| **XML 파서** | [src/sillok/data/xml_parser.py](file:///home/nsua/Dev/sillok/src/sillok/data/xml_parser.py) | UTF-16 서로게이트 복구, 계층 날짜 상속, 저메모리 병렬 파서 |
| **시퀀스 포맷** | [src/sillok/data/formatting.py](file:///home/nsua/Dev/sillok/src/sillok/data/formatting.py) | 정방향(`제목→한문`) 및 역방향(`한문→제목`) 학습 시퀀스 생성기 |
| **데이터셋** | [src/sillok/data/dataset.py](file:///home/nsua/Dev/sillok/src/sillok/data/dataset.py) | `np.memmap` 기반 제로카피 `PackedMemmapDataset` |
| **토크나이저** | [src/sillok/tokenizer/trainer.py](file:///home/nsua/Dev/sillok/src/sillok/tokenizer/trainer.py) | NFC + Byte-Level BPE (`vocab_size=32768`) 학습/로더 |
| **모델** | [src/sillok/model/config.py](file:///home/nsua/Dev/sillok/src/sillok/model/config.py), [transformer.py](file:///home/nsua/Dev/sillok/src/sillok/model/transformer.py) | `SillokLMConfig`, RMSNorm/RoPE/GQA/SwiGLU `SillokForCausalLM` |
| **학습/생성** | [src/sillok/training/trainer.py](file:///home/nsua/Dev/sillok/src/sillok/training/trainer.py), [generate.py](file:///home/nsua/Dev/sillok/src/sillok/training/generate.py) | `bfloat16` AMP, TF32, `torch.compile`, 자동 재개 및 생성기 |
| **원격 연동** | [src/sillok/remote/colab_env.py](file:///home/nsua/Dev/sillok/src/sillok/remote/colab_env.py) | 호스트/Colab 진단, Cloudflare 터널 다운로드, Drive 심볼릭 링크 |
| **CLI 및 실행환경** | [scripts/01_parse_xml.py](file:///home/nsua/Dev/sillok/scripts/01_parse_xml.py) ~ [11_serve_web_studio.py](file:///home/nsua/Dev/sillok/scripts/11_serve_web_studio.py), [evaluate.py](file:///home/nsua/Dev/sillok/scripts/evaluate.py) | 파싱, 토크나이저, Memmap, 학습, 추론, 정량 평가(PPL/Top-k), 경량 모델 추출, 대화형 터미널, 디지털 춘추관 웹 스튜디오(:8020) |
| **원격 노트북/작업** | [notebooks/colab_remote_runner.ipynb](file:///home/nsua/Dev/sillok/notebooks/colab_remote_runner.ipynb), [configs/colab_job.json](file:///home/nsua/Dev/sillok/configs/colab_job.json) | 로컬 IDE ↔ Colab 원격 커널 실행기 및 `/job.json` 작업 명세 |
| **공용 GPU 브리지** | `/home/nsua/colab-gpu-bridge/` (`colab_gpu_server.py`, `colab_oauth.py`, `colab_mcp_proxy.py`, `colab_t4_supervisor.py`) | 범용 GPU 작업 큐, Colab OAuth/MCP 프록시, 3분 주기 무인 감독 및 청크 백업 데몬 |
| **단위 테스트** | [tests/test_pipeline.py](file:///home/nsua/Dev/sillok/tests/test_pipeline.py) | 로컬 무의존 테스트 및 Colab PyTorch 통합 테스트 |
