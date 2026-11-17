# Sillok-LM: 조선왕조실록 From Scratch 생성형 언어모델

**원천 데이터셋**: `교육부 국사편찬위원회_조선왕조실록 정보_실록원문_20260821` (`data/2nd_*.xml`, `673`개 파일 · `30`개 실록 · `381,701`개 기사 · 한문 `67,707,578`자 · 공공누리 제1유형/AI유형)

본 프로젝트는 위 데이터를 바탕으로 **Llama-3 스타일 Decoder-only Transformer (`SillokForCausalLM`)**를 PyTorch 기반 From Scratch로 학습하며, **로컬/에지 SSH 서버(`tsarkr`) + Cloudflare Tunnel (`localhost:8000`) + Google Colab Pro (A100 GPU)** 3계층 워크플로우로 설계·구축되었습니다.

---

## 1. 교육용 교재 및 기술 문서 (`docs/`)

인문학 전공 학생도 원리부터 실습까지 따라갈 수 있도록 수식을 풀어서 정리한 교재와 장별 안내 문서입니다:

- **[단행본 통합 교재 (전체 합본)](file:///home/nsua/Dev/sillok/docs/BOOK_SILLOK_LM_COMPLETE.md)** — 데이터 정리, 토크나이저, 모델 원리, 서버·Colab 운영, 30개 실록 통계와 문제 해결 사례를 담은 통합 원고
- **[서문 및 총괄 색인](file:///home/nsua/Dev/sillok/docs/00_BOOK_INDEX.md)** — 전체 목차, 실측 수치 요약, 전체 파일 인덱스
- **[제1장. 원천 데이터 규격 및 XML 파싱 명세](file:///home/nsua/Dev/sillok/docs/01_DATA_AND_XML_SPEC.md)** — `history.dtd ver 1.4`, 30개 실록 통계표, 5개 XML 서로게이트 복구(`𥘼`), 날짜 상속, 사론 분리
- **[제2장. 시스템 인프라 및 원격 연결 구조](file:///home/nsua/Dev/sillok/docs/02_SYSTEM_AND_REMOTE_ARCHITECTURE.md)** — 호스트 `tsarkr`(`Ubuntu 24.04.5`, `Celeron J3160`, `3.74 GiB RAM`), `sshd`/`.vscode-server`/`cloudflared` 구성 및 저메모리 처리 방식
- **[제3장. 토크나이저와 언어모델 구조 이해하기](file:///home/nsua/Dev/sillok/docs/03_TOKENIZER_AND_MODEL_ARCHITECTURE.md)** — 11종 제어 토큰, 양방향 학습 형식, Byte-Level BPE, `SillokForCausalLM` 동작 원리 및 `Tiny / 350M / 1B` 비교
- **[제4장. 단계별 실행 및 운영 매뉴얼](file:///home/nsua/Dev/sillok/docs/04_TRAINING_AND_OPERATIONS_MANUAL.md)** — XML 파싱부터 터널 전송, Colab A100 사전학습 및 문장 생성 실행 방법

---

## 2. 프로젝트 구조

```text
sillok/
├── configs/
│   ├── sillok_tiny_debug.yaml      # 스모크 테스트용 (4,981,888 params)
│   ├── sillok_350m_a100.yaml       # Colab A100 권장 (347,132,928 params)
│   └── sillok_1b_a100.yaml         # Colab A100 본격 스케일 (1,014,061,056 params)
├── data/
│   ├── 2nd_*.xml                   # 교육부 국사편찬위원회 실록 원문 XML (673개)
│   ├── history.dtd                 # 한국사DB 통합 DTD ver 1.4
│   └── processed/                  # 01_parse_xml.py 전수 파싱 산출물 (articles.jsonl, parse_summary.json)
├── docs/                           # 도서 집필용 상세 기술 백서 (00~04)
├── notebooks/
│   └── colab_remote_runner.ipynb   # 로컬 IDE ↔ Colab Pro A100 원격 커널 실행기
├── scripts/
│   ├── 01_parse_xml.py             # 673개 실록 XML 저메모리 병렬 파싱 -> JSONL
│   ├── 02_train_tokenizer.py       # 한문·한글 공용 Byte-Level BPE 토크나이저 학습
│   ├── 03_prepare_memmap.py        # blake2b 스트리밍 분할 + uint16 train.bin / val.bin 패킹
│   ├── 04_pretrain.py              # A100 bf16 / torch.compile / Drive 자동 재개 학습
│   ├── 05_generate.py              # 조건부 실록 생성 (제목->한문 / 한문->제목)
│   └── 06_serve_artifacts.py       # cloudflared tunnel (localhost:8000) 연동 읽기 전용 HTTP 서버
├── src/sillok/
│   ├── constants.py                # DATASET_INFO, KING_CODE_MAP(30개 실록), SPECIAL_TOKENS(11종)
│   ├── data/                       # xml_parser.py, formatting.py, dataset.py
│   ├── tokenizer/                  # trainer.py (NFC + ByteLevel BPE)
│   ├── model/                      # config.py, transformer.py (RMSNorm, RoPE, GQA, SwiGLU)
│   ├── training/                   # trainer.py, generate.py
│   └── remote/                     # colab_env.py (호스트/Colab 진단, 터널 다운로드, Drive 링크)
└── tests/test_pipeline.py          # XML 파싱, 포맷팅, 설정, 보안 및 PyTorch 모델 테스트
```

---

## 3. 빠른 실행 요약

### Step 1. 에지 호스트(`tsarkr`)에서 전수 XML 파싱 (완료 가능: 표준 라이브러리만 사용)
```bash
PYTHONPATH=src python3 scripts/01_parse_xml.py --data-dir data --output-dir data/processed --workers 2
```

### Step 2. Cloudflare Tunnel(`localhost:8000`)로 Colab에 전처리 아티팩트 서빙
```bash
PYTHONPATH=src python3 scripts/06_serve_artifacts.py --host 127.0.0.1 --port 8000
```

### Step 3. Colab Pro A100 원격 커널에서 학습 파이프라인 실행
[notebooks/colab_remote_runner.ipynb](file:///home/nsua/Dev/sillok/notebooks/colab_remote_runner.ipynb)을 열어 Colab 원격 커널에 연결한 뒤 `pip install -e .` 설치 후 토크나이저 학습(`02_train_tokenizer.py`), 스트리밍 Memmap 패킹(`03_prepare_memmap.py`), 사전학습(`04_pretrain.py`)을 실행합니다. 세부 절차는 [제4장 운영 매뉴얼](file:///home/nsua/Dev/sillok/docs/04_TRAINING_AND_OPERATIONS_MANUAL.md)을 참조하세요.
