# 제4장. End-to-End 실행 및 운영 매뉴얼

본 장에서는 에지 호스트(`tsarkr`)에서의 원천 XML 전처리부터 Cloudflare 터널 전송, Google Colab Pro A100 원격 커널에서의 토크나이저 학습·패킹·사전학습·체크포인트 복구 및 추론까지의 전체 실행 절차를 단계별로 정리한다.

---

## 4.1 단계 1: 에지 호스트(`tsarkr`)에서 원천 XML 파싱

`tsarkr` 호스트는 `3.7 GiB` RAM 환경이므로, 시스템 Python으로 추가 패키지 설치 없이 즉시 파서를 실행할 수 있다.

```bash
cd /home/nsua/Dev/sillok
PYTHONPATH=src python3 scripts/01_parse_xml.py \
  --data-dir data \
  --output-dir data/processed \
  --workers 2
```

- **주요 옵션**:
  - `--workers 2`: 3.7 GiB RAM 환경에서 스왑 사용을 억제하기 위한 권장 값 (미지정 시 `/proc/meminfo`를 감지하여 자동 `2`로 설정).
  - `--no-annotations`: 본문 내 `원주` 등 주석을 제외하고 순수 원문만 추출할 때 사용.
  - `--include-collation`: 기본 제외되는 `교감주`까지 포함할 때 사용.
  - `--extract-entities`: `<index>` 태그의 인명(`ref` 코드 포함)·지명·연호 개체명 목록을 JSONL의 `entities` 필드에 저장할 때 사용.
- **생성 산출물**:
  - `data/processed/articles.jsonl` (381,701개 기사, 약 `352 MiB`)
  - `data/processed/parse_summary.json` (왕대별 전수 통계 JSON)

---

## 4.2 단계 2: Cloudflare Tunnel을 통한 원격 Colab 데이터 전송

호스트에 이미 상주 중인 터널(`cloudflared tunnel --url http://localhost:8000`)을 통해 전처리 결과를 Colab Pro 인스턴스로 직접 스트리밍할 수 있다.

### 1) `tsarkr` 호스트에서 읽기 전용 서버 기동
```bash
cd /home/nsua/Dev/sillok
PYTHONPATH=src python3 scripts/06_serve_artifacts.py --host 127.0.0.1 --port 8000
```
- `/manifest.json`: 서빙 중인 파일 목록과 바이트 크기 확인
- `/processed/articles.jsonl`, `/processed/parse_summary.json`: `data/processed/` 하위 파일 제공
- `/artifacts/...`: `artifacts/` 하위 파일 제공

### 2) 원격 Colab 커널에서 수신
`cloudflared`가 출력한 임시 HTTPS 주소(예: `https://xxxx-xxxx.trycloudflare.com`)를 사용하여 Colab 셀 또는 원격 터미널에서 다운로드한다:

```python
from sillok.remote import download_from_tunnel

TUNNEL_URL = "https://<your-trycloudflare-subdomain>.trycloudflare.com"
download_from_tunnel(TUNNEL_URL, "processed/articles.jsonl", "data/processed/articles.jsonl")
download_from_tunnel(TUNNEL_URL, "processed/parse_summary.json", "data/processed/parse_summary.json")
```

*(대안: 로컬에서 `train.bin`, `val.bin`, `tokenizer.json`까지 생성했거나 Google Drive에 직접 업로드한 경우 동일한 `/processed/train.bin`, `/processed/val.bin`, `/artifacts/tokenizer/tokenizer.json` 경로로 수신하거나 Google Drive 심볼릭 링크를 사용한다.)*

---

## 4.3 단계 3: Colab Pro A100 환경 초기화 및 패키지 설치

로컬 VS Code / Antigravity에서 [notebooks/colab_remote_runner.ipynb](file:///home/nsua/Dev/sillok/notebooks/colab_remote_runner.ipynb)을 열고 **Colab 원격 커널(A100 GPU)**을 연결한다.

```bash
# Colab 원격 셀 또는 터미널에서 실행
pip install -e .
PYTHONPATH=src python3 -c "from sillok.remote import detect_runtime, format_runtime_report; print(format_runtime_report(detect_runtime()))"
```

### 단위 테스트 전체 실행 (PyTorch / Tokenizer 포함)
Colab 커널에는 `torch`, `numpy`가 설치되어 있으므로 로컬에서 스킵되었던 모델 Forward/Backward 및 Memmap 테스트까지 모두 검증할 수 있다:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

---

## 4.4 단계 4: BPE 토크나이저 학습 및 바이너리 패킹

### 1) 커스텀 Byte-Level BPE 토크나이저 학습
```bash
PYTHONPATH=src python3 scripts/02_train_tokenizer.py \
  --jsonl data/processed/articles.jsonl \
  --output-dir artifacts/tokenizer \
  --vocab-size 32768 \
  --min-frequency 2 \
  --reverse-title-prob 0.2
```
- 출력: `artifacts/tokenizer/tokenizer.json`, `artifacts/tokenizer/tokenizer_meta.json`
- 실행 완료 시 샘플 문장의 문자당 토큰 압축률(`chars/tok`)이 함께 출력된다.

### 2) 스트리밍 `uint16` Memmap 패킹
```bash
PYTHONPATH=src python3 scripts/03_prepare_memmap.py \
  --jsonl data/processed/articles.jsonl \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --output-dir data/processed \
  --val-ratio 0.02 \
  --reverse-title-prob 0.2 \
  --seed 42
```
- 출력: `data/processed/train.bin`, `data/processed/val.bin`, `data/processed/dataset_meta.json`
- `blake2b` 해시 기반 기사 단위 분할을 수행하므로 Train/Val 간 기사 중복이나 경계 유출이 발생하지 않는다.

---

## 4.5 단계 5: From-Scratch 사전학습 (A100 GPU)

### 1) 350M 모델 학습 (빠른 수렴 및 실험 권장)
```bash
PYTHONPATH=src python3 scripts/04_pretrain.py \
  --config configs/sillok_350m_a100.yaml \
  --drive-ckpt-dir /content/drive/MyDrive/sillok_lm/checkpoints/sillok_350m
```

### 2) 1B 모델 학습 (본격 사전학습 스케일)
```bash
PYTHONPATH=src python3 scripts/04_pretrain.py \
  --config configs/sillok_1b_a100.yaml \
  --drive-ckpt-dir /content/drive/MyDrive/sillok_lm/checkpoints/sillok_1b
```

### 학습 안정성 및 세션 끊김 복구 메커니즘
- **자동 혼합 정밀도**: A100에서 `torch.amp.autocast("cuda", dtype=torch.bfloat16)` 및 TF32 행렬 가속이 자동 활성화된다.
- **`torch.compile` 래핑 해제 안전성**: 모델 컴파일 상태에서 저장할 때도 `_orig_mod`를 언래핑한 순수 `state_dict`를 저장하므로, 추론 또는 다른 GPU 환경으로 체크포인트를 이식할 때 키 불일치(`_orig_mod.` 접두어 오류)가 발생하지 않는다.
- **Google Drive 영구 보존**: `--drive-ckpt-dir` 지정 시 `checkpoints/...` 디렉터리가 Google Drive 경로로 심볼릭 링크된다. Colab 런타임이 12~24시간 후 초기화되더라도 동일 명령을 재실행하면 `resume_from: "auto"` 설정에 의해 `latest.pt`(모델 가중치, AdamW 옵티마이저 상태, `global_step`, `best_val_loss`)에서 즉시 이어서 학습한다.

---

## 4.6 단계 6: 조건부 생성 및 추론

### 1) 한글 제목 + 메타데이터 → 고전 한문 원문 생성
```bash
PYTHONPATH=src python3 scripts/05_generate.py \
  --checkpoint checkpoints/sillok_350m/best.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --king "세종실록(世宗實錄)" \
  --date "세종 28년 9월 29일 | 1446-09-29 | 甲午" \
  --subject "어문학(語文學), 문화-출판(出版)" \
  --title-ko "훈민정음을 반포하다" \
  --mode hanja \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.95
```

### 2) 한문 원문 → 한글 기사 제목 요약 (`--mode title`)
```bash
PYTHONPATH=src python3 scripts/05_generate.py \
  --checkpoint checkpoints/sillok_350m/best.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --king "태조실록(太祖實錄)" \
  --body-hanja "○丙申/十七日丙申, 太祖卽位于壽昌宮。" \
  --mode title \
  --max-new-tokens 64 \
  --temperature 0.3
```
