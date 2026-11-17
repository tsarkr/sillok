# 제3장. 토크나이저와 언어모델 구조 이해하기

이 장에서는 조선왕조실록의 고전 한문과 한글 제목을 하나의 흐름으로 학습하기 위한 특수 토큰 약속, Byte-Level BPE 토크나이저, 그리고 `SillokForCausalLM` 모델의 동작 원리를 설명합니다.

---

## 3.1 제어 토큰과 양방향 학습 형식

### 1) 특수 토큰 목록 (`SPECIAL_TOKENS`)
[src/sillok/constants.py](file:///home/nsua/Dev/sillok/src/sillok/constants.py)에 정의된 11개의 특수 토큰은 `0`번부터 `10`번까지 고정되어 있으며, 문장의 각 부분이 어떤 역할인지 모델에게 알려주는 이정표 역할을 합니다.

| Token ID | 토큰 문자열 | 역할 |
| :---: | :--- | :--- |
| `0` | `<\|pad\|>` | 길이를 맞출 때 쓰는 빈칸 토큰 |
| `1` | `<\|bos\|>` | 기사 시작 표시 |
| `2` | `<\|eos\|>` | 기사 끝 표시 |
| `3` | `<\|unk\|>` | 미등록 토큰용 예비 번호 |
| `4` | `<\|king\|>` | 실록명 (예: `태조실록(太祖實錄)`) |
| `5` | `<\|date\|>` | 날짜 (`재위연도 \| 서기 \| 간지`) |
| `6` | `<\|subject\|>` | 국사편찬위원회 주제 분류 |
| `7` | `<\|title\|>` | 한글 기사 제목 |
| `8` | `<\|hanja\|>` | 한문 원문 본문 |
| `9` | `<\|ko\|>` | 추후 국역본 데이터를 넣을 때 사용할 예비 자리 |
| `10` | `<\|postscript\|>` | 사론(`postScript`, 史臣曰 등) |

### 2) 양방향 조건부 학습 (`format_article_for_lm`)
[src/sillok/data/formatting.py](file:///home/nsua/Dev/sillok/src/sillok/data/formatting.py)는 `reverse_title_prob`(기본 `0.2`) 확률로 두 가지 순서를 섞어 학습 문장을 만듭니다.

- **기본 순서 (80%, 제목 → 한문 원문 생성)**:
  `<|bos|><|king|>태조실록(太祖實錄)<|date|>...<|subject|>...<|title|>한글제목<|hanja|>한문원문<|eos|>`
- **바뀐 순서 (20%, 한문 원문 → 한글 제목 요약)**:
  `<|bos|><|king|>...<|date|>...<|subject|>...<|hanja|>한문원문[<|postscript|>사론]<|title|>한글제목<|eos|>`

---

## 3.2 Byte-Level BPE 토크나이저

[src/sillok/tokenizer/trainer.py](file:///home/nsua/Dev/sillok/src/sillok/tokenizer/trainer.py)는 다음 기준으로 단어장을 만듭니다.

1. **`NFC` 정규화**: `NFKC`를 쓰면 호환 한자나 옛 괄호 기호 형태가 바뀔 수 있으므로, 한글 자모만 완성형으로 맞추고 한자는 그대로 두는 `NFC`를 사용합니다.
2. **`ByteLevel` BPE**: 기본 256개 바이트에서 시작해 자주 함께 나오는 글자를 묶어 주므로, `𥘼` 같은 드문 한자도 누락(`unk`) 없이 처리할 수 있고 `吏曹判書`, `領議政`, `上曰` 같은 관직·표현은 한 단어로 묶입니다.
3. **단어장 크기 `32,768`**: `65,535` 이하이므로 모든 번호를 2바이트 정수(`uint16`)로 작게 저장할 수 있습니다.

---

## 3.3 `SillokForCausalLM`의 주요 구성 요소

[src/sillok/model/transformer.py](file:///home/nsua/Dev/sillok/src/sillok/model/transformer.py)는 앞단어를 보고 다음 단어를 맞히는 Decoder-only Transformer를 순수 PyTorch로 구현한 모델입니다.

1. **RMSNorm (숫자 크기 고르게 맞추기)**: 층을 거치며 내부 숫자가 지나치게 커지거나 작아지지 않도록 평균 크기를 일정하게 맞춰 줍니다. `bfloat16` 사용 시 반올림 오차를 줄이기 위해 잠깐 `float32`로 계산한 뒤 되돌립니다.
2. **RoPE (위치 정보 시계 바늘, `rope_theta = 500,000`)**: 단어 위치에 따라 벡터를 조금씩 회전시켜, 모델이 두 단어 사이의 거리 차이를 자연스럽게 파악하도록 합니다.
3. **GQA (Grouped Query Attention)**: 질문(Query) 헤드 4개당 참고서(Key/Value) 헤드 1개를 함께 보도록 묶어 GPU 메모리 사용량을 1/4로 줄입니다. (구형 CPU에서는 안전한 `SDPBackend.MATH` 경로로, A100 GPU에서는 FlashAttention으로 동작합니다.)
4. **SwiGLU & 잔차 스케일링**: 입력을 두 갈래로 보내 필요한 정보만 걸러 다음 층으로 넘기며, 층이 깊어질수록 각 층이 더하는 값의 크기를 `0.02 / sqrt(2 * L)` 비율로 줄여 안정적으로 출발하도록 합니다.

---

## 3.4 모델 크기별 구성 비교

| 구성 요소 | Tiny Debug (`sillok_tiny_debug.yaml`) | 350M A100 (`sillok_350m_a100.yaml`) | 1B A100 (`sillok_1b_a100.yaml`) |
| :--- | :---: | :---: | :---: |
| `vocab_size` (단어장) | `32,768` | `32,768` | `32,768` |
| `hidden_size` (표현 차원) | `128` | `1,024` | `2,048` |
| `intermediate_size` (중간 차원) | `384` | `3,072` | `5,632` |
| `num_hidden_layers` (층 수) | `4` | `26` | `21` |
| `num_attention_heads` / `kv_heads` | `4` / `2` | `16` / `4` | `16` / `4` |
| `max_seq_len` (최대 길이) | `256` | `2,048` | `2,048` |
| **단어장 임베딩 가중치** | `4,194,304` | `33,554,432` | `67,108,864` |
| **1개 층 가중치 합계** | `196,864` | `12,060,672` | `45,092,864` |
| **전체 층 가중치 합계** | `787,456` | `313,577,472` | `946,950,144` |
| **총 파라미터 수 (최종 Norm 포함)** | **`4,981,888`** (약 500만) | **`347,132,928`** (약 3.5억) | **`1,014,061,056`** (약 10억) |
| **메모리 절약 모드 (Checkpointing)** | `false` | `false` | `true` |
