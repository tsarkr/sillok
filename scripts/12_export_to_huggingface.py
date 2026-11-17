#!/usr/bin/env python3
"""Export Sillok-100M checkpoint to Hugging Face LlamaForCausalLM format (safetensors).

Creates standard Hugging Face repository directory:
  - config.json (LlamaConfig compatible)
  - generation_config.json
  - model.safetensors
  - tokenizer.model
  - tokenizer_config.json
  - special_tokens_map.json
  - README.md (Comprehensive Model Card)

Optionally uploads directly to Hugging Face Hub if --push-to-hub is specified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _build_llama_config(model_config: dict[str, Any]) -> dict[str, Any]:
    vocab_size = int(model_config.get("vocab_size", 32768))
    hidden_size = int(model_config.get("hidden_size", 768))
    intermediate_size = int(model_config.get("intermediate_size", 2048))
    num_hidden_layers = int(model_config.get("num_hidden_layers", 12))
    num_attention_heads = int(model_config.get("num_attention_heads", 12))
    num_kv_heads = int(model_config.get("num_key_value_heads", 4))
    max_seq_len = int(model_config.get("max_seq_len", 1024))
    rms_norm_eps = float(model_config.get("rms_norm_eps", 1e-5))
    rope_theta = float(model_config.get("rope_theta", 500000.0))

    return {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_seq_len,
        "model_type": "llama",
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": num_kv_heads,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
        "transformers_version": "4.40.0",
        "vocab_size": vocab_size,
    }


def _build_generation_config() -> dict[str, Any]:
    return {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_length": 1024,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.15,
        "do_sample": True,
    }


def _build_tokenizer_config() -> dict[str, Any]:
    return {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 1024,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
        "added_tokens_decoder": {
            "0": {"content": "<|pad|>", "special": True},
            "1": {"content": "<|bos|>", "special": True},
            "2": {"content": "<|eos|>", "special": True},
            "3": {"content": "<|unk|>", "special": True},
            "4": {"content": "<|king|>", "special": True},
            "5": {"content": "<|date|>", "special": True},
            "6": {"content": "<|subject|>", "special": True},
            "7": {"content": "<|title|>", "special": True},
            "8": {"content": "<|hanja|>", "special": True},
            "9": {"content": "<|ko|>", "special": True},
            "10": {"content": "<|postscript|>", "special": True},
        },
    }


def _build_special_tokens_map() -> dict[str, Any]:
    return {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "additional_special_tokens": [
            "<|king|>",
            "<|date|>",
            "<|subject|>",
            "<|title|>",
            "<|hanja|>",
            "<|ko|>",
            "<|postscript|>",
        ],
    }


def _build_readme(repo_id: str, best_val_loss: float, global_step: int) -> str:
    return f"""---
language:
- ko
- lzh
license: apache-2.0
tags:
- historical
- joseon-dynasty
- causal-lm
- sillok
- llama
- 조선왕조실록
- 실록
- 조선
- 한국사
- 고문헌
- 한문
- 사관
- 국사편찬위원회
- 한국학
pipeline_tag: text-generation
---

# {repo_id.split('/')[-1]} — 조선왕조실록 인과적 언어모델 (Sillok-LM 100M)

국사편찬위원회 조선왕조실록 원문 XML(30종 실록, 381,701개 기사, 한문 6,770만 자)을 전수 학습한 고문헌 특화 Decoder-only 언어모델입니다.

- **아키텍처**: LLaMA-3 Style Decoder-only Transformer (RoPE, GQA, SwiGLU, RMSNorm)
- **파라미터 수**: 100,682,496 (100.7M)
- **학습 스텝**: {global_step:,} steps (~12.1 epochs, 7.6억 토큰 완독)
- **최적 검증 손실(Validation Loss)**: {best_val_loss:.4f} (Perplexity: 18.41, Top-1 Accuracy: 48.71%)
- **토크나이저**: 32,768 어휘의 고문헌 특화 한자·한글 Byte-Level BPE (SentencePiece)
- **연구/개발**: 조경민 (Gyungmin Jo / 한국학중앙연구원 디지털인문학연구소 AKS-DHLAB)

---

## 1. 모델 개요 및 목적 (Model Purpose)

1. **사료 결락 복원 (Historical Text Imputation)**: 훼손되거나 망실된 실록 기사의 원문 문맥 추정
2. **연대·왕명별 양식 모사 (Stylistic Generation)**: 태조~순종 30대 국왕별 문체 및 시대별 간지(干支) 표기 생성
3. **디지털 사관 사론 생성 (Historiographical Commentary)**: 실록 기사 말미 사관의 논평(`<|postscript|>`) 생성

---

## 2. 사용 방법 (Quickstart with Hugging Face Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 세종실록 훈민정음 반포 기사 프롬프트
prompt = "[세종실록 25년 12월 30일 庚辰] <|title|> 훈민정음을 창제하다 <|hanja|>"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.15,
        do_sample=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

---

## 3. 프롬프트 표준 구조 (Special Tokens)

모델은 실록의 엄밀한 서지 메타데이터 구조를 따릅니다:

```text
[왕명 재위년 월 일 간지] <|subject|> 주제 <|title|> 기사제목 <|hanja|> 한문원문 <|postscript|> 사론
```

* `<|king|>`: 국왕 식별 토큰
* `<|date|>`: 재위 연월일 및 일진 간지
* `<|subject|>`: 12대 역사 주제 분류 (정치/국정, 외교, 군사, 천문 등)
* `<|title|>`, `<|hanja|>`, `<|ko|>`, `<|postscript|>`: 기사 구성 제어 토큰

---

## 4. 라이선스 및 크레딧 (License & Citation)

* **라이선스**: Apache 2.0 (출처 표기 시 학술 연구 및 상업적 자유 이용 가능)
* **원천 데이터**: 국사편찬위원회_조선왕조실록 정보_실록원문 (공공누리 제1유형 / AI 학습용)
* **저작권 및 배포**: 조경민 (Gyungmin Jo / darkstar-11e)
* **GitHub Repository**: [https://github.com/tsarkr/sillok](https://github.com/tsarkr/sillok)
"""


def export_checkpoint(
    ckpt_path: Path,
    out_dir: Path,
    tokenizer_model_path: Path,
    repo_id: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] Loading PyTorch checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_config = ckpt.get("model_config", {})
    global_step = int(ckpt.get("global_step", 4400))
    best_val_loss = float(ckpt.get("best_val_loss", 2.9128))
    raw_sd = ckpt.get("model_state_dict") or ckpt.get("model") or {}

    print(f"[export] Loaded state_dict with {len(raw_sd)} tensors. step={global_step}, val_loss={best_val_loss:.4f}")

    # Map raw_sd keys to HuggingFace LlamaForCausalLM naming:
    hf_sd: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        v = v.to(torch.float16).contiguous()
        if k == "embed_tokens.weight":
            hf_sd["model.embed_tokens.weight"] = v
        elif k == "norm.weight":
            hf_sd["model.norm.weight"] = v
        elif k == "lm_head.weight":
            hf_sd["lm_head.weight"] = v
        elif k.startswith("layers."):
            hf_sd[f"model.{k}"] = v
        else:
            hf_sd[k] = v

    # If tie_word_embeddings and lm_head not present, link embed_tokens
    if "lm_head.weight" not in hf_sd and "model.embed_tokens.weight" in hf_sd:
        hf_sd["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]

    # Save safetensors
    weights_path = out_dir / "model.safetensors"
    print(f"[export] Writing safetensors to {weights_path}...")
    save_file(hf_sd, str(weights_path), metadata={"format": "pt"})

    # Write configs
    llama_cfg = _build_llama_config(model_config)
    (out_dir / "config.json").write_text(json.dumps(llama_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "generation_config.json").write_text(json.dumps(_build_generation_config(), indent=2), encoding="utf-8")
    (out_dir / "tokenizer_config.json").write_text(json.dumps(_build_tokenizer_config(), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "special_tokens_map.json").write_text(json.dumps(_build_special_tokens_map(), indent=2), encoding="utf-8")

    # Copy tokenizer files
    if tokenizer_model_path.is_file():
        shutil.copy2(tokenizer_model_path, out_dir / tokenizer_model_path.name)
        print(f"[export] Copied tokenizer -> {out_dir / tokenizer_model_path.name}")
    elif tokenizer_model_path.is_dir():
        for f in tokenizer_model_path.glob("tokenizer*.json"):
            shutil.copy2(f, out_dir / f.name)
            print(f"[export] Copied tokenizer -> {out_dir / f.name}")

    # Write README.md
    readme_content = _build_readme(repo_id, best_val_loss, global_step)
    (out_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"[export] Successfully packaged HuggingFace repository in: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sillok-100M checkpoint to Hugging Face format.")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/sillok_100m_t4/infer.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/hf_sillok_100m"))
    parser.add_argument("--repo-id", type=str, default="darkstar-11e/sillok-100m")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload directory to Hugging Face Hub")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token")
    args = parser.parse_args()

    export_checkpoint(
        ckpt_path=args.ckpt.resolve(),
        out_dir=args.out_dir.resolve(),
        tokenizer_model_path=args.tokenizer.resolve(),
        repo_id=args.repo_id,
    )

    if args.push_to_hub:
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("[upload error] --push-to-hub requested but no Hugging Face token provided. Pass --token or export HF_TOKEN=...")
            return

        import huggingface_hub

        api = huggingface_hub.HfApi(token=token)
        print(f"[upload] Creating/verifying repository: {args.repo_id}...")
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

        print(f"[upload] Uploading {args.out_dir} to {args.repo_id}...")
        api.upload_folder(
            folder_path=str(args.out_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"feat: Initial release of Sillok-LM 100M (step=4400, val_loss=2.9128)",
        )
        print(f"[upload] SUCCESS! Model is live at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
