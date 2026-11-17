"""End-to-end and unit tests for Sillok-LM parsing, formatting, config, and PyTorch model."""

from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

from sillok.constants import DATASET_INFO, KING_CODE_MAP
from sillok.data.formatting import format_article_for_lm, format_generation_prompt
from sillok.data.xml_parser import parse_sillok_xml, sanitize_xml_surrogates
from sillok.model.config import SillokLMConfig
from sillok.remote.colab_env import detect_runtime, format_runtime_report

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "configs"

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class TestSillokStandardPipeline(unittest.TestCase):
    """Tests that run locally using only the Python standard library and PyYAML."""

    def test_sanitize_xml_surrogates_decodes_cjk_extension_pair(self) -> None:
        raw = "<name>李&#55381;&#56892;</name>"
        cleaned = sanitize_xml_surrogates(raw)
        self.assertEqual(cleaned, "<name>李\U0002563C</name>")

    @unittest.skipUnless((DATA_DIR / "2nd_waa_101.xml").exists(), "Sillok XML data not present")
    def test_parse_taejo_xml_and_date_inheritance(self) -> None:
        articles = parse_sillok_xml(DATA_DIR / "2nd_waa_101.xml")
        self.assertEqual(len(articles), 187)

        first = articles[0]
        self.assertEqual(first.article_id, "waa_10107017_001")
        self.assertEqual(first.king_code, "waa")
        self.assertEqual(first.king_name_ko, "태조실록")
        self.assertEqual(first.date_iso, "1392-07-17L0")
        self.assertEqual(first.date_ganji, "丙申")
        self.assertEqual(first.date_reign, "태조 1년 7월 17일")
        self.assertIn("수창궁", first.title_ko)
        self.assertIn("太祖卽位于壽昌宮", first.body_hanja)
        # Ensure collation note 〔璽〕 is omitted by default while keeping surrounding text intact
        self.assertIn("奉國(寶)詣太祖邸", first.body_hanja)

        second = articles[1]
        # Pretty-print newlines between <index> tags must not insert spaces inside Hanja compounds
        self.assertIn("慶侍中復興", second.body_hanja)
        self.assertIn("崔都統瑩", second.body_hanja)

    @unittest.skipUnless((DATA_DIR / "2nd_wqa_106.xml").exists(), "Hyojong XML file not present")
    def test_parse_surrogate_error_files_cleanly(self) -> None:
        for fname in ("2nd_wqa_106.xml", "2nd_wra_104.xml"):
            path = DATA_DIR / fname
            if not path.exists():
                continue
            articles = parse_sillok_xml(path)
            self.assertGreater(len(articles), 0)
            joined = "\n".join(a.body_hanja for a in articles)
            self.assertIn("李\U0002563C", joined)

    @unittest.skipUnless((DATA_DIR / "2nd_wba_102.xml").exists(), "Jeongjong XML file not present")
    def test_postscript_extraction_and_formatting(self) -> None:
        articles = parse_sillok_xml(DATA_DIR / "2nd_wba_102.xml")
        with_ps = [a for a in articles if a.postscripts]
        self.assertGreater(len(with_ps), 0)

        art = with_ps[0]
        self.assertNotIn("史臣", art.body_hanja)
        self.assertTrue(any("史臣" in ps for ps in art.postscripts))

        fwd = format_article_for_lm(art, reverse_title_prob=0.0, rng=random.Random(0))
        self.assertTrue(fwd.startswith("<|bos|><|king|>"))
        self.assertLess(fwd.index("<|title|>"), fwd.index("<|hanja|>"))
        self.assertIn("<|postscript|>", fwd)
        self.assertTrue(fwd.endswith("<|eos|>"))

        rev = format_article_for_lm(art, reverse_title_prob=1.0, rng=random.Random(0))
        self.assertLess(rev.index("<|hanja|>"), rev.index("<|title|>"))
        self.assertTrue(rev.endswith("<|eos|>"))

        prompt = format_generation_prompt(king="태조실록", title_ko="테스트", target_mode="hanja")
        self.assertTrue(prompt.endswith("<|hanja|>"))

    def test_yaml_configs_and_parameter_estimates(self) -> None:
        tiny = SillokLMConfig.from_yaml(CONFIG_DIR / "sillok_tiny_debug.yaml")
        m350 = SillokLMConfig.from_yaml(CONFIG_DIR / "sillok_350m_a100.yaml")
        m1b = SillokLMConfig.from_yaml(CONFIG_DIR / "sillok_1b_a100.yaml")

        self.assertLess(tiny.estimate_num_parameters(), 10_000_000)
        self.assertGreater(m350.estimate_num_parameters(), 300_000_000)
        self.assertLess(m350.estimate_num_parameters(), 400_000_000)
        self.assertGreater(m1b.estimate_num_parameters(), 950_000_000)
        self.assertLess(m1b.estimate_num_parameters(), 1_150_000_000)

    def test_remote_runtime_detection(self) -> None:
        info = detect_runtime()
        report = format_runtime_report(info)
        self.assertIn("Runtime Mode", report)
        self.assertIn("System Memory", report)
        self.assertIn("Host Services", report)
        self.assertGreater(info.cpu_count, 0)

    def test_dataset_provenance_constants(self) -> None:
        self.assertEqual(
            DATASET_INFO["dataset_name"],
            "교육부 국사편찬위원회_조선왕조실록 정보_실록원문_20260821",
        )
        self.assertEqual(DATASET_INFO["total_xml_files"], 673)
        self.assertEqual(DATASET_INFO["total_articles"], 381701)
        self.assertEqual(len(KING_CODE_MAP), 30)

    def test_streaming_memmap_split_and_artifact_path_guard(self) -> None:
        spec_split = importlib.util.spec_from_file_location(
            "prepare_memmap", ROOT_DIR / "scripts" / "03_prepare_memmap.py"
        )
        assert spec_split is not None and spec_split.loader is not None
        mod_split = importlib.util.module_from_spec(spec_split)
        spec_split.loader.exec_module(mod_split)

        self.assertFalse(mod_split.is_validation_article("waa_10107017_001", 0.0, 42))
        self.assertTrue(mod_split.is_validation_article("waa_10107017_001", 1.0, 42))
        a = mod_split.is_validation_article("waa_10107017_001", 0.02, 42)
        b = mod_split.is_validation_article("waa_10107017_001", 0.02, 42)
        self.assertEqual(a, b)

        spec_srv = importlib.util.spec_from_file_location(
            "serve_artifacts", ROOT_DIR / "scripts" / "06_serve_artifacts.py"
        )
        assert spec_srv is not None and spec_srv.loader is not None
        mod_srv = importlib.util.module_from_spec(spec_srv)
        spec_srv.loader.exec_module(mod_srv)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proc_dir = root / "processed"
            proc_dir.mkdir()
            (proc_dir / "sample.json").write_text("{}", encoding="utf-8")
            handler = mod_srv._ArtifactHandler.__new__(mod_srv._ArtifactHandler)
            handler.mounts = {"processed": proc_dir}
            self.assertEqual(
                handler._resolve_mount_path("/processed/sample.json"),
                (proc_dir / "sample.json").resolve(),
            )
            self.assertIsNone(handler._resolve_mount_path("/processed/../secret.txt"))
            self.assertIsNone(handler._resolve_mount_path("/etc/passwd"))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed in local environment")
class TestSillokPyTorchModel(unittest.TestCase):
    """Tests executed when PyTorch is installed (e.g., Colab remote kernel)."""

    def test_model_forward_backward_and_generate(self) -> None:
        import torch
        from sillok.model.transformer import SillokForCausalLM

        cfg = SillokLMConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=192,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=32,
        )
        model = SillokForCausalLM(cfg)
        self.assertEqual(model.num_parameters(), cfg.estimate_num_parameters())

        input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(input_ids, labels=input_ids)
        self.assertEqual(tuple(logits.shape), (2, 16, cfg.vocab_size))
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()

        out = model.generate(input_ids[:, :4], max_new_tokens=6, temperature=0.8)
        self.assertEqual(out.shape[0], 2)
        self.assertLessEqual(out.shape[1], 10)


@unittest.skipUnless(
    HAS_TORCH and HAS_TOKENIZERS and HAS_NUMPY and (DATA_DIR / "2nd_waa_101.xml").exists(),
    "Requires torch, tokenizers, numpy, and sample XML",
)
class TestSillokTokenizerAndDataset(unittest.TestCase):
    """Tests executed when full training stack is installed."""

    def test_tokenizer_and_memmap_roundtrip(self) -> None:
        import json
        import numpy as np
        from sillok.data.dataset import PackedMemmapDataset
        from sillok.tokenizer.trainer import load_sillok_tokenizer, train_sillok_tokenizer

        articles = parse_sillok_xml(DATA_DIR / "2nd_waa_101.xml")[:20]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            jsonl_path = tmp / "sample.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as fout:
                for art in articles:
                    fout.write(json.dumps(art.to_dict(), ensure_ascii=False) + "\n")

            tok_path = train_sillok_tokenizer(jsonl_path, tmp / "tok", vocab_size=512)
            tok = load_sillok_tokenizer(tok_path)

            all_ids: list[int] = []
            for art in articles:
                all_ids.extend(tok.encode(format_article_for_lm(art, reverse_title_prob=0.0)).ids)

            bin_path = tmp / "train.bin"
            np.asarray(all_ids, dtype=np.uint16).tofile(bin_path)
            ds = PackedMemmapDataset(bin_path, max_seq_len=32)
            self.assertGreater(len(ds), 0)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape), (32,))
            self.assertEqual(tuple(y.shape), (32,))


if __name__ == "__main__":
    unittest.main()
