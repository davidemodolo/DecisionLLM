"""Equivalence tests on tiny random models built from the real configs (CPU, fp32): the optimised forward
paths must give the same letter probabilities as the plain one."""
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import core
from core import Engine, ModelSpec, Question

TINY = {
    "qwen3.5": ("Qwen/Qwen3.5-0.8B", dict(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, mtp_num_hidden_layers=0,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])),
    "granite": ("ibm-granite/granite-4.2-3b", dict(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        logits_scaling=4.0)),
    "lfm2": ("LiquidAI/LFM2.5-1.2B-Instruct", dict(
        hidden_size=64, intermediate_size=128, block_dim=64, conv_dim=64, num_hidden_layers=4, num_attention_heads=4,
        num_heads=4, num_key_value_heads=2, block_auto_adjust_ff_dim=False,
        layer_types=["conv", "conv", "full_attention", "conv"])),
}


def tiny_engine(name: str) -> Engine:
    repo, overrides = TINY[name]
    config = AutoConfig.from_pretrained(repo)
    config = getattr(config, "text_config", config)
    for k, v in overrides.items():
        setattr(config, k, v)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    engine = Engine()
    engine.device = "cpu"
    engine.spec = ModelSpec(name, repo, name, 0.0, set_attention=name == "granite")
    engine.model = model
    engine.tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="left")
    if engine.tokenizer.pad_token is None:
        engine.tokenizer.pad_token = engine.tokenizer.eos_token
    return engine


QUESTIONS = [
    Question("Qual è la capitale della Francia?", ["Roma", "Parigi", "Berlino", "Madrid"]),
    Question("Quanto fa 7 per 8, considerando anche i casi limite della moltiplicazione?", ["54", "56", "64"]),
    Question("Il Sole è una stella?", type="yesno"),
    Question("Quanto è caldo il Sahara?", ["Freddo", "Mite", "Caldo", "Torrido"], type="score"),
]
STATE = "Recensione: il risotto era ottimo, ma abbiamo aspettato un'ora e il conto era alto. " * 3
STATE_QUESTIONS = [
    Question("Aspetto più criticato?", ["Cibo", "Attesa", "Prezzo"], none_option=False, state=STATE),
    Question("Il cliente chiede un rimborso?", type="yesno", state=STATE),
    Question("Soddisfazione?", ["Bassa", "Media", "Alta"], type="score", state=STATE),
]


def probs(engine, questions, **kw):
    variants, timing = engine.run_variants(questions, max_questions=None, **kw)
    return torch.tensor([p for vs in variants for v in vs for p in v["probs"]]), timing


@pytest.fixture(scope="module", params=list(TINY))
def engine(request):
    return tiny_engine(request.param)


@pytest.mark.parametrize("questions", [QUESTIONS, STATE_QUESTIONS], ids=["per-question", "shared-state"])
def test_prefix_cache_matches_plain_forward(engine, questions, monkeypatch):
    monkeypatch.setattr(core, "MIN_PREFIX_SAVING", 0.0)
    plain, _ = probs(engine, questions, scramble=5)
    cached, timing = probs(engine, questions, scramble=5, prefix_cache=True)
    assert timing["prefix_mode"] in ("per-question", "shared")
    assert timing["tokens_saved"] > 0
    torch.testing.assert_close(cached, plain, atol=2e-4, rtol=1e-3)


def test_short_common_prefix_falls_back_to_plain_forward(engine):
    _, timing = probs(engine, QUESTIONS, scramble=5, prefix_cache=True)
    assert timing["prefix_mode"] == "off" and timing["tokens_saved"] == 0


def test_shared_state_uses_one_prefix(engine):
    _, timing = probs(engine, STATE_QUESTIONS, scramble=5, prefix_cache=True)
    assert timing["prefix_mode"] == "shared"


def test_token_budget_and_row_limit_do_not_change_results(engine, monkeypatch):
    monkeypatch.setattr(core, "MIN_PREFIX_SAVING", 0.0)
    reference, _ = probs(engine, QUESTIONS, scramble=3, batch_size=64, batch_tokens=100_000)
    small, timing = probs(engine, QUESTIONS, scramble=3, batch_size=2, batch_tokens=256)
    assert timing["max_batch_rows"] <= 2
    torch.testing.assert_close(small, reference, atol=2e-4, rtol=1e-3)
    cached, timing = probs(engine, QUESTIONS, scramble=3, prefix_cache=True, batch_size=2, batch_tokens=256)
    assert timing["max_batch_rows"] <= 2
    torch.testing.assert_close(cached, reference, atol=2e-4, rtol=1e-3)


def test_set_attention_rejected_for_recurrent_models(engine):
    if engine.spec.set_attention:
        pytest.skip("modello solo attention")
    with pytest.raises(ValueError):
        engine.run_variants(QUESTIONS, set_attention=True)


def test_set_attention_is_invariant_to_segment_order():
    engine = tiny_engine("granite")
    q = Question("Capitale?", ["Roma", "Parigi", "Berlino"], none_option=False)
    prefix, segments, suffix = engine._prompt_parts(q, q.all_options, "it-strict")
    enc = lambda segs: (engine._encode(prefix), [engine._encode(s) for s in segs], engine._encode(suffix))  # noqa: E731
    stats = {"tokens": 0, "padded_tokens": 0, "max_rows": 0, "forward_s": 0.0, "batches": 0, "oom_retries": 0}
    letters, tops = torch.empty(2, 8), torch.empty(2, dtype=torch.long)
    reordered = [segments[2], segments[0], segments[1]]
    engine._forward_set([enc(segments), enc(reordered)], 8, 4096, stats, letters, tops)
    torch.testing.assert_close(letters[0], letters[1], atol=1e-5, rtol=1e-5)


def test_forward_keeps_only_letter_columns():
    engine = tiny_engine("granite")
    stats = {"tokens": 0, "padded_tokens": 0, "max_rows": 0, "forward_s": 0.0, "batches": 0, "oom_retries": 0}
    letters, tops = torch.empty(1, 8), torch.empty(1, dtype=torch.long)
    engine._forward_plain([engine._encode("ciao")], 8, 4096, stats, letters, tops)
    assert letters.shape == (1, 8) and torch.isfinite(letters).all()


def test_prefix_cache_splits_rows_of_one_shared_prefix_on_oom(monkeypatch):
    engine = tiny_engine("granite")
    reference, _ = probs(engine, STATE_QUESTIONS, scramble=5)
    select = core._select_cache

    def tight_memory(cache, idx):
        if idx.numel() > 2:
            raise torch.OutOfMemoryError("simulated")
        select(cache, idx)

    monkeypatch.setattr(core, "_select_cache", tight_memory)
    cached, timing = probs(engine, STATE_QUESTIONS, scramble=5, prefix_cache=True)
    assert timing["prefix_mode"] == "shared" and timing["oom_retries"] > 0
    torch.testing.assert_close(cached, reference, atol=2e-4, rtol=1e-3)
