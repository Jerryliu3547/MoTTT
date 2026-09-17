"""Tests for Premise/Query decomposition and Distractor Needle Synthesis."""

from mottt.data.distractor_generator import (
    PremiseQueryDecomposer,
    DistractorNeedleSynthesizer,
)


def test_premise_query_decomposition():
    gsm_problem = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
    )
    decomposed = PremiseQueryDecomposer.decompose(gsm_problem)

    assert "Natalia sold clips to 48" in decomposed.premise
    assert "How many clips did Natalia sell altogether in April and May?" in decomposed.query


def test_distractor_needle_synthesizer_and_chunking():
    synthesizer = DistractorNeedleSynthesizer(chunk_size=32)
    premise = "The secret code is 42."

    # Build synthetic context without needing any external dataset
    context_text, depth = synthesizer.build_synthetic_context(
        premise=premise,
        target_token_count=128,
        depth_ratio=0.5,
    )

    assert premise in context_text
    assert depth == 0.5

    # Mock token ids
    dummy_token_ids = list(range(100))
    chunked = synthesizer.chunk_token_ids(dummy_token_ids, premise_start_idx=48)

    # 100 tokens with chunk size 32 -> ceil(100 / 32) = 4 chunks
    assert len(chunked.chunks) == 4
    assert len(chunked.chunks[0]) == 32
    assert len(chunked.chunks[1]) == 32
    assert len(chunked.chunks[2]) == 32
    assert len(chunked.chunks[3]) == 4
    assert chunked.premise_chunk_index == 1  # 48 // 32 = 1
    assert chunked.total_tokens == 100
