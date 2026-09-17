"""Distractor-Injected Long-Context Needle Synthesis, Premise Decomposition, and Multi-Model Record Creation."""

from dataclasses import dataclass, asdict
import random
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DecomposedProblem:
    premise: str
    query: str
    solution: Optional[str] = None
    gold_answer: Optional[str] = None


@dataclass
class ChunkedContext:
    chunks: List[List[int]]
    premise_chunk_index: int
    depth_ratio: float
    total_tokens: int


@dataclass
class LongContextGSM8KRecord:
    """Standardized record format for testing both MoTTT and external baseline models."""
    id: str
    original_question: str
    premise: str
    query: str
    needle_depth_ratio: float
    target_context_tokens: int
    distractor_context: str
    full_prompt: str
    solution: str
    gold_answer: str
    premise_char_start: int
    premise_char_end: int
    mottt_chunks: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_baseline_prompt(context: str, query: str) -> str:
    """Format prompt for standard baseline models (e.g. Qwen2.5, LLaMA, RAG models)."""
    return (
        "Background Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Please solve the problem step by step and end your response with '#### [final numerical answer]'."
    )


class PremiseQueryDecomposer:
    """Separates interwoven problem statements into Premise P and Query Q.

    Example:
    Original: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    Premise P: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."
    Query Q: "How many clips did Natalia sell altogether in April and May?"
    """

    QUESTION_PATTERN = re.compile(
        r"(?:(?:How many|How much|What is|What was|Find the|Calculate|Compute|Determine|If |What are|In total|How long)\b[^\n?.!]*\?)",
        re.IGNORECASE,
    )

    @classmethod
    def decompose(
        cls,
        problem_text: str,
        solution_text: Optional[str] = None,
        gold_answer: Optional[str] = None,
    ) -> DecomposedProblem:
        text = problem_text.strip()
        matches = list(cls.QUESTION_PATTERN.finditer(text))

        if matches:
            last_match = matches[-1]
            split_idx = last_match.start()
            premise = text[:split_idx].strip()
            query = text[split_idx:].strip()

            # Ensure premise is not empty
            if not premise:
                # If question starts immediately, try sentence split
                sentences = re.split(r"(?<=[.?!])\s+", text)
                if len(sentences) > 1:
                    premise = " ".join(sentences[:-1]).strip()
                    query = sentences[-1].strip()
                else:
                    # Single sentence question, split at comma before query word if present
                    comma_match = re.search(r",\s*(how|what|find|calculate)\b", text, re.IGNORECASE)
                    if comma_match:
                        c_idx = comma_match.start()
                        premise = text[:c_idx].strip() + "."
                        query = text[c_idx + 1:].strip()
                        query = query[0].upper() + query[1:]
                    else:
                        premise = text
                        query = text
        else:
            # Fallback: split by last sentence boundary
            sentences = re.split(r"(?<=[.?!])\s+", text)
            if len(sentences) > 1:
                premise = " ".join(sentences[:-1]).strip()
                query = sentences[-1].strip()
            else:
                premise = text
                query = text

        return DecomposedProblem(
            premise=premise,
            query=query,
            solution=solution_text,
            gold_answer=gold_answer,
        )


def is_non_numerical(text: str) -> bool:
    """Check if text is completely free of digits and numerical symbols.

    Adheres to ideas.md Section 4.2:
    'pre-filtered of mathematical and numerical tokens that could induce spurious reasoning paths'
    """
    # Exclude any digits
    if re.search(r"\d", text):
        return False
    # Exclude arithmetic and equation symbols
    if re.search(r"[=+*/%$#<>]", text):
        return False
    # Exclude spelled out digit words that act as numbers
    spelled_numbers = r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|twenty|hundred|thousand)\b"
    if re.search(spelled_numbers, text, re.IGNORECASE):
        return False
    return True


class DistractorNeedleSynthesizer:
    """Synthesizes long context with distractor sentences and embeds Premise P at depth delta in [0.1, 0.9].

    Adheres strictly to ideas.md Section 4.2 by pre-filtering distractor reservoirs of any
    mathematical or numerical tokens.
    """

    DEFAULT_SYNTHETIC_DISTRACTORS = [
        "The ancient lighthouse stood steadily against the roaring coastal winds.",
        "A quiet forest path wound gently through the towering pine trees.",
        "The architectural committee reviewed the blueprints for the historic restoration.",
        "Deep beneath the ocean surface, hydrothermal vents release warm minerals into the current.",
        "The celestial observatory recorded unusual atmospheric fluctuations across the northern sky.",
        "Vibrant wildflowers blossomed across the vast valley following the spring rains.",
        "The antique clock in the library ticked rhythmically through the afternoon silence.",
        "Scientists explored the remote cavern to map subterranean geological formations.",
        "The merchant vessel sailed smoothly across the calm waters of the outer archipelago.",
        "Sunlight filtered softly through the stained glass windows of the grand cathedral.",
        "Flocks of migratory swallows soared gracefully over the tranquil emerald lake.",
        "The artisan carefully carved intricate floral motifs into the polished mahogany cabinet.",
        "Dense morning mist rolled slowly down the slopes of the snowcapped mountain ridge.",
        "The botanical gardens showcased rare tropical orchids with iridescent violet petals.",
        "Scholars gathered in the courtyard to discuss classical literature and philosophy.",
        "The wandering caravan rested beside a refreshing oasis sheltered by date palms.",
        "Gentle ocean waves lapped against the shores of the secluded coral cove.",
        "The old windmill turned lazily under the warm breeze of the harvest season.",
        "A winding cobblestone street led travelers toward the bustling harbor quarter.",
        "Silver moonlight illuminated the quiet courtyard fountain throughout the tranquil night.",
    ]

    def __init__(
        self,
        chunk_size: int = 256,
        distractor_sentences: Optional[List[str]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        raw_distractors = distractor_sentences or self.DEFAULT_SYNTHETIC_DISTRACTORS
        # Enforce non-numerical constraint
        self.distractor_sentences = [s for s in raw_distractors if is_non_numerical(s)]
        if not self.distractor_sentences:
            raise ValueError("All provided distractor sentences contained numerical/mathematical tokens!")

    def build_synthetic_context(
        self,
        premise: str,
        target_token_count: int = 4096,
        depth_ratio: float = 0.5,
        return_char_spans: bool = False,
    ) -> Any:
        """Construct long-context text with Premise P inserted at specified depth ratio delta.

        Args:
            premise: Context premise text to embed
            target_token_count: Target token length
            depth_ratio: Insertion depth ratio delta in [0.0, 1.0]
            return_char_spans: If True, returns (context, depth, start, end); else (context, depth)

        Returns:
            (context, depth_ratio) by default, or (context, depth_ratio, char_start, char_end)
        """
        if not (0.0 <= depth_ratio <= 1.0):
            raise ValueError(f"depth_ratio must be in [0.0, 1.0], got {depth_ratio}")

        # Estimate words needed (approx 1.3 tokens per word)
        words_needed = max(10, int(target_token_count / 1.3))
        distractor_pool = list(self.distractor_sentences)

        total_words = 0
        all_sentences = []
        while total_words < words_needed:
            s = random.choice(distractor_pool)
            all_sentences.append(s)
            total_words += len(s.split())

        insert_idx = int(len(all_sentences) * depth_ratio)
        insert_idx = max(0, min(insert_idx, len(all_sentences)))

        prefix_text = " ".join(all_sentences[:insert_idx])
        suffix_text = " ".join(all_sentences[insert_idx:])

        if prefix_text:
            char_start = len(prefix_text) + 1  # plus space
            full_context = f"{prefix_text} {premise} {suffix_text}".strip()
        else:
            char_start = 0
            full_context = f"{premise} {suffix_text}".strip()

        char_end = char_start + len(premise)

        if return_char_spans:
            return full_context, depth_ratio, char_start, char_end
        return full_context, depth_ratio

    def create_record(
        self,
        example_id: str,
        original_question: str,
        premise: str,
        query: str,
        solution: str,
        gold_answer: str,
        depth_ratio: float = 0.5,
        target_token_count: int = 4096,
        tokenizer: Optional[Any] = None,
    ) -> LongContextGSM8KRecord:
        """Create a complete LongContextGSM8KRecord for multi-model evaluation."""
        context, applied_depth, c_start, c_end = self.build_synthetic_context(
            premise=premise,
            target_token_count=target_token_count,
            depth_ratio=depth_ratio,
            return_char_spans=True,
        )

        full_prompt = format_baseline_prompt(context=context, query=query)

        # Optional chunk metadata if tokenizer provided
        chunk_metadata = None
        if tokenizer is not None:
            token_ids = tokenizer.encode(context)
            chunked = self.chunk_token_ids(token_ids)
            chunk_metadata = {
                "num_chunks": len(chunked.chunks),
                "chunk_size": self.chunk_size,
                "premise_chunk_index": chunked.premise_chunk_index,
                "total_tokens": chunked.total_tokens,
            }

        rec_id = f"{example_id}_depth_{applied_depth:.2f}"

        return LongContextGSM8KRecord(
            id=rec_id,
            original_question=original_question,
            premise=premise,
            query=query,
            needle_depth_ratio=applied_depth,
            target_context_tokens=target_token_count,
            distractor_context=context,
            full_prompt=full_prompt,
            solution=solution,
            gold_answer=gold_answer,
            premise_char_start=c_start,
            premise_char_end=c_end,
            mottt_chunks=chunk_metadata,
        )

    def chunk_token_ids(
        self,
        token_ids: List[int],
        premise_start_idx: Optional[int] = None,
    ) -> ChunkedContext:
        """Divide continuous token ID sequence into chunks of length K."""
        chunks = [
            token_ids[i : i + self.chunk_size]
            for i in range(0, len(token_ids), self.chunk_size)
        ]

        premise_chunk_idx = 0
        depth_ratio = 0.0
        if premise_start_idx is not None and len(token_ids) > 0:
            premise_chunk_idx = min(premise_start_idx // self.chunk_size, max(0, len(chunks) - 1))
            depth_ratio = premise_start_idx / len(token_ids)

        return ChunkedContext(
            chunks=chunks,
            premise_chunk_index=premise_chunk_idx,
            depth_ratio=depth_ratio,
            total_tokens=len(token_ids),
        )
