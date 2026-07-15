import csv
import random
import urllib.request
from pathlib import Path

from src.config import CONFIG
from src.data.schema import Row
from src.logging import get_logger

logger = get_logger(__name__)

CORPUS_URL = (
    "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets"
    "/resolve/main/dataset-tickets-multi-lang-4-20k.csv"
)
CORPUS_LICENSE = "CC BY-NC 4.0 (Tobi-Bueck/customer-support-tickets on Hugging Face)"

STORE_SIZE = 60
TEST_SIZE = 18
MIN_ROWS_BEFORE_SYNTH = STORE_SIZE + TEST_SIZE

REWORD_SYSTEM = (
    "You rewrite customer support emails. Keep the exact same underlying issue, "
    "intent, and any concrete facts (product names, error messages, dates) the "
    "customer mentions. Change the wording, sentence structure, and phrasing "
    "substantially so the text is not a near-duplicate of the original. "
    "Return only the rewritten email body, no preamble."
)

SYNTH_SYSTEM = (
    "You write a single realistic customer support email and the reply a "
    "support agent would send. Match the tone and register of a real "
    "helpdesk ticket. Return exactly two lines: 'EMAIL: <text>' then "
    "'REPLY: <text>'."
)


def _download_corpus() -> Path:
    cache_path = Path(CONFIG.data_cache_dir) / "corpus.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        logger.info(f"downloading corpus from {CORPUS_URL}")
        urllib.request.urlretrieve(CORPUS_URL, cache_path)
    return cache_path


def _load_english_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            r
            for r in reader
            if r.get("language", "").strip().lower() == "en"
            and r.get("body", "").strip()
            and r.get("answer", "").strip()
        ]
    return rows


def _to_row(raw: dict, idx: int, id_prefix: str, source: str) -> Row:
    subject = raw.get("subject", "").strip()
    body = raw.get("body", "").strip()
    email = f"Subject: {subject}\n\n{body}" if subject else body
    return Row(
        id=f"{id_prefix}-{idx:04d}",
        category=raw.get("queue", "uncategorized").strip() or "uncategorized",
        email=email,
        sent_reply=raw.get("answer", "").strip(),
        source=source,
    )


def _synthesize_rows(n: int, generator, start_idx: int) -> list[Row]:
    """Only invoked if the corpus is thinner than MIN_ROWS_BEFORE_SYNTH."""
    rows = []
    for i in range(n):
        text, _provider_used = generator.complete(
            SYNTH_SYSTEM,
            "Write one new customer support email + reply, distinct from any previous one.",
        )
        email_line, _, reply_line = text.partition("\nREPLY:")
        email = email_line.replace("EMAIL:", "").strip()
        reply = reply_line.strip()
        rows.append(
            Row(
                id=f"synth-{start_idx + i:04d}",
                category="uncategorized",
                email=email,
                sent_reply=reply,
                source="synthesized",
            )
        )
    return rows


def _reword_email(email: str, generator) -> str:
    text, _provider_used = generator.complete(REWORD_SYSTEM, email)
    return text.strip()


def split_indices(n_rows: int, store_size: int, test_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Pure, seeded partition of row indices into disjoint store/test sets.
    Split out from build_dataset so split integrity (AC-4: no overlap, and
    determinism across runs) is unit-testable without network/LLM calls.
    """
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    store_idx = indices[:store_size]
    test_idx = indices[store_size : store_size + test_size]
    return store_idx, test_idx


def build_dataset(generator=None) -> tuple[list[Row], list[Row]]:
    """Returns (store_rows, test_rows). Deterministic given CONFIG.seed.

    Test rows are reworded versions of held-out corpus rows (FR-2): same
    intent as a row that is NOT in the store, but different wording, so
    retrieval can surface relevant precedent without ever handing back a
    copyable reply.
    """
    csv_path = _download_corpus()
    english_rows = _load_english_rows(csv_path)
    logger.info(f"loaded {len(english_rows)} english rows from corpus")

    if len(english_rows) < MIN_ROWS_BEFORE_SYNTH:
        if generator is None:
            raise RuntimeError(
                "corpus coverage too thin and no generator provided to synthesize rows"
            )
        needed = MIN_ROWS_BEFORE_SYNTH - len(english_rows)
        logger.info(f"corpus thin ({len(english_rows)} rows) -- synthesizing {needed} more")
        synth = _synthesize_rows(needed, generator, start_idx=len(english_rows))
    else:
        synth = []

    store_idx, test_idx = split_indices(len(english_rows), STORE_SIZE, TEST_SIZE, CONFIG.seed)
    store_raw = [english_rows[i] for i in store_idx]
    test_raw = [english_rows[i] for i in test_idx]

    store_rows = [_to_row(r, i, "store", "corpus") for i, r in enumerate(store_raw)]
    store_rows += synth[: max(0, STORE_SIZE - len(store_raw))]

    test_rows_raw = [_to_row(r, i, "test", "corpus") for i, r in enumerate(test_raw)]

    if generator is not None:
        test_rows = [
            Row(
                id=row.id,
                category=row.category,
                email=_reword_email(row.email, generator),
                sent_reply=row.sent_reply,
                source=row.source,
            )
            for row in test_rows_raw
        ]
    else:
        # No generator available (e.g. offline dry run) -- keep raw test
        # rows. NOT used for real scoring: rewording is required for split
        # integrity (AC-4) whenever an LLM key is available.
        logger.warning("no generator provided -- test rows are NOT reworded (dry-run only)")
        test_rows = test_rows_raw

    return store_rows, test_rows
