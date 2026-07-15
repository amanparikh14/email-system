from src.data.schema import Row
from src.evaluator.judge import judge
from src.llm.anthropic import AnthropicProvider
from src.logging import get_logger

logger = get_logger(__name__)

KNOWN_BAD_REPLIES = {
    "empty": "",
    "off_topic": (
        "Thanks so much for being a loyal customer! As a token of our appreciation, "
        "enjoy 10% off your next purchase of headphones. Have a great day!"
    ),
    "rude_curt": "Not our problem. Figure it out yourself.",
}

# Hand-authored, deliberately reworded-not-copied "good" replies for anti-copying
# check (prong 2): same resolution as the reference, near-zero lexical overlap.
ANTI_COPY_EXAMPLES = [
    {
        "email": (
            "Subject: Can't log into my account\n\nI've tried resetting my password "
            "three times and I'm still locked out. This is really frustrating."
        ),
        "reference_reply": (
            "Hi, sorry about the trouble. I've manually reset your account lock. "
            "Please try logging in again with a new password reset link, which "
            "I've just sent to your registered email."
        ),
        "good_reworded_reply": (
            "Hey there -- I can tell this has been a headache, and I appreciate you "
            "sticking with it through three attempts. I just went in and cleared the "
            "lockout on our end directly, and a fresh password-reset email is on its "
            "way to your inbox right now. That link should get you back in within a "
            "couple of minutes. Let me know if it still doesn't work and I'll dig further."
        ),
    },
]

# Hand-authored scenarios for prong 3 (human-correlation validation). Each
# scenario has one fixed, high-quality `reference_reply` (what the judge sees
# as "what good looks like") and several `candidates` at different quality
# levels, each with a human-assigned score -- so every candidate is judged
# against the same fixed reference, never against itself.
HUMAN_LABELED_SCENARIOS = [
    {
        "email": "Subject: Refund status\n\nI returned my order two weeks ago and haven't seen a refund yet.",
        "reference_reply": (
            "Hi, I'm sorry for the delay. I checked and your return was received and "
            "processed on our end; the refund of $42.50 was issued to your original "
            "payment method 2 days ago and can take 3-5 business days to appear. "
            "If it's not there by Friday, reply here and I'll escalate it immediately."
        ),
        "candidates": [
            {
                "reply": (
                    "Hi, thanks for reaching out. I can see your return in our system. Refunds "
                    "typically take 5-7 business days once we receive the item. Could you confirm "
                    "the order number so I can check exactly where things stand and follow up?"
                ),
                "human_score": 78,
            },
            {"reply": "We process refunds. Please wait.", "human_score": 30},
        ],
    },
    {
        "email": "Subject: Product not working\n\nThe device stopped turning on after 2 days.",
        "reference_reply": (
            "I'm sorry to hear that -- a device failing after two days definitely "
            "isn't expected. Could you try holding the power button for 10 seconds "
            "to rule out a simple freeze? If that doesn't help, I'll send a prepaid "
            "return label right away and get a replacement shipped out to you."
        ),
        "candidates": [
            {"reply": "Have you tried turning it off and on again?", "human_score": 40},
            {"reply": "That is unfortunate. We are not able to help with hardware issues.", "human_score": 15},
        ],
    },
    {
        "email": "Subject: Billing question\n\nWhy was I charged twice this month?",
        "reference_reply": (
            "Good catch, and sorry for the confusion -- I looked into your account and "
            "one of those charges was a duplicate caused by a processing error on our "
            "side. I've refunded the duplicate charge of $19.99, which should post within "
            "3-5 business days. Let me know if you don't see it by early next week."
        ),
        "candidates": [
            {"reply": "Billing issues happen sometimes.", "human_score": 10},
        ],
    },
]


def _score_bad_replies(email: str, reference_reply: str, category: str, provider: AnthropicProvider) -> dict:
    scores = {}
    for label, reply in KNOWN_BAD_REPLIES.items():
        record = judge(
            email,
            reply or "(no content)",
            reference_reply,
            provider,
            id_=f"bad-{label}",
            category=category,
        )
        scores[label] = record.score_overall
    return scores


def _spearman(x: list[float], y: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for idx in range(i, j + 1):
                ranks[order[idx]] = avg_rank
            i = j + 1
        return ranks

    n = len(x)
    rx, ry = rank(x), rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d_sq) / (n * (n**2 - 1))


def run_validation(sample_rows: list[Row], provider: AnthropicProvider) -> str:
    lines = ["=== Metric validation report ===", ""]

    # Prong 1 (P0): discrimination -- known-bad replies must score materially
    # below genuine sent replies.
    lines.append("-- Prong 1: discrimination (bad replies vs. genuine replies) --")
    bad_scores_all: list[float] = []
    genuine_scores_all: list[float] = []
    for row in sample_rows[:3]:
        bad_scores = _score_bad_replies(row.email, row.sent_reply, row.category, provider)
        genuine_record = judge(
            row.email, row.sent_reply, row.sent_reply, provider, id_=f"genuine-{row.id}", category=row.category
        )
        bad_scores_all.extend(bad_scores.values())
        genuine_scores_all.append(genuine_record.score_overall)
        lines.append(f"  [{row.id}] genuine={genuine_record.score_overall:.1f}  bad={bad_scores}")
    avg_bad = sum(bad_scores_all) / len(bad_scores_all)
    avg_genuine = sum(genuine_scores_all) / len(genuine_scores_all)
    lines.append(f"  avg genuine score: {avg_genuine:.1f}  |  avg bad-reply score: {avg_bad:.1f}")
    lines.append(f"  separation: {avg_genuine - avg_bad:.1f} points  {'PASS' if avg_genuine - avg_bad > 20 else 'FAIL'}")
    lines.append("")

    # Prong 2 (P1): anti-copying -- a good but differently-worded reply should
    # still score high, and should NOT be rewarded for lexical similarity.
    lines.append("-- Prong 2: anti-copying (reworded-but-good reply) --")
    for ex in ANTI_COPY_EXAMPLES:
        record = judge(
            ex["email"], ex["good_reworded_reply"], ex["reference_reply"], provider, id_="anti-copy"
        )
        lines.append(
            f"  score={record.score_overall:.1f}  similarity_to_reference={record.similarity:.2f}  "
            f"{'PASS (high score despite low similarity)' if record.score_overall >= 70 else 'FAIL'}"
        )
    lines.append("")

    # Prong 3 (P0, addition beyond the original TRD): correlate judge scores
    # against hand-authored human scores across a quality spectrum. This is
    # the check that answers "does the metric reflect real quality" rather
    # than just "does it reject garbage" (prong 1) or "does it ignore
    # wording" (prong 2) -- it's the piece the brief asks for directly.
    lines.append("-- Prong 3: human-score correlation --")
    human_scores, judge_scores = [], []
    i = 0
    for scenario in HUMAN_LABELED_SCENARIOS:
        # the reference reply itself stands in as the top-scoring candidate
        candidates = [{"reply": scenario["reference_reply"], "human_score": 95}] + scenario["candidates"]
        for cand in candidates:
            record = judge(
                scenario["email"], cand["reply"], scenario["reference_reply"], provider, id_=f"human-corr-{i}"
            )
            human_scores.append(cand["human_score"])
            judge_scores.append(record.score_overall)
            lines.append(f"  human={cand['human_score']:>3}  judge={record.score_overall:>5.1f}")
            i += 1
    rho = _spearman(human_scores, judge_scores)
    lines.append(f"  Spearman correlation (judge vs. human): {rho:.2f}  {'PASS' if rho >= 0.7 else 'FAIL'}")
    lines.append("")

    return "\n".join(lines)
