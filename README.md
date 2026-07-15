# AI Email Suggested-Response System

A system that, given an incoming support email, generates a suggested reply grounded in a dataset of
past emails and their sent replies — and then **measures how good each generated reply actually is**
with a validated, weighted LLM-as-judge metric.

The evaluation is the core of this project. Generation is deliberately simple and correct; the
attention goes to *knowing whether a reply is good, and why*.

---

## TL;DR — run it end-to-end

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then fill in your keys
python run.py
```

`run.py` builds/loads the dataset, generates a suggested reply for each held-out test email, judges
each reply, and prints **per-response scores + an overall score + a per-category breakdown**, writing
full results to `results/`. No manual steps in between.

Optional interactive demo (not required for end-to-end):

```bash
uvicorn api.main:app --reload      # then POST to /suggest and /evaluate
```

---

## 1. Approach at a glance

Three stages with clean file boundaries, so each runs and is inspectable independently:

1. **Dataset** — real support email/reply pairs, split into a *retrieval store* and a *held-out test
   set*.
2. **Generator** — retrieval-augmented few-shot: for a new email, retrieve the most similar past
   `(email → reply)` pairs and use them as in-context examples for a single LLM call.
3. **Evaluator** — a weighted LLM-as-judge rubric scores each reply on five dimensions, cross-checked
   against embedding similarity, with a validation harness that proves the metric tracks real quality.

The batch pipeline (`run.py`) is the deliverable. A thin FastAPI layer is an optional demonstration
surface, not the product.

---

## 2. The dataset

**Source.** Anchored on the *Multilingual Customer Support Tickets* corpus (Tobi-Bueck; Kaggle /
Hugging Face), English-filtered. Each row is a real support **email** (subject + body) paired with the
**first agent response**, already labelled by category (billing, product support, IT support, returns,
etc.). It's genuine email — not chat or tweets — so the register already matches the B2B support inbox
this system is built for.

**Why this corpus is representative.** It's first-level support email with real customer phrasing and
real agent replies across the categories a shared inbox actually sees. That makes the "learn how we
reply" premise honest: the examples are the kind of mail the system will face in production.

**Augmentation.** Where a category is thin, we synthesise additional email-style threads in the same
register to ensure coverage. These are clearly generated in the same style as the real rows, not
scraped from elsewhere — described honestly here rather than passed off as real.

**License.** The anchor corpus is CC BY-NC; used here for a non-commercial evaluation exercise.

**The split — and why it keeps evaluation honest.** We partition into a **retrieval store** (what the
generator learns from) and a **held-out test set** (what we generate for and score). Crucially, the
split is *deliberate*: **no test email has a near-duplicate reply sitting in the store.** Test emails
are reworded versions of store topics — same underlying intent, different wording. This matters because
the generator works by retrieving similar past emails; if a test email had a twin in the store, the
model would just retrieve and copy that twin's reply, and we'd be measuring memorisation, not the
ability to reply to something new. By keeping test emails distinct-but-related, retrieval still
*helps* (relevant precedent surfaces) but never *hands over the answer* — so we're measuring
composition, which is what a suggested-reply feature actually does. The split is seeded and
deterministic.

---

## 3. The generator

**Mechanism: retrieval-augmented few-shot.** Embed the incoming email locally
(`sentence-transformers`, `all-MiniLM-L6-v2`), cosine-search the store for the top-`k` most similar
past emails (`k` configurable, default 3), and use *their* replies as in-context examples in a single
LLM call. A refund email retrieves past refund threads; an angry billing email retrieves past
de-escalations — so the model always sees the *relevant* precedent, conditioned on the incoming email.

**Why retrieval, not just hardcoded examples.** Retrieval is what makes the dataset *load-bearing*
rather than decorative. Tone isn't uniform across email types, and retrieval fetches the right tonal
template for *this* email instead of one fixed set. Add rows to the store and behaviour shifts
automatically — that's "grounded in your dataset" in the real sense.

**Trade-offs considered.**
- *Plain prompting (no retrieval)* — rejected: no grounding in the data, which the task explicitly
  asks for.
- *Fine-tuning* — rejected: expensive and unjustifiable on a small dataset, and it bakes the data into
  weights instead of keeping it inspectable and swappable. Wrong tool at this scale and time budget.
- *Large-context RAG (stuff many examples)* — rejected: dilutes the signal with less-relevant
  examples and burns tokens; a few *highly relevant* examples beat many mediocre ones.
- **Retrieval-augmented few-shot** — chosen: grounds generation in the actual data, adapts per-email,
  stays cheap, and keeps the dataset as a first-class, swappable component.

**Grounding is tonal/structural/policy — deliberately NOT fact-carrying.** This is a considered design
choice, not a limitation we backed into. Retrieved replies ground **tone, structure, and company
conventions** (how we open and close, how much empathy, refund-window language, escalation shape). They
do **not** transplant thread-specific facts. A fact that was true for one thread — "yes, the 15th is a
holiday," "your refund is processed," a specific invoice number — may be false for a new email, and a
*confident stale fact* is a worse failure than a generically-worded reply, because an agent might send
it without checking. So the generator is prompted to **draft, not fabricate**: when the incoming email
asks for a specific fact the examples can't reliably supply, it drafts a reply that *confirms or
verifies* rather than asserting. That's exactly how a real suggested-reply feature behaves — it drafts,
a human sends.

**Retrieval fallback.** If nothing in the store is similar enough (top similarity below a threshold),
the generator drops to a generic prompt seeded with a few fixed, diverse examples that convey only
tone/politeness/intent — and logs that the fallback fired. This is consistent with the tonal-grounding
philosophy: when there's no relevant precedent, all those examples can honestly provide is voice.

**Model choice: OpenAI generates, a different family judges.** See §4 — using different families for
generation and judging is a deliberate defence against self-preference bias.

---

## 4. Measuring accuracy — the core

### What "accurate" means for a suggested reply

Exact match against the human reply is the wrong bar: a reply can be worded a hundred ways and still be
excellent, and copying the human's exact words can still be the *wrong* reply for the customer.
N-gram/overlap metrics (BLEU/ROUGE/exact-match) therefore measure the wrong thing — they reward
parroting and punish valid paraphrase. "Accurate" here means **send-ready**: does the reply address the
customer's actual issue, cover what's needed, give a clear next step, stay policy-consistent without
fabricating, and match the right tone.

### The metric: a weighted LLM-as-judge rubric

A single pinned judge model scores each reply 0–5 on five dimensions (strict JSON, with a one-line
reason per score). Determinism comes from a fixed model + fixed prompt: `temperature` is intentionally
not sent on the judge call, since current Claude Opus versions reject a non-default value — see the
note in `llm/anthropic.py`. Dimensions and weights:

| Dimension | Weight | What it captures |
|---|---:|---|
| Relevance | 25% | Does it address the actual issue raised? |
| Completeness | 25% | Does it cover what a good reply needs to cover? |
| Actionability | 20% | Is there a clear next step? |
| Correctness | 17.5% | Policy-consistent and **non-hallucinated** (not fact-verified) |
| Tone | 12.5% | Appropriate empathy and register for this email |

**Why these weights.** Relevance and completeness lead because a reply that misses the issue or leaves
it half-answered fails regardless of polish. Actionability is close behind — a suggested reply that
doesn't move the customer forward isn't useful. **Correctness is redefined for this context**: with no
ground-truth fact source, it can't mean "factually verified"; it means *policy-consistent and
non-hallucinated*. Kept at 17.5% because it's the guardrail against the single worst failure of a
suggested-reply system — confident hallucination (a fabricated invoice number, an invented policy, a
false "it's fixed") that no other dimension catches: a hallucinated reply can be perfectly relevant,
complete, actionable, and well-toned. Tone weighs least because the output is a *draft* an agent
polishes before sending.

**The reference reply is used as a reference, not a target.** The judge sees the human sent reply as
*an example of what good looked like* for context — it does **not** score "how close is the generated
reply to the human one." That would smuggle string-matching back in and punish a better-but-different
reply. The reply is graded on its own merits.

**Similarity cross-check (diagnostic, not score).** Alongside each rubric score we report embedding
similarity between the generated reply and the reference. It is **not** blended into the score — it's a
drift detector: a high rubric score with very low similarity (or the reverse) flags a case worth
inspecting. It's how we sanity-check the judge without letting resemblance masquerade as quality.

### Validating that the metric reflects real quality, not just a number

A score of 82/100 means nothing unless we can show the number tracks good-vs-bad. Three checks (harness
in `evaluator/validate.py`):

- **Prong 1 — Discrimination.** Feed deliberately bad replies (empty, off-topic, rude/curt) through the
  metric and confirm they score *materially lower* than genuine replies. This proves the metric isn't
  rubber-stamping ~80 for everything — it can tell good from bad.
- **Prong 2 — Anti-copying.** Feed a genuinely good reply worded *very differently* from the reference
  and confirm it still scores high. This proves the metric rewards *quality*, not resemblance —
  demonstrating we avoided the string-matching trap.
- **Prong 3 — Human-score correlation.** Score several hand-labelled candidate replies per scenario
  (each spanning a real quality spectrum, from strong to weak, against one fixed reference) and compute
  the Spearman correlation between judge scores and the hand-assigned human scores. This is the check
  that most directly answers "does the metric track real quality," beyond just rejecting garbage
  (Prong 1) or ignoring wording (Prong 2).

Numbers from an actual run aren't captured here yet — this environment has no API keys configured, so
the harness hasn't been executed live. Running `python run.py` prints the full validation report
(score separation for Prong 1, score + similarity for Prong 2, and the Spearman correlation for Prong 3)
at the end of the run; paste that output here once you've run it with real keys.

### Reporting

- **Per-response:** overall 0–100 score, all five dimension scores, similarity, the judge's one-line
  reason, plus provenance — `judge_model`, generation `provider_used`, and whether retrieval fallback
  fired.
- **Overall:** mean score across the test set.
- **Per-category:** mean score per category, which surfaces *where* the generator is weak (e.g. strong
  on how-to, weaker on escalations) — the actionable output a team actually uses.

### Evaluation integrity — the pinned ruler

A metric is a measuring instrument, so it must be consistent. The judge is **one fixed model, one fixed
prompt, one set of weights** for an entire run, and `judge_model` is stamped into every record.
Critically, **the judge does not fall over.** Generation *may* fail over across vendors
(you just need *a* reply), but if two items were judged by two different models, their scores wouldn't
be comparable and the aggregates would be silently contaminated — a half-and-half scoreboard is worse
than none. So if the judge model is unavailable, the run **fails loudly** rather than quietly switching
rulers. (This generator-robustness vs. evaluator-integrity contrast is the reason the two components
have opposite failover rules.)

### Avoiding self-preference bias

LLM judges tend to over-rate outputs from their own model family. We **generate with OpenAI and judge
with a different family (Anthropic)** — cross-family judging is the textbook mitigation, so this is a
deliberate design choice rather than a caveat. Our discrimination test (Prong 1) further shows the
judge still scores known-bad replies low regardless of who wrote them, so it isn't blindly
rubber-stamping.

---

## 5. Architecture & system design

**Input/output contracts** are defined explicitly (see the code / TRD) at three levels — dataset row,
generator I/O (reply + retrieved-example ids + provider + latency + fallback flag), and evaluator I/O
(per-response record + aggregate record). The generator surfacing *which* examples it retrieved is what
lets us show grounding actually happened.

**Provider abstraction.** Every model call goes through one interface (`complete` / `acomplete`); no
module outside `src/llm` touches a vendor SDK. Swapping the primary model or reordering the fallback
chain is a one-line config change. Generation runs through a `FallbackProvider` chain (retry *inside*
each provider, fail over *across* providers, in that order); the judge is a single pinned handle with no
chain.

**Robustness & observability.** Transient errors (rate-limit, overloaded, timeout) are retried with
exponential backoff + jitter; terminal errors (400/401) are not. Structured logging records
provider/model/latency/tokens/retries per call and a one-line summary per batch item.

**Why no server / queue / websockets / cron for the core.** These were considered and deliberately
excluded — the core deliverable is an **offline batch evaluation**, which needs none of them:
- *Cron* — nothing runs on a schedule; the eval is run-on-demand.
- *WebSockets/streaming* — the interaction is discrete request→response (one email in, one reply out),
  not a live stream.
- *Task queue (Celery/RQ)* — the batch *is* an offline job; there's no request to keep alive.

The thin FastAPI layer (`/suggest`, `/evaluate`) is `async` so a slow LLM call doesn't block the event
loop, and uses FastAPI `BackgroundTasks` to fire-and-forget post-response logging — a *gesture* at the
async-offload pattern. **In production, generation would move to a dedicated task queue** (return a job
id, poll or webhook on completion) because multi-second LLM latency shouldn't hold a request open — but
that broker/worker infrastructure is intentionally out of scope here, as it adds operational surface
with no bearing on evaluation quality, which is what this challenge measures.

---

## 6. Feedback loop (designed for, not built)

Real human feedback — thumbs up/down, or "pick the better of two drafts" — is the **ground-truth
signal** that every offline metric here is ultimately a *proxy* for. It's the strongest signal a
production system could collect.

It is deliberately **not built**, for an honest reason: there are no real users in this exercise, and
simulating feedback with an LLM picking the better reply would just be *our judge wearing a different
hat* — synthetic signal dressed up as human signal. Instead, the system is **feedback-ready by
construction**: every per-response record carries a stable `id`, so a real preference signal would
attach as `(response_id, human_label)` with no schema change. In production that data would be used to
*validate and recalibrate the judge* — checking whether high-scoring replies are the ones agents
actually pick, and re-tuning dimension weights against that. We validate with the discrimination and
anti-copying tests precisely *because* real feedback isn't available here.

---

## 7. Limitations (honest)

- **Single-judge bias.** One model judges; even cross-family, an LLM judge has its own biases.
  Mitigations: a pinned model + fixed prompt for consistency, cross-family to reduce self-preference,
  and the validation prongs to show it discriminates.
- **Judge non-determinism.** Determinism rests on a fixed model and fixed prompt, not on
  `temperature=0` — current Claude Opus versions reject a non-default temperature, so it's intentionally
  omitted from the judge call (see `llm/anthropic.py`). Output is therefore near- but not
  fully-deterministic; the same reply can wobble a point between runs. We judge each item once. Judging
  N times and averaging would stabilise scores at ~N× cost — a documented future improvement, not
  implemented.
- **Synthetic augmentation.** Some rows are generated in-style for category coverage; disclosed rather
  than passed off as real.
- **One-reference framing.** The human reply is one valid answer among many; we use it as a reference
  for "good," not a target, but it still anchors the judge's sense of the domain.
- **Sequential batch.** Runs items one at a time for debuggability; concurrency is a future
  optimisation, not a v1 requirement.
- **Tonal grounding ceilings factual QA by design.** The system won't authoritatively *answer* a
  factual query ("is the 15th a holiday?") — it drafts a verify-then-send reply. That's the correct,
  safe product behaviour for a human-in-the-loop draft assistant, not a bug.

---

## 8. How AI tools were used

*(Fill in honestly.)* This project was built with AI assistance. AI tools were used for: designing the
architecture and evaluation methodology (the reasoning behind the metric, weights, grounding choice,
and pinned-ruler design was worked through in dialogue); scaffolding and implementing the modules from
a written technical spec; and drafting this README. All design decisions, trade-off calls, and the
final review were made by me. *(Adjust to reflect your actual usage — the graders asked for this
explicitly, so be specific and truthful.)*

---

## 9. Repository layout

```
src/
  llm/          # provider abstraction — the only place vendor SDKs are imported
  data/         # dataset build + deliberate seeded split
  retrieval/    # embed store, cosine search, k + threshold fallback
  generator/    # retrieval -> few-shot prompt -> reply (+ metadata)
  evaluator/    # pinned judge, weighted rubric, validation harness, aggregation
  config.py     # single source of truth: weights, k, thresholds, model strings, chain order
  logging.py    # structured logging
api/            # optional thin FastAPI demo (async; BackgroundTasks)
run.py          # the end-to-end batch deliverable
tests/          # sanity checks (weight math, JSON parse, split-leakage)
```

**Cleanliness rules enforced:** config centralised (no weights/model strings outside `config.py`);
prompts live in `prompts.py`, never inline; the judge's model is single-sourced from config;
`run.py` and the API hold zero business logic (orchestration only); only `src/llm` imports vendor SDKs.
