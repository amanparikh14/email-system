from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from src.config import CONFIG
from src.data.build import build_dataset
from src.data.schema import GeneratorOutput, PerResponseRecord
from src.evaluator.judge import ajudge
from src.generator.generate import agenerate
from src.llm.factory import get_generator, get_judge
from src.logging import get_logger
from src.retrieval.store import RetrievalStore

logger = get_logger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Store rows only -- the API doesn't need the held-out/reworded test split,
    # so pass generator=None to skip the LLM reword calls build_dataset() does
    # for that split (see data/build.py).
    store_rows, _test_rows = build_dataset(generator=None)
    _state["store"] = RetrievalStore(store_rows, model_name=CONFIG.embed_model)
    _state["generator"] = get_generator()
    _state["judge_provider"] = get_judge()
    logger.info(f"API ready: store={len(store_rows)} rows")
    yield
    _state.clear()


app = FastAPI(title="AI Email Suggested-Response System", lifespan=lifespan)


class SuggestRequest(BaseModel):
    email: str


class SuggestResponse(BaseModel):
    reply_text: str
    retrieved_example_ids: list[str]
    fallback_used: bool
    provider_used: str
    latency_ms: int


class EvaluateRequest(BaseModel):
    email: str
    reply: str
    reference_reply: str = ""


class EvaluateResponse(BaseModel):
    score_overall: float
    dimension_scores: dict
    similarity: float
    judge_reason: str
    judge_model: str


def _log_suggestion(output: GeneratorOutput) -> None:
    logger.info(
        f"[suggest] fallback={output.fallback_used} provider={output.provider_used} "
        f"latency_ms={output.latency_ms}"
    )


def _log_evaluation(record: PerResponseRecord) -> None:
    logger.info(f"[evaluate] score={record.score_overall:.1f} similarity={record.similarity:.2f}")


@app.post("/suggest", response_model=SuggestResponse)
async def suggest(req: SuggestRequest, background_tasks: BackgroundTasks) -> SuggestResponse:
    output = await agenerate(
        req.email, _state["store"], _state["generator"], k=CONFIG.k, threshold=CONFIG.similarity_threshold
    )
    # Fire-and-forget logging gesture (FR-14) -- in production this would move
    # to a dedicated task queue (Celery/RQ) instead of an in-process background task.
    background_tasks.add_task(_log_suggestion, output)
    return SuggestResponse(
        reply_text=output.reply_text,
        retrieved_example_ids=output.retrieved_example_ids,
        fallback_used=output.fallback_used,
        provider_used=output.provider_used,
        latency_ms=output.latency_ms,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest, background_tasks: BackgroundTasks) -> EvaluateResponse:
    record = await ajudge(req.email, req.reply, req.reference_reply, _state["judge_provider"])
    background_tasks.add_task(_log_evaluation, record)
    return EvaluateResponse(
        score_overall=record.score_overall,
        dimension_scores=record.dimension_scores,
        similarity=record.similarity,
        judge_reason=record.judge_reason,
        judge_model=record.judge_model,
    )
