from src.config import CONFIG
from src.data.build import build_dataset
from src.evaluator.aggregate import aggregate, render_summary_table, write_results
from src.evaluator.judge import judge
from src.evaluator.validate import run_validation
from src.generator.generate import generate
from src.llm.factory import get_generator, get_judge
from src.logging import get_logger
from src.retrieval.store import RetrievalStore

logger = get_logger(__name__)


def main() -> None:
    generator = get_generator()
    judge_provider = get_judge()

    logger.info("building dataset (store + held-out test split)")
    store_rows, test_rows = build_dataset(generator=generator)
    logger.info(f"store={len(store_rows)} rows, test={len(test_rows)} rows")

    store = RetrievalStore(store_rows, model_name=CONFIG.embed_model)

    records = []
    for row in test_rows:
        gen_output = generate(
            row.email, store, generator, k=CONFIG.k, threshold=CONFIG.similarity_threshold
        )
        record = judge(
            row.email,
            gen_output.reply_text,
            row.sent_reply,
            judge_provider,
            id_=row.id,
            category=row.category,
            gen_provider_used=gen_output.provider_used,
            gen_fallback_used=gen_output.fallback_used,
        )
        logger.info(
            f"id={row.id} category={row.category} score={record.score_overall:.1f} "
            f"similarity={record.similarity:.2f} fallback={gen_output.fallback_used} "
            f"provider={gen_output.provider_used} latency_ms={gen_output.latency_ms}"
        )
        records.append(record)

    aggregate_record = aggregate(records)
    out_dir = write_results(records, aggregate_record)
    logger.info(f"wrote results to {out_dir}")

    print()
    print(render_summary_table(records, aggregate_record))
    print()

    validation_report = run_validation(test_rows[:3], judge_provider)
    print(validation_report)


if __name__ == "__main__":
    main()
