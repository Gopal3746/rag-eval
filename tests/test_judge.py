from rag_eval.judge import HeuristicJudge
from rag_eval.models import GoldenExample


def test_heuristic_judge_rewards_grounded_answer() -> None:
    example = GoldenExample(
        id="q",
        question="Which broker is used?",
        answer="Redis is the Celery broker.",
        relevant_doc_ids=["doc"],
        answer_keywords=["Redis", "Celery broker"],
    )
    score = HeuristicJudge().score(
        example,
        "Redis is the Celery broker.",
        ["Redis is the Celery broker and carries task messages."],
    )
    assert score.faithfulness == 5
    assert score.relevance == 5
