"""RAGAS Evaluator for NASA Mission Intelligence RAG System.

Provides real-time and batch evaluation of RAG responses using RAGAS metrics
including Response Relevancy, Faithfulness, BLEU, and ROUGE scores.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        ResponseRelevancy,
        Faithfulness,
        RougeScore,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _run_async(coro):
    """Run an async coroutine, handling both sync and async contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context (e.g., Streamlit) -- use nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


async def _score_metric(metric, sample):
    """Score a single metric asynchronously."""
    return await metric.single_turn_ascore(sample)


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics.

    Computes Response Relevancy and Faithfulness for every call.
    When a reference answer is provided, also computes BLEU and ROUGE.

    Args:
        question: The user question.
        answer: The model-generated answer.
        contexts: List of retrieved context strings.
        reference: Optional reference/expected answer for BLEU/ROUGE.

    Returns:
        Dict mapping metric names to float scores, or an error dict.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Validate inputs
    if not question or not question.strip():
        return {"error": "Empty question provided"}
    if not answer or not answer.strip():
        return {"error": "Empty answer provided"}
    if not contexts or all(not c.strip() for c in contexts):
        return {"error": "No valid contexts provided"}

    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            api_key = os.environ.get("CHROMA_OPENAI_API_KEY", "")

        # Create evaluator LLM and embeddings (gpt-3.5-turbo for cost efficiency)
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        )

        # Define metric instances
        response_relevancy = ResponseRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings
        )
        faithfulness = Faithfulness(llm=evaluator_llm)

        # Build metrics list -- always include relevancy and faithfulness
        metrics_to_run = [
            ("response_relevancy", response_relevancy),
            ("faithfulness", faithfulness),
        ]

        # BLEU and ROUGE require a reference answer
        if reference and reference.strip():
            metrics_to_run.append(("bleu_score", BleuScore()))
            metrics_to_run.append(("rouge_score", RougeScore()))

        # Build the sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference if reference and reference.strip() else None,
        )

        # Evaluate each metric
        results: Dict[str, float] = {}

        for metric_name, metric in metrics_to_run:
            try:
                score = _run_async(_score_metric(metric, sample))
                results[metric_name] = float(score) if score is not None else 0.0
            except Exception as e:
                logger.warning(f"Metric {metric_name} failed: {e}")
                results[metric_name] = 0.0

        return results

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}


def evaluate_batch(
    dataset_path: str,
) -> Dict[str, Any]:
    """Run batch evaluation from a test dataset file.

    Supports JSON (list of dicts with 'question', 'contexts', 'answer')
    or the plain-text evaluation_dataset.txt format.

    Args:
        dataset_path: Path to the dataset file.

    Returns:
        Dict with 'per_question' results and 'aggregate' statistics.
    """
    if not os.path.exists(dataset_path):
        return {"error": f"Dataset file not found: {dataset_path}"}

    questions = _load_dataset(dataset_path)
    if not questions:
        return {"error": "No questions loaded from dataset"}

    per_question: List[Dict[str, Any]] = []
    all_scores: Dict[str, List[float]] = {}

    for item in questions:
        q = item.get("question", "")
        a = item.get("answer", item.get("expected_answer", ""))
        ctxs = item.get("contexts", [])
        ref = item.get("reference", item.get("expected_answer", ""))

        if not a:
            per_question.append({
                "question": q,
                "scores": {"error": "No answer provided for evaluation"},
            })
            continue

        scores = evaluate_response_quality(
            q, a, ctxs if ctxs else [a], reference=ref
        )
        per_question.append({"question": q, "scores": scores})

        for metric, val in scores.items():
            if metric != "error" and isinstance(val, (int, float)):
                all_scores.setdefault(metric, []).append(val)

    # Compute aggregates
    aggregate: Dict[str, Dict[str, float]] = {}
    for metric, vals in all_scores.items():
        aggregate[metric] = {
            "mean": sum(vals) / len(vals) if vals else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "count": len(vals),
        }

    return {"per_question": per_question, "aggregate": aggregate}


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load evaluation questions from JSON or plain-text file.

    For .json files, expects a list of dicts.
    For .txt files, parses the Q:/A:/Category: format used by
    evaluation_dataset.txt.
    """
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)

    # Parse plain-text format
    questions: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                if current.get("question"):
                    questions.append(current)
                    current = {}
                continue

            if line.lower().startswith("q:") or line.lower().startswith("question:"):
                if current.get("question"):
                    questions.append(current)
                    current = {}
                current["question"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("a:") or line.lower().startswith("expected answer:"):
                current["expected_answer"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("category:"):
                current["category"] = line.split(":", 1)[1].strip()

    if current.get("question"):
        questions.append(current)

    return questions
