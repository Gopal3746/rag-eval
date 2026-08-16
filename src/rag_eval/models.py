from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    id: str
    title: str
    text: str


class GoldenExample(BaseModel):
    id: str
    question: str
    answer: str
    relevant_doc_ids: list[str] = Field(min_length=1)
    answer_keywords: list[str] = Field(default_factory=list)


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    rationale: str


@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    score: float
    text: str


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    question: str
    generated_answer: str
    retrieved_doc_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    faithfulness: float
    relevance: float
    judge_rationale: str
