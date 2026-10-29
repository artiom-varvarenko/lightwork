"""Trial labels shared by collect, evaluation and dataset building.

Labels are operator/trial-script facts, never predictions relabelled as ground truth. A physical collection session
has an explicit session_id shared across its runs, so dataset splits cannot leak that session into holdout.
Records live in the run's labels.jsonl (manifest-covered) and may also be carried by ui/trial_label events.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .backend import Verdict


class TrialCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected: Verdict


class TrialLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    frame_sha: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected: Verdict
    source: Literal["trial-script", "operator"]
    reviewed: bool = False
