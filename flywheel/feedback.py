"""
Feedback Collector (Phase 5)
============================

Collect and process human feedback for model improvement.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback."""
    RATING = "rating"              # 1-5 star rating
    THUMBS = "thumbs"              # Thumbs up/down
    CORRECTION = "correction"      # User correction
    COMMENT = "comment"            # Free text comment
    PREFERENCE = "preference"      # A vs B preference


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    feedback_id: str = ""
    interaction_id: str = ""       # Related agent interaction
    feedback_type: FeedbackType = FeedbackType.RATING
    rating: Optional[int] = None   # 1-5 for RATING, 0/1 for THUMBS
    original_response: str = ""
    corrected_response: str = ""   # For CORRECTION type
    comment: str = ""
    user_id: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.feedback_id:
            self.feedback_id = f"fb_{int(self.timestamp * 1000)}"


@dataclass
class FeedbackSummary:
    """Summary of collected feedback."""
    total_feedback: int = 0
    average_rating: float = 0.0
    thumbs_up_pct: float = 0.0
    corrections_count: int = 0
    rating_distribution: Dict[int, int] = field(default_factory=dict)
    common_issues: List[str] = field(default_factory=list)
    period_start: float = 0.0
    period_end: float = 0.0


class FeedbackCollector:
    """Collects and processes human feedback.

    Feedback is used to:
    1. Improve auto-scoring accuracy
    2. Identify training examples needing correction
    3. Track user satisfaction over time
    4. Generate RLHF training data
    """

    def __init__(self, storage_dir: str = "data/feedback") -> None:
        """Initialize feedback collector.

        Args:
            storage_dir: Directory for feedback storage
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._feedback: List[FeedbackEntry] = []
        self._load_feedback()

    def submit_rating(
        self,
        interaction_id: str,
        rating: int,
        comment: str = "",
        user_id: str = ""
    ) -> FeedbackEntry:
        """Submit a star rating (1-5).

        Args:
            interaction_id: Related interaction
            rating: 1-5 rating
            comment: Optional comment
            user_id: User identifier

        Returns:
            Created feedback entry
        """
        rating = max(1, min(5, rating))

        entry = FeedbackEntry(
            interaction_id=interaction_id,
            feedback_type=FeedbackType.RATING,
            rating=rating,
            comment=comment,
            user_id=user_id,
        )

        self._feedback.append(entry)
        self._save_entry(entry)

        log.info("Received rating %d for interaction %s", rating, interaction_id)
        return entry

    def submit_thumbs(
        self,
        interaction_id: str,
        thumbs_up: bool,
        comment: str = "",
        user_id: str = ""
    ) -> FeedbackEntry:
        """Submit thumbs up/down feedback.

        Args:
            interaction_id: Related interaction
            thumbs_up: True for thumbs up
            comment: Optional comment
            user_id: User identifier

        Returns:
            Created feedback entry
        """
        entry = FeedbackEntry(
            interaction_id=interaction_id,
            feedback_type=FeedbackType.THUMBS,
            rating=1 if thumbs_up else 0,
            comment=comment,
            user_id=user_id,
        )

        self._feedback.append(entry)
        self._save_entry(entry)

        log.info("Received thumbs %s for interaction %s",
                "up" if thumbs_up else "down", interaction_id)
        return entry

    def submit_correction(
        self,
        interaction_id: str,
        original_response: str,
        corrected_response: str,
        comment: str = "",
        user_id: str = ""
    ) -> FeedbackEntry:
        """Submit a corrected response.

        Args:
            interaction_id: Related interaction
            original_response: Original AI response
            corrected_response: User's corrected version
            comment: Optional comment
            user_id: User identifier

        Returns:
            Created feedback entry
        """
        entry = FeedbackEntry(
            interaction_id=interaction_id,
            feedback_type=FeedbackType.CORRECTION,
            original_response=original_response,
            corrected_response=corrected_response,
            comment=comment,
            user_id=user_id,
        )

        self._feedback.append(entry)
        self._save_entry(entry)

        log.info("Received correction for interaction %s", interaction_id)
        return entry

    def submit_preference(
        self,
        interaction_id: str,
        response_a: str,
        response_b: str,
        preferred: str,  # "a" or "b"
        comment: str = "",
        user_id: str = ""
    ) -> FeedbackEntry:
        """Submit A/B preference.

        Args:
            interaction_id: Related interaction
            response_a: First response option
            response_b: Second response option
            preferred: Which is preferred ("a" or "b")
            comment: Optional comment
            user_id: User identifier

        Returns:
            Created feedback entry
        """
        entry = FeedbackEntry(
            interaction_id=interaction_id,
            feedback_type=FeedbackType.PREFERENCE,
            rating=0 if preferred == "a" else 1,
            original_response=response_a,
            corrected_response=response_b,  # Using as storage for response_b
            comment=comment,
            user_id=user_id,
            metadata={"preferred": preferred},
        )

        self._feedback.append(entry)
        self._save_entry(entry)

        log.info("Received preference for interaction %s", interaction_id)
        return entry

    def get_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> FeedbackSummary:
        """Get feedback summary for a time period.

        Args:
            start_time: Period start (None = all time)
            end_time: Period end (None = now)

        Returns:
            Feedback summary
        """
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = 0.0

        # Filter by time period
        period_feedback = [
            f for f in self._feedback
            if start_time <= f.timestamp <= end_time
        ]

        summary = FeedbackSummary(
            total_feedback=len(period_feedback),
            period_start=start_time,
            period_end=end_time,
        )

        # Calculate rating stats
        ratings = [f.rating for f in period_feedback
                  if f.feedback_type == FeedbackType.RATING and f.rating]
        if ratings:
            summary.average_rating = sum(ratings) / len(ratings)
            for r in range(1, 6):
                summary.rating_distribution[r] = ratings.count(r)

        # Calculate thumbs stats
        thumbs = [f.rating for f in period_feedback
                 if f.feedback_type == FeedbackType.THUMBS and f.rating is not None]
        if thumbs:
            summary.thumbs_up_pct = sum(thumbs) / len(thumbs) * 100

        # Count corrections
        summary.corrections_count = sum(
            1 for f in period_feedback
            if f.feedback_type == FeedbackType.CORRECTION
        )

        # Extract common issues from comments
        summary.common_issues = self._extract_common_issues(period_feedback)

        return summary

    def get_corrections_for_training(
        self,
        limit: int = 1000
    ) -> List[Dict[str, str]]:
        """Get corrections formatted for training.

        Returns data suitable for supervised fine-tuning.

        Args:
            limit: Maximum number of corrections

        Returns:
            List of training examples
        """
        corrections = [
            f for f in self._feedback
            if f.feedback_type == FeedbackType.CORRECTION
            and f.corrected_response
        ]

        training_data = []
        for corr in corrections[:limit]:
            # Get the original query from interaction
            # In production, would look up from interaction store
            training_data.append({
                "original": corr.original_response,
                "corrected": corr.corrected_response,
                "interaction_id": corr.interaction_id,
            })

        return training_data

    def get_preferences_for_rlhf(
        self,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get preference pairs for RLHF training.

        Args:
            limit: Maximum number of preferences

        Returns:
            List of preference pairs
        """
        preferences = [
            f for f in self._feedback
            if f.feedback_type == FeedbackType.PREFERENCE
        ]

        rlhf_data = []
        for pref in preferences[:limit]:
            preferred = pref.metadata.get("preferred", "a")
            rlhf_data.append({
                "interaction_id": pref.interaction_id,
                "chosen": pref.original_response if preferred == "a" else pref.corrected_response,
                "rejected": pref.corrected_response if preferred == "a" else pref.original_response,
            })

        return rlhf_data

    def _extract_common_issues(
        self,
        feedback: List[FeedbackEntry],
        top_k: int = 5
    ) -> List[str]:
        """Extract common issues from feedback comments."""
        # Simple keyword extraction
        issue_keywords = {
            "incorrect": "Factual errors",
            "wrong": "Factual errors",
            "inaccurate": "Factual errors",
            "slow": "Performance issues",
            "long": "Verbosity",
            "verbose": "Verbosity",
            "unclear": "Clarity issues",
            "confusing": "Clarity issues",
            "incomplete": "Missing information",
            "missing": "Missing information",
            "hallucin": "Hallucinations",
            "made up": "Hallucinations",
        }

        issue_counts: Dict[str, int] = {}
        for f in feedback:
            comment = f.comment.lower()
            for keyword, issue in issue_keywords.items():
                if keyword in comment:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Sort by count
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues[:top_k]]

    def _load_feedback(self) -> None:
        """Load feedback from storage."""
        feedback_file = self._storage_dir / "feedback.jsonl"
        if feedback_file.exists():
            try:
                with open(feedback_file) as f:
                    for line in f:
                        data = json.loads(line)
                        entry = FeedbackEntry(
                            feedback_id=data["feedback_id"],
                            interaction_id=data["interaction_id"],
                            feedback_type=FeedbackType(data["feedback_type"]),
                            rating=data.get("rating"),
                            original_response=data.get("original_response", ""),
                            corrected_response=data.get("corrected_response", ""),
                            comment=data.get("comment", ""),
                            user_id=data.get("user_id", ""),
                            timestamp=data["timestamp"],
                            metadata=data.get("metadata", {}),
                        )
                        self._feedback.append(entry)
                log.info("Loaded %d feedback entries", len(self._feedback))
            except Exception as e:
                log.warning("Failed to load feedback: %s", e)

    def _save_entry(self, entry: FeedbackEntry) -> None:
        """Save feedback entry to storage."""
        feedback_file = self._storage_dir / "feedback.jsonl"
        with open(feedback_file, "a") as f:
            data = {
                "feedback_id": entry.feedback_id,
                "interaction_id": entry.interaction_id,
                "feedback_type": entry.feedback_type.value,
                "rating": entry.rating,
                "original_response": entry.original_response,
                "corrected_response": entry.corrected_response,
                "comment": entry.comment,
                "user_id": entry.user_id,
                "timestamp": entry.timestamp,
                "metadata": entry.metadata,
            }
            f.write(json.dumps(data) + "\n")
