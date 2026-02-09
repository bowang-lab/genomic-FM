"""
Surprise-Based Replay Buffer for Continual Learning

Based on SuRe (ICLR 2025): "Surprise-Driven Prioritised Replay for Continual LLM Learning"
https://arxiv.org/abs/2511.22367

Key insight: Store samples with highest loss (most "surprising") - these are:
- Hard examples the model struggles with
- Rare/underrepresented samples
- Samples at task boundaries where interference is highest

Usage:
    buffer = ReplayBuffer(max_size=1000)

    # During training, add samples with their loss
    buffer.add(sample, loss=2.5, task_id="CLNSIG")

    # Get replay samples for training
    replay_samples = buffer.sample(batch_size=16)
"""

import torch
import random
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


@dataclass
class ReplaySample:
    """A sample stored in the replay buffer."""
    ref_input_ids: torch.Tensor
    ref_attention_mask: torch.Tensor
    alt_input_ids: torch.Tensor
    alt_attention_mask: torch.Tensor
    labels: torch.Tensor
    task_id: str
    loss: float  # Surprise score (higher = more important to replay)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for batch collation."""
        return {
            "ref_input_ids": self.ref_input_ids,
            "ref_attention_mask": self.ref_attention_mask,
            "alt_input_ids": self.alt_input_ids,
            "alt_attention_mask": self.alt_attention_mask,
            "labels": self.labels,
        }


class ReplayBuffer:
    """
    Surprise-prioritized replay buffer for continual learning.

    Stores the most "surprising" (high-loss) samples from each task.
    When buffer is full, lowest-loss samples are evicted.

    Args:
        max_size: Maximum total samples across all tasks
        max_per_task: Maximum samples per task (None = no limit)
        selection_strategy: How to select samples
            - "surprise": Keep highest loss samples (SuRe-style)
            - "reservoir": Random reservoir sampling
            - "balanced": Equal samples per task
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_per_task: Optional[int] = None,
        selection_strategy: str = "surprise",
    ):
        self.max_size = max_size
        self.max_per_task = max_per_task or max_size
        self.selection_strategy = selection_strategy

        # Storage: task_id -> list of (loss, sample) tuples
        # Using negative loss for min-heap (we want to keep highest loss)
        self._buffers: Dict[str, List[Tuple[float, int, ReplaySample]]] = defaultdict(list)
        self._counter = 0  # Unique ID for heap ordering

    @property
    def total_size(self) -> int:
        """Total samples across all tasks."""
        return sum(len(buf) for buf in self._buffers.values())

    def task_sizes(self) -> Dict[str, int]:
        """Number of samples per task."""
        return {task: len(buf) for task, buf in self._buffers.items()}

    def add(
        self,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: torch.Tensor,
        alt_input_ids: torch.Tensor,
        alt_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        task_id: str,
        loss: float,
    ):
        """
        Add a sample to the buffer.

        If buffer is full, evicts lowest-loss sample from the same task.
        """
        sample = ReplaySample(
            ref_input_ids=ref_input_ids.cpu().clone(),
            ref_attention_mask=ref_attention_mask.cpu().clone(),
            alt_input_ids=alt_input_ids.cpu().clone(),
            alt_attention_mask=alt_attention_mask.cpu().clone(),
            labels=labels.cpu().clone(),
            task_id=task_id,
            loss=loss,
        )

        task_buffer = self._buffers[task_id]
        self._counter += 1

        if self.selection_strategy == "surprise":
            # Min-heap by negative loss (keeps highest loss samples)
            if len(task_buffer) < self.max_per_task:
                heapq.heappush(task_buffer, (-loss, self._counter, sample))
            elif loss > -task_buffer[0][0]:  # New sample has higher loss
                heapq.heapreplace(task_buffer, (-loss, self._counter, sample))

        elif self.selection_strategy == "reservoir":
            # Reservoir sampling
            if len(task_buffer) < self.max_per_task:
                task_buffer.append((-loss, self._counter, sample))
            else:
                # Replace random element with probability max_per_task / n
                n = self._counter
                j = random.randint(0, n)
                if j < self.max_per_task:
                    task_buffer[j] = (-loss, self._counter, sample)

        # Enforce global max_size by removing lowest-loss samples across tasks
        while self.total_size > self.max_size:
            self._evict_lowest_loss()

    def _evict_lowest_loss(self):
        """Remove the sample with lowest loss across all tasks."""
        min_loss = float('inf')
        min_task = None

        for task_id, buf in self._buffers.items():
            if buf:
                # Remember: stored as negative loss
                loss = -buf[0][0]
                if loss < min_loss:
                    min_loss = loss
                    min_task = task_id

        if min_task:
            heapq.heappop(self._buffers[min_task])

    def sample(
        self,
        batch_size: int,
        task_id: Optional[str] = None,
    ) -> List[ReplaySample]:
        """
        Sample from the buffer.

        Args:
            batch_size: Number of samples to return
            task_id: If provided, sample only from this task

        Returns:
            List of ReplaySample objects
        """
        if task_id:
            candidates = [s for _, _, s in self._buffers.get(task_id, [])]
        else:
            candidates = [s for buf in self._buffers.values() for _, _, s in buf]

        if not candidates:
            return []

        # Sample with replacement if batch_size > available
        k = min(batch_size, len(candidates))
        return random.sample(candidates, k)

    def sample_balanced(self, samples_per_task: int) -> List[ReplaySample]:
        """Sample equal number from each task."""
        samples = []
        for task_id in self._buffers:
            task_samples = self.sample(samples_per_task, task_id=task_id)
            samples.extend(task_samples)
        return samples

    def get_all(self, task_id: Optional[str] = None) -> List[ReplaySample]:
        """Get all samples, optionally filtered by task."""
        if task_id:
            return [s for _, _, s in self._buffers.get(task_id, [])]
        return [s for buf in self._buffers.values() for _, _, s in buf]

    def clear(self, task_id: Optional[str] = None):
        """Clear buffer, optionally only for a specific task."""
        if task_id:
            self._buffers[task_id] = []
        else:
            self._buffers.clear()

    def save(self, path: str):
        """Save buffer to disk."""
        data = {
            "max_size": self.max_size,
            "max_per_task": self.max_per_task,
            "selection_strategy": self.selection_strategy,
            "buffers": {
                task_id: [(loss, sample.to_dict(), sample.task_id, sample.loss)
                         for loss, _, sample in buf]
                for task_id, buf in self._buffers.items()
            }
        }
        torch.save(data, path)
        print(f"Saved replay buffer to {path} ({self.total_size} samples)")

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        """Load buffer from disk."""
        data = torch.load(path)
        buffer = cls(
            max_size=data["max_size"],
            max_per_task=data["max_per_task"],
            selection_strategy=data["selection_strategy"],
        )

        for task_id, samples in data["buffers"].items():
            for neg_loss, sample_dict, task, loss in samples:
                buffer._buffers[task_id].append((
                    neg_loss,
                    buffer._counter,
                    ReplaySample(
                        ref_input_ids=sample_dict["ref_input_ids"],
                        ref_attention_mask=sample_dict["ref_attention_mask"],
                        alt_input_ids=sample_dict["alt_input_ids"],
                        alt_attention_mask=sample_dict["alt_attention_mask"],
                        labels=sample_dict["labels"],
                        task_id=task,
                        loss=loss,
                    )
                ))
                buffer._counter += 1

        print(f"Loaded replay buffer from {path} ({buffer.total_size} samples)")
        return buffer


def collate_replay_samples(
    samples: List[ReplaySample],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate replay samples into a batch.

    Args:
        samples: List of ReplaySample objects
        device: Device to move tensors to

    Returns:
        Dictionary with batched tensors
    """
    if not samples:
        return {}

    batch = {
        "ref_input_ids": torch.stack([s.ref_input_ids for s in samples]),
        "ref_attention_mask": torch.stack([s.ref_attention_mask for s in samples]),
        "alt_input_ids": torch.stack([s.alt_input_ids for s in samples]),
        "alt_attention_mask": torch.stack([s.alt_attention_mask for s in samples]),
        "labels": torch.stack([s.labels for s in samples]),
    }

    if device:
        batch = {k: v.to(device) for k, v in batch.items()}

    return batch
