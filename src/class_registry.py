"""Single source of truth for class names, IDs, and many-to-one prompt remapping.

``ClassRegistry`` is the canonical store for all class-name/ID logic in the
pipeline.  Every module that needs class information receives a ``ClassRegistry``
instance rather than maintaining its own list of names.

Key guarantee: **numeric IDs are never in the user config**.  The user supplies
only ``prompts`` (a list of text strings) and an optional ``class_remapping``
dict (name → name).  All numeric assignments are computed here, automatically.

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)


class ClassRegistry:
    """Maps SAM3 prompt indices to output class IDs (many-to-one supported).

    The remapping pipeline is:
    1. User provides ``prompts`` and optional ``class_remapping`` (name→name).
    2. ``from_config()`` applies the name remapping to each prompt.
    3. Deduplicated output names are assigned sequential integer IDs.
    4. Every method then converts between prompt index / class name / output ID.

    Example (5 prompts → 2 output classes)::

        prompts = ["teacher", "student", "kid", "child", "Adult"]
        class_remapping = {"Adult": "teacher", "kid": "student", "child": "student"}

        # Auto-computed:
        output_classes = ["teacher", "student"]  # first-seen order
        prompt_index_to_output_id = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}

    Attributes:
        prompts: Original prompt strings exactly as given in config.
        class_remapping: Name-to-name remapping supplied by user (may be empty).
        _output_names: Deduplicated output class names in first-seen order.
        _prompt_to_output_name: Resolved output name for each prompt.
        _output_name_to_id: Mapping from output class name → sequential int ID.
        _prompt_index_to_output_id: Fast lookup: prompt index → output int ID.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        prompts: List[str],
        class_remapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialise and compute all derived mappings.

        Args:
            prompts: SAM3 text prompts in order (index 0, 1, …).
            class_remapping: Optional name-to-name remapping.  Keys must be
                existing prompt names; values are the target output class names.

        Raises:
            ValueError: If ``prompts`` is empty or remapping keys are unknown.
        """
        self.prompts: List[str] = list(prompts)
        self.class_remapping: Dict[str, str] = dict(class_remapping or {})

        self.validate()
        self._build_mappings()

        _logger.info(
            "ClassRegistry built: %d prompt(s) → %d output class(es): %s",
            len(self.prompts),
            self.num_classes,
            self._output_names,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @trace
    def validate(self) -> None:
        """Validate prompts and class_remapping for obvious user mistakes.

        Raises:
            ValueError: On empty prompts, unknown remapping keys, or empty output.
        """
        if not self.prompts:
            raise ValueError("ClassRegistry: 'prompts' must contain at least one entry.")

        for p in self.prompts:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(
                    f"ClassRegistry: all prompt entries must be non-empty strings, got {p!r}."
                )

        prompt_set = set(self.prompts)
        for key, value in self.class_remapping.items():
            if key not in prompt_set:
                raise ValueError(
                    f"ClassRegistry: class_remapping key {key!r} is not in prompts "
                    f"{self.prompts}. Possible typo?"
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ClassRegistry: class_remapping value for {key!r} must be a "
                    f"non-empty string, got {value!r}."
                )

        # Warn on suspicious remapping targets (not in prompts and not a target elsewhere)
        all_targets = set(self.class_remapping.values())
        unmapped_prompts = {p for p in self.prompts if p not in self.class_remapping}
        valid_targets = unmapped_prompts | all_targets
        for key, value in self.class_remapping.items():
            if value not in valid_targets:
                _logger.warning(
                    "ClassRegistry: remapping target %r for prompt %r is not an "
                    "existing (unmapped) prompt. Possible typo?",
                    value,
                    key,
                )

    # ------------------------------------------------------------------
    # Internal mapping construction
    # ------------------------------------------------------------------

    def _build_mappings(self) -> None:
        """Compute all derived mappings from prompts + class_remapping."""
        # Step 1 — resolve each prompt to its output name
        self._prompt_to_output_name: List[str] = []
        for prompt in self.prompts:
            self._prompt_to_output_name.append(
                self.class_remapping.get(prompt, prompt)
            )

        # Step 2 — deduplicate in first-seen order
        seen: Dict[str, int] = {}
        for name in self._prompt_to_output_name:
            if name not in seen:
                seen[name] = len(seen)
        self._output_names: List[str] = list(seen.keys())
        self._output_name_to_id: Dict[str, int] = seen

        # Step 3 — build fast prompt-index → output-id lookup
        self._prompt_index_to_output_id: Dict[int, int] = {
            idx: self._output_name_to_id[name]
            for idx, name in enumerate(self._prompt_to_output_name)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> List[str]:
        """Output class names in assignment order (index = output class ID)."""
        return list(self._output_names)

    @property
    def num_classes(self) -> int:
        """Number of distinct output classes (M ≤ N prompts)."""
        return len(self._output_names)

    @property
    def num_prompts(self) -> int:
        """Number of SAM3 prompts (N)."""
        return len(self.prompts)

    @trace
    def remap_prompt_index(self, prompt_index: int) -> int:
        """Convert a raw SAM3 prompt index to an output class ID.

        This is the **only** place where raw → remapped conversion happens.
        Call this in the pipeline remap stage immediately after segmentation.

        Args:
            prompt_index: Raw class index returned by SAM3 (0 … N-1).

        Returns:
            Output class ID (0 … M-1).

        Raises:
            ValueError: If ``prompt_index`` is out of range.
        """
        if prompt_index not in self._prompt_index_to_output_id:
            raise ValueError(
                f"ClassRegistry: prompt_index {prompt_index} is out of range "
                f"(0 … {self.num_prompts - 1})."
            )
        return self._prompt_index_to_output_id[prompt_index]

    @trace
    def get_class_name(self, output_class_id: int) -> str:
        """Return the human-readable name for an output class ID.

        Args:
            output_class_id: Output class ID (0 … M-1).

        Returns:
            Class name string.

        Raises:
            ValueError: If ``output_class_id`` is out of range.
        """
        if output_class_id < 0 or output_class_id >= self.num_classes:
            raise ValueError(
                f"ClassRegistry: output_class_id {output_class_id} is out of range "
                f"(0 … {self.num_classes - 1})."
            )
        return self._output_names[output_class_id]

    @trace
    def get_yolo_names(self) -> Dict[int, str]:
        """Return the ``names`` dict for a YOLO ``data.yaml`` file.

        Returns:
            ``{0: 'class_a', 1: 'class_b', …}``
        """
        return {i: name for i, name in enumerate(self._output_names)}

    @trace
    def get_output_id_for_prompt_name(self, prompt_name: str) -> int:
        """Look up the output class ID for a prompt name (not prompt index).

        Args:
            prompt_name: A string from ``self.prompts``.

        Returns:
            Output class ID.

        Raises:
            ValueError: If ``prompt_name`` is not in prompts.
        """
        if prompt_name not in self.prompts:
            raise ValueError(
                f"ClassRegistry: {prompt_name!r} is not in prompts {self.prompts}."
            )
        idx = self.prompts.index(prompt_name)
        return self._prompt_index_to_output_id[idx]

    # ------------------------------------------------------------------
    # Serialisation — for multiprocessing IPC
    # ------------------------------------------------------------------

    @trace
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for IPC (pickle-free).

        Returns:
            Dict with ``prompts`` and ``class_remapping`` keys.
        """
        return {
            "prompts": list(self.prompts),
            "class_remapping": dict(self.class_remapping),
        }

    @classmethod
    @trace
    def from_dict(cls, data: Dict[str, Any]) -> "ClassRegistry":
        """Reconstruct a ``ClassRegistry`` from a serialised dict.

        Args:
            data: Dict as returned by :meth:`to_dict`.

        Returns:
            New ``ClassRegistry`` instance.
        """
        return cls(
            prompts=data["prompts"],
            class_remapping=data.get("class_remapping", {}),
        )

    # ------------------------------------------------------------------
    # Factory — primary construction path
    # ------------------------------------------------------------------

    @classmethod
    @trace
    def from_config(cls, config: Any) -> "ClassRegistry":
        """Build a ``ClassRegistry`` from a ``ModelConfig`` or any object with
        ``prompts`` and optionally ``class_remapping`` attributes.

        Args:
            config: ``ModelConfig`` dataclass (or any object with ``.prompts``
                and optional ``.class_remapping``).

        Returns:
            Configured ``ClassRegistry`` instance.

        Raises:
            ValueError: If the config is invalid.
        """
        prompts: List[str] = list(config.prompts)
        remapping: Dict[str, str] = {}

        raw = getattr(config, "class_remapping", None)
        if raw:
            if isinstance(raw, dict):
                remapping = raw
            else:
                _logger.warning(
                    "ClassRegistry.from_config: class_remapping is not a dict (%s), ignoring.",
                    type(raw).__name__,
                )

        return cls(prompts=prompts, class_remapping=remapping)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ClassRegistry(prompts={self.prompts!r}, "
            f"class_remapping={self.class_remapping!r}, "
            f"output_classes={self._output_names!r})"
        )

    # ------------------------------------------------------------------
    # Stats pattern
    # ------------------------------------------------------------------

    @trace
    def get_stats(self) -> Dict[str, Any]:
        """Return structural statistics for this registry.

        Returns:
            Dict with ``num_prompts``, ``num_output_classes``,
            ``has_remapping``, and ``output_class_names``.
        """
        return {
            "num_prompts": len(self.prompts),
            "num_output_classes": self.num_output_classes,
            "has_remapping": bool(self.class_remapping),
            "output_class_names": list(self._output_names),
        }

    @trace
    def reset_stats(self) -> None:
        """No-op — :class:`ClassRegistry` has no mutable counters."""

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassRegistry):
            return NotImplemented
        return (
            self.prompts == other.prompts
            and self.class_remapping == other.class_remapping
        )
