"""
Golden sets: the queries you evaluate against and the answers you expect.

A golden set is a list of queries paired with the documents that should come
back. Twenty hand-written entries covering the questions your users actually
ask is worth more than a thousand generated ones, because the whole point is
to encode judgement that the system cannot infer for itself.

The file format is YAML or JSON:

.. code-block:: yaml

    - query: "What was Apple's net income in fiscal 2025?"
      expected: ["apple_10k_2025.pdf"]

    - query: "How did Amazon describe AWS growth?"
      expected: ["amazon_10q_q3.pdf", "amazon_10k_2025.pdf"]
      filters: {company_name: "amazon"}

``expected`` holds whatever identifies a correct chunk. By default that is the
``source`` metadata field, compared on file name alone so paths do not have to
match. Point ``match_field`` at another field to score on something else.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)


def _basename(value: str) -> str:
    """Strip any directory prefix, tolerating both separators on any platform."""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


class GoldenQuery:
    """
    One query and the documents that should be retrieved for it.

    Attributes:
        query: The search query
        expected: Identifiers that count as a correct hit
        filters: Metadata filters to pass to retrieve(), if any
        note: Free text explaining why this case is in the set
    """

    def __init__(
        self,
        query: str,
        expected: Iterable[str],
        filters: Optional[Dict[str, Any]] = None,
        note: str = "",
    ):
        if not query or not str(query).strip():
            raise ValueError("A golden query cannot be empty")

        self.query = str(query)
        self.expected = [str(e) for e in expected]
        self.filters = filters
        self.note = note

        if not self.expected:
            raise ValueError(
                f"Golden query {self.query!r} lists no expected documents, so "
                f"nothing it retrieves could ever be scored as correct"
            )

    def __repr__(self) -> str:
        return f"GoldenQuery(query={self.query!r}, expected={self.expected!r})"


class GoldenSet:
    """
    A collection of golden queries plus the rules for judging a match.

    Attributes:
        queries: The GoldenQuery entries
        match_field: Metadata field compared against ``expected``
        match_mode: ``"basename"`` compares file names only, ``"exact"``
            compares the stored value verbatim, ``"contains"`` accepts a
            substring match.

    Example:
        >>> golden = GoldenSet([GoldenQuery("revenue?", ["a.pdf"])])
        >>> len(golden)
        1
    """

    MATCH_MODES = ("basename", "exact", "contains")

    def __init__(
        self,
        queries: List[GoldenQuery],
        match_field: str = "source",
        match_mode: str = "basename",
    ):
        if match_mode not in self.MATCH_MODES:
            raise ValueError(
                f"Unknown match_mode: '{match_mode}'. "
                f"Available: {', '.join(self.MATCH_MODES)}"
            )

        self.queries = queries
        self.match_field = match_field
        self.match_mode = match_mode

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "GoldenSet":
        """
        Load a golden set from YAML or JSON.

        The file is either a bare list of entries, or a mapping with a
        ``queries`` key alongside ``match_field`` and ``match_mode`` settings.

        Args:
            path: Path to a .yaml, .yml or .json file

        Returns:
            The loaded GoldenSet

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the structure is not recognised

        Example:
            >>> GoldenSet.from_file("golden.yaml")  # doctest: +SKIP
            <GoldenSet with 20 queries>
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Golden set not found: {file_path}")

        raw_text = file_path.read_text(encoding="utf-8")

        if file_path.suffix.lower() == ".json":
            data = json.loads(raw_text)
        else:
            import yaml

            data = yaml.safe_load(raw_text)

        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: Any) -> "GoldenSet":
        """
        Build a golden set from already-parsed data.

        Args:
            data: A list of entries, or a mapping containing ``queries``

        Returns:
            The GoldenSet

        Raises:
            ValueError: If the structure is not a list or a mapping with queries

        Example:
            >>> g = GoldenSet.from_data([{"query": "q", "expected": ["a.pdf"]}])
            >>> len(g)
            1
        """
        settings: Dict[str, Any] = {}

        if isinstance(data, dict):
            entries = data.get("queries")
            if entries is None:
                raise ValueError(
                    "A golden set mapping must have a 'queries' key holding "
                    "the list of query entries"
                )
            settings = {
                key: data[key]
                for key in ("match_field", "match_mode")
                if key in data
            }
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(
                f"A golden set must be a list of entries or a mapping with a "
                f"'queries' key, got {type(data).__name__}"
            )

        queries = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Golden set entry {index} must be a mapping, got "
                    f"{type(entry).__name__}"
                )

            expected = entry.get("expected", entry.get("expected_sources"))
            if expected is None:
                raise ValueError(
                    f"Golden set entry {index} ({entry.get('query')!r}) has no "
                    f"'expected' key"
                )
            # A single file name is the common case, so accept it unwrapped.
            if isinstance(expected, str):
                expected = [expected]

            queries.append(
                GoldenQuery(
                    query=entry.get("query", ""),
                    expected=expected,
                    filters=entry.get("filters"),
                    note=entry.get("note", ""),
                )
            )

        return cls(queries, **settings)

    def identify(self, document: Any) -> str:
        """
        Reduce a retrieved document to the identifier used for scoring.

        Args:
            document: A retrieved Document

        Returns:
            The value of ``match_field``, normalised for the configured
            match_mode. Returns an empty string when the field is missing,
            which will never match an expected value.
        """
        value = document.metadata.get(self.match_field, "")
        if value is None:
            return ""
        value = str(value)
        return _basename(value) if self.match_mode == "basename" else value

    def matches(self, identifier: str, expected: str) -> bool:
        """
        Decide whether a retrieved identifier satisfies an expected one.

        Args:
            identifier: Output of :meth:`identify`
            expected: One entry from a query's ``expected`` list

        Returns:
            True if this counts as a correct hit
        """
        if not identifier:
            return False

        if self.match_mode == "basename":
            return identifier.lower() == _basename(expected).lower()
        if self.match_mode == "exact":
            return identifier == expected
        return expected.lower() in identifier.lower()

    def __repr__(self) -> str:
        return (
            f"<GoldenSet with {len(self.queries)} queries, "
            f"match_field={self.match_field!r}, match_mode={self.match_mode!r}>"
        )
