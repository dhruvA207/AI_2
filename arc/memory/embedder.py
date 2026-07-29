"""Local text embedding.

Runs BAAI/bge-small-en-v1.5 through ONNX Runtime rather than PyTorch. That choice is
load-bearing for two reasons the brief cares about:

- **Cross-platform** (§2). The same ONNX file and the same code run on Apple Silicon,
  Windows, and Linux. MLX would be faster here but exists only on Macs, and Track A
  is supposed to survive the Windows move without a rewrite.
- **Weight** (§7). onnxruntime is ~19 MB and tokenizers ~3 MB. Pulling PyTorch to
  embed short strings would add ~2.5 GB for a 33M-parameter model.

Embeddings never leave the machine — no API calls, per §3.
"""

from __future__ import annotations

import threading
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from arc.errors import MemoryError as ArcMemoryError
from arc.log import get_logger
from arc.paths import models_dir

_log = get_logger(__name__)

#: bge-small-en-v1.5. MIT, 33.4M params, 384 dimensions, 512 max tokens.
#: Licence verified 2026-07-29 (docs/DEPENDENCIES.md).
DEFAULT_REPO = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSION = 384
DEFAULT_MAX_TOKENS = 512

#: Files needed to run the model. Fetching only these keeps the download ~130 MB
#: instead of pulling the PyTorch and Safetensors copies we will never load.
_REQUIRED_FILES = ("onnx/model.onnx", "tokenizer.json", "tokenizer_config.json")

#: bge models were trained with this prefix on *queries* but not on stored passages.
#: Skipping it measurably degrades retrieval, and it is the kind of detail that is
#: invisible until recall is quietly bad.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Turns text into vectors, entirely locally."""

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        *,
        dimension: int = DEFAULT_DIMENSION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache_dir: Path | None = None,
    ) -> None:
        self._repo = repo
        self._dimension = dimension
        self._max_tokens = max_tokens
        self._cache_dir = cache_dir or (models_dir() / repo.replace("/", "--"))
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None

    @property
    def dimension(self) -> int:
        """Vector width. Must match the store's."""
        return self._dimension

    @property
    def name(self) -> str:
        """Model identifier, for stats and logs."""
        return self._repo

    def ensure_downloaded(self) -> Path:
        """Fetch the ONNX model and tokenizer if absent, returning the local directory."""
        target = self._cache_dir
        if (target / "onnx" / "model.onnx").is_file() and (target / "tokenizer.json").is_file():
            return target

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ArcMemoryError(
                "huggingface_hub is needed to download the embedding model."
            ) from exc

        try:
            snapshot_download(
                self._repo, local_dir=str(target), allow_patterns=list(_REQUIRED_FILES)
            )
        except Exception as exc:
            raise ArcMemoryError(f"could not download embedder {self._repo!r}: {exc}") from exc

        _log.info("downloaded embedder", extra={"repo": self._repo, "path": str(target)})
        return target

    def _load(self) -> None:
        """Load the ONNX session and tokenizer once, under a lock.

        Lazy because opening the session costs ~200 ms and not every command needs an
        embedder — ``arc memory stats`` should not pay for it.

        The check happens inside the lock rather than as a double-checked fast path.
        An uncontended lock acquire is tens of nanoseconds against milliseconds of
        inference, so the fast path bought nothing and made the concurrency harder to
        reason about.
        """
        with self._lock:
            if self._session is not None:
                return

            directory = self.ensure_downloaded()

            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise ArcMemoryError(
                    "onnxruntime and tokenizers are needed to embed text. "
                    "Install them with: pip install 'arc[memory]'"
                ) from exc

            tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=self._max_tokens)
            tokenizer.enable_padding()

            options = ort.SessionOptions()
            # Deterministic and polite: embedding runs alongside inference on a laptop,
            # and letting ORT grab every core makes the model stutter.
            options.intra_op_num_threads = 2
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._tokenizer = tokenizer
            self._session = ort.InferenceSession(
                str(directory / "onnx" / "model.onnx"),
                options,
                providers=["CPUExecutionProvider"],
            )
            _log.info("loaded embedder", extra={"repo": self._repo, "dim": self._dimension})

    def embed(self, text: str, *, is_query: bool = False) -> list[float]:
        """Embed a single string."""
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Embed many strings at once.

        Batching matters: the per-call ONNX overhead dominates for short texts, so
        embedding 1,000 memories one at a time is several times slower than in batches.
        """
        if not texts:
            return []

        self._load()
        prepared = [QUERY_PREFIX + t if is_query else t for t in texts]

        encodings = self._tokenizer.encode_batch(prepared)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Some BERT exports require token_type_ids; others omit it entirely.
        expected = {i.name for i in self._session.get_inputs()}
        if "token_type_ids" in expected:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        try:
            outputs = self._session.run(None, feed)
        except Exception as exc:
            raise ArcMemoryError(f"embedding failed: {exc}") from exc

        # bge uses the CLS token (position 0) as the sentence representation, not mean
        # pooling. Using the wrong one produces vectors that look fine and retrieve
        # badly, which is the worst kind of wrong.
        hidden = outputs[0]
        cls = hidden[:, 0, :]

        # Normalise to unit length so cosine similarity reduces to a dot product, and
        # so sqlite-vec's L2 distance is monotonic with cosine distance.
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = cls / norms

        if normalized.shape[1] != self._dimension:
            raise ArcMemoryError(
                f"embedder produced {normalized.shape[1]} dimensions, expected {self._dimension}"
            )

        return [row.tolist() for row in normalized.astype(np.float32)]


class HashEmbedder:
    """A deterministic, dependency-free stand-in used by tests.

    Not a toy for production: it has no semantic understanding at all. It exists so the
    test suite can exercise storage, retrieval ranking, and consolidation without
    downloading a model or paying ONNX startup on every test.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return "hash-embedder (test only)"

    def embed(self, text: str, *, is_query: bool = False) -> list[float]:
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Bag-of-words hashing, so texts sharing words get similar vectors.

        Uses crc32 rather than the builtin ``hash``, which is salted per process for
        strings. With ``hash`` a vector written in one run would not match the same
        text embedded in the next, producing tests that pass alone and fail in CI.
        """
        vectors: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self._dimension, dtype=np.float32)
            for word in text.lower().split():
                vec[zlib.crc32(word.encode("utf-8")) % self._dimension] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors
