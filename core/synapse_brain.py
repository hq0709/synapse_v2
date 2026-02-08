"""
Synapse Brain - Memory-Driven Scientific Thinking

Architecture:
- Semantic retrieval: Gemini text-embedding-004 (768D) + cosine similarity
- Keyword retrieval: BM25 (k1=1.5, b=0.75)
- Hybrid fusion: RRF (k=60) over 4 signals: embedding + BM25 + importance + recency
- Agentic retrieval: LLM judges sufficiency → generates refined queries → re-retrieve
- Persistence: SQLite (structured data, ACID) + FAISS HNSW (vector search)
- Consolidation: Semantic clustering → LLM merge/contradiction detection
- QA: Scientist persona (expertise + memory) → self-feedback → sentence-level citation
- Explorer: Evidence-chain hypotheses → critical evaluation → cross-doc synthesis

Core innovations from:
- EverMemOS: MemCell atomicity, Episode narrative, Foresight, RRF hybrid retrieval,
             Agentic multi-round recall, Profile with evidence, Consolidation
- OpenScholar: Self-feedback loop, Post-hoc sentence-level citation,
              Edit with feedback + retrieval, Passage format [idx]
"""

import time
import json
import hashlib
import re
import math
import os
import sqlite3
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# FAISS
try:
    import faiss
except ImportError:
    faiss = None

# Gemini API
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env file into os.environ."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_load_env_file(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MEMORY_DIR = os.getenv("SYNAPSE_MEMORY_DIR", ".synapse_memory")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
SYNAPSE_AUTO_BACKFILL_EMBEDDINGS = os.getenv(
    "SYNAPSE_AUTO_BACKFILL_EMBEDDINGS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
SYNAPSE_EXPLORER_MODE = os.getenv("SYNAPSE_EXPLORER_MODE", "epistemic_tools").strip().lower()
SYNAPSE_EPISTEMIC_LAMBDA_COST = float(os.getenv("SYNAPSE_EPISTEMIC_LAMBDA_COST", "0.15"))
SYNAPSE_EPISTEMIC_MU_RISK = float(os.getenv("SYNAPSE_EPISTEMIC_MU_RISK", "0.10"))
SYNAPSE_EPISTEMIC_STOP_ENTROPY = float(os.getenv("SYNAPSE_EPISTEMIC_STOP_ENTROPY", "0.35"))
LLM_REQUIRED_ERROR = (
    "LLM is required but unavailable. Please install dependencies and set "
    "GEMINI_API_KEY in .env."
)
EMBEDDINGS_REQUIRED_ERROR = (
    "Embedding service is required but unavailable. "
    "Check GEMINI_EMBEDDING_MODEL and API permissions."
)


def _detail_suffix(message: Optional[str]) -> str:
    if not message:
        return ""
    return f" Details: {message}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_first_json(
    text: str,
    expected: Optional[Union[type, Tuple[type, ...]]] = None
) -> Optional[Any]:
    """Extract the first valid JSON object/array from arbitrary model text."""
    if not text:
        return None
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        if expected is None or isinstance(obj, expected):
            return obj
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if expected is None or isinstance(obj, expected):
                return obj
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


# ======================== Data Structures ========================

class MemoryType(Enum):
    ATOMIC_FACT = "atomic_fact"
    EPISODE = "episode"
    PROFILE = "profile"
    FORESIGHT = "foresight"
    QA_PAIR = "qa_pair"
    CONTRADICTION = "contradiction"


@dataclass
class Evidence:
    content: str
    source: str
    type: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemCell:
    id: str
    memory_type: MemoryType
    content: str
    evidence: List[Evidence]
    source_doc: Optional[str]
    source_section: Optional[str]
    timestamp: datetime
    keywords: List[str]
    embedding: Optional[List[float]] = None
    connections: List[str] = field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def access(self):
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.importance = min(1.0, self.importance + 0.02 * (1.0 / (1 + self.access_count * 0.1)))


@dataclass
class Episode:
    id: str
    subject: str
    summary: str
    narrative: str
    memcell_ids: List[str]
    timestamp: datetime
    keywords: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicProfile:
    topic: str
    facts: List[Evidence]
    open_questions: List[str]
    knowledge_level: float
    last_updated: datetime
    episode_ids: List[str] = field(default_factory=list)


# ======================== BM25 Retrieval ========================

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lens: Dict[str, int] = {}
        self.avg_dl: float = 0.0
        self.corpus_size: int = 0
        self.doc_keywords: Dict[str, List[str]] = {}

    def add_document(self, doc_id: str, keywords: List[str]):
        self.doc_keywords[doc_id] = keywords
        self.doc_lens[doc_id] = len(keywords)
        for kw in set(keywords):
            self.doc_freqs[kw] += 1
        self.corpus_size += 1
        total = sum(self.doc_lens.values())
        self.avg_dl = total / self.corpus_size if self.corpus_size > 0 else 0

    def score(self, query_keywords: List[str], doc_id: str) -> float:
        if doc_id not in self.doc_keywords:
            return 0.0
        doc_kws = self.doc_keywords[doc_id]
        dl = self.doc_lens[doc_id]
        s = 0.0
        kw_freq = defaultdict(int)
        for kw in doc_kws:
            kw_freq[kw] += 1
        for qw in query_keywords:
            if qw not in kw_freq:
                continue
            tf = kw_freq[qw]
            df = self.doc_freqs.get(qw, 0)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1)))
            s += idf * tf_norm
        return s

    def search(self, query_keywords: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        scores = [(doc_id, self.score(query_keywords, doc_id))
                  for doc_id in self.doc_keywords]
        scores = [(d, s) for d, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ======================== Embedding Provider ========================

class EmbeddingProvider:
    """Single-model embedding provider with strict availability checks."""
    def __init__(self, api_key: str = None):
        self.available = False
        self.client = None
        self.dim = 0
        self.model_name: Optional[str] = None
        self.error_message: Optional[str] = None
        self.document_config = None
        self.query_config = None
        if api_key and genai and genai_types:
            try:
                self.client = genai.Client(api_key=api_key)
                self.model_name = GEMINI_EMBEDDING_MODEL
                if GEMINI_EMBEDDING_DIM <= 0:
                    raise ValueError("GEMINI_EMBEDDING_DIM must be a positive integer.")
                if self._uses_mrl_embedding():
                    self.document_config = genai_types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=GEMINI_EMBEDDING_DIM,
                    )
                    self.query_config = genai_types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=GEMINI_EMBEDDING_DIM,
                    )
                probe = self.client.models.embed_content(
                    model=self.model_name,
                    contents="test",
                    config=self.query_config,
                )
                values = self._extract_embedding_values(probe)
                if not values:
                    raise ValueError("Embedding probe returned empty vector.")
                self.dim = len(values)
                self.available = True
                print(f"  Embeddings: Gemini {self.dim}D connected ({self.model_name})")
            except Exception as e:
                self._mark_unavailable(str(e))
                print(f"  Embeddings: Failed ({self.error_message})")
        else:
            print("  Embeddings: Offline")

    def _mark_unavailable(self, message: str):
        self.available = False
        self.error_message = message

    def _uses_mrl_embedding(self) -> bool:
        if not self.model_name:
            return False
        return "gemini-embedding-001" in self.model_name

    def _extract_embedding_values(self, response: Any) -> Optional[List[float]]:
        if response is None:
            return None
        embeddings = getattr(response, "embeddings", None)
        if embeddings and len(embeddings) > 0:
            first = embeddings[0]
            vals = getattr(first, "values", None)
            if vals:
                return list(vals)
        if isinstance(response, dict):
            maybe_embeddings = response.get("embeddings")
            if isinstance(maybe_embeddings, list) and maybe_embeddings:
                first = maybe_embeddings[0]
                if isinstance(first, dict) and isinstance(first.get("values"), list):
                    return first["values"]
        return None

    def embed_document(self, text: str) -> Optional[List[float]]:
        if not self.available or not self.client or not self.model_name:
            return None
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text[:2000],
                config=self.document_config,
            )
            return self._extract_embedding_values(result)
        except Exception as e:
            self._mark_unavailable(str(e))
            return None

    def embed_query(self, text: str) -> Optional[List[float]]:
        if not self.available or not self.client or not self.model_name:
            return None
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text[:2000],
                config=self.query_config,
            )
            return self._extract_embedding_values(result)
        except Exception as e:
            self._mark_unavailable(str(e))
            return None


# ======================== SQLite Storage ========================

class SQLiteStorage:
    """SQLite backend for structured data — replaces JSON files."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        c = self.conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS memcells (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source_doc TEXT,
            source_section TEXT,
            timestamp TEXT NOT NULL,
            keywords TEXT NOT NULL,
            connections TEXT NOT NULL DEFAULT '[]',
            importance REAL NOT NULL DEFAULT 0.5,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed TEXT,
            has_embedding INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memcell_id TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            type TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (memcell_id) REFERENCES memcells(id)
        );
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            summary TEXT NOT NULL,
            narrative TEXT NOT NULL,
            memcell_ids TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            keywords TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS profiles (
            topic TEXT PRIMARY KEY,
            open_questions TEXT NOT NULL DEFAULT '[]',
            knowledge_level REAL NOT NULL DEFAULT 0.0,
            last_updated TEXT NOT NULL,
            episode_ids TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS profile_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            type TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (topic) REFERENCES profiles(topic)
        );
        CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_memcells_type ON memcells(memory_type);
        CREATE INDEX IF NOT EXISTS idx_memcells_source ON memcells(source_doc);
        CREATE INDEX IF NOT EXISTS idx_memcells_importance ON memcells(importance DESC);
        CREATE INDEX IF NOT EXISTS idx_memcells_timestamp ON memcells(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_evidence_memcell ON evidence(memcell_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_profile_facts_topic ON profile_facts(topic);
        """)
        c.commit()

    @contextmanager
    def transaction(self):
        """Context manager for atomic multi-statement transactions."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---- MemCells ----

    def insert_memcell(self, mc: 'MemCell'):
        self.conn.execute(
            """INSERT OR REPLACE INTO memcells
               (id, memory_type, content, source_doc, source_section, timestamp,
                keywords, connections, importance, access_count, last_accessed,
                has_embedding, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (mc.id, mc.memory_type.value, mc.content, mc.source_doc,
             mc.source_section, mc.timestamp.isoformat(),
             json.dumps(mc.keywords), json.dumps(mc.connections),
             mc.importance, mc.access_count,
             mc.last_accessed.isoformat() if mc.last_accessed else None,
             1 if mc.embedding else 0, json.dumps(mc.metadata))
        )
        # Insert evidence rows
        self.conn.execute("DELETE FROM evidence WHERE memcell_id=?", (mc.id,))
        for ev in mc.evidence:
            self.conn.execute(
                """INSERT INTO evidence (memcell_id, content, source, type, confidence, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (mc.id, ev.content, ev.source, ev.type, ev.confidence,
                 ev.timestamp.isoformat())
            )
        self.conn.commit()

    def get_memcell(self, mid: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM memcells WHERE id=?", (mid,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data['keywords'] = json.loads(data['keywords'])
        data['connections'] = json.loads(data['connections'])
        data['metadata'] = json.loads(data['metadata'])
        # Load evidence
        ev_rows = self.conn.execute(
            "SELECT * FROM evidence WHERE memcell_id=?", (mid,)).fetchall()
        data['evidence'] = [dict(r) for r in ev_rows]
        return data

    def get_all_memcells(self) -> List[dict]:
        rows = self.conn.execute("SELECT id FROM memcells").fetchall()
        return [self.get_memcell(r['id']) for r in rows]

    def get_memcells_by_type(self, memory_type: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT id FROM memcells WHERE memory_type=?", (memory_type,)).fetchall()
        return [r['id'] for r in rows]

    def get_memcells_by_source(self, source_doc: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT id FROM memcells WHERE source_doc=?", (source_doc,)).fetchall()
        return [r['id'] for r in rows]

    def update_memcell_connections(self, mid: str, connections: List[str]):
        self.conn.execute(
            "UPDATE memcells SET connections=? WHERE id=?",
            (json.dumps(connections), mid))
        self.conn.commit()

    def update_memcell_access(self, mid: str, access_count: int,
                              last_accessed: str, importance: float):
        self.conn.execute(
            "UPDATE memcells SET access_count=?, last_accessed=?, importance=? WHERE id=?",
            (access_count, last_accessed, importance, mid))
        self.conn.commit()

    def update_memcell_importance(self, mid: str, importance: float):
        self.conn.execute(
            "UPDATE memcells SET importance=? WHERE id=?", (importance, mid))
        self.conn.commit()

    def get_top_by_importance(self, limit: int) -> List[str]:
        rows = self.conn.execute(
            "SELECT id FROM memcells ORDER BY importance DESC LIMIT ?",
            (limit,)).fetchall()
        return [r['id'] for r in rows]

    def get_top_by_recency(self, limit: int) -> List[str]:
        rows = self.conn.execute(
            "SELECT id FROM memcells ORDER BY timestamp DESC LIMIT ?",
            (limit,)).fetchall()
        return [r['id'] for r in rows]

    # ---- Episodes ----

    def insert_episode(self, ep: 'Episode'):
        self.conn.execute(
            """INSERT OR REPLACE INTO episodes
               (id, subject, summary, narrative, memcell_ids, timestamp, keywords, metadata)
               VALUES (?,?,?,?,?,?,?,?)""",
            (ep.id, ep.subject, ep.summary, ep.narrative,
             json.dumps(ep.memcell_ids), ep.timestamp.isoformat(),
             json.dumps(ep.keywords), json.dumps(ep.metadata))
        )
        self.conn.commit()

    def get_all_episodes(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM episodes").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d['memcell_ids'] = json.loads(d['memcell_ids'])
            d['keywords'] = json.loads(d['keywords'])
            d['metadata'] = json.loads(d['metadata'])
            result.append(d)
        return result

    # ---- Profiles ----

    def upsert_profile(self, prof: 'TopicProfile'):
        self.conn.execute(
            """INSERT OR REPLACE INTO profiles
               (topic, open_questions, knowledge_level, last_updated, episode_ids)
               VALUES (?,?,?,?,?)""",
            (prof.topic, json.dumps(prof.open_questions), prof.knowledge_level,
             prof.last_updated.isoformat(), json.dumps(prof.episode_ids))
        )
        # Refresh facts
        self.conn.execute("DELETE FROM profile_facts WHERE topic=?", (prof.topic,))
        for ev in prof.facts:
            self.conn.execute(
                """INSERT INTO profile_facts (topic, content, source, type, confidence, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (prof.topic, ev.content, ev.source, ev.type, ev.confidence,
                 ev.timestamp.isoformat())
            )
        self.conn.commit()

    def get_all_profiles(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM profiles").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d['open_questions'] = json.loads(d['open_questions'])
            d['episode_ids'] = json.loads(d['episode_ids'])
            # Load facts
            fact_rows = self.conn.execute(
                "SELECT * FROM profile_facts WHERE topic=?", (d['topic'],)).fetchall()
            d['facts'] = [dict(fr) for fr in fact_rows]
            result.append(d)
        return result

    # ---- Stats ----

    def get_stat(self, key: str, default: int = 0) -> int:
        row = self.conn.execute(
            "SELECT value FROM stats WHERE key=?", (key,)).fetchone()
        return row['value'] if row else default

    def set_stat(self, key: str, value: int):
        self.conn.execute(
            "INSERT OR REPLACE INTO stats (key, value) VALUES (?,?)", (key, value))
        self.conn.commit()

    def increment_stat(self, key: str, delta: int = 1):
        self.conn.execute(
            """INSERT INTO stats (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = value + ?""",
            (key, delta, delta))
        self.conn.commit()

    def close(self):
        self.conn.close()


# ======================== FAISS Vector Store ========================

class FAISSVectorStore:
    """FAISS HNSW index — replaces linear-scan VectorStore."""

    def __init__(self, dim: int = 768, m: int = 32):
        self.dim = dim
        self.last_error: Optional[str] = None
        self.id_to_idx: Dict[str, int] = {}   # doc_id → FAISS int index
        self.idx_to_id: Dict[int, str] = {}   # FAISS int index → doc_id
        self.next_idx = 0

        if faiss is not None:
            self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
        else:
            self.index = None

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize so inner product = cosine similarity."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def add(self, doc_id: str, embedding: List[float]):
        if self.index is None or doc_id in self.id_to_idx:
            return
        vec = self._normalize(np.array(embedding, dtype=np.float32)).reshape(1, -1)
        self.index.add(vec)
        idx = self.next_idx
        self.id_to_idx[doc_id] = idx
        self.idx_to_id[idx] = doc_id
        self.next_idx += 1

    def add_batch(self, doc_ids: List[str], embeddings: List[List[float]]):
        if self.index is None:
            return
        vecs = []
        ids_added = []
        for doc_id, emb in zip(doc_ids, embeddings):
            if doc_id in self.id_to_idx:
                continue
            vecs.append(self._normalize(np.array(emb, dtype=np.float32)))
            ids_added.append(doc_id)
        if not vecs:
            return
        matrix = np.vstack(vecs).astype(np.float32)
        self.index.add(matrix)
        for doc_id in ids_added:
            idx = self.next_idx
            self.id_to_idx[doc_id] = idx
            self.idx_to_id[idx] = doc_id
            self.next_idx += 1

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query = self._normalize(np.array(query_embedding, dtype=np.float32)).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.idx_to_id.get(int(idx))
            if doc_id:
                results.append((doc_id, float(score)))
        return results

    def similarity(self, id1: str, id2: str) -> float:
        if self.index is None:
            return 0.0
        idx1 = self.id_to_idx.get(id1)
        idx2 = self.id_to_idx.get(id2)
        if idx1 is None or idx2 is None:
            return 0.0
        v1 = self.index.reconstruct(idx1)
        v2 = self.index.reconstruct(idx2)
        return float(np.dot(v1, v2))

    def contains(self, doc_id: str) -> bool:
        return doc_id in self.id_to_idx

    @property
    def size(self) -> int:
        return len(self.id_to_idx)

    def save(self, index_path: str, ids_path: str):
        if self.index is None:
            return
        # Atomic write: mkstemp + os.replace
        d = os.path.dirname(index_path) or '.'
        fd, tmp_index = tempfile.mkstemp(dir=d, suffix='.faiss')
        os.close(fd)
        try:
            faiss.write_index(self.index, tmp_index)
            os.replace(tmp_index, index_path)
        except Exception:
            os.unlink(tmp_index)
            raise

        ids_data = {
            'id_to_idx': self.id_to_idx,
            'idx_to_id': {int(k): v for k, v in self.idx_to_id.items()},
            'next_idx': self.next_idx
        }
        fd, tmp_ids = tempfile.mkstemp(dir=d, suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(ids_data, f)
            os.replace(tmp_ids, ids_path)
        except Exception:
            os.unlink(tmp_ids)
            raise

    def load(self, index_path: str, ids_path: str) -> bool:
        if faiss is None:
            return False
        if not os.path.exists(index_path) or not os.path.exists(ids_path):
            return False
        try:
            self.index = faiss.read_index(index_path)
            self.index.hnsw.efSearch = 64
            with open(ids_path, 'r') as f:
                ids_data = json.load(f)
            self.id_to_idx = ids_data['id_to_idx']
            self.idx_to_id = {int(k): v for k, v in ids_data['idx_to_id'].items()}
            self.next_idx = ids_data['next_idx']
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = str(e)
            return False


# ======================== Vector Store (legacy fallback) ========================

class VectorStore:
    """In-memory cosine similarity search — used only if FAISS unavailable."""
    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}

    def add(self, doc_id: str, embedding: List[float]):
        self.embeddings[doc_id] = np.array(embedding, dtype=np.float32)

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.embeddings:
            return []
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        scores = []
        for doc_id, emb in self.embeddings.items():
            doc_norm = np.linalg.norm(emb)
            if doc_norm == 0:
                continue
            sim = float(np.dot(query, emb) / (query_norm * doc_norm))
            scores.append((doc_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def similarity(self, id1: str, id2: str) -> float:
        if id1 not in self.embeddings or id2 not in self.embeddings:
            return 0.0
        e1, e2 = self.embeddings[id1], self.embeddings[id2]
        n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))


# ======================== LLM Interface ========================

class LLMProvider:
    def __init__(self, api_key: str = None):
        self.client = None
        self.call_count = 0
        self.model_name = GEMINI_MODEL
        self.last_error: Optional[str] = None
        self.generation_config = None
        if api_key and genai and genai_types:
            try:
                self.client = genai.Client(api_key=api_key)
                self.generation_config = genai_types.GenerateContentConfig(
                    temperature=GEMINI_TEMPERATURE,
                    top_p=GEMINI_TOP_P,
                    top_k=GEMINI_TOP_K,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    candidate_count=1,
                )
                print(f"  LLM: Gemini connected ({self.model_name}, temp={GEMINI_TEMPERATURE})")
            except Exception as e:
                self.last_error = str(e)
                print(f"  LLM: Failed ({e})")
        else:
            print("  LLM: Offline")

    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        if not self.client:
            return None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config,
                )
                self.call_count += 1
                text = getattr(response, "text", None)
                if text:
                    return text.strip()
            except Exception as e:
                self.last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        return None

    @property
    def is_available(self) -> bool:
        return self.client is not None


# ======================== Memory System ========================

class MemorySystem:
    def __init__(self, llm: LLMProvider, embedder: EmbeddingProvider,
                 storage_dir: str = None):
        self.llm = llm
        self.embedder = embedder
        self.storage_dir = storage_dir

        # In-memory caches (write-through)
        self.memcells: Dict[str, MemCell] = {}
        self.episodes: Dict[str, Episode] = {}
        self.profiles: Dict[str, TopicProfile] = {}

        # Indices
        self.bm25 = BM25()
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.source_index: Dict[str, Set[str]] = defaultdict(set)

        # SQLite + FAISS backends
        self.db: Optional[SQLiteStorage] = None
        self.faiss_vectors: Optional[FAISSVectorStore] = None
        self.vectors = VectorStore()  # legacy fallback
        self._pending_access: Set[str] = set()  # batched access writes

        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            db_path = os.path.join(storage_dir, 'synapse.db')
            self.db = SQLiteStorage(db_path)
            vector_dim = embedder.dim if embedder.dim > 0 else GEMINI_EMBEDDING_DIM
            self.faiss_vectors = FAISSVectorStore(dim=vector_dim, m=32)
            # Load FAISS index if exists
            idx_path = os.path.join(storage_dir, 'vectors.faiss')
            ids_path = os.path.join(storage_dir, 'vector_ids.json')
            if os.path.exists(idx_path):
                loaded = self.faiss_vectors.load(idx_path, ids_path)
                if not loaded and self.faiss_vectors.last_error:
                    print(f"  Warning: failed to load FAISS index ({self.faiss_vectors.last_error})")
            # Populate in-memory caches from SQLite
            self._load_from_sqlite()

        self.stats = {
            'total_memcells': 0, 'total_episodes': 0,
            'total_connections': 0, 'total_retrievals': 0,
            'total_contradictions': 0, 'llm_calls_for_memory': 0
        }
        if self.db:
            self._load_stats_from_sqlite()

    def _load_from_sqlite(self):
        """Populate in-memory caches from SQLite on startup."""
        if not self.db:
            return
        # MemCells
        for data in self.db.get_all_memcells():
            evidence = [Evidence(
                content=e['content'], source=e['source'],
                type=e['type'], confidence=e['confidence'],
                timestamp=datetime.fromisoformat(e['timestamp'])
            ) for e in data['evidence']]
            mc = MemCell(
                id=data['id'], memory_type=MemoryType(data['memory_type']),
                content=data['content'], evidence=evidence,
                source_doc=data['source_doc'], source_section=data['source_section'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                keywords=data['keywords'], connections=data['connections'],
                importance=data['importance'], access_count=data['access_count'],
                last_accessed=datetime.fromisoformat(data['last_accessed']) if data['last_accessed'] else None,
                metadata=data['metadata']
            )
            self.memcells[mc.id] = mc
            self.type_index[mc.memory_type].add(mc.id)
            if mc.source_doc:
                self.source_index[mc.source_doc].add(mc.id)
            self.bm25.add_document(mc.id, mc.keywords)

        # Episodes
        for data in self.db.get_all_episodes():
            ep = Episode(
                id=data['id'], subject=data['subject'], summary=data['summary'],
                narrative=data['narrative'], memcell_ids=data['memcell_ids'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                keywords=data['keywords'], metadata=data['metadata']
            )
            self.episodes[ep.id] = ep
            self.bm25.add_document(f"ep_{ep.id}", ep.keywords)

        # Profiles
        for data in self.db.get_all_profiles():
            facts = [Evidence(
                content=f['content'], source=f['source'],
                type=f['type'], confidence=f['confidence'],
                timestamp=datetime.fromisoformat(f['timestamp'])
            ) for f in data['facts']]
            self.profiles[data['topic']] = TopicProfile(
                topic=data['topic'], facts=facts,
                open_questions=data['open_questions'],
                knowledge_level=data['knowledge_level'],
                last_updated=datetime.fromisoformat(data['last_updated']),
                episode_ids=data['episode_ids']
            )

    def _load_stats_from_sqlite(self):
        """Load stats counters from SQLite."""
        for key in self.stats:
            self.stats[key] = self.db.get_stat(key, 0)

    def _chunk_text(self, content: str, chunk_size: int = 2600,
                    overlap: int = 240) -> List[str]:
        """Split long text into overlap-aware chunks to avoid prefix truncation."""
        text = re.sub(r'\n{3,}', '\n\n', content.strip())
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                split_para = text.rfind('\n\n', start, end)
                split_sent = text.rfind('. ', start, end)
                split = max(split_para, split_sent)
                if split > start + chunk_size // 2:
                    end = split + (2 if split == split_sent else 0)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _build_coverage_excerpt(self, chunks: List[str], max_chunks: int = 4,
                                chunk_chars: int = 1200,
                                include_headers: bool = True) -> str:
        """Build a distributed excerpt covering beginning/middle/end chunks."""
        if not chunks:
            return ""
        if len(chunks) <= max_chunks:
            selected_indices = list(range(len(chunks)))
        else:
            selected_indices = sorted({
                round(i * (len(chunks) - 1) / (max_chunks - 1))
                for i in range(max_chunks)
            })
        selected = []
        for idx in selected_indices:
            text = chunks[idx][:chunk_chars]
            if include_headers:
                selected.append(f"[Chunk {idx+1}/{len(chunks)}]\n{text}")
            else:
                selected.append(text)
        return "\n\n".join(selected)

    def _build_hierarchical_document_summary(self, chunks: List[str]) -> str:
        """
        Two-level document summary:
        1) chunk-level structured notes
        2) global synthesis over notes
        """
        if not chunks:
            return ""

        coverage_excerpt = self._build_coverage_excerpt(chunks, max_chunks=4, chunk_chars=1200)
        if not self.llm.is_available:
            return ""

        chunk_notes = []
        for idx, chunk in enumerate(chunks[:8]):
            prompt = f"""Summarize scientific chunk {idx+1}/{len(chunks)}.

Return JSON:
{{
  "key_points": ["2-4 concise findings/methods with specific values when present"],
  "entities": ["important materials/methods/datasets"]
}}
Return ONLY JSON.

Chunk:
{chunk[:2600]}"""
            response = self.llm.generate(prompt)
            self._incr_stat('llm_calls_for_memory')
            if not response:
                continue
            data = _extract_first_json(response, dict)
            if not isinstance(data, dict):
                continue
            points = data.get('key_points', [])
            entities = data.get('entities', [])
            point_text = " | ".join([p for p in points if isinstance(p, str)][:4])
            entity_text = ", ".join([e for e in entities if isinstance(e, str)][:6])
            if point_text or entity_text:
                chunk_notes.append(
                    f"Chunk {idx+1}: {point_text}"
                    + (f" [Entities: {entity_text}]" if entity_text else "")
                )

        if not chunk_notes:
            return coverage_excerpt

        synth_prompt = f"""Synthesize these chunk notes into a document-level overview.
Focus on findings, methods, bottlenecks, and quantitative evidence.
Keep it under 250 words.

Chunk notes:
{chr(10).join(chunk_notes)}

Document-level overview:"""
        synthesis = self.llm.generate(synth_prompt)
        self._incr_stat('llm_calls_for_memory')
        if synthesis:
            return synthesis.strip()
        return "\n".join(chunk_notes[:6])

    def _dedupe_fact_pairs(self, pairs: List[Tuple[str, str]],
                           max_items: int = 24) -> List[Tuple[str, str]]:
        seen: Set[str] = set()
        deduped: List[Tuple[str, str]] = []
        for fact, evidence in pairs:
            fact_text = re.sub(r'\s+', ' ', str(fact)).strip()
            if len(fact_text) < 15:
                continue
            key = fact_text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((fact_text, str(evidence).strip()))
            if len(deduped) >= max_items:
                break
        return deduped

    # ---- Extraction ----

    def extract_from_document(self, doc_path: str,
                              status_callback: Callable = None) -> Dict[str, Any]:
        if not self.llm.is_available:
            return {'error': LLM_REQUIRED_ERROR}
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {'error': str(e)}

        result = {'memcells': [], 'episodes': [], 'profile_updates': []}
        try:
            chunks = self._chunk_text(content)

            if status_callback:
                status_callback("Summarizing document chunks")
            doc_summary = self._build_hierarchical_document_summary(chunks)

            # Step 1: Extract atomic facts
            if status_callback:
                status_callback("Extracting atomic facts")
            facts = self._extract_atomic_facts(content, doc_path, chunks=chunks)
            for fact_text, evidence_text in facts:
                mem = self._store_memcell(
                    content=fact_text,
                    memory_type=MemoryType.ATOMIC_FACT,
                    evidence=[Evidence(content=evidence_text, source=doc_path,
                                       type="direct", confidence=0.9)],
                    source_doc=doc_path
                )
                result['memcells'].append(mem.id)

            # Step 2: Connect related facts (semantic + keyword)
            if status_callback:
                status_callback("Building connections")
            self._auto_connect(result['memcells'])

            # Step 3: Construct episode
            if status_callback:
                status_callback("Constructing episode narrative")
            episode = self._construct_episode(
                content, doc_path, result['memcells'],
                chunks=chunks, doc_summary=doc_summary
            )
            if episode:
                result['episodes'].append(episode.id)

            # Step 4: Topic profiles
            if status_callback:
                status_callback("Building topic profiles")
            topics = self._extract_topics(content, chunks=chunks, doc_summary=doc_summary)
            for topic in topics:
                self._update_profile(topic, result['memcells'])
                result['profile_updates'].append(topic)

            # Step 5: Foresight
            if status_callback:
                status_callback("Generating foresight")
            self._extract_foresight(
                content, doc_path, result['memcells'],
                chunks=chunks, doc_summary=doc_summary
            )

            # Step 6: Consolidate with existing memories (cross-document)
            if self.stats['total_memcells'] > len(result['memcells']):
                if status_callback:
                    status_callback("Consolidating memories")
                self._consolidate(result['memcells'], status_callback)

            return {
                'memcells_created': len(result['memcells']),
                'episodes_created': len(result['episodes']),
                'profiles_updated': len(result['profile_updates']),
                'topics': topics
            }
        except Exception as e:
            return {'error': str(e)}

    def _extract_atomic_facts(self, content: str, source: str,
                              chunks: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        chunks = chunks or self._chunk_text(content)
        if not chunks:
            return []

        all_facts: List[Tuple[str, str]] = []
        for idx, chunk in enumerate(chunks[:12]):
            prompt = f"""You are a scientific knowledge extractor. Extract atomic facts from this chunk ({idx+1}/{len(chunks)}).

RULES (atomicity enforcement):
- Each fact MUST be a single, independent, verifiable statement
- Each fact MUST be self-contained (understandable without reading other facts)
- Include specific numbers, names, methods, materials when available
- Preserve the precision of scientific claims (e.g., "12.4 mS/cm" not "high conductivity")
- For each fact, provide the exact text evidence from the document
- Extract 3-6 facts for this chunk

Chunk text:
{chunk[:2800]}

Return as JSON array:
[{{"fact": "...", "evidence": "exact quote or paraphrase from document", "section": "which section"}}]
Return ONLY the JSON array."""

            response = self.llm.generate(prompt)
            self._incr_stat('llm_calls_for_memory')
            if not response:
                continue
            facts_data = _extract_first_json(response, list)
            if not isinstance(facts_data, list):
                continue
            for item in facts_data:
                if isinstance(item, dict) and 'fact' in item:
                    all_facts.append((item['fact'], item.get('evidence', '')))
        return self._dedupe_fact_pairs(all_facts, max_items=24)

    def _auto_connect(self, memcell_ids: List[str]):
        """Connect MemCells by semantic similarity + keyword overlap."""
        for i, id1 in enumerate(memcell_ids):
            for id2 in memcell_ids[i+1:]:
                if id1 not in self.memcells or id2 not in self.memcells:
                    continue
                # Semantic similarity (FAISS or legacy)
                if self.faiss_vectors and self.faiss_vectors.contains(id1) and self.faiss_vectors.contains(id2):
                    sim = self.faiss_vectors.similarity(id1, id2)
                else:
                    sim = self.vectors.similarity(id1, id2)
                if sim > 0.75:
                    self._connect(id1, id2)
                    continue
                # Keyword overlap fallback
                kw1 = set(self.memcells[id1].keywords)
                kw2 = set(self.memcells[id2].keywords)
                if len(kw1 & kw2) >= 2:
                    self._connect(id1, id2)

    def _construct_episode(self, content: str, source: str,
                           memcell_ids: List[str],
                           chunks: Optional[List[str]] = None,
                           doc_summary: Optional[str] = None) -> Optional[Episode]:
        episode_id = self._gen_id(f"episode_{source}_{datetime.now()}")
        chunks = chunks or self._chunk_text(content)
        doc_summary = doc_summary or self._build_hierarchical_document_summary(chunks)
        prompt = f"""Construct a research episode narrative from chunk-level notes and global summary.

Return JSON:
{{
  "subject": "concise title (one line)",
  "summary": "2-3 sentence overview capturing the key contribution",
  "narrative": "A coherent paragraph (150-250 words) telling the research story. Preserve critical numerical details."
}}

Document summary:
{doc_summary[:2500]}

Coverage excerpt:
{self._build_coverage_excerpt(chunks, max_chunks=4, chunk_chars=900)}

Return ONLY JSON."""

        response = self.llm.generate(prompt)
        self._incr_stat('llm_calls_for_memory')
        if not response:
            return None
        data = _extract_first_json(response, dict)
        if not isinstance(data, dict):
            return None
        episode = Episode(
            id=episode_id,
            subject=data.get('subject', 'Research Document'),
            summary=data.get('summary', ''),
            narrative=data.get('narrative', ''),
            memcell_ids=memcell_ids,
            timestamp=datetime.now(),
            keywords=self._tokenize(
                data.get('subject', '') + ' ' + data.get('summary', ''))
        )
        self.episodes[episode_id] = episode
        self._incr_stat('total_episodes')
        self.bm25.add_document(f"ep_{episode_id}", episode.keywords)
        if self.db:
            self.db.insert_episode(episode)
        return episode

    def _extract_foresight(self, content: str, source: str, memcell_ids: List[str],
                           chunks: Optional[List[str]] = None,
                           doc_summary: Optional[str] = None):
        chunks = chunks or self._chunk_text(content)
        doc_summary = doc_summary or self._build_hierarchical_document_summary(chunks)
        prompt = f"""Based on this research, predict 2-3 future research directions.
For each: what will be investigated, timeline (near/mid/long-term), supporting evidence.

Document summary:
{doc_summary[:2200]}

Coverage excerpt:
{self._build_coverage_excerpt(chunks, max_chunks=3, chunk_chars=800)}

Return JSON array:
[{{"prediction": "...", "timeline": "near-term/mid-term/long-term", "evidence": "..."}}]
Return ONLY JSON array."""

        response = self.llm.generate(prompt)
        self._incr_stat('llm_calls_for_memory')
        if response:
            predictions = _extract_first_json(response, list)
            if isinstance(predictions, list):
                for pred in predictions[:3]:
                    if isinstance(pred, dict) and 'prediction' in pred:
                        mem = self._store_memcell(
                            content=f"[Foresight] {pred['prediction']} (Timeline: {pred.get('timeline', 'unknown')})",
                            memory_type=MemoryType.FORESIGHT,
                            evidence=[Evidence(content=pred.get('evidence', ''),
                                               source=source, type="inferred", confidence=0.5)],
                            source_doc=source, importance=0.6
                        )
                        for mid in memcell_ids[:3]:
                            self._connect(mem.id, mid)

    def _extract_topics(self, content: str, chunks: Optional[List[str]] = None,
                        doc_summary: Optional[str] = None) -> List[str]:
        chunks = chunks or self._chunk_text(content)
        doc_summary = doc_summary or self._build_hierarchical_document_summary(chunks)
        prompt = f"""What are the 2-3 main research topics in this document?
Return as JSON array of strings. Return ONLY the JSON array.
Document summary: {doc_summary[:1800]}
Coverage excerpt:
{self._build_coverage_excerpt(chunks, max_chunks=3, chunk_chars=700)}"""
        response = self.llm.generate(prompt)
        self._incr_stat('llm_calls_for_memory')
        if response:
            parsed_topics = _extract_first_json(response, list)
            if isinstance(parsed_topics, list):
                return [t for t in parsed_topics if isinstance(t, str)]
        return []

    def _update_profile(self, topic: str, memcell_ids: List[str]):
        if topic not in self.profiles:
            self.profiles[topic] = TopicProfile(
                topic=topic, facts=[], open_questions=[],
                knowledge_level=0.0, last_updated=datetime.now()
            )
        profile = self.profiles[topic]
        for mid in memcell_ids:
            if mid in self.memcells:
                for ev in self.memcells[mid].evidence:
                    profile.facts.append(ev)
        profile.knowledge_level = min(1.0, len(profile.facts) * 0.05)
        profile.last_updated = datetime.now()
        if self.db:
            self.db.upsert_profile(profile)

    # ---- Consolidation (cross-document) ----

    def _consolidate(self, new_ids: List[str], status_callback: Callable = None):
        """
        EverMemOS-inspired consolidation:
        Compare new MemCells against existing ones.
        - High similarity (>0.90): Merge (same fact from different sources)
        - Medium similarity (0.75-0.90) + different claims: Flag contradiction
        - Medium similarity: Connect (related facts)
        """
        if not self.embedder.available:
            return

        existing_ids = [mid for mid in self.memcells if mid not in new_ids
                        and self.memcells[mid].memory_type == MemoryType.ATOMIC_FACT]

        if not existing_ids:
            return

        merge_pairs = []
        existing_set = set(existing_ids)

        # Use FAISS search(top_k=20) per new_id instead of pairwise O(n*m)
        for new_id in new_ids:
            if new_id not in self.memcells:
                continue
            mc = self.memcells[new_id]
            has_faiss = (self.faiss_vectors and self.faiss_vectors.size > 0
                         and self.faiss_vectors.contains(new_id))
            if has_faiss:
                # Reconstruct vector from FAISS (works even if mc.embedding is None)
                vec = self.faiss_vectors.index.reconstruct(
                    self.faiss_vectors.id_to_idx[new_id])
                neighbors = self.faiss_vectors.search(vec.tolist(), top_k=20)
                for old_id, sim in neighbors:
                    if old_id not in existing_set:
                        continue
                    if sim > 0.90:
                        merge_pairs.append((new_id, old_id, sim))
                    elif sim > 0.75:
                        self._connect(new_id, old_id)
            elif mc.embedding:
                # Fallback to legacy pairwise
                for old_id in existing_ids:
                    sim = self.vectors.similarity(new_id, old_id)
                    if sim > 0.90:
                        merge_pairs.append((new_id, old_id, sim))
                    elif sim > 0.75:
                        self._connect(new_id, old_id)

        # Use LLM to check contradictions among high-similarity pairs
        if self.llm.is_available and merge_pairs:
            for new_id, old_id, sim in merge_pairs[:5]:
                new_content = self.memcells[new_id].content
                old_content = self.memcells[old_id].content
                new_source = self.memcells[new_id].source_doc or "unknown"
                old_source = self.memcells[old_id].source_doc or "unknown"

                prompt = f"""Compare these two scientific claims:
Claim A (from {Path(old_source).stem}): {old_content}
Claim B (from {Path(new_source).stem}): {new_content}

Are they: (1) saying the same thing, (2) complementary/related, or (3) contradictory?
If contradictory, explain the conflict.

Return JSON: {{"relation": "same/complementary/contradictory", "explanation": "brief explanation"}}
Return ONLY JSON."""

                response = self.llm.generate(prompt)
                self._incr_stat('llm_calls_for_memory')
                if response:
                    data = _extract_first_json(response, dict)
                    if isinstance(data, dict):
                        relation = data.get('relation', 'same')
                        if relation == 'contradictory':
                            self._store_memcell(
                                content=f"[Contradiction] {data.get('explanation', 'Conflicting claims detected')}\nClaim A: {old_content}\nClaim B: {new_content}",
                                memory_type=MemoryType.CONTRADICTION,
                                evidence=[
                                    Evidence(content=old_content, source=old_source,
                                             type="direct", confidence=0.9),
                                    Evidence(content=new_content, source=new_source,
                                             type="direct", confidence=0.9)
                                ],
                                importance=0.8
                            )
                            self._incr_stat('total_contradictions')
                        elif relation == 'same':
                            # Strengthen connection, boost importance
                            self._connect(new_id, old_id)
                            self.memcells[old_id].importance = min(1.0,
                                self.memcells[old_id].importance + 0.1)
                            if self.db:
                                self.db.update_memcell_importance(
                                    old_id, self.memcells[old_id].importance)
                        else:
                            self._connect(new_id, old_id)
                    else:
                        self._connect(new_id, old_id)

    # ---- Storage ----

    def _store_memcell(self, content: str, memory_type: MemoryType,
                       evidence: List[Evidence], source_doc: str = None,
                       source_section: str = None, importance: float = 0.5,
                       metadata: Optional[Dict[str, Any]] = None) -> MemCell:
        mem_id = self._gen_id(content + str(datetime.now()))
        keywords = self._tokenize(content)

        if not self.embedder.available:
            raise RuntimeError(EMBEDDINGS_REQUIRED_ERROR + _detail_suffix(self.embedder.error_message))
        embedding = self.embedder.embed_document(content)
        if embedding is None:
            raise RuntimeError("Embedding generation failed while storing memory." + _detail_suffix(self.embedder.error_message))

        memcell = MemCell(
            id=mem_id, memory_type=memory_type, content=content,
            evidence=evidence, source_doc=source_doc,
            source_section=source_section, timestamp=datetime.now(),
            keywords=keywords, embedding=embedding, importance=importance,
            metadata=metadata or {}
        )

        # In-memory
        self.memcells[mem_id] = memcell
        self.type_index[memory_type].add(mem_id)
        if source_doc:
            self.source_index[source_doc].add(mem_id)
        self.bm25.add_document(mem_id, keywords)

        # FAISS
        if embedding and self.faiss_vectors:
            self.faiss_vectors.add(mem_id, embedding)
        elif embedding:
            self.vectors.add(mem_id, embedding)

        # SQLite write-through
        if self.db:
            self.db.insert_memcell(memcell)

        self._incr_stat('total_memcells')
        return memcell

    def _connect(self, id1: str, id2: str):
        if id1 in self.memcells and id2 in self.memcells:
            changed = False
            if id2 not in self.memcells[id1].connections:
                self.memcells[id1].connections.append(id2)
                changed = True
            if id1 not in self.memcells[id2].connections:
                self.memcells[id2].connections.append(id1)
                changed = True
            if changed:
                self._incr_stat('total_connections')
                if self.db:
                    self.db.update_memcell_connections(id1, self.memcells[id1].connections)
                    self.db.update_memcell_connections(id2, self.memcells[id2].connections)

    # ---- Retrieval ----

    def _access_memcell(self, mid: str):
        """Update access stats in-memory (SQLite flush deferred to _flush_access)."""
        if mid in self.memcells:
            self.memcells[mid].access()
            self._pending_access.add(mid)

    def _flush_access(self):
        """Batch-write pending access updates to SQLite in one transaction."""
        if not self.db or not self._pending_access:
            return
        for mid in self._pending_access:
            if mid in self.memcells:
                mc = self.memcells[mid]
                self.db.conn.execute(
                    "UPDATE memcells SET access_count=?, last_accessed=?, importance=? WHERE id=?",
                    (mc.access_count,
                     mc.last_accessed.isoformat() if mc.last_accessed else None,
                     mc.importance, mid))
        self.db.conn.commit()
        self._pending_access.clear()

    def _normalize_allowed_types(self, allowed_types: Optional[Set[MemoryType]]) -> Optional[Set[MemoryType]]:
        if not allowed_types:
            return None
        normalized: Set[MemoryType] = set()
        for item in allowed_types:
            if isinstance(item, MemoryType):
                normalized.add(item)
            elif isinstance(item, str):
                try:
                    normalized.add(MemoryType(item))
                except ValueError:
                    continue
        return normalized or None

    def _is_allowed_memcell(self, mem_id: str,
                            allowed_types: Optional[Set[MemoryType]]) -> bool:
        if mem_id not in self.memcells:
            return False
        if not allowed_types:
            return True
        return self.memcells[mem_id].memory_type in allowed_types

    def retrieve(self, query: str, top_k: int = 10,
                 strategy: str = "hybrid",
                 allowed_types: Optional[Set[MemoryType]] = None) -> List[Tuple[MemCell, float]]:
        query_keywords = self._tokenize(query)
        self._incr_stat('total_retrievals')
        allowed_types = self._normalize_allowed_types(allowed_types)
        internal_top_k = top_k if not allowed_types else top_k * 5

        if strategy == "keyword":
            results = self._bm25_retrieval(query_keywords, internal_top_k, allowed_types)
        elif strategy == "hybrid":
            results = self._hybrid_retrieval(query, query_keywords, internal_top_k, allowed_types)
        elif strategy == "agentic":
            results = self._agentic_retrieval(
                query, query_keywords, internal_top_k, allowed_types
            )
        else:
            results = self._bm25_retrieval(query_keywords, internal_top_k, allowed_types)

        if allowed_types:
            results = [(m, s) for m, s in results if m.memory_type in allowed_types]

        self._flush_access()
        return results[:top_k]

    def _bm25_retrieval(self, query_keywords: List[str],
                        top_k: int,
                        allowed_types: Optional[Set[MemoryType]] = None) -> List[Tuple[MemCell, float]]:
        bm25_results = self.bm25.search(query_keywords, top_k * 2)
        results = []
        for doc_id, score in bm25_results:
            if self._is_allowed_memcell(doc_id, allowed_types):
                self._access_memcell(doc_id)
                results.append((self.memcells[doc_id], score))
        return results[:top_k]

    def _embedding_retrieval(self, query: str,
                             top_k: int) -> List[Tuple[str, float]]:
        """Semantic retrieval via Gemini embeddings + FAISS HNSW."""
        if not self.embedder.available:
            raise RuntimeError(EMBEDDINGS_REQUIRED_ERROR + _detail_suffix(self.embedder.error_message))
        query_emb = self.embedder.embed_query(query)
        if query_emb is None:
            raise RuntimeError("Embedding query failed." + _detail_suffix(self.embedder.error_message))
        if self.faiss_vectors and self.faiss_vectors.size > 0:
            return self.faiss_vectors.search(query_emb, top_k)
        if self.vectors.embeddings:
            return self.vectors.search(query_emb, top_k)
        return []

    def _hybrid_retrieval(self, query: str, query_keywords: List[str],
                          top_k: int,
                          allowed_types: Optional[Set[MemoryType]] = None) -> List[Tuple[MemCell, float]]:
        """
        4-signal RRF fusion (k=60):
        1. BM25 keyword matching
        2. Embedding cosine similarity
        3. Importance score
        4. Recency
        """
        # Signal 1: BM25
        bm25_results = self.bm25.search(query_keywords, top_k * 3)

        # Signal 2: Embedding similarity
        emb_results = self._embedding_retrieval(query, top_k * 3)

        # Signal 3: Importance
        importance_ranking = sorted(
            self.memcells.values(), key=lambda m: m.importance, reverse=True
        )[:top_k * 3]

        # Signal 4: Recency
        recency_ranking = sorted(
            self.memcells.values(), key=lambda m: m.timestamp, reverse=True
        )[:top_k * 3]

        # RRF fusion (k=60)
        rrf_k = 60
        rrf_scores: Dict[str, float] = defaultdict(float)

        # BM25: weight 1.0
        for rank, (doc_id, _) in enumerate(bm25_results):
            if self._is_allowed_memcell(doc_id, allowed_types):
                rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

        # Embedding: weight 1.0 (equally important as BM25)
        for rank, (doc_id, _) in enumerate(emb_results):
            if self._is_allowed_memcell(doc_id, allowed_types):
                rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

        # Importance: weight 0.5
        for rank, mem in enumerate(importance_ranking):
            if allowed_types and mem.memory_type not in allowed_types:
                continue
            rrf_scores[mem.id] += 0.5 / (rrf_k + rank + 1)

        # Recency: weight 0.3
        for rank, mem in enumerate(recency_ranking):
            if allowed_types and mem.memory_type not in allowed_types:
                continue
            rrf_scores[mem.id] += 0.3 / (rrf_k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for mem_id, score in sorted_results[:top_k]:
            if self._is_allowed_memcell(mem_id, allowed_types):
                self._access_memcell(mem_id)
                results.append((self.memcells[mem_id], score))
        return results

    def _agentic_retrieval(self, query: str, query_keywords: List[str],
                           top_k: int,
                           allowed_types: Optional[Set[MemoryType]] = None) -> List[Tuple[MemCell, float]]:
        """
        Round 1: Hybrid → LLM judges sufficiency
        Round 2: If insufficient, refined queries → re-retrieve → RRF merge
        """
        initial_results = self._hybrid_retrieval(query, query_keywords, top_k, allowed_types)

        if not initial_results:
            return initial_results
        if not self.llm.is_available:
            return []

        context = "\n".join([f"- {mem.content}" for mem, _ in initial_results[:5]])
        prompt = f"""Assess if this information is sufficient to answer the question.
Question: {query}
Retrieved: {context}
Return JSON: {{"sufficient": true/false, "missing": "what's missing", "refined_queries": ["query1", "query2"]}}
Return ONLY JSON."""

        response = self.llm.generate(prompt)
        self._incr_stat('llm_calls_for_memory')

        if response:
            data = _extract_first_json(response, dict)
            if isinstance(data, dict) and not data.get('sufficient', True):
                additional: Dict[str, float] = {}
                for rq in data.get('refined_queries', [])[:3]:
                    rq_results = self._hybrid_retrieval(
                        rq, self._tokenize(rq), top_k, allowed_types
                    )
                    for mem, score in rq_results:
                        if mem.id not in additional or score > additional[mem.id]:
                            additional[mem.id] = score

                merged: Dict[str, float] = {}
                for mem, score in initial_results:
                    merged[mem.id] = score
                for mem_id, score in additional.items():
                    if mem_id in merged:
                        merged[mem_id] += score * 0.5
                    else:
                        merged[mem_id] = score * 0.5

                sorted_merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)
                return [(self.memcells[mid], s) for mid, s in sorted_merged[:top_k]
                        if self._is_allowed_memcell(mid, allowed_types)]

        return initial_results

    # ---- Memory Tracing ----

    def trace(self, query: str) -> Dict[str, Any]:
        results = self.retrieve(query, top_k=5, strategy="hybrid")
        if not results:
            return {'found': False, 'message': 'No memories found'}

        traces = []
        for mem, score in results:
            trace = {
                'content': mem.content,
                'type': mem.memory_type.value,
                'relevance': score,
                'source': mem.source_doc or 'conversation',
                'evidence': [{'content': ev.content, 'source': ev.source,
                              'type': ev.type, 'confidence': ev.confidence}
                             for ev in mem.evidence],
                'connections': len(mem.connections),
                'access_count': mem.access_count,
                'timestamp': mem.timestamp.isoformat()
            }
            traces.append(trace)

        related_episodes = []
        for ep in self.episodes.values():
            for mem, _ in results:
                if mem.id in ep.memcell_ids:
                    related_episodes.append({'subject': ep.subject, 'summary': ep.summary})
                    break

        return {
            'found': True, 'query': query, 'traces': traces,
            'related_episodes': related_episodes,
            'total_memories': self.stats['total_memcells']
        }

    # ---- Persistence ----

    def save_vectors(self):
        """Save FAISS index to disk. SQLite is already durable via write-through."""
        if self.faiss_vectors and self.storage_dir:
            idx_path = os.path.join(self.storage_dir, 'vectors.faiss')
            ids_path = os.path.join(self.storage_dir, 'vector_ids.json')
            self.faiss_vectors.save(idx_path, ids_path)

    def backfill_vectors(self, max_items: int = 5000) -> Dict[str, int]:
        """
        Rebuild missing vectors for existing memories using current embedder.
        Useful when legacy data was loaded without a FAISS index.
        """
        if not self.embedder.available:
            return {"checked": 0, "embedded": 0, "failed": 0}

        has_faiss = bool(self.faiss_vectors and self.faiss_vectors.index is not None)
        checked = embedded = failed = 0
        for mem in self.memcells.values():
            if checked >= max_items:
                break
            checked += 1
            if has_faiss and self.faiss_vectors.contains(mem.id):
                continue
            if (not has_faiss) and mem.id in self.vectors.embeddings:
                continue
            emb = self.embedder.embed_document(mem.content)
            if emb is None:
                failed += 1
                continue
            mem.embedding = emb
            added = False
            if has_faiss:
                self.faiss_vectors.add(mem.id, emb)
                added = self.faiss_vectors.contains(mem.id)
            else:
                self.vectors.add(mem.id, emb)
                added = mem.id in self.vectors.embeddings

            if not added:
                failed += 1
                continue
            if self.db and has_faiss:
                self.db.conn.execute(
                    "UPDATE memcells SET has_embedding=1 WHERE id=?", (mem.id,))
            embedded += 1

        if self.db and has_faiss:
            self.db.conn.commit()
        if embedded > 0 and has_faiss:
            self.save_vectors()
        return {"checked": checked, "embedded": embedded, "failed": failed}

    def migrate_from_json(self, directory: str) -> bool:
        """One-time migration: read old JSON files into SQLite + FAISS."""
        memcells_path = os.path.join(directory, 'memcells.json')
        if not os.path.exists(memcells_path):
            return False

        try:
            print("  Migrating JSON → SQLite + FAISS...")

            # MemCells
            with open(memcells_path, 'r') as f:
                memcells_data = json.load(f)

            emb_ids, emb_vecs = [], []
            for mid, data in memcells_data.items():
                evidence = [Evidence(
                    content=e['content'], source=e['source'],
                    type=e['type'], confidence=e['confidence'],
                    timestamp=datetime.fromisoformat(e['timestamp'])
                ) for e in data['evidence']]

                mc = MemCell(
                    id=data['id'], memory_type=MemoryType(data['memory_type']),
                    content=data['content'], evidence=evidence,
                    source_doc=data['source_doc'], source_section=data['source_section'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    keywords=data['keywords'], connections=data['connections'],
                    importance=data['importance'], access_count=data['access_count'],
                    last_accessed=datetime.fromisoformat(data['last_accessed']) if data['last_accessed'] else None,
                    metadata=data.get('metadata', {})
                )
                self.memcells[mid] = mc
                self.type_index[mc.memory_type].add(mid)
                if mc.source_doc:
                    self.source_index[mc.source_doc].add(mid)
                self.bm25.add_document(mid, mc.keywords)
                if self.db:
                    self.db.insert_memcell(mc)

            # Embeddings → FAISS
            emb_path = os.path.join(directory, 'embeddings.json')
            if os.path.exists(emb_path):
                with open(emb_path, 'r') as f:
                    emb_data = json.load(f)
                for mid, emb in emb_data.items():
                    if mid in self.memcells:
                        self.memcells[mid].embedding = emb
                        emb_ids.append(mid)
                        emb_vecs.append(emb)
            if self.faiss_vectors and emb_ids:
                self.faiss_vectors.add_batch(emb_ids, emb_vecs)
            # Update has_embedding in SQLite (was 0 when inserted above)
            if self.db and emb_ids:
                for mid in emb_ids:
                    self.db.conn.execute(
                        "UPDATE memcells SET has_embedding=1 WHERE id=?", (mid,))
                self.db.conn.commit()

            # Episodes
            ep_path = os.path.join(directory, 'episodes.json')
            if os.path.exists(ep_path):
                with open(ep_path, 'r') as f:
                    episodes_data = json.load(f)
                for eid, data in episodes_data.items():
                    ep = Episode(
                        id=data['id'], subject=data['subject'], summary=data['summary'],
                        narrative=data['narrative'], memcell_ids=data['memcell_ids'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        keywords=data['keywords'], metadata=data.get('metadata', {})
                    )
                    self.episodes[eid] = ep
                    self.bm25.add_document(f"ep_{eid}", ep.keywords)
                    if self.db:
                        self.db.insert_episode(ep)

            # Profiles
            prof_path = os.path.join(directory, 'profiles.json')
            if os.path.exists(prof_path):
                with open(prof_path, 'r') as f:
                    profiles_data = json.load(f)
                for topic, data in profiles_data.items():
                    facts = [Evidence(
                        content=e['content'], source=e['source'],
                        type=e['type'], confidence=e['confidence'],
                        timestamp=datetime.fromisoformat(e['timestamp'])
                    ) for e in data['facts']]
                    prof = TopicProfile(
                        topic=data['topic'], facts=facts,
                        open_questions=data['open_questions'],
                        knowledge_level=data['knowledge_level'],
                        last_updated=datetime.fromisoformat(data['last_updated']),
                        episode_ids=data.get('episode_ids', [])
                    )
                    self.profiles[topic] = prof
                    if self.db:
                        self.db.upsert_profile(prof)

            # Stats
            stats_path = os.path.join(directory, 'stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.stats.update(json.load(f))
                if self.db:
                    for key, val in self.stats.items():
                        self.db.set_stat(key, val)

            # Save FAISS index
            self.save_vectors()

            print(f"  Migrated: {len(self.memcells)} memcells, "
                  f"{len(self.episodes)} episodes, "
                  f"{len(self.profiles)} profiles, "
                  f"{len(emb_ids)} vectors")
            return True

        except Exception as e:
            print(f"  Warning: Migration failed ({e})")
            return False

    # ---- Utilities ----

    def _incr_stat(self, key: str, delta: int = 1):
        """Increment a stat in-memory + SQLite."""
        self.stats[key] = self.stats.get(key, 0) + delta
        if self.db:
            self.db.increment_stat(key, delta)

    def _gen_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _tokenize(self, text: str) -> List[str]:
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
            'to', 'for', 'of', 'and', 'or', 'not', 'with', 'by', 'from',
            'this', 'that', 'it', 'its', 'be', 'as', 'has', 'have', 'had',
            'but', 'if', 'can', 'will', 'do', 'does', 'did', 'been', 'being',
            'which', 'what', 'when', 'where', 'who', 'how', 'than', 'then',
            'so', 'no', 'yes', 'just', 'more', 'also', 'very', 'about'
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in stopwords]


# ======================== Q&A System ========================

class QAEngine:
    """Scientist persona: background knowledge + research memory."""
    EVIDENCE_TYPES = {MemoryType.ATOMIC_FACT, MemoryType.EPISODE}

    def __init__(self, memory: MemorySystem, llm: LLMProvider):
        self.memory = memory
        self.llm = llm
        self.conversation_history: List[Dict] = []

    def ask(self, question: str,
            status_callback: Callable = None) -> Dict[str, Any]:
        start_time = time.time()

        # Step 1: Retrieve
        if status_callback:
            status_callback("Retrieving memories")
        strategy = "agentic"
        retrieved = self.memory.retrieve(
            question,
            top_k=10,
            strategy=strategy,
            allowed_types=self.EVIDENCE_TYPES
        )

        # Step 2: Format context
        context = self._format_context(retrieved)
        has_memory = len(retrieved) > 0

        # Step 3: Generate with scientist persona
        if status_callback:
            status_callback("Reasoning")
        answer, mode = self._generate_scientist_answer(question, context, has_memory)

        # Step 4: Self-feedback → refine
        if answer and has_memory:
            if status_callback:
                status_callback("Self-evaluation")
            feedback = self._get_feedback(question, context, answer)
            if feedback and not feedback.get('sufficient', True):
                if status_callback:
                    status_callback("Refining answer")
                additional = self._retrieve_for_feedback(feedback)
                full_ctx = context + ("\n" + additional if additional else "")
                answer = self._refine_answer(question, full_ctx, answer, feedback)

        # Step 5: Sentence-level citation verification
        if answer and has_memory:
            if status_callback:
                status_callback("Verifying citations")
            answer = self._verify_citations_sentence_level(answer, context)

        evidence_contract = self._build_evidence_contract(
            question=question,
            answer=answer or "",
            retrieved=retrieved
        )

        # Step 6: Store
        self._store_qa(question, answer, retrieved, evidence_contract)

        return {
            'answer': answer or "I need more information. Try uploading relevant documents.",
            'memories_used': len(retrieved),
            'sources': list(set(mem.source_doc for mem, _ in retrieved if mem.source_doc)),
            'evidence': [{'content': mem.content[:100],
                          'confidence': mem.evidence[0].confidence if mem.evidence else 0.5}
                         for mem, _ in retrieved[:3]],
            'elapsed': time.time() - start_time,
            'mode': mode,
            'evidence_contract': evidence_contract
        }

    def _format_context(self, retrieved: List[Tuple[MemCell, float]]) -> str:
        if not retrieved:
            return ""
        parts = []
        for idx, (mem, score) in enumerate(retrieved):
            source = Path(mem.source_doc).stem if mem.source_doc else "memory"
            parts.append(f"[{idx}] (Source: {source}) {mem.content}")
        return "\n".join(parts)

    def _generate_scientist_answer(self, question: str, context: str,
                                   has_memory: bool) -> Tuple[Optional[str], str]:
        history_text = ""
        if self.conversation_history:
            recent = self.conversation_history[-3:]
            history_text = "Previous conversation:\n"
            for item in recent:
                history_text += f"Q: {item['question']}\nA: {item['answer'][:300]}\n\n"

        if has_memory and context:
            mode = "expertise + memory"
            prompt = f"""You are an expert scientist with deep domain knowledge. You have BOTH your own scientific expertise AND specific research materials you have studied.

YOUR ROLE:
- You are a knowledgeable researcher, NOT a search engine
- You have read and internalized the research materials listed below
- Cite research materials using [0], [1], etc. for specific findings, numbers, or claims
- For general/foundational knowledge, use your expertise directly without citations
- If the question is general (e.g., "what is X?"), first explain from expertise, then connect to specific research
- If the question is specific, ground primarily in the research materials

{history_text}
Your Research Materials:
{context}

Question: {question}

Answer as a scientist who has thoroughly studied these materials and has broad domain expertise:"""

        else:
            mode = "expertise"
            prompt = f"""You are an expert scientist with deep domain knowledge across multiple fields.
Answer this question using your scientific expertise. Be thorough but concise.
A real scientist always has something insightful to say.

{history_text}
Question: {question}

Answer as a knowledgeable scientist:"""

        return self.llm.generate(prompt), mode

    def _get_feedback(self, question: str, context: str, answer: str) -> Optional[Dict]:
        prompt = f"""Evaluate this scientific answer:
1. CONTENT: Are claims grounded in the research materials? Citations accurate?
2. COMPLETENESS: Anything important missing?
3. ACCURACY: Any potentially incorrect statements?

Question: {question}
Research Materials: {context[:1500]}
Answer: {answer[:1500]}

Return JSON: {{"sufficient": true/false, "content_feedback": "...", "missing_info": "...", "retrieval_query": "search query or null"}}
Return ONLY JSON."""

        response = self.llm.generate(prompt)
        if response:
            parsed = _extract_first_json(response, dict)
            if isinstance(parsed, dict):
                return parsed
        return None

    def _retrieve_for_feedback(self, feedback: Dict) -> Optional[str]:
        query = feedback.get('retrieval_query')
        if not query:
            return None
        additional = self.memory.retrieve(
            query,
            top_k=5,
            strategy="hybrid",
            allowed_types=self.EVIDENCE_TYPES
        )
        if additional:
            parts = []
            for idx, (mem, score) in enumerate(additional):
                source = Path(mem.source_doc).stem if mem.source_doc else "memory"
                parts.append(f"[{10+idx}] (Source: {source}) {mem.content}")
            return "\n".join(parts)
        return None

    def _refine_answer(self, question: str, context: str,
                       answer: str, feedback: Dict) -> str:
        prompt = f"""Improve this scientific answer based on the feedback.
Only modify parts that need improvement. Keep correct information. Maintain citations.

Question: {question}
Research Materials: {context[:2000]}
Current Answer: {answer}
Feedback: {feedback.get('content_feedback', '')}
Missing: {feedback.get('missing_info', '')}

Improved Answer:"""

        refined = self.llm.generate(prompt)
        return refined if refined else answer

    def _verify_citations_sentence_level(self, answer: str, context: str) -> str:
        """
        OpenScholar post-hoc: split into sentences, verify each citation
        against the actual passages. Fix wrong citations, add missing ones.
        """
        if not re.search(r'\[\d+\]', answer):
            return answer

        prompt = f"""Verify citations in this answer against the research materials.

For each sentence with a citation [N]:
- Check if passage [N] actually supports the claim
- If wrong, fix to the correct passage number
- If a claim has no citation but a passage supports it, add the citation
- Do NOT modify text content, only fix citation numbers

Research Materials:
{context[:2000]}

Answer:
{answer}

Return ONLY the corrected answer text."""

        verified = self.llm.generate(prompt)
        return verified if verified else answer

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _build_evidence_contract(self, question: str, answer: str,
                                 retrieved: List[Tuple[MemCell, float]]) -> Dict[str, Any]:
        evidence_items = []
        for idx, (mem, score) in enumerate(retrieved[:8]):
            evidence_items.append({
                'index': idx,
                'memcell_id': mem.id,
                'memory_type': mem.memory_type.value,
                'source': mem.source_doc or "memory",
                'relevance': round(float(score), 4),
                'confidence': round(mem.evidence[0].confidence, 3) if mem.evidence else 0.5,
                'excerpt': mem.content[:180]
            })

        cited_indices = {
            int(x) for x in re.findall(r'\[(\d+)\]', answer)
            if x.isdigit() and int(x) < len(evidence_items)
        }
        cited_evidence = [item for item in evidence_items if item['index'] in cited_indices]

        sentences = self._split_sentences(answer)
        cited_sentence_count = 0
        for sent in sentences:
            if re.search(r'\[\d+\]', sent):
                cited_sentence_count += 1

        assumptions = []
        hedge_hits = re.findall(r'\b(may|might|could|likely|possible|possibly)\b', answer.lower())
        if hedge_hits:
            assumptions.append("Answer includes inferential language and should be treated as a hypothesis.")
        if not cited_evidence and retrieved:
            assumptions.append("No explicit citation anchors were detected in answer sentences.")

        reasoning_risks = []
        if not retrieved:
            reasoning_risks.append("No memory evidence retrieved; answer relies on generic expertise.")
        if retrieved and not cited_evidence:
            reasoning_risks.append("Retrieved evidence exists but none was explicitly cited.")
        if len(cited_indices) < max(1, len(sentences) // 3):
            reasoning_risks.append("Citation coverage is sparse relative to answer length.")

        follow_up = []
        if retrieved:
            follow_up.append("Validate strongest claim against the top-ranked source passage.")
        if reasoning_risks:
            follow_up.append("Run targeted retrieval for missing claims and regenerate answer.")

        contract = {
            'question': question,
            'evidence_items': evidence_items,
            'cited_evidence': cited_evidence,
            'assumptions': assumptions,
            'reasoning_risks': reasoning_risks,
            'follow_up_checks': follow_up,
            'citation_coverage': {
                'sentences_total': len(sentences),
                'sentences_with_citation': cited_sentence_count,
                'evidence_items_total': len(evidence_items),
                'evidence_items_cited': len(cited_evidence)
            }
        }

        if retrieved and answer:
            llm_contract = self._llm_contract_review(question, answer, retrieved)
            if llm_contract:
                if isinstance(llm_contract.get('assumptions'), list):
                    contract['assumptions'].extend(
                        [a for a in llm_contract['assumptions'] if isinstance(a, str)]
                    )
                if isinstance(llm_contract.get('reasoning_risks'), list):
                    contract['reasoning_risks'].extend(
                        [r for r in llm_contract['reasoning_risks'] if isinstance(r, str)]
                    )
                if isinstance(llm_contract.get('follow_up_checks'), list):
                    contract['follow_up_checks'].extend(
                        [f for f in llm_contract['follow_up_checks'] if isinstance(f, str)]
                    )

        # Stable dedupe keeps output compact and deterministic.
        for key in ('assumptions', 'reasoning_risks', 'follow_up_checks'):
            deduped = []
            seen = set()
            for item in contract[key]:
                norm = re.sub(r'\s+', ' ', item.strip())
                if not norm:
                    continue
                if norm.lower() in seen:
                    continue
                seen.add(norm.lower())
                deduped.append(norm)
            contract[key] = deduped[:6]
        return contract

    def _llm_contract_review(self, question: str, answer: str,
                             retrieved: List[Tuple[MemCell, float]]) -> Optional[Dict[str, Any]]:
        context = self._format_context(retrieved[:6])
        prompt = f"""Create an evidence contract review for this scientific answer.

Question: {question}
Evidence context:
{context[:1800]}
Answer:
{answer[:1600]}

Return JSON only:
{{
  "assumptions": ["2-4 assumptions in answer"],
  "reasoning_risks": ["2-4 risks or unsupported leaps"],
  "follow_up_checks": ["2-4 concrete verification actions"]
}}"""
        response = self.llm.generate(prompt)
        if not response:
            return None
        parsed = _extract_first_json(response, dict)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _store_qa(self, question: str, answer: str, retrieved: List,
                  evidence_contract: Optional[Dict[str, Any]] = None):
        if not answer:
            return
        self.memory._store_memcell(
            content=f"Q: {question}\nA: {answer[:500]}",
            memory_type=MemoryType.QA_PAIR,
            evidence=[Evidence(content=question, source="conversation",
                               type="direct", confidence=0.7)],
            importance=0.6,
            metadata={
                'contract': {
                    'assumptions': (evidence_contract or {}).get('assumptions', []),
                    'reasoning_risks': (evidence_contract or {}).get('reasoning_risks', []),
                    'follow_up_checks': (evidence_contract or {}).get('follow_up_checks', []),
                    'cited_evidence_count': (evidence_contract or {}).get('citation_coverage', {}).get('evidence_items_cited', 0),
                }
            }
        )
        self.conversation_history.append({
            'question': question, 'answer': answer,
            'timestamp': datetime.now().isoformat()
        })


# ======================== Scientific Explorer ========================

class ScientificExplorer:
    """Deep exploration with evidence-chain hypotheses and epistemic tool policy."""
    EVIDENCE_TYPES = {MemoryType.ATOMIC_FACT, MemoryType.EPISODE}
    TOOL_ACTIONS = (
        "retrieve_evidence",
        "propose_hypotheses",
        "audit_contradictions",
        "design_experiments",
        "recalibrate_beliefs",
    )

    def __init__(self, memory: MemorySystem, llm: LLMProvider):
        self.memory = memory
        self.llm = llm
        self.mode = SYNAPSE_EXPLORER_MODE
        self.lambda_cost = SYNAPSE_EPISTEMIC_LAMBDA_COST
        self.mu_risk = SYNAPSE_EPISTEMIC_MU_RISK
        self.stop_entropy = SYNAPSE_EPISTEMIC_STOP_ENTROPY

    def explore(self, topic: str, depth: int = 3,
                status_callback: Callable = None) -> Dict[str, Any]:
        if self.mode in {"legacy", "classic"}:
            return self._explore_legacy(topic, depth, status_callback)
        return self._explore_epistemic_tools(topic, depth, status_callback)

    def _explore_legacy(self, topic: str, depth: int = 3,
                        status_callback: Callable = None) -> Dict[str, Any]:
        result = {
            'topic': topic, 'start_time': datetime.now(),
            'iterations': [], 'final_synthesis': None,
            'mode': 'legacy'
        }
        accumulated = []

        for i in range(depth):
            if status_callback:
                status_callback(f"Iteration {i+1}/{depth}: Reviewing literature")

            iter_start = time.time()
            ir = self._run_iteration(topic, i+1, depth, accumulated, status_callback)
            ir['duration'] = time.time() - iter_start
            result['iterations'].append(ir)
            accumulated.extend(ir.get('insights', []))

            for insight in ir.get('insights', []):
                self.memory._store_memcell(
                    content=insight, memory_type=MemoryType.FORESIGHT,
                    evidence=[Evidence(
                        content=f"Exploration of {topic}, iteration {i+1}",
                        source="exploration", type="inferred", confidence=0.6)],
                    importance=0.7,
                    metadata={
                        'lifecycle': 'insight',
                        'topic': topic,
                        'iteration': i + 1
                    }
                )
            for exp in ir.get('experiments', [])[:2]:
                hypothesis = exp.get('hypothesis', '')
                plan = exp.get('experiment', '')
                metric = exp.get('measurable_outcome', '')
                content = (
                    f"[Experiment Plan] {hypothesis}\n"
                    f"Plan: {plan}\n"
                    f"Outcome metric: {metric}"
                ).strip()
                self.memory._store_memcell(
                    content=content,
                    memory_type=MemoryType.FORESIGHT,
                    evidence=[Evidence(
                        content=f"Exploration experiment design for {topic}, iteration {i+1}",
                        source="exploration", type="inferred", confidence=0.65)],
                    importance=0.72,
                    metadata={
                        'lifecycle': 'experiment_plan',
                        'topic': topic,
                        'iteration': i + 1,
                        'priority_score': exp.get('priority_score', 0.0),
                        'failure_signal': exp.get('failure_signal', '')
                    }
                )

        if status_callback:
            status_callback("Writing final synthesis")
        result['final_synthesis'] = self._synthesize(topic, result['iterations'])
        result['end_time'] = datetime.now()
        result['total_duration'] = (result['end_time'] - result['start_time']).total_seconds()

        self.memory._store_memcell(
            content=result['final_synthesis'] or f"Exploration of {topic} completed",
            memory_type=MemoryType.EPISODE,
            evidence=[Evidence(content=f"Synthesis from {depth}-iteration exploration",
                               source="exploration", type="inferred", confidence=0.7)],
            importance=0.8,
            metadata={
                'lifecycle': 'synthesis',
                'topic': topic,
                'iterations': depth
            }
        )
        return result

    def _explore_epistemic_tools(self, topic: str, depth: int = 3,
                                 status_callback: Callable = None) -> Dict[str, Any]:
        result = {
            'topic': topic,
            'start_time': datetime.now(),
            'iterations': [],
            'final_synthesis': None,
            'mode': 'epistemic_tools',
            'epistemic_policy': {
                'objective': 'maximize(expected_information_gain - lambda_cost*cost - mu_risk*risk)',
                'lambda_cost': self.lambda_cost,
                'mu_risk': self.mu_risk,
                'stop_entropy': self.stop_entropy,
                'tool_trace': [],
                'total_information_gain': 0.0,
                'final_entropy': 1.0,
                'stop_reason': 'budget_exhausted'
            }
        }
        state = self._init_epistemic_state(topic, depth)
        max_steps = max(4, depth * 4)
        no_gain_steps = 0

        for step in range(max_steps):
            if status_callback:
                status_callback(f"Epistemic policy {step+1}/{max_steps}")

            entropy_before = self._belief_entropy(state['hypotheses'])
            action, expected = self._select_epistemic_tool(state, step, max_steps)
            ir = self._execute_epistemic_tool(
                action=action, topic=topic, state=state, step=step + 1, total=max_steps
            )
            entropy_after = self._belief_entropy(state['hypotheses'])
            realized_gain = max(0.0, entropy_before - entropy_after)
            state['total_information_gain'] += realized_gain
            ir['selected_tool'] = action
            ir['expected_utility'] = expected['utility']
            ir['expected_information_gain'] = expected['expected_information_gain']
            ir['expected_cost'] = expected['cost']
            ir['expected_risk'] = expected['risk']
            ir['entropy_before'] = entropy_before
            ir['entropy_after'] = entropy_after
            ir['realized_information_gain'] = realized_gain
            ir['belief_snapshot'] = self._rank_state_hypotheses(state['hypotheses'])[:3]

            if realized_gain < 0.01:
                no_gain_steps += 1
            else:
                no_gain_steps = 0

            result['iterations'].append(ir)
            result['epistemic_policy']['tool_trace'].append({
                'step': step + 1,
                'action': action,
                'expected_utility': expected['utility'],
                'expected_information_gain': expected['expected_information_gain'],
                'realized_information_gain': realized_gain,
                'entropy_before': entropy_before,
                'entropy_after': entropy_after,
                'reason': expected['reason'],
            })

            if entropy_after <= self.stop_entropy and len(state['experiments']) > 0:
                result['epistemic_policy']['stop_reason'] = 'low_entropy_with_experiment_plan'
                break
            if no_gain_steps >= 2 and step + 1 >= max(3, depth):
                result['epistemic_policy']['stop_reason'] = 'information_gain_plateau'
                break

        result['epistemic_policy']['total_information_gain'] = state['total_information_gain']
        result['epistemic_policy']['final_entropy'] = self._belief_entropy(state['hypotheses'])
        result['epistemic_policy']['final_hypotheses'] = self._rank_state_hypotheses(state['hypotheses'])[:5]
        result['epistemic_policy']['experiments_generated'] = len(state['experiments'])

        if status_callback:
            status_callback("Writing final synthesis")
        result['final_synthesis'] = self._synthesize_epistemic(topic, result['iterations'], state)
        result['end_time'] = datetime.now()
        result['total_duration'] = (result['end_time'] - result['start_time']).total_seconds()

        for step_result in result['iterations']:
            for insight in step_result.get('insights', [])[:3]:
                self.memory._store_memcell(
                    content=insight,
                    memory_type=MemoryType.FORESIGHT,
                    evidence=[Evidence(
                        content=f"Epistemic tool policy exploration of {topic}",
                        source="epistemic_exploration", type="inferred", confidence=0.65)],
                    importance=0.72,
                    metadata={
                        'lifecycle': 'insight',
                        'topic': topic,
                        'mode': 'epistemic_tools',
                        'tool': step_result.get('selected_tool')
                    }
                )
            for exp in step_result.get('experiments', [])[:2]:
                content = (
                    f"[Epistemic Experiment Plan] {exp.get('hypothesis', '')}\n"
                    f"Plan: {exp.get('experiment', '')}\n"
                    f"Outcome metric: {exp.get('measurable_outcome', '')}"
                ).strip()
                self.memory._store_memcell(
                    content=content,
                    memory_type=MemoryType.FORESIGHT,
                    evidence=[Evidence(
                        content=f"Epistemic tool policy experiment plan for {topic}",
                        source="epistemic_exploration", type="inferred", confidence=0.7)],
                    importance=0.75,
                    metadata={
                        'lifecycle': 'experiment_plan',
                        'topic': topic,
                        'mode': 'epistemic_tools',
                        'tool': step_result.get('selected_tool'),
                        'priority_score': exp.get('priority_score', 0.0),
                        'failure_signal': exp.get('failure_signal', '')
                    }
                )

        self.memory._store_memcell(
            content=result['final_synthesis'] or f"Epistemic exploration of {topic} completed",
            memory_type=MemoryType.EPISODE,
            evidence=[Evidence(
                content=f"Epistemic synthesis for {topic} with stop reason {result['epistemic_policy']['stop_reason']}",
                source="epistemic_exploration", type="inferred", confidence=0.75)],
            importance=0.82,
            metadata={
                'lifecycle': 'synthesis',
                'topic': topic,
                'mode': 'epistemic_tools',
                'total_information_gain': result['epistemic_policy']['total_information_gain'],
                'final_entropy': result['epistemic_policy']['final_entropy'],
                'stop_reason': result['epistemic_policy']['stop_reason']
            }
        )
        return result

    def _init_epistemic_state(self, topic: str, depth: int) -> Dict[str, Any]:
        retrieved = self.memory.retrieve(
            topic,
            top_k=max(10, depth * 5),
            strategy="agentic",
            allowed_types=self.EVIDENCE_TYPES
        )
        knowledge = "\n".join([f"[{i}] {mem.content}" for i, (mem, _) in enumerate(retrieved[:12])])
        hyp_prompt = f"""Initialize a hypothesis space for scientific exploration of "{topic}".
Use the evidence to generate 3-5 candidate hypotheses.

Evidence:
{knowledge if knowledge else "No retrieved evidence. Use cautious priors."}

Return JSON array:
[{{"hypothesis":"...", "rationale":"...", "prior_belief":0.0, "uncertainty":0.0, "testability_score":0.0}}]
Rules:
- prior_belief in [0,1], uncertainty in [0,1]
- uncertainty should be high when evidence is weak
- hypotheses must be falsifiable
Return ONLY JSON array."""
        hypotheses: List[Dict[str, Any]] = []
        response = self.llm.generate(hyp_prompt)
        if response:
            parsed = _extract_first_json(response, list)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    added = self._add_state_hypothesis(hypotheses, item)
                    if not added:
                        continue

        if not hypotheses:
            for i, (mem, _) in enumerate(retrieved[:3], start=1):
                self._add_state_hypothesis(hypotheses, {
                    "hypothesis": f"H{i}: {mem.content[:180]}",
                    "rationale": "Initialized from retrieved evidence",
                    "prior_belief": 0.45,
                    "uncertainty": 0.65,
                    "testability_score": 0.5
                })

        if not hypotheses:
            self._add_state_hypothesis(hypotheses, {
                "hypothesis": f"Mechanistic drivers of {topic} remain under-specified.",
                "rationale": "Fallback prior due to sparse evidence",
                "prior_belief": 0.4,
                "uncertainty": 0.75,
                "testability_score": 0.4
            })

        return {
            'hypotheses': hypotheses,
            'retrieved': retrieved,
            'knowledge': knowledge,
            'experiments': [],
            'action_history': [],
            'total_information_gain': 0.0,
            'last_gaps': []
        }

    def _select_epistemic_tool(self, state: Dict[str, Any], step: int, max_steps: int) -> Tuple[str, Dict[str, Any]]:
        if len(state['hypotheses']) < 2:
            forced = self._estimate_tool_utility(state, "propose_hypotheses", step, max_steps)
            forced['reason'] = "Hypothesis space too small; force expansion."
            return "propose_hypotheses", forced

        if step == 0 and not state.get('knowledge'):
            forced = self._estimate_tool_utility(state, "retrieve_evidence", step, max_steps)
            forced['reason'] = "No starting evidence context; force retrieval."
            return "retrieve_evidence", forced

        scored = {
            action: self._estimate_tool_utility(state, action, step, max_steps)
            for action in self.TOOL_ACTIONS
        }
        best_action = max(scored.items(), key=lambda kv: kv[1]['utility'])[0]
        return best_action, scored[best_action]

    def _estimate_tool_utility(self, state: Dict[str, Any], action: str,
                               step: int, max_steps: int) -> Dict[str, Any]:
        hypotheses = state.get('hypotheses', [])
        entropy = self._belief_entropy(hypotheses)
        if hypotheses:
            uncertainty_values = [_safe_float(h.get('uncertainty', 0.5), 0.5) for h in hypotheses]
            avg_uncertainty = sum(uncertainty_values) / max(len(uncertainty_values), 1)
        else:
            avg_uncertainty = 0.8
        max_belief = max([h.get('belief', 0.5) for h in hypotheses], default=0.5)
        num_experiments = len(state.get('experiments', []))
        progress = (step + 1) / max(max_steps, 1)

        if action == "retrieve_evidence":
            expected_gain = 0.25 + 0.45 * avg_uncertainty + 0.20 * entropy
            cost = 0.35
            risk = 0.10
            reason = "Evidence retrieval lowers epistemic uncertainty and supports posterior updates."
        elif action == "propose_hypotheses":
            expected_gain = 0.40 if len(hypotheses) < 4 else 0.18
            expected_gain += 0.20 * entropy
            cost = 0.28
            risk = 0.22
            reason = "Hypothesis expansion increases search breadth when belief mass is diffuse."
        elif action == "audit_contradictions":
            expected_gain = 0.20 + 0.35 * max_belief + 0.15 * avg_uncertainty
            cost = 0.30
            risk = 0.08
            reason = "Contradiction audits prevent overconfident belief collapse."
        elif action == "design_experiments":
            expected_gain = 0.08 + 0.20 * (1.0 - entropy) + (0.15 if num_experiments == 0 else 0.05)
            expected_gain += 0.10 * progress
            cost = 0.20
            risk = 0.05
            reason = "Experiment design converts uncertain beliefs into testable next actions."
        else:
            expected_gain = 0.15 + 0.35 * entropy
            cost = 0.24
            risk = 0.12
            reason = "Belief recalibration improves posterior calibration and stopping confidence."

        utility = expected_gain - self.lambda_cost * cost - self.mu_risk * risk
        return {
            'expected_information_gain': expected_gain,
            'cost': cost,
            'risk': risk,
            'utility': utility,
            'reason': reason
        }

    def _execute_epistemic_tool(self, action: str, topic: str,
                                state: Dict[str, Any], step: int,
                                total: int) -> Dict[str, Any]:
        base = {
            'iteration': step,
            'gaps': [],
            'hypotheses': [],
            'ranked_hypotheses': [],
            'experiments': [],
            'insights': [],
            'feedback': None
        }
        if action == "retrieve_evidence":
            base.update(self._tool_retrieve_evidence(topic, state, step, total))
        elif action == "propose_hypotheses":
            base.update(self._tool_propose_hypotheses(topic, state, step, total))
        elif action == "audit_contradictions":
            base.update(self._tool_audit_contradictions(topic, state, step, total))
        elif action == "design_experiments":
            base.update(self._tool_design_experiments(topic, state, step, total))
        else:
            base.update(self._tool_recalibrate_beliefs(topic, state, step, total))
        state['action_history'].append(action)
        return base

    def _tool_retrieve_evidence(self, topic: str, state: Dict[str, Any],
                                step: int, total: int) -> Dict[str, Any]:
        ranked = sorted(state['hypotheses'], key=lambda h: h.get('uncertainty', 0.0), reverse=True)
        query_focus = " ".join([h.get('hypothesis', '')[:90] for h in ranked[:2]])
        query = f"{topic} {query_focus}".strip()
        retrieved = self.memory.retrieve(
            query,
            top_k=15,
            strategy="agentic",
            allowed_types=self.EVIDENCE_TYPES
        )
        knowledge = "\n".join([f"[{i}] {mem.content}" for i, (mem, _) in enumerate(retrieved[:12])])
        state['retrieved'] = retrieved
        state['knowledge'] = knowledge

        prompt = f"""Update hypothesis beliefs using new evidence for topic "{topic}".
Step {step}/{total}.

Hypotheses:
{json.dumps(state['hypotheses'], ensure_ascii=True)}

Evidence:
{knowledge if knowledge else "No new evidence retrieved."}

Return JSON:
{{
  "updates":[{{"hypothesis":"...", "support_delta":0.0, "uncertainty_delta":0.0, "evidence_note":"..."}}],
  "gaps":["..."],
  "insights":["..."]
}}
Rules:
- support_delta in [-0.25, 0.25]
- uncertainty_delta in [-0.2, 0.2]
- be conservative; avoid extreme changes
Return ONLY JSON."""
        updates, gaps, insights = [], [], []
        response = self.llm.generate(prompt)
        if response:
            parsed = _extract_first_json(response, dict)
            if isinstance(parsed, dict):
                updates = [u for u in parsed.get('updates', []) if isinstance(u, dict)]
                gaps = [g for g in parsed.get('gaps', []) if isinstance(g, str)]
                insights = [i for i in parsed.get('insights', []) if isinstance(i, str)]
        self._apply_hypothesis_updates(state['hypotheses'], updates)
        state['last_gaps'] = gaps

        if not insights and retrieved:
            insights = [f"Retrieved {len(retrieved)} evidence items to refine posterior beliefs."]
        return {
            'gaps': gaps[:3],
            'insights': insights[:3],
            'feedback': {'retrieved_items': len(retrieved)}
        }

    def _tool_propose_hypotheses(self, topic: str, state: Dict[str, Any],
                                 step: int, total: int) -> Dict[str, Any]:
        prompt = f"""Propose 1-3 new falsifiable hypotheses for "{topic}".
Avoid duplicates with existing hypotheses.

Existing hypotheses:
{json.dumps(state['hypotheses'], ensure_ascii=True)}
Known gaps:
{json.dumps(state.get('last_gaps', []), ensure_ascii=True)}

Return JSON array:
[{{"hypothesis":"...", "rationale":"...", "prior_belief":0.0, "uncertainty":0.0, "testability_score":0.0}}]
Return ONLY JSON array."""
        response = self.llm.generate(prompt)
        new_hypotheses: List[Dict[str, Any]] = []
        if response:
            parsed = _extract_first_json(response, list)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    if self._add_state_hypothesis(state['hypotheses'], item):
                        new_hypotheses.append(item)
        insights = []
        if new_hypotheses:
            insights.append(f"Expanded hypothesis space with {len(new_hypotheses)} candidate mechanisms.")
        return {
            'hypotheses': new_hypotheses[:3],
            'insights': insights[:3]
        }

    def _tool_audit_contradictions(self, topic: str, state: Dict[str, Any],
                                   step: int, total: int) -> Dict[str, Any]:
        contradiction_query = f"{topic} contradiction conflict inconsistent"
        retrieved = self.memory.retrieve(
            contradiction_query,
            top_k=10,
            strategy="hybrid",
            allowed_types=None
        )
        contradiction_notes = []
        for mem, _ in retrieved:
            mem_type = getattr(mem.memory_type, "value", "")
            if mem_type == MemoryType.CONTRADICTION.value:
                contradiction_notes.append(mem.content)
        contradiction_text = "\n".join([f"- {c}" for c in contradiction_notes[:6]])
        prompt = f"""Audit current hypotheses for contradictions in topic "{topic}".

Hypotheses:
{json.dumps(state['hypotheses'], ensure_ascii=True)}
Contradiction evidence:
{contradiction_text if contradiction_text else "No explicit contradiction memcells found."}

Return JSON:
{{
  "updates":[{{"hypothesis":"...", "support_delta":0.0, "uncertainty_delta":0.0, "evidence_note":"..."}}],
  "weaknesses":["..."],
  "insights":["..."]
}}
Rules:
- Penalize overconfident hypotheses when contradiction signals exist.
- support_delta in [-0.25, 0.10], uncertainty_delta in [-0.05, 0.20]
Return ONLY JSON."""
        updates, weaknesses, insights = [], [], []
        response = self.llm.generate(prompt)
        if response:
            parsed = _extract_first_json(response, dict)
            if isinstance(parsed, dict):
                updates = [u for u in parsed.get('updates', []) if isinstance(u, dict)]
                weaknesses = [w for w in parsed.get('weaknesses', []) if isinstance(w, str)]
                insights = [i for i in parsed.get('insights', []) if isinstance(i, str)]
        self._apply_hypothesis_updates(state['hypotheses'], updates)
        if not insights and contradiction_notes:
            insights = [f"Detected {len(contradiction_notes)} contradiction signals to down-weight brittle hypotheses."]
        return {
            'insights': insights[:3],
            'feedback': {'weaknesses': weaknesses[:4], 'contradiction_signals': len(contradiction_notes)}
        }

    def _tool_design_experiments(self, topic: str, state: Dict[str, Any],
                                 step: int, total: int) -> Dict[str, Any]:
        ranked = self._rank_state_hypotheses(state['hypotheses'])[:4]
        experiments = self._design_experiments(topic, ranked)
        if experiments:
            state['experiments'].extend(experiments[:3])
            for exp in experiments[:3]:
                idx = self._find_hypothesis_index(state['hypotheses'], exp.get('hypothesis', ''))
                if idx is not None:
                    h = state['hypotheses'][idx]
                    h['uncertainty'] = self._clip(h.get('uncertainty', 0.5) - 0.05, 0.01, 0.99)
        insights = []
        if experiments:
            insights = [f"Designed {len(experiments[:3])} minimum-viable experiments to maximize falsifiability."]
        return {
            'ranked_hypotheses': ranked[:3],
            'experiments': experiments[:3],
            'insights': insights[:2]
        }

    def _tool_recalibrate_beliefs(self, topic: str, state: Dict[str, Any],
                                  step: int, total: int) -> Dict[str, Any]:
        prompt = f"""Recalibrate posterior belief and uncertainty for hypotheses on "{topic}".
Use conservative Bayesian-style judgment from current evidence.

Hypotheses:
{json.dumps(state['hypotheses'], ensure_ascii=True)}

Return JSON array:
[{{"hypothesis":"...", "belief":0.0, "uncertainty":0.0, "rationale":"..."}}]
Rules:
- belief and uncertainty must remain in [0,1]
- avoid collapsing all belief mass to one hypothesis without strong evidence
Return ONLY JSON array."""
        response = self.llm.generate(prompt)
        insights = []
        if response:
            parsed = _extract_first_json(response, list)
            if isinstance(parsed, list):
                updates = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    updates.append({
                        'hypothesis': item.get('hypothesis', ''),
                        'support_delta': self._clip(_safe_float(item.get('belief', 0.5), 0.5), 0.0, 1.0),
                        'uncertainty_delta': self._clip(_safe_float(item.get('uncertainty', 0.5), 0.5), 0.0, 1.0),
                        'absolute': True
                    })
                self._apply_hypothesis_updates(state['hypotheses'], updates, absolute=True)
        ranked = self._rank_state_hypotheses(state['hypotheses'])
        if ranked:
            strongest = ranked[0].get('hypothesis', '')[:130]
            insights.append(f"Posterior recalibrated; strongest current hypothesis: {strongest}")
        return {
            'ranked_hypotheses': ranked[:3],
            'insights': insights[:2]
        }

    def _rank_state_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []
        for item in hypotheses:
            belief = self._clip(_safe_float(item.get('belief', 0.5), 0.5), 0.0, 1.0)
            uncertainty = self._clip(_safe_float(item.get('uncertainty', 0.5), 0.5), 0.0, 1.0)
            testability = self._clip(_safe_float(item.get('testability_score', 0.5), 0.5), 0.0, 1.0)
            priority = belief * (1.0 - uncertainty) * (0.5 + 0.5 * testability)
            ranked.append({
                **item,
                'belief': belief,
                'uncertainty': uncertainty,
                'testability_score': testability,
                'priority_score': priority,
                'status': item.get('status', 'ranked')
            })
        ranked.sort(key=lambda x: x.get('priority_score', 0.0), reverse=True)
        return ranked

    def _apply_hypothesis_updates(self, hypotheses: List[Dict[str, Any]],
                                  updates: List[Dict[str, Any]],
                                  absolute: bool = False):
        for upd in updates:
            idx = self._find_hypothesis_index(hypotheses, upd.get('hypothesis', ''))
            if idx is None:
                continue
            h = hypotheses[idx]
            if absolute or upd.get('absolute'):
                h['belief'] = self._clip(_safe_float(upd.get('support_delta', h.get('belief', 0.5)), h.get('belief', 0.5)), 0.0, 1.0)
                h['uncertainty'] = self._clip(_safe_float(upd.get('uncertainty_delta', h.get('uncertainty', 0.5)), h.get('uncertainty', 0.5)), 0.0, 1.0)
            else:
                h['belief'] = self._clip(h.get('belief', 0.5) + _safe_float(upd.get('support_delta', 0.0), 0.0), 0.0, 1.0)
                h['uncertainty'] = self._clip(h.get('uncertainty', 0.5) + _safe_float(upd.get('uncertainty_delta', 0.0), 0.0), 0.0, 1.0)
            note = upd.get('evidence_note')
            if note and isinstance(note, str):
                h['latest_update_note'] = note[:240]

    def _add_state_hypothesis(self, hypotheses: List[Dict[str, Any]], item: Dict[str, Any]) -> bool:
        hypothesis_text = str(item.get('hypothesis', '')).strip()
        if len(hypothesis_text) < 10:
            return False
        if self._find_hypothesis_index(hypotheses, hypothesis_text) is not None:
            return False
        entry = {
            'id': f"hyp_{len(hypotheses) + 1}",
            'hypothesis': hypothesis_text,
            'rationale': str(item.get('rationale', '')),
            'belief': self._clip(_safe_float(item.get('prior_belief', item.get('belief', 0.45)), 0.45), 0.0, 1.0),
            'uncertainty': self._clip(_safe_float(item.get('uncertainty', 0.65), 0.65), 0.0, 1.0),
            'testability_score': self._clip(_safe_float(item.get('testability_score', 0.5), 0.5), 0.0, 1.0),
            'status': item.get('status', 'active')
        }
        hypotheses.append(entry)
        return True

    def _find_hypothesis_index(self, hypotheses: List[Dict[str, Any]], text: str) -> Optional[int]:
        needle = re.sub(r'[^a-z0-9 ]+', ' ', str(text).lower()).strip()
        if not needle:
            return None
        needle_tokens = set(needle.split())
        best_idx, best_score = None, 0.0
        for idx, item in enumerate(hypotheses):
            cand = re.sub(r'[^a-z0-9 ]+', ' ', str(item.get('hypothesis', '')).lower()).strip()
            if not cand:
                continue
            if cand == needle or needle in cand or cand in needle:
                return idx
            cand_tokens = set(cand.split())
            overlap = len(needle_tokens & cand_tokens)
            denom = max(len(needle_tokens), 1)
            score = overlap / denom
            if score > best_score:
                best_idx, best_score = idx, score
        if best_score >= 0.5:
            return best_idx
        return None

    @staticmethod
    def _belief_entropy(hypotheses: List[Dict[str, Any]]) -> float:
        if not hypotheses:
            return 1.0
        weights = [max(1e-6, _safe_float(h.get('belief', 0.0), 0.0)) for h in hypotheses]
        total = sum(weights)
        if total <= 0:
            return 1.0
        probs = [w / total for w in weights]
        n = len(probs)
        if n <= 1:
            return 0.0
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0) / math.log(n)
        return float(max(0.0, min(1.0, entropy)))

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _synthesize_epistemic(self, topic: str, iterations: List[Dict[str, Any]],
                              state: Dict[str, Any]) -> Optional[str]:
        ranked = self._rank_state_hypotheses(state.get('hypotheses', []))[:5]
        tool_trace = [
            {
                'tool': it.get('selected_tool'),
                'expected_information_gain': it.get('expected_information_gain', 0.0),
                'realized_information_gain': it.get('realized_information_gain', 0.0),
                'entropy_after': it.get('entropy_after', 1.0),
            }
            for it in iterations
        ]
        prompt = f"""Write a scientific synthesis for topic "{topic}" using an epistemic tool policy trajectory.

Top posterior hypotheses:
{json.dumps(ranked, ensure_ascii=True)}

Tool trajectory:
{json.dumps(tool_trace[:10], ensure_ascii=True)}

Generated experiments:
{json.dumps(state.get('experiments', [])[:5], ensure_ascii=True)}

Write 4 concise paragraphs:
1) Posterior state of knowledge and remaining uncertainty
2) Highest-priority hypotheses and confidence rationale
3) Contradictions/risks that still threaten validity
4) Next experimental actions with measurable outcomes

Write as a principal investigator preparing a lab decision memo."""
        synthesis = self.llm.generate(prompt)
        if synthesis:
            return synthesis
        return (
            f"Epistemic exploration of {topic} completed. "
            f"Final entropy={self._belief_entropy(state.get('hypotheses', [])):.3f}. "
            f"Generated {len(state.get('experiments', []))} experiments."
        )

    def _run_iteration(self, topic: str, iteration: int, total: int,
                       accumulated: List[str],
                       status_callback: Callable = None) -> Dict[str, Any]:
        ir = {
            'iteration': iteration,
            'gaps': [],
            'hypotheses': [],
            'ranked_hypotheses': [],
            'experiments': [],
            'insights': [],
            'feedback': None
        }

        # Phase 1: Literature review
        if status_callback:
            status_callback(f"Iteration {iteration}/{total}: Reviewing literature")
        retrieved = self.memory.retrieve(
            topic,
            top_k=15,
            strategy="hybrid",
            allowed_types=self.EVIDENCE_TYPES
        )
        retrieved = [
            (mem, score) for mem, score in retrieved
            if mem.metadata.get('lifecycle') not in {'synthesis', 'experiment_plan'}
        ]
        knowledge = "\n".join([f"[{i}] {mem.content}" for i, (mem, _) in enumerate(retrieved)])
        acc_text = "\n".join([f"- {a}" for a in accumulated[-5:]])

        # Phase 2: Gap analysis
        if status_callback:
            status_callback(f"Iteration {iteration}/{total}: Analyzing gaps")

        gaps_response = self.llm.generate(f"""You are a scientist reviewing "{topic}".
Research materials: {knowledge if knowledge else "General expertise only."}
Previous insights: {acc_text if acc_text else "First iteration."}
Iteration {iteration}/{total}.
What are the 2-3 most critical unanswered questions?
Return as JSON array of strings. Return ONLY JSON array.""")

        if gaps_response:
            parsed_gaps = _extract_first_json(gaps_response, list)
            if isinstance(parsed_gaps, list):
                ir['gaps'] = [g for g in parsed_gaps if isinstance(g, str)]
            else:
                ir['gaps'] = [gaps_response[:200]]

        # Phase 3: Evidence-chain hypotheses
        if status_callback:
            status_callback(f"Iteration {iteration}/{total}: Generating hypotheses")

        hyp_response = self.llm.generate(f"""Generate 2-3 hypotheses about "{topic}".

IMPORTANT: Each hypothesis must cite specific evidence from the research materials using [N] numbers.
Build evidence chains: "Based on [0] which shows X, and [3] which shows Y, I hypothesize that Z."

Research materials: {knowledge[:2000] if knowledge else "General knowledge only."}
Knowledge gaps: {json.dumps(ir['gaps'])}

Return JSON array: [{{"hypothesis": "...", "evidence_chain": "Based on [N]... and [M]...", "rationale": "...", "testable": true/false, "key_prediction": "..."}}]
Return ONLY JSON.""")

        if hyp_response:
            parsed_hyp = _extract_first_json(hyp_response, list)
            if isinstance(parsed_hyp, list):
                ir['hypotheses'] = [h for h in parsed_hyp if isinstance(h, dict)]

        # Phase 4: Critical evaluation
        if ir['hypotheses']:
            if status_callback:
                status_callback(f"Iteration {iteration}/{total}: Critical evaluation")

            fb_response = self.llm.generate(f"""Critically evaluate these hypotheses about "{topic}":
{json.dumps(ir['hypotheses'], indent=2)}

For each: evidence strength (strong/moderate/weak), key assumptions that could be wrong,
alternative explanations, what experiment would test it.

Return JSON: {{"evaluation": "...", "strongest": "...", "weaknesses": ["..."], "suggested_experiments": ["..."]}}
Return ONLY JSON.""")

            if fb_response:
                parsed_fb = _extract_first_json(fb_response, dict)
                if isinstance(parsed_fb, dict):
                    ir['feedback'] = parsed_fb

        # Phase 5: Hypothesis lifecycle scoring
        if ir['hypotheses']:
            if status_callback:
                status_callback(f"Iteration {iteration}/{total}: Scoring hypotheses")
            ir['ranked_hypotheses'] = self._score_hypotheses(
                topic=topic,
                hypotheses=ir['hypotheses'],
                knowledge=knowledge
            )

        # Phase 6: Design minimum viable experiments
        if ir['ranked_hypotheses']:
            if status_callback:
                status_callback(f"Iteration {iteration}/{total}: Designing experiments")
            ir['experiments'] = self._design_experiments(
                topic=topic,
                ranked_hypotheses=ir['ranked_hypotheses']
            )

        # Phase 7: Synthesize
        if status_callback:
            status_callback(f"Iteration {iteration}/{total}: Synthesizing")

        syn_response = self.llm.generate(f"""Based on this exploration iteration of "{topic}":
Gaps: {json.dumps(ir['gaps'])}
Hypotheses: {json.dumps(ir['hypotheses'][:2])}
Ranked hypotheses: {json.dumps(ir['ranked_hypotheses'][:2])}
Experiment plans: {json.dumps(ir['experiments'][:2])}
Evaluation: {json.dumps(ir.get('feedback', {}))}
What are the 2-3 most important insights?
Return as JSON array of strings. Return ONLY JSON array.""")

        if syn_response:
            parsed_syn = _extract_first_json(syn_response, list)
            if isinstance(parsed_syn, list):
                ir['insights'] = [i for i in parsed_syn if isinstance(i, str)]
            else:
                ir['insights'] = [syn_response[:200]]

        return ir

    def _score_hypotheses(self, topic: str, hypotheses: List[Dict[str, Any]],
                          knowledge: str) -> List[Dict[str, Any]]:
        if not hypotheses:
            return []

        prompt = f"""Score hypotheses for scientific discovery planning.
For each hypothesis, assign scores in [0,1]:
- testability_score: can it be tested with concrete measurements?
- novelty_score: non-obvious contribution relative to current knowledge.
- falsifiability_score: can it be disproved clearly?
- priority_score: overall priority for next experiments.

Topic: {topic}
Knowledge: {knowledge[:1500]}
Hypotheses: {json.dumps(hypotheses[:5], ensure_ascii=True)}

Return JSON array:
[{{"hypothesis":"...", "testability_score":0.0, "novelty_score":0.0, "falsifiability_score":0.0, "priority_score":0.0, "status":"ranked"}}]
Return ONLY JSON array."""
        response = self.llm.generate(prompt)
        if response:
            items = _extract_first_json(response, list)
            if isinstance(items, list):
                cleaned = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    cleaned.append({
                        **item,
                        'status': 'ranked',
                        'testability_score': _safe_float(item.get('testability_score', 0.0)),
                        'novelty_score': _safe_float(item.get('novelty_score', 0.0)),
                        'falsifiability_score': _safe_float(item.get('falsifiability_score', 0.0)),
                        'priority_score': _safe_float(item.get('priority_score', 0.0))
                    })
                cleaned.sort(key=lambda x: x.get('priority_score', 0.0), reverse=True)
                return cleaned[:4]
        return []

    def _design_experiments(self, topic: str,
                            ranked_hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ranked_hypotheses:
            return []
        top = ranked_hypotheses[:3]

        prompt = f"""Design minimum viable experiments for top-ranked hypotheses on "{topic}".
Each experiment must include measurable outcome and failure signal.

Hypotheses:
{json.dumps(top, ensure_ascii=True)}

Return JSON array:
[{{
  "hypothesis":"...",
  "experiment":"stepwise minimal experiment design",
  "measurable_outcome":"quantitative metric",
  "failure_signal":"what falsifies the hypothesis",
  "resources":"minimal resources required",
  "priority_score":0.0,
  "status":"experiment_designed"
}}]
Return ONLY JSON array."""
        response = self.llm.generate(prompt)
        if response:
            data = _extract_first_json(response, list)
            if isinstance(data, list):
                experiments = []
                for item in data:
                    if isinstance(item, dict):
                        experiments.append({
                            'hypothesis': item.get('hypothesis', ''),
                            'experiment': item.get('experiment', ''),
                            'measurable_outcome': item.get('measurable_outcome', ''),
                            'failure_signal': item.get('failure_signal', ''),
                            'resources': item.get('resources', ''),
                            'priority_score': _safe_float(item.get('priority_score', 0.0)),
                            'status': 'experiment_designed'
                        })
                if experiments:
                    return experiments[:3]
        return []

    def _synthesize(self, topic: str, iterations: List[Dict]) -> Optional[str]:
        all_insights, all_hyps, all_gaps, all_ranked, all_experiments = [], [], [], [], []
        for it in iterations:
            all_insights.extend(it.get('insights', []))
            all_hyps.extend(it.get('hypotheses', []))
            all_gaps.extend(it.get('gaps', []))
            all_ranked.extend(it.get('ranked_hypotheses', []))
            all_experiments.extend(it.get('experiments', []))

        return self.llm.generate(f"""Write a scientific synthesis of exploring "{topic}" ({len(iterations)} iterations).

Insights: {json.dumps(all_insights)}
Hypotheses: {json.dumps(all_hyps[:6])}
Ranked hypotheses: {json.dumps(all_ranked[:6])}
Experiment plans: {json.dumps(all_experiments[:6])}
Gaps: {json.dumps(all_gaps[:6])}

Write 3-4 paragraphs:
1. Current state of knowledge
2. Strongest hypotheses with evidence
3. High-priority experiments and failure criteria
4. Critical remaining gaps and next steps

Write as a scientist at a research meeting — authoritative, precise, forward-looking.""")


# ======================== Main Brain ========================

class SynapseBrain:
    def __init__(self):
        print("\n  Initializing Synapse Brain...")
        self.llm = LLMProvider(api_key=GEMINI_API_KEY)
        self.embedder = EmbeddingProvider(api_key=GEMINI_API_KEY)

        # Storage directory
        self.memory_dir = os.path.join(os.getcwd(), MEMORY_DIR)
        os.makedirs(self.memory_dir, exist_ok=True)
        self._init_run_logging()

        # Create MemorySystem with SQLite + FAISS backend
        self.memory = MemorySystem(self.llm, self.embedder,
                                   storage_dir=self.memory_dir)

        # Auto-migrate from JSON if needed
        db_path = os.path.join(self.memory_dir, 'synapse.db')
        json_path = os.path.join(self.memory_dir, 'memcells.json')
        db_has_data = (os.path.exists(db_path) and
                       len(self.memory.memcells) > 0)
        if not db_has_data and os.path.exists(json_path):
            self.memory.migrate_from_json(self.memory_dir)
        elif db_has_data:
            stats = self.memory.stats
            print(f"  Memory loaded: {stats['total_memcells']} facts, "
                  f"{stats['total_episodes']} episodes, "
                  f"{stats['total_connections']} connections")

        if SYNAPSE_AUTO_BACKFILL_EMBEDDINGS and self.embedder.available:
            backfill = self.memory.backfill_vectors(max_items=5000)
            if backfill.get("embedded", 0) > 0:
                print(
                    f"  Vector backfill: embedded {backfill['embedded']} "
                    f"(failed {backfill['failed']})"
                )

        self.qa = QAEngine(self.memory, self.llm)
        self.explorer = ScientificExplorer(self.memory, self.llm)
        print("  Brain ready.\n")

    def _init_run_logging(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.memory_dir, "runs")
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_log_path = os.path.join(self.run_dir, f"run_{self.run_id}.jsonl")

    def _log_event(self, event_type: str, payload: Dict[str, Any]):
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "model": self.llm.model_name,
            "llm_available": self.llm.is_available,
            "run_id": self.run_id,
            "payload": payload,
        }
        try:
            with open(self.run_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=True) + "\n")
        except Exception as e:
            print(f"  Warning: Failed to write run log ({e})")

    def _auto_save(self):
        """Save FAISS index. SQLite is already durable via write-through."""
        self.memory.save_vectors()

    def _ensure_services_available(self) -> Optional[Dict[str, Any]]:
        if not self.llm.is_available:
            return {'error': LLM_REQUIRED_ERROR}
        if not self.embedder.available:
            return {'error': EMBEDDINGS_REQUIRED_ERROR + _detail_suffix(self.embedder.error_message)}
        return None

    def upload(self, doc_path: str,
               status_callback: Callable = None) -> Dict[str, Any]:
        service_err = self._ensure_services_available()
        if service_err:
            self._log_event("upload_error", {"doc_path": doc_path, "error": service_err["error"]})
            return service_err
        try:
            result = self.memory.extract_from_document(doc_path, status_callback)
        except Exception as e:
            err = str(e)
            self._log_event("upload_error", {"doc_path": doc_path, "error": err})
            return {'error': err}
        self._log_event("upload", {
            "doc_path": doc_path,
            "memcells_created": result.get("memcells_created", 0),
            "episodes_created": result.get("episodes_created", 0),
            "profiles_updated": result.get("profiles_updated", 0),
            "topics": result.get("topics", []),
            "error": result.get("error")
        })
        if 'error' not in result:
            if status_callback:
                status_callback("Saving to long-term memory")
            self._auto_save()
        return result

    def ask(self, question: str,
            status_callback: Callable = None) -> Dict[str, Any]:
        service_err = self._ensure_services_available()
        if service_err:
            self._log_event("ask_error", {"question": question, "error": service_err["error"]})
            return service_err
        try:
            result = self.qa.ask(question, status_callback)
        except Exception as e:
            err = str(e)
            self._log_event("ask_error", {"question": question, "error": err})
            return {'error': err}
        self._log_event("ask", {
            "question": question,
            "mode": result.get("mode"),
            "memories_used": result.get("memories_used", 0),
            "elapsed": result.get("elapsed", 0.0),
            "sources": result.get("sources", []),
            "citation_coverage": (result.get("evidence_contract") or {}).get("citation_coverage", {}),
            "error": result.get("error")
        })
        self._auto_save()
        return result

    def explore(self, topic: str, depth: int = 3,
                status_callback: Callable = None) -> Dict[str, Any]:
        service_err = self._ensure_services_available()
        if service_err:
            self._log_event("explore_error", {"topic": topic, "depth": depth, "error": service_err["error"]})
            return service_err
        try:
            result = self.explorer.explore(topic, depth, status_callback)
        except Exception as e:
            err = str(e)
            self._log_event("explore_error", {"topic": topic, "depth": depth, "error": err})
            return {'error': err}
        first_iter = (result.get("iterations") or [{}])[0] if result.get("iterations") else {}
        self._log_event("explore", {
            "topic": topic,
            "depth": depth,
            "mode": result.get("mode"),
            "iterations": len(result.get("iterations", [])),
            "duration": result.get("total_duration", 0.0),
            "ranked_hypotheses_count": len(first_iter.get("ranked_hypotheses", [])),
            "experiments_count": len(first_iter.get("experiments", [])),
            "epistemic_information_gain": (result.get("epistemic_policy") or {}).get("total_information_gain"),
            "epistemic_final_entropy": (result.get("epistemic_policy") or {}).get("final_entropy"),
            "epistemic_stop_reason": (result.get("epistemic_policy") or {}).get("stop_reason"),
            "error": result.get("error")
        })
        self._auto_save()
        return result

    def trace(self, query: str) -> Dict[str, Any]:
        service_err = self._ensure_services_available()
        if service_err:
            self._log_event("trace_error", {"query": query, "error": service_err["error"]})
            return {'found': False, 'message': service_err["error"]}
        try:
            return self.memory.trace(query)
        except Exception as e:
            err = str(e)
            self._log_event("trace_error", {"query": query, "error": err})
            return {'found': False, 'message': err}

    def get_stats(self) -> Dict[str, Any]:
        has_faiss_backend = bool(
            self.memory.faiss_vectors and self.memory.faiss_vectors.index is not None
        )
        stats = {
            'memory': self.memory.stats,
            'llm_available': self.llm.is_available,
            'embeddings_available': self.embedder.available,
            'llm_calls': self.llm.call_count,
            'profiles': list(self.memory.profiles.keys()),
            'episodes': len(self.memory.episodes),
            'conversation_length': len(self.qa.conversation_history),
            'vector_backend': 'faiss' if has_faiss_backend else 'in-memory',
            'storage': (
                'sqlite+faiss' if self.memory.db and has_faiss_backend
                else 'sqlite+in-memory' if self.memory.db
                else 'memory-only'
            ),
        }
        if has_faiss_backend and self.memory.faiss_vectors:
            stats['faiss_vectors'] = self.memory.faiss_vectors.size
        stats['in_memory_vectors'] = len(self.memory.vectors.embeddings)
        return stats

    def preflight_check(self) -> Dict[str, Any]:
        run_log_parent = os.path.dirname(self.run_log_path)
        checks: Dict[str, Any] = {
            'llm_available': self.llm.is_available,
            'llm_model': self.llm.model_name,
            'llm_error': self.llm.last_error,
            'embedding_available': self.embedder.available,
            'embedding_model': self.embedder.model_name,
            'embedding_error': self.embedder.error_message,
            'storage_dir_exists': os.path.isdir(self.memory_dir),
            'run_log_parent_exists': os.path.isdir(run_log_parent),
            'run_log_writable': os.access(run_log_parent, os.W_OK) if os.path.isdir(run_log_parent) else False,
            'vector_backend': 'faiss' if self.memory.faiss_vectors and self.memory.faiss_vectors.index is not None else 'in-memory',
            'faiss_vectors': self.memory.faiss_vectors.size if self.memory.faiss_vectors else 0,
            'in_memory_vectors': len(self.memory.vectors.embeddings),
            'errors': []
        }
        if not checks['llm_available']:
            checks['errors'].append(LLM_REQUIRED_ERROR)
        if not checks['embedding_available']:
            checks['errors'].append(EMBEDDINGS_REQUIRED_ERROR + _detail_suffix(self.embedder.error_message))
        if not checks['storage_dir_exists']:
            checks['errors'].append(f"Storage directory missing: {self.memory_dir}")
        if not checks['run_log_parent_exists']:
            checks['errors'].append(f"Run log directory missing: {run_log_parent}")
        elif not checks['run_log_writable']:
            checks['errors'].append(f"Run log directory not writable: {run_log_parent}")
        checks['ok'] = len(checks['errors']) == 0
        return checks

    def close(self):
        """Flush FAISS index and close SQLite connection."""
        self._auto_save()
        if self.memory.db:
            self.memory.db.close()
