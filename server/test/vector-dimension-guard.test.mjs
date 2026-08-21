import test, { afterEach, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import {
  configureRagDataDirectory,
  getRagDataDirectory,
  getRagDataPath,
  writeJsonFileSync,
} from "../rag/storage.js";
import {
  resetLocalVectorStore,
  resetDimensionMismatchReports,
  searchLocalDocuments,
} from "../rag/vector-store-local.js";

// DocCompare supports any OpenAI-compatible endpoint, which makes changing
// OPENAI_BASE_URL or OPENAI_EMBEDDING_MODEL a normal user action rather than a
// misconfiguration. When that happens, freshly embedded queries meet stored chunks
// from a different embedding space. These tests pin what happens then.

const originalDataDirectory = getRagDataDirectory();
let tempRoot = null;
let warnings = [];
let originalWarn = null;

const seedIndex = (entries) => {
  writeJsonFileSync(getRagDataPath("vector-index.json"), entries);
  resetLocalVectorStore();
};

const ones = (length) => Array.from({ length }, () => 1);

const entry = (id, vector) => ({
  id: `${id}:0`,
  pageContent: `content for ${id}`,
  vector,
  metadata: { docId: id, fileName: `${id}.pdf`, pageNumber: 1 },
});

beforeEach(async () => {
  tempRoot = await mkdtemp(path.join(os.tmpdir(), "dimension-guard-test-"));
  configureRagDataDirectory(path.join(tempRoot, "rag-data"));
  resetLocalVectorStore();
  resetDimensionMismatchReports();
  warnings = [];
  originalWarn = console.warn;
  console.warn = (message) => {
    warnings.push(String(message));
  };
});

afterEach(async () => {
  console.warn = originalWarn;
  configureRagDataDirectory(originalDataDirectory);
  resetLocalVectorStore();
  resetDimensionMismatchReports();
  if (tempRoot) {
    await rm(tempRoot, { recursive: true, force: true });
    tempRoot = null;
  }
});

test("a chunk embedded at a different dimension scores zero, not a plausible score", async () => {
  // The assertion that fails before the guard exists. Truncating the dot product to
  // the shorter vector while dividing by both full magnitudes gives these two
  // all-ones vectors 0.7071 -- above the 0.32 default of getMinRelevanceScore(), so
  // the stale chunk qualified as evidence and was cited. A wrong citation presented
  // with confidence is worse than no answer.
  seedIndex([entry("same", ones(64)), entry("stale", ones(32))]);

  const results = await searchLocalDocuments({
    queryVector: ones(64),
    queryText: "content",
    docIds: ["same", "stale"],
    topK: 10,
    scoringMode: "vector",
  });

  const byDocId = new Map(results.map((result) => [result.document.metadata.docId, result]));

  // Equal-length vectors must keep their true cosine: this half of the assertion
  // guards against a fix that simply zeroes everything.
  assert.equal(byDocId.get("same").vectorScore, 1);
  assert.equal(byDocId.get("stale").vectorScore, 0);
  assert.ok(
    byDocId.get("stale").vectorScore < 0.32,
    "a mismatched chunk must fall below the qualifying relevance score"
  );
});

test("an archive at a uniform different dimension ranks nothing by meaning", async () => {
  // The headline upgrade case: every stored chunk predates an endpoint change.
  seedIndex([entry("a", ones(1536)), entry("b", ones(1536))]);

  const results = await searchLocalDocuments({
    queryVector: ones(3072),
    queryText: "content",
    docIds: ["a", "b"],
    topK: 10,
    scoringMode: "vector",
  });

  assert.ok(results.length > 0, "results are still returned for keyword scoring to act on");
  assert.ok(
    results.every((result) => result.vectorScore === 0),
    "no stored chunk may claim semantic similarity to a query from a different embedding space"
  );
});

test("the mismatch is reported once per shape rather than once per query", async () => {
  // Scoring 0 is correct but silent, and silence is what made the original bug
  // expensive. The warning has to name both dimensions and the remedy -- and must
  // not print on every query, or it becomes noise that gets filtered out.
  seedIndex([entry("stale", ones(1536)), entry("fresh", ones(3072))]);

  for (let attempt = 0; attempt < 3; attempt += 1) {
    await searchLocalDocuments({
      queryVector: ones(3072),
      queryText: "content",
      docIds: ["stale", "fresh"],
      topK: 10,
      scoringMode: "vector",
    });
  }

  assert.equal(warnings.length, 1, "one warning per mismatch shape, not per query");
  assert.match(warnings[0], /3072-dimensional/);
  assert.match(warnings[0], /1536/);
  assert.match(warnings[0], /re-ingest/i);
  assert.match(warnings[0], /OPENAI_EMBEDDING_MODEL|OPENAI_BASE_URL/);
});

test("a consistent archive is scored and reported exactly as before", async () => {
  // The no-op case. If this warns or changes a score, the guard has overreached.
  seedIndex([entry("a", [1, 0, 0, 0]), entry("b", [0, 1, 0, 0])]);

  const results = await searchLocalDocuments({
    queryVector: [1, 0, 0, 0],
    queryText: "content",
    docIds: ["a", "b"],
    topK: 10,
    scoringMode: "vector",
  });

  const byDocId = new Map(results.map((result) => [result.document.metadata.docId, result]));

  assert.equal(byDocId.get("a").vectorScore, 1);
  assert.equal(byDocId.get("b").vectorScore, 0);
  assert.deepEqual(warnings, []);
});
