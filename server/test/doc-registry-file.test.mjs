import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "fs/promises";
import os from "os";
import path from "path";

import { createFileDocumentRegistryStore } from "../rag/doc-registry-file.js";
import {
  configureDocumentRegistryStore,
  getDocument,
  getDocumentFile,
  initializeDocumentRegistry,
  listDocuments,
  registerDocument,
  resetDocumentRegistryStore,
} from "../rag/doc-registry.js";

// The store is exercised through injected paths rather than
// configureRagDataDirectory so these tests touch no module-level state. The one
// test that goes through the shared registry resets it in a finally block.
const withTemporaryStore = async (run) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "doc-registry-file-"));
  const registryFilePath = path.join(root, "documents.json");
  const documentsDirectory = path.join(root, "documents");
  const createStore = (overrides = {}) =>
    createFileDocumentRegistryStore({
      getDocumentsDirectory: () => documentsDirectory,
      getRegistryFilePath: () => registryFilePath,
      ...overrides,
    });

  try {
    await run({
      createStore,
      documentsDirectory,
      registryFilePath,
      root,
    });
  } finally {
    await rm(root, {
      force: true,
      recursive: true,
    });
  }
};

const pathExists = async (targetPath) => {
  try {
    await stat(targetPath);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") {
      return false;
    }

    throw error;
  }
};

test("file document registry store survives a fresh store instance", async () => {
  await withTemporaryStore(async ({ createStore, documentsDirectory, registryFilePath }) => {
    const writer = createStore();
    await writer.initialize();

    const upsertedDocument = await writer.upsert({
      docId: " doc-alpha ",
      fileName: " Alpha.pdf ",
      fileBuffer: Buffer.from("%PDF-1.7 alpha"),
      chunkCount: "4",
      pageCount: "2",
      ownerUserId: "alice",
      workspaceId: "workspace-a",
      uploadedAt: "2024-01-01T00:00:00.000Z",
      profile: {
        summary: "Alpha summary",
        tags: ["alpha"],
      },
    });

    assert.equal(upsertedDocument.docId, "doc-alpha");
    assert.equal(upsertedDocument.fileName, "Alpha.pdf");
    assert.equal(upsertedDocument.mimeType, "application/pdf");
    assert.equal(upsertedDocument.fileSize, Buffer.byteLength("%PDF-1.7 alpha"));
    assert.equal(upsertedDocument.chunkCount, 4);
    assert.equal(upsertedDocument.pageCount, 2);
    assert.equal(upsertedDocument.filePath, "documents/doc-alpha/file");
    assert.equal(upsertedDocument.publicFilePath, "documents/doc-alpha/file");
    assert.equal(upsertedDocument.storageBackend, "filesystem");

    // The layout is the contract: buildPublicFilePath() promises
    // documents/<docId>/file, and this is what actually has to be on disk for a
    // second process to find it.
    assert.equal(await pathExists(registryFilePath), true);
    assert.equal(
      await pathExists(path.join(documentsDirectory, "doc-alpha", "file")),
      true
    );

    // The whole point of the store: a brand new instance, holding no cache, reads
    // the document and its exact bytes back off disk.
    const reader = createStore();
    const documents = await reader.list();

    assert.deepEqual(
      documents.map((document) => document.docId),
      ["doc-alpha"]
    );
    assert.equal(documents[0].storageBackend, "filesystem");
    assert.deepEqual(documents[0].profile.tags, ["alpha"]);

    const storedFile = await reader.getFile("doc-alpha");

    assert.equal(storedFile.fileBuffer.toString("utf8"), "%PDF-1.7 alpha");
    assert.equal(storedFile.fileName, "Alpha.pdf");
    assert.equal(storedFile.mimeType, "application/pdf");
    assert.equal(storedFile.fileSize, Buffer.byteLength("%PDF-1.7 alpha"));
    assert.equal(storedFile.document.docId, "doc-alpha");
  });
});

test("file document registry store reads bytes from a source file path and replaces them on re-upsert", async () => {
  await withTemporaryStore(async ({ createStore, documentsDirectory, root }) => {
    const sourceFilePath = path.join(root, "source.pdf");
    await writeFile(sourceFilePath, "%PDF-1.7 from-disk");

    const store = createStore();

    const fromSourceFile = await store.upsert({
      docId: "doc-source",
      fileName: "Source.pdf",
      sourceFilePath,
      mimeType: "  ",
      uploadedAt: "2024-01-01T00:00:00.000Z",
    });

    assert.equal(fromSourceFile.fileSize, Buffer.byteLength("%PDF-1.7 from-disk"));
    assert.equal(fromSourceFile.mimeType, "application/pdf");
    assert.equal(
      (await store.getFile("doc-source")).fileBuffer.toString("utf8"),
      "%PDF-1.7 from-disk"
    );

    // docIds are UUIDs, not content hashes, so re-ingesting one has to overwrite
    // its bytes. A write primitive that refuses to clobber an existing file would
    // silently serve the stale PDF here.
    await store.upsert({
      docId: "doc-source",
      fileName: "Source.pdf",
      fileBuffer: Buffer.from("%PDF-1.7 replaced"),
      uploadedAt: "2024-02-01T00:00:00.000Z",
    });

    const documents = await store.list();

    assert.equal(documents.length, 1);
    assert.equal(documents[0].uploadedAt, "2024-02-01T00:00:00.000Z");
    assert.equal(
      (await store.getFile("doc-source")).fileBuffer.toString("utf8"),
      "%PDF-1.7 replaced"
    );

    // Uint8Array is the other shape doc-registry.js accepts.
    await store.upsert({
      docId: "doc-bytes",
      fileName: "Bytes.pdf",
      fileBuffer: new Uint8Array([1, 2, 3]),
      uploadedAt: "2024-03-01T00:00:00.000Z",
    });

    assert.deepEqual(
      [...(await store.getFile("doc-bytes")).fileBuffer],
      [1, 2, 3]
    );
    assert.equal(
      await pathExists(path.join(documentsDirectory, "doc-bytes", "file")),
      true
    );
  });
});

test("file document registry store rejects incomplete documents and unusable ids", async () => {
  await withTemporaryStore(async ({ createStore }) => {
    const store = createStore();

    await assert.rejects(
      () =>
        store.upsert({
          docId: "   ",
          fileName: "Missing.pdf",
          fileBuffer: Buffer.from("pdf"),
        }),
      /docId and fileName/
    );
    await assert.rejects(
      () =>
        store.upsert({
          docId: "doc-a",
          fileName: "   ",
          fileBuffer: Buffer.from("pdf"),
        }),
      /docId and fileName/
    );
    await assert.rejects(
      () =>
        store.upsert({
          docId: "doc-a",
          fileName: "Alpha.pdf",
        }),
      /PDF buffer or source file path/
    );

    // encodeURIComponent leaves "." alone, so ".." would resolve outside the
    // documents directory if it were allowed through to the filesystem.
    await assert.rejects(
      () =>
        store.upsert({
          docId: "..",
          fileName: "Escape.pdf",
          fileBuffer: Buffer.from("pdf"),
        }),
      /cannot be used as a storage path/
    );

    assert.deepEqual(await store.list(), []);
  });
});

test("file document registry store enforces access scope on reads, deletes, and clears", async () => {
  await withTemporaryStore(async ({ createStore, documentsDirectory }) => {
    const store = createStore();

    await store.upsert({
      docId: "doc-alice",
      fileName: "Alice.pdf",
      fileBuffer: Buffer.from("alice"),
      ownerUserId: "alice",
      workspaceId: "workspace-a",
      uploadedAt: "2024-01-01T00:00:00.000Z",
    });
    await store.upsert({
      docId: "doc-bob",
      fileName: "Bob.pdf",
      fileBuffer: Buffer.from("bob"),
      ownerUserId: "bob",
      workspaceId: "workspace-b",
      uploadedAt: "2024-02-01T00:00:00.000Z",
    });
    await store.upsert({
      docId: "doc-shared",
      fileName: "Shared.pdf",
      fileBuffer: Buffer.from("shared"),
      uploadedAt: "2024-01-01T00:00:00.000Z",
    });

    // Ordered by uploadedAt then docId, matching the PostgreSQL store's
    // ORDER BY uploaded_at ASC, doc_id ASC.
    assert.deepEqual(
      (await store.list()).map((document) => document.docId),
      ["doc-alice", "doc-shared", "doc-bob"]
    );
    assert.deepEqual(
      (
        await store.list({
          userId: "alice",
          workspaceId: "workspace-a",
        })
      ).map((document) => document.docId),
      ["doc-alice"]
    );

    assert.equal(await store.getFile(""), null);
    assert.equal(await store.getFile("doc-missing"), null);
    assert.equal(
      await store.getFile("doc-alice", {
        userId: "bob",
        workspaceId: "workspace-b",
      }),
      null
    );
    assert.equal(
      (
        await store.getFile("doc-alice", {
          userId: "alice",
          workspaceId: "workspace-a",
        })
      ).fileBuffer.toString("utf8"),
      "alice"
    );

    assert.equal(await store.delete(""), null);
    assert.equal(await store.delete("doc-missing"), null);
    assert.equal(
      await store.delete("doc-alice", {
        userId: "bob",
        workspaceId: "workspace-b",
      }),
      null
    );
    assert.equal(
      await pathExists(path.join(documentsDirectory, "doc-alice", "file")),
      true
    );

    const deletedDocument = await store.delete("doc-alice", {
      userId: "alice",
      workspaceId: "workspace-a",
    });

    assert.equal(deletedDocument.docId, "doc-alice");
    assert.equal(await store.getFile("doc-alice"), null);
    assert.equal(
      await pathExists(path.join(documentsDirectory, "doc-alice")),
      false
    );

    // A scope that matches nothing is a no-op, not a wipe.
    assert.equal(
      await store.clear({
        userId: "nobody",
        workspaceId: "workspace-z",
      }),
      true
    );
    assert.deepEqual(
      (await store.list()).map((document) => document.docId),
      ["doc-shared", "doc-bob"]
    );

    await store.clear({
      userId: "bob",
      workspaceId: "workspace-b",
    });

    assert.deepEqual(
      (await store.list()).map((document) => document.docId),
      ["doc-shared"]
    );
    assert.equal(await pathExists(path.join(documentsDirectory, "doc-bob")), false);

    await store.clear();

    assert.deepEqual(await store.list(), []);
    assert.equal(await pathExists(documentsDirectory), false);
  });
});

test("file document registry store tolerates a damaged registry and keeps files on reset", async () => {
  await withTemporaryStore(async ({ createStore, registryFilePath }) => {
    const store = createStore();

    await store.upsert({
      docId: "doc-keep",
      fileName: "Keep.pdf",
      fileBuffer: Buffer.from("keep"),
      uploadedAt: "2024-01-01T00:00:00.000Z",
    });

    // reset() drops in-process state, and this store has none. Wiping the user's
    // archive here would be data loss dressed up as cleanup -- clear() is the
    // destructive call.
    await store.reset();

    assert.deepEqual(
      (await store.list()).map((document) => document.docId),
      ["doc-keep"]
    );

    // Records missing a docId or fileName cannot be served; dropping them beats
    // failing every read, the same way vector-store-local.js filters its entries.
    await writeFile(
      registryFilePath,
      JSON.stringify([
        {
          docId: "doc-keep",
          fileName: "Keep.pdf",
          uploadedAt: "2024-01-01T00:00:00.000Z",
        },
        {
          docId: "",
          fileName: "Nameless.pdf",
        },
        {
          docId: "doc-untitled",
          fileName: "",
        },
      ])
    );

    assert.deepEqual(
      (await store.list()).map((document) => document.docId),
      ["doc-keep"]
    );

    // A hand-edited registry that is valid JSON but not an array reads as empty
    // rather than throwing on property access.
    await writeFile(registryFilePath, JSON.stringify({ documents: [] }));

    assert.deepEqual(await store.list(), []);

    // A registered document whose bytes are gone is corruption. It reports as
    // absent rather than as an empty PDF, and says so on stderr -- silenced here
    // so the expected complaint does not read as a test failure.
    await writeFile(
      registryFilePath,
      JSON.stringify([
        {
          docId: "doc-orphan",
          fileName: "Orphan.pdf",
          uploadedAt: "2024-01-01T00:00:00.000Z",
        },
      ])
    );

    const originalConsoleError = console.error;
    console.error = () => {};

    try {
      assert.equal(await store.getFile("doc-orphan"), null);
    } finally {
      console.error = originalConsoleError;
    }
  });
});

test("file document registry store round-trips document profiles through the shared registry", async () => {
  await withTemporaryStore(async ({ createStore, documentsDirectory }) => {
    configureDocumentRegistryStore(createStore());

    try {
      assert.deepEqual(await initializeDocumentRegistry(), []);

      // The flat spelling: summary/tags/entities/source at the top level, which is
      // what doc-registry.js's own normalizer accepts alongside a nested profile.
      const registeredDocument = await registerDocument({
        docId: "doc-arxiv",
        fileName: "Paper.pdf",
        fileBuffer: Buffer.from("%PDF-1.7 paper"),
        chunkCount: 3,
        pageCount: 2,
        ownerUserId: "alice",
        workspaceId: "workspace-a",
        summary: "Paper summary",
        tags: ["ml", "ml", " "],
        entities: ["Alice"],
        source: {
          sourceType: "arxiv",
          arxivId: "2401.00001v1",
          absUrl: "https://arxiv.org/abs/2401.00001v1",
        },
        uploadedAt: "2024-01-01T00:00:00.000Z",
      });

      assert.equal(registeredDocument.summary, "Paper summary");
      assert.deepEqual(registeredDocument.tags, ["ml"]);
      assert.equal(registeredDocument.source.arxivId, "2401.00001v1");
      // The registry renormalizes every document on the way in, so a hardcoded
      // backend there would make these documents claim a database that is not
      // running -- which is exactly what a standalone install does not have.
      assert.equal(registeredDocument.storageBackend, "filesystem");

      const storedFile = await getDocumentFile("doc-arxiv", {
        userId: "alice",
        workspaceId: "workspace-a",
      });

      assert.equal(storedFile.fileBuffer.toString("utf8"), "%PDF-1.7 paper");
      assert.equal(
        (await readFile(path.join(documentsDirectory, "doc-arxiv", "file"))).toString(
          "utf8"
        ),
        "%PDF-1.7 paper"
      );

      // A fresh registry backed by a fresh store instance over the same directory
      // is the in-process stand-in for a second process starting up: nothing is
      // cached, everything comes from disk.
      configureDocumentRegistryStore(createStore());
      const reloadedDocuments = await initializeDocumentRegistry();

      assert.deepEqual(
        reloadedDocuments.map((document) => document.docId),
        ["doc-arxiv"]
      );
      assert.equal(getDocument("doc-arxiv").summary, "Paper summary");
      assert.deepEqual(getDocument("doc-arxiv").tags, ["ml"]);
      assert.equal(getDocument("doc-arxiv").source.sourceType, "arxiv");
      assert.equal(getDocument("doc-arxiv").pageCount, 2);
      assert.equal(getDocument("doc-arxiv").storageBackend, "filesystem");
      assert.deepEqual(
        listDocuments({
          userId: "bob",
          workspaceId: "workspace-b",
        }),
        []
      );
    } finally {
      await resetDocumentRegistryStore();
    }
  });
});
