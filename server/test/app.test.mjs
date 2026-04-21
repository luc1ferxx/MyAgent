import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createApp } from "../app.js";

const okHealthService = {
  buildHealthReport: async () => ({
    status: "ok",
    checks: {},
  }),
  runStartupHealthChecks: async () => ({
    status: "ok",
    checks: {},
  }),
};

const startServer = async (app) => {
  const server = createServer(app);

  await new Promise((resolve) => {
    server.listen(0, "127.0.0.1", resolve);
  });

  const address = server.address();

  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: async () => {
      await new Promise((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error);
            return;
          }

          resolve();
        });
      });
    },
  };
};

test("upload flow stores chunks, completes ingestion, and deletes documents", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "agentai-app-test-"));
  const uploadsDirectory = path.join(tempRoot, "uploads");
  const uploadSessionDirectory = path.join(tempRoot, "upload-sessions");
  const documents = new Map();
  let mergedContent = null;

  const ragService = {
    chat: async () => ({
      text: "stub",
      citations: [],
    }),
    clearDocuments: async () => {
      const cleared = [...documents.values()];
      documents.clear();
      return cleared;
    },
    clearSessionMemory: () => true,
    deleteDocument: async (docId) => {
      const document = documents.get(docId) ?? null;
      documents.delete(docId);
      return document;
    },
    getDocument: (docId) => documents.get(docId) ?? null,
    getDocumentFile: async (docId) => {
      const document = documents.get(docId);

      if (!document) {
        return null;
      }

      return {
        document,
        fileBuffer: Buffer.from(mergedContent ?? "", "utf8"),
        fileName: document.fileName,
        mimeType: "application/pdf",
        fileSize: Buffer.byteLength(mergedContent ?? "", "utf8"),
      };
    },
    ingestDocument: async ({ docId, filePath, fileName }) => {
      mergedContent = await readFile(filePath, "utf8");
      const document = {
        docId,
        fileName,
        filePath: `documents/${docId}/file`,
        publicFilePath: `documents/${docId}/file`,
        fileSize: Buffer.byteLength(mergedContent ?? "", "utf8"),
        pageCount: 1,
        chunkCount: 1,
        uploadedAt: new Date().toISOString(),
        storageBackend: "postgresql",
      };

      documents.set(docId, document);
      return document;
    },
    initializeSessionMemory: async () => true,
    listDocuments: () => [...documents.values()],
  };

  const app = await createApp({
    ragService,
    chatMcp: async () => ({
      text: "web",
    }),
    healthService: okHealthService,
    uploadSessionDirectory,
    uploadsDirectory,
  });
  const server = await startServer(app);

  try {
    const fileId = "test-file-id";
    const content = "fake-pdf-content";

    let response = await fetch(`${server.baseUrl}/upload/init`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        fileId,
        fileName: "notes.pdf",
        fileSize: content.length,
        lastModified: 0,
        totalChunks: 2,
        chunkSize: 8,
      }),
    });

    assert.equal(response.status, 201);

    const chunkOne = new FormData();
    chunkOne.append("fileId", fileId);
    chunkOne.append("chunkIndex", "0");
    chunkOne.append("totalChunks", "2");
    chunkOne.append("chunk", new Blob([content.slice(0, 8)]), "notes.pdf.part-0");

    response = await fetch(`${server.baseUrl}/upload/chunk`, {
      method: "POST",
      body: chunkOne,
    });

    assert.equal(response.status, 201);

    const chunkTwo = new FormData();
    chunkTwo.append("fileId", fileId);
    chunkTwo.append("chunkIndex", "1");
    chunkTwo.append("totalChunks", "2");
    chunkTwo.append("chunk", new Blob([content.slice(8)]), "notes.pdf.part-1");

    response = await fetch(`${server.baseUrl}/upload/chunk`, {
      method: "POST",
      body: chunkTwo,
    });

    assert.equal(response.status, 201);

    response = await fetch(`${server.baseUrl}/upload/complete`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        fileId,
      }),
    });

    assert.equal(response.status, 201);
    const uploadedDocument = await response.json();
    assert.equal(uploadedDocument.fileName, "notes.pdf");
    assert.equal(mergedContent, content);

    response = await fetch(`${server.baseUrl}/documents`);
    assert.equal(response.status, 200);
    assert.equal((await response.json()).length, 1);

    response = await fetch(
      `${server.baseUrl}/documents/${uploadedDocument.docId}`,
      {
        method: "DELETE",
      }
    );

    assert.equal(response.status, 200);
    assert.equal(documents.size, 0);
  } finally {
    await server.close();
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("chat endpoint exposes explicit rag abstain fields", async () => {
  const documents = new Map([
    [
      "doc-1",
      {
        docId: "doc-1",
        fileName: "notes.pdf",
      },
    ],
  ]);
  const app = await createApp({
    ragService: {
      chat: async () => ({
        text: 'I found related material, but I still cannot confirm "NULPAR-DZ" reliably.',
        citations: [],
        abstained: true,
        abstainReason:
          'I found related material, but I still cannot confirm "NULPAR-DZ" reliably.',
        resolvedQuery: "What is the NULPAR-DZ allocation amount?",
        memoryApplied: false,
        gapPlan: {
          missingAspects: [
            {
              label: "NULPAR-DZ",
            },
          ],
          supplementalSearches: [],
        },
      }),
      clearDocuments: async () => [],
      clearSessionMemory: () => true,
      deleteDocument: async () => null,
      getDocument: (docId) => documents.get(docId) ?? null,
      ingestDocument: async () => null,
      initializeSessionMemory: async () => true,
      listDocuments: () => [...documents.values()],
    },
    chatMcp: async () => ({
      text: "web",
    }),
    healthService: okHealthService,
  });
  const server = await startServer(app);

  try {
    const response = await fetch(`${server.baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        docId: "doc-1",
        question: "What is the NULPAR-DZ allocation amount?",
      }),
    });

    assert.equal(response.status, 200);

    const body = await response.json();

    assert.equal(body.ragAbstained, true);
    assert.match(body.ragAbstainReason, /NULPAR-DZ/);
    assert.equal(body.ragResolvedQuestion, "What is the NULPAR-DZ allocation amount?");
    assert.equal(body.ragGapPlan.missingAspects[0].label, "NULPAR-DZ");
    assert.equal("possibleLocations" in body.ragGapPlan, false);
  } finally {
    await server.close();
  }
});

test("memory endpoints list, create, and delete long-term memories", async () => {
  const memories = [
    {
      memoryId: "memory-1",
      userId: "user-1",
      category: "preference",
      memoryKey: "reply_language",
      memoryValue: "zh",
      text: "Reply in Chinese by default.",
      source: "user_explicit",
      confidence: 1,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      lastUsedAt: null,
    },
  ];
  const app = await createApp({
    ragService: {
      listLongMemories: async ({ userId }) =>
        memories.filter((memory) => memory.userId === userId),
      rememberLongMemory: async ({ userId, text, category, memoryKey, memoryValue }) => {
        const memory = {
          memoryId: "memory-2",
          userId,
          category: category ?? "note",
          memoryKey: memoryKey ?? null,
          memoryValue: memoryValue ?? null,
          text,
          source: "user_explicit",
          confidence: 1,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          lastUsedAt: null,
        };

        memories.push(memory);
        return memory;
      },
      deleteLongMemory: async ({ userId, memoryId }) => {
        const memoryIndex = memories.findIndex(
          (memory) => memory.userId === userId && memory.memoryId === memoryId
        );

        if (memoryIndex === -1) {
          return null;
        }

        const [deletedMemory] = memories.splice(memoryIndex, 1);
        return deletedMemory;
      },
      clearLongMemories: async ({ userId }) => {
        const matchingMemories = memories.filter((memory) => memory.userId === userId);
        memories.splice(
          0,
          memories.length,
          ...memories.filter((memory) => memory.userId !== userId)
        );
        return matchingMemories.length;
      },
      initializeSessionMemory: async () => true,
    },
    chatMcp: async () => ({
      text: "web",
    }),
    healthService: okHealthService,
  });
  const server = await startServer(app);

  try {
    let response = await fetch(`${server.baseUrl}/memory?userId=user-1`);

    assert.equal(response.status, 200);
    assert.equal((await response.json()).memories.length, 1);

    response = await fetch(`${server.baseUrl}/memory`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        userId: "user-1",
        text: "Keep answers concise by default.",
        category: "preference",
        memoryKey: "answer_style",
        memoryValue: "concise",
      }),
    });

    assert.equal(response.status, 201);
    assert.equal((await response.json()).memory.memoryId, "memory-2");

    response = await fetch(`${server.baseUrl}/memory/memory-2?userId=user-1`, {
      method: "DELETE",
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).deleted, true);

    response = await fetch(`${server.baseUrl}/memory?userId=user-1`, {
      method: "DELETE",
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).deletedCount, 1);
  } finally {
    await server.close();
  }
});

test("health and ready endpoints expose startup status", async () => {
  const healthService = {
    buildHealthReport: async () => ({
      status: "error",
      checkedAt: new Date().toISOString(),
      checks: {
        vectorStore: {
          status: "error",
          message: "Qdrant is unreachable.",
        },
      },
    }),
    runStartupHealthChecks: async () => ({
      status: "ok",
      checks: {},
    }),
  };
  const app = await createApp({
    chatMcp: async () => ({
      text: "web",
    }),
    healthService,
    ragService: {
      initializeDocumentRegistry: async () => [],
      initializeSessionMemory: async () => true,
    },
  });
  const server = await startServer(app);

  try {
    let response = await fetch(`${server.baseUrl}/health`);
    assert.equal(response.status, 200);
    assert.equal((await response.json()).status, "error");

    response = await fetch(`${server.baseUrl}/ready`);
    assert.equal(response.status, 503);
    assert.equal((await response.json()).status, "error");
  } finally {
    await server.close();
  }
});

test("api auth protects document routes while leaving health public", async () => {
  const originalAuthEnabled = process.env.API_AUTH_ENABLED;
  const originalAuthToken = process.env.API_AUTH_TOKEN;

  process.env.API_AUTH_ENABLED = "true";
  process.env.API_AUTH_TOKEN = "local-test-token";

  try {
    const app = await createApp({
      healthService: okHealthService,
      ragService: {
        initializeDocumentRegistry: async () => [],
        initializeSessionMemory: async () => true,
      },
    });
    const server = await startServer(app);

    try {
      let response = await fetch(`${server.baseUrl}/health`);
      assert.equal(response.status, 200);

      response = await fetch(`${server.baseUrl}/documents`);
      assert.equal(response.status, 401);

      response = await fetch(`${server.baseUrl}/documents`, {
        headers: {
          "x-api-key": "local-test-token",
        },
      });
      assert.equal(response.status, 200);
    } finally {
      await server.close();
    }
  } finally {
    if (originalAuthEnabled === undefined) {
      delete process.env.API_AUTH_ENABLED;
    } else {
      process.env.API_AUTH_ENABLED = originalAuthEnabled;
    }

    if (originalAuthToken === undefined) {
      delete process.env.API_AUTH_TOKEN;
    } else {
      process.env.API_AUTH_TOKEN = originalAuthToken;
    }
  }
});

test("document file route streams stored PDFs before auth middleware", async () => {
  const fileBuffer = Buffer.from("%PDF-test-document", "utf8");
  const app = await createApp({
    healthService: okHealthService,
    ragService: {
      initializeDocumentRegistry: async () => [],
      initializeSessionMemory: async () => true,
      getDocumentFile: async (docId) =>
        docId === "doc-1"
          ? {
              document: {
                docId,
                fileName: "stored.pdf",
              },
              fileBuffer,
              fileName: "stored.pdf",
              mimeType: "application/pdf",
              fileSize: fileBuffer.byteLength,
            }
          : null,
    },
  });
  const server = await startServer(app);

  try {
    const response = await fetch(`${server.baseUrl}/documents/doc-1/file`, {
      headers: {
        Range: "bytes=0-3",
      },
    });

    assert.equal(response.status, 206);
    assert.equal(response.headers.get("content-type"), "application/pdf");
    assert.equal(response.headers.get("accept-ranges"), "bytes");
    assert.equal(await response.text(), "%PDF");
  } finally {
    await server.close();
  }
});
