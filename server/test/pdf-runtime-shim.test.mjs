import assert from "node:assert/strict";
import test from "node:test";

import { installPdfRuntimeShim } from "../rag/pdf-runtime-shim.js";

// The shim exists so that pdfjs survives being bundled into a single-file
// executable, where its native canvas dependency cannot be embedded. Bundling is
// not reproducible from a unit test, so these tests pin the two properties that
// make the bundled build work, both of which are observable in-process.

test("shim installs tripwire globals only where nothing already provides them", () => {
  const existing = class RealDOMMatrix {};
  const target = { DOMMatrix: existing };

  const result = installPdfRuntimeShim({ target });

  // A real implementation must win. pdfjs polyfills behind `if (!globalThis.X)`,
  // so an unconditional overwrite here would strip capability from any runtime
  // that actually has it.
  assert.equal(target.DOMMatrix, existing);
  assert.deepEqual(result.installedTripwires, ["ImageData", "Path2D"]);
  assert.equal(typeof target.ImageData, "function");
  assert.equal(target.Path2D.name, "Path2D");
});

test("tripwire names the missing dependency instead of faking it", () => {
  const target = {};
  installPdfRuntimeShim({ target });

  const matrix = new target.DOMMatrix();

  // The failure mode being prevented: a plausible fake would let pdfjs "succeed"
  // and emit wrong text. The tripwire has to be impossible to mistake for output.
  assert.throws(
    () => matrix.translate(1, 1),
    (error) =>
      error instanceof Error &&
      error.message.includes("DOMMatrix.translate") &&
      error.message.includes("cannot render PDFs") &&
      error.message.includes("pdf-runtime-shim.js")
  );

  // Await and symbol-keyed protocol probes are the runtime inspecting the object,
  // not code trying to render, so they must stay silent.
  assert.equal(matrix.then, undefined);
  assert.equal(matrix[Symbol.toPrimitive], undefined);
  assert.doesNotThrow(() => {
    matrix.anything = 1;
  });
});

test("importing the shim registers a worker pdfjs can use without a dynamic import", () => {
  // pdfjs resolves its worker as the relative specifier "./pdf.worker.mjs" and
  // imports it at runtime, which a bundler cannot follow into a virtual
  // filesystem. It prefers this global when present, so this assertion is what
  // keeps single-file builds from failing with "Cannot find module".
  assert.equal(
    typeof globalThis.pdfjsWorker?.WorkerMessageHandler,
    "function",
    "importing rag/pdf-runtime-shim.js must leave a usable WorkerMessageHandler on globalThis"
  );

  // The regression that matters most -- that preempting the real canvas polyfills
  // does not degrade extraction -- is covered by pdf-loader.test.mjs rather than
  // duplicated here. pdf-loader.js imports this shim, so those tests already parse
  // real PDFs with these tripwires standing in for the canvas globals.
  assert.equal(globalThis.DOMMatrix.name, "DOMMatrix");
});
