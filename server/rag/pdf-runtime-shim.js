// Makes pdfjs-dist work when this backend is bundled into a single-file
// executable. Imported for its side effects, and it must run BEFORE pdfjs is
// evaluated -- see the import order in pdf-loader.js.
//
// pdfjs reaches outside itself twice, and both reaches fail inside a bundle:
//
// 1. It require()s @napi-rs/canvas, a native .node binding, purely to polyfill
//    DOMMatrix, ImageData and Path2D. A native binding cannot be embedded in a
//    single-file executable, so the polyfills are skipped -- and then pdfjs runs
//    `new DOMMatrix` at module scope and dies with a ReferenceError before it can
//    parse anything. Each polyfill is guarded by `if (!globalThis.X)`, so
//    defining them first makes pdfjs skip the native module entirely.
//
//    These are TRIPWIRES, not implementations. This backend extracts text and
//    never renders, and a compiled binary has no canvas to render onto anyway. A
//    plausible fake would turn a missing dependency into silently wrong text; a
//    tripwire turns it into a loud error naming the cause. They are installed on
//    every runtime, not just compiled ones, so the behaviour under test is the
//    behaviour that ships.
//
// 2. It resolves its worker as the relative specifier "./pdf.worker.mjs" and
//    imports it dynamically, which no bundler can follow into a virtual
//    filesystem. But the loader prefers globalThis.pdfjsWorker's
//    WorkerMessageHandler when one exists, so importing the worker statically here
//    both embeds it in the binary and stops the dynamic import from running.

import * as pdfjsWorker from "pdfjs-dist/legacy/build/pdf.worker.mjs";

const TRIPWIRE_GLOBALS = ["DOMMatrix", "ImageData", "Path2D"];

const createTripwire = (name) => {
  const explode = (property) => {
    throw new Error(
      `${name}.${String(property)} was used, but ${name} is a text-extraction-only ` +
        `stub installed by rag/pdf-runtime-shim.js. This build cannot render PDFs. ` +
        `If rendering is now required, install a real implementation instead of ` +
        `removing the stub, or pdfjs will fail to load in bundled builds.`
    );
  };

  return class {
    constructor() {
      return new Proxy(this, {
        get: (target, property) =>
          // Symbols and `then` are probed by the runtime itself (await, logging,
          // instanceof); answering those honestly keeps the error from firing on
          // something that never touched the object on purpose.
          property === "then" || typeof property === "symbol"
            ? undefined
            : explode(property),
        set: () => true,
      });
    }
  };
};

export const installPdfRuntimeShim = ({ target = globalThis } = {}) => {
  const installed = [];

  for (const name of TRIPWIRE_GLOBALS) {
    if (target[name]) {
      continue;
    }

    const tripwire = createTripwire(name);
    Object.defineProperty(tripwire, "name", {
      value: name,
    });
    target[name] = tripwire;
    installed.push(name);
  }

  if (!target.pdfjsWorker) {
    target.pdfjsWorker = pdfjsWorker;
  }

  return {
    installedTripwires: installed,
    workerRegistered: Boolean(target.pdfjsWorker?.WorkerMessageHandler),
  };
};

installPdfRuntimeShim();
