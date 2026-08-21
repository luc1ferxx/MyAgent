import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";

import {
  APP_DIRECTORY_NAME,
  APP_DISPLAY_DIRECTORY_NAME,
  getUserDataDirectory,
  isBundledSourceDirectory,
  resolveDataDirectory,
} from "../runtime-paths.js";

// The condition this module exists for -- a bundled single-file executable whose
// source directory is a read-only virtual path -- cannot be produced from a normal
// test run. So the resolver takes its filesystem and platform as injectable
// parameters and these tests drive it through every branch directly, including the
// bundled one.

const fakeFileSystem = ({ existing = [], unwritable = [] } = {}) => {
  const existingSet = new Set(existing.map((entry) => path.resolve(entry)));
  const unwritableSet = new Set(unwritable.map((entry) => path.resolve(entry)));

  return {
    existsSync: (target) => existingSet.has(path.resolve(target)),
    isWritable: (target) => !unwritableSet.has(path.resolve(target)),
  };
};

test("an explicit path wins over every other consideration", () => {
  // The escape hatch has to work even when everything else about the environment
  // looks broken, because it is what an operator reaches for when it is.
  const resolved = resolveDataDirectory({
    explicitPath: "  /srv/archive  ",
    derivedPath: "/app/server/data/rag",
    fallbackSegments: ["rag"],
    sourceDirectory: "/does/not/exist",
    fileSystem: fakeFileSystem(),
  });

  assert.equal(resolved, path.resolve("/srv/archive"));
});

test("a data directory already on disk is never relocated", () => {
  // The most dangerous outcome of this whole module would be moving a populated
  // archive, which a user would experience as every document disappearing. An
  // existing directory therefore outranks even the bundled check.
  const resolved = resolveDataDirectory({
    derivedPath: "/app/server/data/rag",
    fallbackSegments: ["rag"],
    sourceDirectory: "/$bunfs/root/rag",
    fileSystem: fakeFileSystem({ existing: ["/app/server/data/rag"] }),
    platform: "linux",
    environment: {},
    homeDirectory: "/home/u",
  });

  assert.equal(resolved, path.resolve("/app/server/data/rag"));
});

test("a bundled source directory falls back to the user data directory", () => {
  // Reproduces what a compiled binary actually reports: dirname "/$bunfs/root",
  // which does not exist, with nothing on the real filesystem yet.
  const resolved = resolveDataDirectory({
    derivedPath: "/$bunfs/data/rag",
    fallbackSegments: ["rag"],
    sourceDirectory: "/$bunfs/root",
    fileSystem: fakeFileSystem(),
    platform: "linux",
    environment: {},
    homeDirectory: "/home/u",
  });

  assert.equal(resolved, path.join("/home/u", ".local", "share", APP_DIRECTORY_NAME, "rag"));
});

test("a bundled build does not write to a writable filesystem root", () => {
  // The failure mode that ruled out a writability-only rule: running as root, the
  // nearest existing ancestor of "/$bunfs/data/rag" is "/" and it IS writable, so
  // a writability check alone would create a literal "$bunfs" directory at the
  // filesystem root. The bundled check has to be consulted first.
  const resolved = resolveDataDirectory({
    derivedPath: "/$bunfs/data/rag",
    fallbackSegments: ["rag"],
    sourceDirectory: "/$bunfs/root",
    fileSystem: fakeFileSystem({ existing: ["/"] }),
    platform: "linux",
    environment: {},
    homeDirectory: "/root",
  });

  assert.equal(resolved, path.join("/root", ".local", "share", APP_DIRECTORY_NAME, "rag"));
});

test("an unwritable install directory falls back even when not bundled", () => {
  // A root-owned global npm prefix, or a .app's Resources directory: the source is
  // really on disk, so the bundled check does not fire, but writing beside it would
  // fail or be discarded on upgrade.
  const resolved = resolveDataDirectory({
    derivedPath: "/usr/local/lib/node_modules/doccompare/server/data/rag",
    fallbackSegments: ["rag"],
    sourceDirectory: "/usr/local/lib/node_modules/doccompare/server/rag",
    fileSystem: fakeFileSystem({
      existing: ["/usr/local/lib/node_modules/doccompare/server/rag", "/usr/local/lib"],
      unwritable: ["/usr/local/lib"],
    }),
    platform: "linux",
    environment: {},
    homeDirectory: "/home/u",
  });

  assert.equal(resolved, path.join("/home/u", ".local", "share", APP_DIRECTORY_NAME, "rag"));
});

test("a writable development checkout keeps its historical path", () => {
  // The no-op case, and the one that matters most for not disrupting anyone: a
  // normal clone must behave exactly as it did before this module existed.
  const resolved = resolveDataDirectory({
    derivedPath: "/home/u/project/server/data/rag",
    fallbackSegments: ["rag"],
    sourceDirectory: "/home/u/project/server/rag",
    fileSystem: fakeFileSystem({
      existing: ["/home/u/project/server/rag", "/home/u/project/server"],
    }),
    platform: "linux",
    environment: {},
    homeDirectory: "/home/u",
  });

  assert.equal(resolved, path.resolve("/home/u/project/server/data/rag"));
});

test("user data directories follow each platform's convention", () => {
  assert.equal(
    getUserDataDirectory({ platform: "darwin", environment: {}, homeDirectory: "/Users/u" }),
    path.join("/Users/u", "Library", "Application Support", APP_DISPLAY_DIRECTORY_NAME)
  );

  assert.equal(
    getUserDataDirectory({
      platform: "win32",
      environment: { APPDATA: "C:\\Users\\u\\AppData\\Roaming" },
      homeDirectory: "C:\\Users\\u",
    }),
    path.join("C:\\Users\\u\\AppData\\Roaming", APP_DISPLAY_DIRECTORY_NAME)
  );

  // Windows without APPDATA set still has to land somewhere sensible.
  assert.equal(
    getUserDataDirectory({ platform: "win32", environment: {}, homeDirectory: "C:\\Users\\u" }),
    path.join("C:\\Users\\u", "AppData", "Roaming", APP_DISPLAY_DIRECTORY_NAME)
  );

  assert.equal(
    getUserDataDirectory({
      platform: "linux",
      environment: { XDG_DATA_HOME: "/custom/data" },
      homeDirectory: "/home/u",
    }),
    path.join("/custom/data", APP_DIRECTORY_NAME)
  );

  assert.equal(
    getUserDataDirectory({ platform: "linux", environment: {}, homeDirectory: "/home/u" }),
    path.join("/home/u", ".local", "share", APP_DIRECTORY_NAME)
  );
});

test("the bundled check reports on existence, not on path spelling", () => {
  // Keyed off the semantic property rather than a "$bunfs" substring, so it also
  // holds for a bundler that names its virtual root something else.
  assert.equal(
    isBundledSourceDirectory({
      sourceDirectory: "/anything/virtual",
      fileSystem: fakeFileSystem(),
    }),
    true
  );

  assert.equal(
    isBundledSourceDirectory({
      sourceDirectory: "/real/dir",
      fileSystem: fakeFileSystem({ existing: ["/real/dir"] }),
    }),
    false
  );

  // A path that merely looks like a bundle but exists is not treated as one.
  assert.equal(
    isBundledSourceDirectory({
      sourceDirectory: "/$bunfs/root",
      fileSystem: fakeFileSystem({ existing: ["/$bunfs/root"] }),
    }),
    false
  );
});
