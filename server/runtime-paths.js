import { accessSync, constants, existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";

// One place decides where user data lives. Four modules previously derived their
// own default from import.meta.url, which meant a shipped build had four separate
// chances to write into a directory it does not own.
//
// Two directories are wrong to write into, for different reasons:
//   - a single-file executable's bundle, which is a read-only virtual filesystem
//     (mkdir there fails with EROFS)
//   - a global npm install or a macOS .app, where the program directory is
//     root-owned or replaced wholesale on upgrade, so an archive stored beside
//     the source is unwritable or silently discarded
//
// Existing installs must not be disturbed, so a data directory that is already on
// disk always wins over any of this logic.

// Phase 2 renames the product; these are the only strings that encode the name in
// a filesystem path, deliberately mirroring how opencode keeps its own app name in
// exactly one place.
export const APP_DIRECTORY_NAME = "doccompare";
export const APP_DISPLAY_DIRECTORY_NAME = "DocCompare";

const defaultFileSystem = {
  existsSync,
  isWritable: (target) => {
    try {
      accessSync(target, constants.W_OK);
      return true;
    } catch {
      return false;
    }
  },
};

const nearestExistingAncestor = (target, fileSystem) => {
  let current = path.resolve(target);

  for (;;) {
    if (fileSystem.existsSync(current)) {
      return current;
    }

    const parent = path.dirname(current);

    if (parent === current) {
      return current;
    }

    current = parent;
  }
};

export const getUserDataDirectory = ({
  platform = process.platform,
  environment = process.env,
  homeDirectory = os.homedir(),
} = {}) => {
  if (platform === "darwin") {
    return path.join(
      homeDirectory,
      "Library",
      "Application Support",
      APP_DISPLAY_DIRECTORY_NAME
    );
  }

  if (platform === "win32") {
    const appData =
      environment.APPDATA?.trim() ||
      path.join(homeDirectory, "AppData", "Roaming");
    return path.join(appData, APP_DISPLAY_DIRECTORY_NAME);
  }

  const xdgDataHome = environment.XDG_DATA_HOME?.trim();

  return path.join(
    xdgDataHome || path.join(homeDirectory, ".local", "share"),
    APP_DIRECTORY_NAME
  );
};

// A module's own directory always exists when the program runs from disk, and
// never exists when the program was bundled into a single file. Preferred over
// matching "$bunfs" in the path because it states the property that actually
// matters and does not depend on one bundler's naming.
//
// Verified three ways: node from disk and bun from disk both see their own
// directory; a bun-compiled binary reports dirname "/$bunfs/root" with
// existsSync false, nearest existing ancestor "/", and EROFS on mkdir.
export const isBundledSourceDirectory = ({
  sourceDirectory,
  fileSystem = defaultFileSystem,
}) => Boolean(sourceDirectory) && !fileSystem.existsSync(sourceDirectory);

export const resolveDataDirectory = ({
  explicitPath,
  derivedPath,
  fallbackSegments = [],
  sourceDirectory,
  platform,
  environment = process.env,
  homeDirectory,
  fileSystem = defaultFileSystem,
}) => {
  const explicit = explicitPath?.trim();

  if (explicit) {
    return path.resolve(explicit);
  }

  const derived = path.resolve(derivedPath);

  // An archive that is already here keeps being found, whatever the reasoning
  // below would otherwise conclude. Relocating a populated data directory would
  // present to the user as every document silently disappearing.
  if (fileSystem.existsSync(derived)) {
    return derived;
  }

  const userDataPath = () =>
    path.join(
      getUserDataDirectory({ platform, environment, homeDirectory }),
      ...fallbackSegments
    );

  if (isBundledSourceDirectory({ sourceDirectory, fileSystem })) {
    return userDataPath();
  }

  // Covers the read-only install that is not a bundle: a root-owned global npm
  // prefix, or a .app bundle's Resources directory. Checked on the nearest
  // ancestor that exists, because the derived path itself does not yet.
  if (!fileSystem.isWritable(nearestExistingAncestor(derived, fileSystem))) {
    return userDataPath();
  }

  return derived;
};
