import crypto from "crypto";
import { getApiAuthToken, isApiAuthEnabled } from "./rag/config.js";

const PUBLIC_PATH_PREFIXES = ["/health", "/ready", "/uploads"];

const getProvidedToken = (req) => {
  const apiKeyHeader = req.get("x-api-key")?.trim();

  if (apiKeyHeader) {
    return apiKeyHeader;
  }

  const authorizationHeader = req.get("authorization")?.trim() ?? "";
  const bearerMatch = authorizationHeader.match(/^bearer\s+(.+)$/i);

  return bearerMatch?.[1]?.trim() ?? "";
};

const constantTimeEqual = (left, right) => {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);

  if (leftBuffer.length !== rightBuffer.length) {
    return false;
  }

  return crypto.timingSafeEqual(leftBuffer, rightBuffer);
};

export const requireApiAuth = (req, res, next) => {
  if (
    !isApiAuthEnabled() ||
    PUBLIC_PATH_PREFIXES.some((prefix) => req.path.startsWith(prefix))
  ) {
    next();
    return;
  }

  const configuredToken = getApiAuthToken().trim();

  if (!configuredToken) {
    res.status(500).json({
      error: "API authentication is enabled, but API_AUTH_TOKEN is not configured.",
    });
    return;
  }

  const providedToken = getProvidedToken(req);

  if (!providedToken || !constantTimeEqual(providedToken, configuredToken)) {
    res.status(401).json({
      error: "Unauthorized.",
    });
    return;
  }

  next();
};
