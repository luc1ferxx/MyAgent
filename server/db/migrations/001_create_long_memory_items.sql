CREATE TABLE IF NOT EXISTS __LONG_MEMORY_TABLE__ (
  memory_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  category TEXT NOT NULL,
  memory_key TEXT NULL,
  memory_value TEXT NULL,
  text TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'user_explicit',
  confidence REAL NOT NULL DEFAULT 1,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_used_at TIMESTAMPTZ NULL,
  archived BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS __LONG_MEMORY_TABLE___user_updated_idx
ON __LONG_MEMORY_TABLE__ (user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS __LONG_MEMORY_TABLE___user_category_key_idx
ON __LONG_MEMORY_TABLE__ (user_id, category, memory_key)
WHERE memory_key IS NOT NULL AND archived = FALSE;

CREATE INDEX IF NOT EXISTS __LONG_MEMORY_TABLE___user_category_text_idx
ON __LONG_MEMORY_TABLE__ (user_id, category, text)
WHERE memory_key IS NULL AND archived = FALSE;
