CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    source      text         NOT NULL,  -- file path inside docs/, used for citations
    chunk_index int          NOT NULL,  -- position of the chunk inside its file
    content     text         NOT NULL,
    embedding   vector(1024) NOT NULL,  -- bge-m3 dimensions
    UNIQUE (source, chunk_index)        -- makes re-ingestion idempotent
);
