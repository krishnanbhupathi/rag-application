CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
	id 				uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	source_type 	text NOT NULL CHECK (source_type IN ('pdf', 'url', 'markdown')),
	source_uri		text NOT NULL UNIQUE,
	title			text,
	content_hash	text NOT NULL,
	metadata 		jsonb NOT NULL DEFAULT '{}'::jsonb,
	created_at 		timestamptz NOT NULL DEFAULT now(),
	ingested_at 	timestamptz NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS chunks (
	id 				bigserial PRIMARY KEY,
	document_id		uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
	chunk_index 	int NOT NULL,
	content 		text NOT NULL,
	token_count		int NOT NULL,
	embedding 		vector(384),
	tsv 			tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
	metadata		jsonb NOT NULL DEFAULT '{}'::jsonb,
	created_at 		timestamptz NOT NULL DEFAULT now(),
	UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx		  ON chunks USING GIN (tsv);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
	ON chunks USING hnsw (embedding vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);
