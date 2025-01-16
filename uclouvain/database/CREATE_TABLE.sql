-- make sure pgvector is enabled
CREATE EXTENSION vector;
-- create the documents table which is used to hold all embeddings data.
CREATE TABLE  IF NOT EXISTS documents
(
	id 				bigserial 	    primary key not null,
	embedding 		vector(1024)	not null,
	text 			varchar		    not null,
	path_to_doc		varchar         not null
);

-- create index
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops);