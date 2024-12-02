-- CREATE EXTENSION vector;
CREATE TABLE documents
(
	id 				bigserial 	primary key not null,
	embedding 		vector(768)	not null,
	text 			varchar		not null,
	path_to_doc		varchar     not null
);