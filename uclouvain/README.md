# RAG-UCLouvain

Les documents sont dans le sous dossier 'corpus'

## Notes Techniques


### Extraction des données

L'essentiel des données consiste en des documents de type pdf. Ceux-ci peuvent 
être traités avec pymupdf qui intèrgre les capacités de Tesseract-OCR pour 
traiter les pages potentiellement problématiques.

De plus, pymupdf permet aussi d'extraire le texte des document:
* .pdf
* .docx
* .pptx
* .xlsx
* .html

#### Caveat
Une solution doit toutefois être trouvée pour les documents: 
* .jpg
* .7z
* .zip
* .db

### Vector Database

Pour faciliter l'indexation et la recherche sémantique, il est nécessaire d'avoir 
une vector database. La solution retenue actuellement est postgresql avec
l'extension pgvector. **MAIS** l'extension n'est pas compatible avec PG17 
sous windows et c'est pour cette raison que j'ai décidé de rester avec la version 
16.6 de postgres.

Pour l'installation de l'extension, il faut se référer à la page github du projet
[pgvector](https://github.com/pgvector/pgvector).

**Note:**
Apres l'installation sur windows, il faut installer l'extension, puis on devra aussi 
démarrer le server. Le plus simple c'est de le faire via: 
`cmd + r` , `services.msc` puis de trouver le service postgres et le run.


**Queries:**
Les queries par similarité se font dans la clause ORDER BY. D'un coté ca a du sens,
de l'autre, ca m'a un peu surpris.

#### Exemple: 
```
select path_to_doc, text from documents
order by embedding <=> '[0.0, 0.0, ... , 0.0, 0.0 ]'
limit 3;
```