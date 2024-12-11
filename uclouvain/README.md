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


### Lancement du conteneur Docker pour la vector db (postgres + pgvector)

```
docker run -e POSTGRES_PASSWORD="admin" -p 6666:5432 --name "rag_ucl" pgvector/pgvect
or:pg17
```

Note: la syntaxe pour le port mapping est {host}:{container}

## Modele d'embeddings

Il est important de noter qu'il ne suffit pas de prendre n'importe quel modele 
(ex camembert) pour calculer les embeddings des différents bouts de texte. Ce qui
fonctionne le mieux est d'utiliser un modele d'embedding pré-entrainé
(par ex `intfloat/multilingual-e5-large-instruct`). Pour ce modèle, il est 
nécessaire de donner un petit prompt avant la requete de l'utilisateur afin de primer
le modele pour qu'il soit capable de donner un embedding qui soit similaire
à l'embedding calculé pour le texte.

**note:** 
Celui ci semble intéressant aussi (et sans doute un peu plus performant.)

### Amélioration des performances

Pour que la technique marche bien, et parce qu'on recoit un embedding par token,
on va devoir faire un petit peu de *pooling* à la sortie du modèle. J'ai considéré
trois approches distinctes: 
1. average pooling (on prend la moyenne = ce qui est recommandé par le modele)
2. max pooling (qui pourrait permettre de faire ressortir au mieux les infos)
3. pas de pooling, on prend uniquement l'embedding du token de classification [CLS].

L'autre étape qui est nécessaire si on veut pouvoir améliorer les performances
de la technique, est de normaliser les vecteurs d'embeddings pour qu'ils aient
tous une magnitude de 1 (ce qui donne du coup un sens au calcul de la similarité
cosinus). A cet effet, j'ai simplement utilisé la normalisation L2 qui consiste
à diviser les poids du vecteurs par la norme L2 (= taille du vecteur dans un
espace euclidien généralisé ==> je l'oublie toujours).

### Calcul des embeddings

Vu qu'on veut calculer un embedding pour un texte complet, on doit utiliser un
modele de type "sentence transformer". Techniquement le modele n'a rien de 
spécial, mais il a été entrainé pour fournir des embedding d'un texte complet.
Un exemple d'algo pour entrainer ce type de modele est l'algo e5. 

La performance des models sentence transformers est mesurée et benchmarkée 
avec le bench MTEB (voir MTEB leaderboard).

Il est possible d'utiliser la lib sentence_transformers pour faire le calcul
des embeddings. Mais pour l'instant, j'ai préféré utiliser mon propre code
parce que je trouve que ca me permet de mieux comprendre tout ce qui se passe
et que par ailleurs, ca me permet de mieux controler comment les embeddings
sont crées.

