import psycopg2
from psycopg2.extensions import connection
from transformers import AutoModel, AutoTokenizer

from embedding import EmbeddedChunk, EmbeddingPipeline, embedding_pipeline


def get_database() -> connection:
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="rag_ucl",
        user="postgres",
        password="admin",
    )


if __name__ == "__main__":
    device: str = "cuda"
    checkpoint: str = "intfloat/multilingual-e5-large-instruct"  # "xaviergillard/rag_ucl"  # "FacebookAI/xlm-roberta-large"  # "FacebookAI/xlm-roberta-base"  # "xaviergillard/rag_ucl"
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model: AutoModel = AutoModel.from_pretrained(checkpoint)
    pipe: EmbeddingPipeline = embedding_pipeline(tokenizer, model, device)
    text: str = """
    Instruct: Given a web search query, retrieve relevant passages that answer the query.
    Query: Quelles sont les décisions qui ont été prises par rapport au cours d'informatique pour les étudiants TAL ?
    """  # Path("./dummy.txt").read_text(encoding="utf8")
    chunk: EmbeddedChunk = pipe(text)[0]
    with get_database() as con:
        cur = con.cursor()
        qry = """
            select path_to_doc, text
            from documents
            order by embedding <=> %s
            limit 3;"""
        param: str = ", ".join(str(x) for x in chunk.embedding.tolist())
        param: str = f"[{param}]"
        cur.execute(qry, (param,))
        for record in cur.fetchall():
            print("#" * 80)
            print(record[0])
            print("-" * 80)
            print(record[1])
            print("#" * 80)
