"""
An example script to ask a question and retrieve documents in the corpus.
"""

from argparse import ArgumentParser
from requests import post

def ask_question(host_port: str, nb_search: int, nb_tokens: int, sources: bool, question: str) -> None:
    """Asks a question about the corpus."""
    response = post(
        f"http://{host_port}/rag",
        json={
            "question": question,
            "search_results": nb_search,
            "generation_tokens": nb_tokens,
            "append_sources": sources,
        },
        stream=True,
    )
    for txt in response.iter_content(decode_unicode=True):
        if txt:
            print(txt, end="", flush=True)

def main(host_port: str, nb_search: int, nb_tokens: int, sources: bool) -> None:
    """
    Main function: reads a question from stdin and
    search for answers in the corpus.
    """
    question: str = input("Your Question: ")
    ask_question(host_port, nb_search, nb_tokens, sources, question)


if __name__ == "__main__":
    parser = ArgumentParser("ask question: une petite démo de RAG à l'UCL")
    parser.add_argument("-n", '--nb-search', default=5,   help="Nombre de résultats lors de la phase de recherche")
    parser.add_argument("-t", "--nb-tokens", default=128, help="Nombre max de tokens générés pour la réponse")
    parser.add_argument("-s", "--sources", default=False, help="Afficher le détail des sources apres la réponse ?")

    args = parser.parse_args()
    main("localhost:8000", args.nb_search, args.nb_tokens, args.sources)
    #main("automemo.sipr.ucl.ac.be:8000")
