"""
An example script to ask a question and retrieve documents in the corpus.
"""

from sys import stderr

from requests import post


def ask_question(host_port: str, question: str) -> None:
    """Asks a question about the corpus."""
    response = post(
        f"http://{host_port}/retrieval",
        params={
            "user_request": question,
            "nb_results": 5,
        },
    )
    if response.status_code == 200:
        messages = response.json()
        for m in messages:
            print("#" * 80)
            print(f"{m['path_to_doc']}")
            print("~" * 80)
            print(f"{m['text']}")

    else:
        print("!" * 80)
        print(response.text, file=stderr)
        print("!" * 80)


def main(host_port: str) -> None:
    """
    Main function: reads a question from stdin and
    search for answers in the corpus.
    """
    question: str = input("Your Question: ")
    ask_question(host_port, question)


if __name__ == "__main__":
    main("localhost:8000")
