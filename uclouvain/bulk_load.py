"""
An example script to bulk load files from a folder.
"""

from pathlib import Path
from sys import stderr

from requests import post


def index_file(host_port: str, file: Path) -> None:
    """Executes the indexation of one file using the API endpoints."""
    response = post(
        f"http://{host_port}/indexation",
        params={"path_to_doc": file.absolute().as_uri()},
        files={"file": (file.name, file.open("rb"))},
    )
    if response.status_code == 200:
        print(response.text)
    else:
        print(response.text, file=stderr)


def main(host_port: str, folder: Path) -> None:
    """
    Main function: bulk loads the files in the embeddings database using
    the API endpoints defined in the backend.
    """
    for f in folder.rglob("*.pdf"):
        index_file(host_port, f)


if __name__ == "__main__":
    main("localhost:8000", Path("./corpus-example"))
