import argparse
import json
import sys

from src.query import answer_question, query_rag_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Q&A Bot Query CLI")
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The question to ask the document knowledge base."
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=3,
        help="Number of retrieved documents/chunks to use (default: 3)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the response as raw JSON."
    )

    args = parser.parse_args()

    try:
        result = query_rag_pipeline(user_query=args.query, k=args.k)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n=== Grounded Answer ===")
            print(result["answer"])
            print("\n=== Sources ===")
            if not result["sources"]:
                print("No source documents retrieved.")
            for i, src in enumerate(result["sources"], start=1):
                file_name = src["file"]
                page_num = src["page"]
                snippet = src["snippet"].strip()
                # Print a clean, formatted representation of the sources
                print(f"[{i}] File: {file_name} | Page: {page_num if page_num != -1 else 'N/A'}")
                print(f"    Snippet: {snippet[:150]}..." if len(snippet) > 150 else f"    Snippet: {snippet}")
                print("-" * 40)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
