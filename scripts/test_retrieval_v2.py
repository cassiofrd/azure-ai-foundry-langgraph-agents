from shared.search_service_v2 import SearchServiceV2


QUERY = "current inventory of BOLT-M10 at PLANT-BH"
TOP = 5


def print_results(title, results):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"#{rank} score={result.get('@search.score')}")
        print(f"chunk_id: {result.get('chunk_id')}")
        print(f"domain: {result.get('domain')}")
        print(f"document_type: {result.get('document_type')}")
        print(f"entity_ids: {result.get('entity_ids')}")
        print(f"aliases: {result.get('aliases')}")
        print(f"location_ids: {result.get('location_ids')}")
        print(f"is_global: {result.get('is_global')}")
        print("content:")
        print(result.get("content"))
        print("-" * 80)


def main():
    service = SearchServiceV2()

    print(f"Query: {QUERY}")

    lexical = service.lexical_search(QUERY, top=TOP)
    vector = service.vector_search(QUERY, top=TOP)
    hybrid = service.hybrid_search(QUERY, top=TOP)

    print_results("LEXICAL SEARCH", lexical)
    print_results("VECTOR SEARCH", vector)
    print_results("HYBRID SEARCH", hybrid)


if __name__ == "__main__":
    main()
