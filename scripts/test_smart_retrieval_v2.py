from shared.search_service_v2 import SearchServiceV2


QUERIES = [
    "current inventory of BOLT-M10 at PLANT-BH",
    "reorder point for BOLT-M10 at PLANT-BH",
    "shipping options for M10",
    "replenishment procedure for M10",
]


def print_result(result):
    print()
    print("=" * 90)
    print(f"Query: {result['query']}")
    print(f"Resolved entity: {result['entity_id']}")
    print(f"Resolved location: {result['location_id']}")
    print(f"Resolved domain: {result['domain']}")
    print(f"Filter: {result['filter']}")
    print("=" * 90)

    for rank, item in enumerate(result["results"], start=1):
        print(
            f"#{rank} score={item.get('@search.score')} "
            f"[{item.get('domain')}/{item.get('document_type')}]"
        )
        print(f"chunk_id: {item.get('chunk_id')}")
        print(f"entity_ids: {item.get('entity_ids')}")
        print(f"location_ids: {item.get('location_ids')}")
        print(f"is_global: {item.get('is_global')}")
        print("content:")
        print(item.get("content"))
        print("-" * 90)


def main():
    service = SearchServiceV2()

    for query in QUERIES:
        result = service.smart_hybrid_search(query, top=5)
        print_result(result)


if __name__ == "__main__":
    main()
