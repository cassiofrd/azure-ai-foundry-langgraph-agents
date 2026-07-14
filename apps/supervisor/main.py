from __future__ import annotations

from graphs.supervisor_graph import build_supervisor_graph
from shared.foundry_client import get_openai_client
from shared.settings import load_settings


def main() -> None:
    settings = load_settings()
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: get_openai_client(settings),
    )

    print(
        f"{settings.app_name} v{settings.app_version} "
        f"({settings.app_environment})"
    )
    print("Type 'exit' to finish.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = graph.invoke(
            {
                "user_input": user_input,
                "intent": "general",
                "answer": "",
            }
        )

        print(f"Route: {result['intent']}")
        print(f"Agent: {result['answer']}\n")


if __name__ == "__main__":
    main()
