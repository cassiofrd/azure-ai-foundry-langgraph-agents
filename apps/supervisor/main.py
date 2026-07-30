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

    conversation_response_id: str | None = None

    print(
        f"{settings.app_name} v{settings.app_version} "
        f"({settings.app_environment})"
    )
    print("Type 'exit' to finish.")
    print("Type 'reset' to clear the current conversation memory.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if user_input.lower() == "reset":
            conversation_response_id = None
            print("Conversation memory cleared.\n")
            continue

        if not user_input:
            continue

        result = graph.invoke(
            {
                "user_input": user_input,
                "intent": "general",
                "agent": "general",
                "answer": "",
                "conversation_response_id": conversation_response_id,
            }
        )

        conversation_response_id = result["conversation_response_id"]

        print(f"Route: {result['intent']}")
        print(f"Specialist: {result['agent']}")
        if result.get("specialist_outputs"):
            participants = ", ".join(result["specialist_outputs"].keys())
            print(f"Participants: {participants}")
        if result.get("specialist_queries"):
            print("Specialist queries:")
            for name, query in result["specialist_queries"].items():
                print(f"- {name}: {query}")
        print(f"Answer: {result['answer']}\n")


if __name__ == "__main__":
    main()
