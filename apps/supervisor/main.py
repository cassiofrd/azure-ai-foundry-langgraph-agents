from __future__ import annotations

from graphs.supervisor_graph import build_supervisor_graph
from shared.conversation_store import build_conversation_store
from shared.foundry_client import get_openai_client
from shared.settings import load_settings


def main() -> None:
    settings = load_settings()
    conversation_store = build_conversation_store(settings)
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: get_openai_client(settings),
        conversation_store=conversation_store,
    )

    session_id = settings.default_session_id
    conversation_response_id = conversation_store.load(
        session_id
    ).previous_response_id

    print(
        f"{settings.app_name} v{settings.app_version} "
        f"({settings.app_environment})"
    )
    print(f"Conversation store: {conversation_store.backend_name}")
    print(
        "Conversation persistence: "
        + ("enabled" if conversation_store.is_persistent else "disabled")
    )
    if conversation_store.startup_warning:
        print(f"WARNING: {conversation_store.startup_warning}")
    print(f"Active session: {session_id}")
    print("Type 'exit' to finish.")
    print("Type 'reset' to clear the active session.")
    print("Type 'session <id>' to switch sessions.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if user_input.lower() == "reset":
            conversation_store.clear(session_id)
            conversation_response_id = None
            print(f"Session '{session_id}' cleared.\n")
            continue

        if user_input.lower().startswith("session "):
            requested_session = user_input[8:].strip()
            if not requested_session:
                print("Session ID cannot be empty.\n")
                continue
            session = conversation_store.load(requested_session)
            session_id = session.session_id
            conversation_response_id = session.previous_response_id
            print(f"Active session: {session_id}\n")
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
                "session_id": session_id,
            }
        )

        conversation_response_id = result["conversation_response_id"]

        print(f"Route: {result['intent']}")
        print(f"Specialist: {result['agent']}")
        print(f"Session: {result.get('session_id', session_id)}")
        if result.get("memory_last_entity"):
            print(f"Memory entity: {result['memory_last_entity']}")
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
