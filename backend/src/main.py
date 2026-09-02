import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# Paths / Environment
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


# --------------------------------------------------
# Knowledge Base Preparation
# --------------------------------------------------

def prepare_knowledge_base():
    """
    Transcribe all supplied audio files and build the
    vector database used by the conversational agent.
    """

    print("\n" + "=" * 60)
    print("Preparing Fermi Podcast Companion Knowledge Base")
    print("=" * 60)

    try:
        from transcribe import transcribe_all
        from ingestion import ingest_transcripts

        print("\n[1/2] Transcribing audio files...")
        transcribe_all()

        print("\n[2/2] Building vector database...")
        ingest_transcripts()

        print("\nKnowledge base prepared successfully.")

    except Exception as e:
        print("\n[KNOWLEDGE BASE ERROR]")
        print(f"{type(e).__name__}: {e}")
        raise


# --------------------------------------------------
# CLI Chat
# --------------------------------------------------

def run_chat():
    """
    Run the conversational agent from the terminal.
    """

    from graph import (
        app_graph,
        LLMServiceError,
        RetrievalServiceError,
    )

    print("\n" + "=" * 60)
    print("Fermi Podcast Companion")
    print("=" * 60)
    print("Ask questions about the supplied podcast episodes.")
    print("Type 'exit' or 'quit' to stop.\n")

    chat_history = []

    while True:
        try:
            question = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = app_graph.invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                }
            )

            answer = result.get(
                "answer",
                "I couldn't find enough information in the provided episodes."
            )

            status = result.get("status", "success")

            print(f"\nAssistant: {answer}")

            # Keep CLI history synchronized with the graph.
            chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            # Keep history bounded.
            chat_history = chat_history[-10:]

            if status != "success":
                print(f"\n[STATUS] {status}")

        except LLMServiceError as e:
            print(
                "\n[LLM ERROR] "
                "The AI service is temporarily unavailable. "
                "Please try again later."
            )
            print(f"[DEBUG] {type(e).__name__}: {e}")

        except RetrievalServiceError as e:
            print(
                "\n[RETRIEVAL ERROR] "
                "I couldn't search the podcast episodes right now."
            )
            print(f"[DEBUG] {type(e).__name__}: {e}")

        except Exception as e:
            print(
                "\n[UNEXPECTED ERROR] "
                "Something went wrong while processing your question."
            )
            print(f"[DEBUG] {type(e).__name__}: {e}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    """
    Prepare the knowledge base and start the CLI.
    """

    prepare_knowledge_base()
    run_chat()


if __name__ == "__main__":
    main()