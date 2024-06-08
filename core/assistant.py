from typing import Any, Optional
from queue import Queue
import threading

# Add phi.
from phi.assistant import Assistant

# Information Access
from phi.llm.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

# Tools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k

# Import yaver-stuff.
from speech_to_text import S2TSettings, speech2text, S2TResult

# Constants
SETTINGS: dict[str, Any] = {
    "DB_URL": "postgresql+psycopg://ai:ai@localhost:5532/ai",
}

VECTOR_DATABASE = PgVector2(
    collection="information",
    db_url=SETTINGS["DB_URL"],
    embedder=OllamaEmbedder(model="llama3")
)

STORAGE = PgAssistantStorage(
    table_name="pdf_assistant",
    db_url=SETTINGS["DB_URL"]
)


def yaver(new: bool = False, user: str = "default"):
    """Run the yaver."""
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: list[str] = STORAGE.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        llm=Ollama(model="knoopx/hermes-2-pro-mistral:7b-q8_0"),
        tools=[DuckDuckGo(), Newspaper4k()],
        run_id=run_id,
        user_id=user,
        storage=STORAGE,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        description="You are an helpful assistant that answers questions and provides information."
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    # Create a locked mutex to control the while loop in the
    # process function.
    mutex = threading.Lock()
    mutex.acquire()

    # Create a queue to pass transcribed text back to the main thread.
    speech_queue = Queue()

    # Start the process function in a separate thread.
    thread = threading.Thread(
        target=speech2text,
        args=(S2TSettings(), speech_queue, mutex)
    )
    thread.start()

    print("Listening...")

    while True:
        try:
            request: S2TResult = speech_queue.get()

            print("Duration: ", str(request.time_delta))
            if request.time_delta < 3:
                continue
            if len(request.text) <= 1:
                continue

            request.text = request.text.lower()
            assistant.print_response(request.text, markdown=True)
        except KeyboardInterrupt:
            mutex.release()
            print("Exiting...")
            break


if __name__ == "__main__":
    yaver()
