import os
import time
from pathlib import Path
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        f"GROQ_API_KEY not found in {ENV_FILE}"
    )


# ============================================================
# CUSTOM ERRORS
# ============================================================

class LLMServiceError(Exception):
    """Raised when the LLM/API service fails."""
    pass


class RetrievalServiceError(Exception):
    """Raised when the retrieval pipeline fails."""
    pass


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    groq_api_key=GROQ_API_KEY
)


# ============================================================
# GRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):
    question: str
    chat_history: List[Dict]
    rewritten_query: str
    retrieved_chunks: List[Dict]
    context: str
    answer: str

    # Observability / error state
    status: str
    error_type: str
    error_message: str


# ============================================================
# CASUAL ROUTER
# ============================================================

CASUAL_PHRASES = {
    "hi", "hello", "hey", "thanks", "thank you", 
    "ok", "okay", "cool", "nice", "great", "got it",
    "good", "alright", "sure"
}

def is_casual(text: str) -> bool:
    """Check if message is casual greeting/acknowledgment."""
    cleaned = text.lower().strip().strip("?!.,")
    return cleaned in CASUAL_PHRASES or len(cleaned.split()) <= 2


# ============================================================
# NODE 1 — QUERY REWRITING
# ============================================================

REWRITE_PROMPT = """
You are the QUERY REWRITER for a podcast-grounded question answering system.

Your ONLY task is to transform the user's latest message into a standalone
search query for retrieving relevant transcript passages.

You MUST NOT answer the question.

========================
CORE PRINCIPLE
========================

Preserve the user's exact intent.

Use conversation history ONLY when necessary to understand references such as:

- "why?"
- "why so?"
- "how?"
- "how does that work?"
- "what about Einstein?"
- "is it related?"
- "what did you mean?"
- "are those related?"
- "what about the previous one?"

Do not add facts merely because they are commonly associated with the topic.

========================
RULES
========================

1. SELF-CONTAINED QUESTIONS

If the latest user message is already a complete, understandable question,
return it unchanged.

Example:

History:
USER: What does Shannon's theory say about entropy?

Latest:
USER: What is the difference between entropy and information?

Return:

What is the difference between entropy and information?


2. FOLLOW-UP QUESTIONS

If the latest message depends on the conversation, resolve the reference
using the minimum amount of necessary context.

Example:

USER: What is the Unruh effect?
ASSISTANT: [answer about Unruh effect]

USER: Why does that happen?

Return:

Why does the Unruh effect happen?


3. IMMEDIATELY RELEVANT CONTEXT

Prefer the most recent topic that clearly answers the reference.

Do NOT combine unrelated older topics unless the user explicitly connects them.

Example:

USER: Tell me about black hole radiation.
ASSISTANT: [answer]
USER: What about DNA?
USER: Why is that important?

The final question should refer to DNA, not black hole radiation.


4. EXPLICIT COMPARISONS

If the user explicitly asks whether two or more topics are related,
preserve all mentioned topics.

Example:

"Is Shannon's information theory related to Transformers?"

Return:

Is Shannon's information theory related to Transformers?


5. PRONOUNS AND SHORT REFERENCES

Resolve:

"it"
"that"
"this"
"they"
"those"
"above"
"previous"
"the other one"

using the conversation context.

Do not guess when the reference is genuinely ambiguous.


6. AMBIGUOUS REFERENCES

If a reference cannot be resolved confidently, preserve the user's wording
rather than inventing a topic.

Example:

USER: Tell me about Shannon.
USER: What about the other one?

Return:

What about the other one?


7. REFUSAL FOLLOW-UPS

If the previous assistant response was a refusal such as:

"I couldn't find enough information in the provided episodes."

and the user asks:

"why?"
"why man?"
"why not?"
"what do you mean?"

rewrite the question as:

Why couldn't the previous question be answered from the provided podcast episodes?

Do NOT invent a podcast topic.


8. UNRELATED QUESTIONS

If the latest question is clearly a new, unrelated question, preserve it as
a new standalone query.

Do not force it into the previous podcast topic.


9. DO NOT BROADEN

Never add concepts, entities, papers, people, or relationships that the user
did not imply.

Bad:

User: "What is attention?"

Bad rewrite:
"What is attention in Transformers, neural networks, and information theory?"

Good rewrite:
"What is attention?"


10. DO NOT ANSWER

Your output must be ONLY the search query.

Do not provide:

- explanations
- reasoning
- commentary
- quotation marks
- prefixes
- bullet points
- "Rewritten query:"
- answers

========================
CONVERSATION
========================

{history_text}

========================
LATEST USER MESSAGE
========================

{question}

========================
OUTPUT
========================

Return ONLY the final standalone search query.
"""


def rewrite_query(state: GraphState):

    start_time = time.perf_counter()

    question = state["question"].strip()
    history = state.get("chat_history", [])

    # --------------------------------------------------------
    # Casual greeting router
    # --------------------------------------------------------
    if is_casual(question):
        print(f"\n[QUERY REWRITE]")
        print(f"Input: {question}")
        print("[QUERY REWRITE] Casual greeting detected — skipping LLM")
        return {
            "rewritten_query": question,
            "status": "casual"
        }

    # --------------------------------------------------------
    # No conversation history
    # --------------------------------------------------------

    if not history:

        print(
            f"\n[QUERY REWRITE]"
            f"\nInput: {question}"
        )

        print(
            f"[QUERY REWRITE] "
            f"Skipped — no conversation history"
        )

        return {
            "rewritten_query": question,
            "status": "success"
        }

    # --------------------------------------------------------
    # Keep recent conversation only (last 4 messages, truncated)
    # --------------------------------------------------------

    history_text = "\n".join(
        f"{message['role']}: {message['content'][:1200]}"
        for message in history[-4:]
    )

    prompt = REWRITE_PROMPT.format(
        history_text=history_text,
        question=question
    )

    print("\n[QUERY REWRITE]")
    print(f"Input: {question}")

    try:

        response = llm.invoke(prompt)

        rewritten_query = response.content.strip()

        if not rewritten_query:
            rewritten_query = question

        elapsed = time.perf_counter() - start_time

        print(
            f"Output: {rewritten_query}"
        )

        print(
            f"[QUERY REWRITE] "
            f"latency={elapsed:.3f}s"
        )

        return {
            "rewritten_query": rewritten_query,
            "status": "success"
        }

    except Exception as e:

        elapsed = time.perf_counter() - start_time

        print(
            f"[QUERY REWRITE ERROR] "
            f"type={type(e).__name__} "
            f"latency={elapsed:.3f}s"
        )

        print(
            f"[QUERY REWRITE ERROR DETAILS] {e}"
        )

        raise LLMServiceError(
            "The language model failed while rewriting the query."
        ) from e


# ============================================================
# NODE 2 — RETRIEVAL
# ============================================================

def retrieve(state: GraphState):

    start_time = time.perf_counter()

    from retrieval import retrieve as retrieve_chunks

    query = state["rewritten_query"]

    print(
        f"\n[RETRIEVAL]"
        f"\nQuery: {query}"
    )

    try:

        chunks = retrieve_chunks(query)

        elapsed = time.perf_counter() - start_time

        print(
            f"[RETRIEVAL] "
            f"Retrieved {len(chunks)} chunks"
        )

        print(
            f"[RETRIEVAL] "
            f"latency={elapsed:.3f}s"
        )

        # ----------------------------------------------------
        # No chunks is NOT a retrieval crash.
        # It simply means no useful evidence was found.
        # ----------------------------------------------------

        if not chunks:

            print(
                "[RETRIEVAL] No chunks returned"
            )

            return {
                "retrieved_chunks": [],
                "status": "no_evidence"
            }

        return {
            "retrieved_chunks": chunks,
            "status": "success"
        }

    except Exception as e:

        elapsed = time.perf_counter() - start_time

        print(
            f"[RETRIEVAL ERROR] "
            f"type={type(e).__name__} "
            f"latency={elapsed:.3f}s"
        )

        print(
            f"[RETRIEVAL ERROR DETAILS] {e}"
        )

        raise RetrievalServiceError(
            "The retrieval system failed while searching the episodes."
        ) from e


# ============================================================
# NODE 3 — BUILD CONTEXT
# ============================================================

def format_timestamp(seconds: float) -> str:

    seconds = max(0, float(seconds))

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


def clean_episode_name(title: str) -> str:

    if not title:
        return "Unknown episode"

    for extension in [
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".aac",
        ".ogg",
        ".opus",
        ".wma"
    ]:

        if title.lower().endswith(extension):

            title = title[
                :-len(extension)
            ]

            break

    return title.replace("_", " ").strip()


def build_context(state: GraphState):

    start_time = time.perf_counter()

    context_parts = []

    for index, chunk in enumerate(
        state.get("retrieved_chunks", []),
        start=1
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        episode = metadata.get(
            "title",
            metadata.get(
                "video_id",
                "Unknown episode"
            )
        )

        episode = clean_episode_name(
            episode
        )

        start = metadata.get(
            "start",
            0
        )

        end = metadata.get(
            "end",
            0
        )

        start_time_formatted = format_timestamp(
            start
        )

        end_time_formatted = format_timestamp(
            end
        )

        text = chunk.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        context_parts.append(
            f"""
SOURCE {index}

Episode:
{episode}

Timestamp:
{start_time_formatted} - {end_time_formatted}

Transcript:
{text}
"""
        )

    context = "\n---\n".join(
        context_parts
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"\n[CONTEXT]"
        f"\nSources used: {len(context_parts)}"
        f"\nCharacters: {len(context)}"
        f"\nlatency={elapsed:.3f}s"
    )

    if not context.strip():

        print(
            "[CONTEXT] No usable transcript evidence"
        )

        return {
            "context": "",
            "status": "no_evidence"
        }

    return {
        "context": context,
        "status": "success"
    }


# ============================================================
# NODE 4 — GENERATE ANSWER
# ============================================================

ANSWER_PROMPT = """
You are Fermi Podcast Companion, a trustworthy conversational assistant
that answers questions ONLY from the supplied podcast transcript evidence.

Your job is to answer the user's CURRENT question using the provided
transcript sources.

You are NOT a general-purpose knowledge assistant.

========================
SOURCE OF TRUTH
========================

The transcript sources are the ONLY authoritative source.

You may use:

- facts explicitly stated in the transcripts
- explanations explicitly given in the transcripts
- relationships explicitly stated in the transcripts
- comparisons that can be directly supported by multiple transcript sources

You MUST NOT use outside knowledge, even if you know the answer.
You can do normal grettings, You are a teacher.
Your internal knowledge must never be used to fill missing evidence.

========================
QUESTION
========================

{question}

========================
SEARCH QUERY
========================

{rewritten_query}

========================
TRANSCRIPT EVIDENCE
========================

{context}

========================
EVIDENCE RULES
========================

1. DIRECT SUPPORT

Every factual claim in your answer must be supported by one or more
provided transcript sources.

If a claim cannot be supported by the evidence, do not make the claim.


2. MULTIPLE SOURCES

When the question asks about multiple topics or episodes, use evidence
from each relevant source.

Clearly distinguish what each episode contributes.

Do not assume that two concepts are related merely because they appear
in different episodes.


3. CROSS-EPISODE SYNTHESIS

You may synthesize information across episodes ONLY when the relationship
is directly supported by the transcripts.

For example:

If Episode A explains information theory and Episode B explains DNA,
you may explain both concepts separately.

But do NOT claim:

"DNA is an implementation of Shannon's theory"

unless the supplied evidence explicitly supports that relationship.


4. INFERENCE

Limited reasoning is allowed only when it is a direct logical consequence
of the supplied evidence.

Do NOT introduce outside scientific, historical, mathematical, or technical
knowledge to strengthen an answer.


5. WEAK EVIDENCE

If retrieved passages are only loosely related to the question, DO NOT
construct an answer from them.

Instead return exactly:

I couldn't find enough information in the provided episodes.


6. MISSING EVIDENCE

If the transcripts do not provide enough information to answer the question,
return exactly:

I couldn't find enough information in the provided episodes.


7. UNSUPPORTED TOPICS

If the user asks about something outside the supplied podcast material and
the evidence does not support an answer, return exactly:

I couldn't find enough information in the provided episodes.


8. FOLLOW-UP QUESTIONS

Answer the CURRENT question using the conversation context already reflected
in the question.

Do not repeat the entire previous conversation.

If the current question is:

"why?"

answer why the immediately relevant topic behaves as discussed.

If the current question cannot be grounded in the provided evidence,
refuse.


9. SIMPLE EXPLANATIONS

If the user asks for a simpler explanation, simplify ONLY what the transcript
supports.

Do not add external examples or analogies that introduce new facts.


10. COMPARISONS

For questions asking:

- "how are X and Y related?"
- "what is the difference?"
- "compare X and Y"
- "are these connected?"

first identify evidence for X and evidence for Y.

Only state a relationship if the relationship is supported by the sources.

If the evidence supports only the individual concepts, say so explicitly.


========================
TIMESTAMPS
========================

Whenever making a claim based on a specific transcript passage, include its
source timestamp.

Use exactly this format:

[Episode: <episode name>, <MM:SS>-<MM:SS>]

Never invent timestamps.

Only use timestamps provided in the transcript evidence.

If several claims come from different passages, cite the relevant passage
after each claim or group of claims.


========================
SOURCE NAMES
========================

Use the episode name supplied in the evidence.

Do not invent episode titles.

Do not mention filenames unless they are explicitly presented as episode names.


========================
ANSWER STYLE
========================

Be:

- concise
- conversational
- technically accurate
- evidence-grounded
- transparent about uncertainty

Do not dump transcript passages.

Do not mention internal implementation details.

Never mention:

- ChromaDB
- embeddings
- vector search
- reranking
- retrieval
- prompts
- LangGraph
- query rewriting
- system architecture


========================
FINAL SAFETY CHECK
========================

Before answering, internally verify:

1. Is every factual claim supported by the transcript evidence?
2. Did I accidentally use outside knowledge?
3. Did I correctly distinguish different episodes?
4. Did I invent any relationship between concepts?
5. Did I invent a timestamp?
6. Does the answer actually answer the CURRENT question?
7. Is the evidence strong enough to justify an answer?

If any important claim fails these checks, refuse rather than guess.

========================
ANSWER
========================
"""


def generate_answer(state: GraphState):

    start_time = time.perf_counter()

    question = state["question"]
    rewritten_query = state["rewritten_query"]
    context = state["context"]

    # --------------------------------------------------------
    # Casual greeting response
    # --------------------------------------------------------
    if state.get("status") == "casual":
        print(
            "\n[ANSWER]"
            "\nStatus: casual"
        )
        return {
            "answer": "Hello! How can I help you learn from the podcast?",
            "status": "casual"
        }

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not context.strip():

        print(
            "\n[ANSWER]"
            "\nStatus: no_evidence"
        )

        return {
            "answer": (
                "I couldn't find enough information "
                "in the provided episodes."
            ),
            "status": "no_evidence"
        }

    prompt = ANSWER_PROMPT.format(
        question=question,
        rewritten_query=rewritten_query,
        context=context
    )

    print(
        "\n[GENERATING ANSWER]"
    )

    try:

        response = llm.invoke(prompt)

        answer = response.content.strip()

        elapsed = time.perf_counter() - start_time

        print(
            f"[LLM] "
            f"latency={elapsed:.3f}s"
        )

        if not answer:

            print(
                "[LLM] Empty response received"
            )

            answer = (
                "I couldn't find enough information "
                "in the provided episodes."
            )

        print(
            "[ANSWER] Successfully generated"
        )

        return {
            "answer": answer,
            "status": "success"
        }

    except Exception as e:

        elapsed = time.perf_counter() - start_time

        print(
            f"\n[LLM ERROR]"
            f"\ntype={type(e).__name__}"
            f"\nlatency={elapsed:.3f}s"
        )

        print(
            f"[LLM ERROR DETAILS] {e}"
        )

        raise LLMServiceError(
            "The language model service failed while generating the answer."
        ) from e


# ============================================================
# NODE 5 — SAVE MEMORY
# ============================================================

def save_memory(state: GraphState):

    history = state.get(
        "chat_history",
        []
    )

    updated_history = history + [

        {
            "role": "user",
            "content": state["question"]
        },

        {
            "role": "assistant",
            "content": state["answer"]
        }

    ]

    # Keep the latest 10 messages
    updated_history = updated_history[-10:]

    return {
        "chat_history": updated_history
    }


# ============================================================
# GRAPH
# ============================================================

def create_graph():

    graph = StateGraph(
        GraphState
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query
    )

    graph.add_node(
        "retrieve",
        retrieve
    )

    graph.add_node(
        "build_context",
        build_context
    )

    graph.add_node(
        "generate_answer",
        generate_answer
    )

    graph.add_node(
        "save_memory",
        save_memory
    )

    graph.set_entry_point(
        "rewrite_query"
    )

    graph.add_edge(
        "rewrite_query",
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "build_context"
    )

    graph.add_edge(
        "build_context",
        "generate_answer"
    )

    graph.add_edge(
        "generate_answer",
        "save_memory"
    )

    graph.add_edge(
        "save_memory",
        END
    )

    # --------------------------------------------------------
    # LangGraph checkpoint memory
    # --------------------------------------------------------

    memory = MemorySaver()

    return graph.compile(
        checkpointer=memory
    )


# ============================================================
# APPLICATION GRAPH
# ============================================================

app_graph = create_graph()