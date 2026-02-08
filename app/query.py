from pathlib import Path
from collections import defaultdict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# Config
DB_PATH = "vectorstore/chroma"
OUT_DIR = "eval"
OUT_DIR_PATH = Path(OUT_DIR)
OUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4.1-nano"
EMBED_MODEL = "text-embedding-3-small"

K = 8
FETCH_K = 30
LAMBDA_MULT = 0.5

ABSTENTION_TEXT = "Ich weiß es nicht auf Basis der vorliegenden Quellen."
MIN_DOCS = 2


# Setup
llm = ChatOpenAI(model=MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT},
)


def summarize_sources(source_documents):
    pages_by_src = defaultdict(set)
    topic_by_src = {}

    for doc in source_documents:
        src = doc.metadata.get("source", "unknown")
        src = Path(src).as_posix()
        topic = doc.metadata.get("topic")

        page = doc.metadata.get("page")
        if page is not None:
            page = page + 1
            pages_by_src[src].add(page)
        else:
            pages_by_src[src].add(None)
        if topic:
            topic_by_src[src] = topic

    lines = []
    for src in sorted(pages_by_src.keys()):
        topic = topic_by_src.get(src)
        pages = sorted([p for p in pages_by_src[src] if p is not None])

        if pages:
            pages_str = ", ".join(str(p) for p in pages)
            line = (
                f"- {src}"
                + (f" (topic={topic})" if topic else "")
                + f", S. {pages_str}"
            )
        else:
            line = f"- {src}" + (f" (topic={topic})" if topic else "")

        lines.append(line)

    return "\n".join(lines)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


prompt = ChatPromptTemplate.from_template(
    """
Du bist ein wissenschaftlich vorsichtiger Assistent.
Beantworte die Frage ausschließlich anhand des bereitgestellten Kontexts.

Regeln:
1) Wenn der Kontext nicht ausreicht, antworte exakt:
"Ich weiß es nicht auf Basis der vorliegenden Quellen."
2) Erfinde keine Fakten oder Zahlen.
3) Antworte in ein bis zwei kurzen, präzisen Sätzen und auf Deutsch.

Kontext:
{context}

Frage:
{question}

Antwort:
"""
)

answer_chain = (
    {
        "context": RunnableLambda(lambda x: format_docs(x["docs"])),
        "question": RunnableLambda(lambda x: x["question"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)


def add_docs(x):
    docs = retriever.invoke(x["question"])
    return {**x, "docs": docs, "abstain": len(docs) < MIN_DOCS}


def answer_or_abstain(x):
    if x["abstain"]:
        return ABSTENTION_TEXT
    else:
        return answer_chain.invoke(x)


def finalize_abstain(result):
    if result["answer"].strip() == ABSTENTION_TEXT:
        result["abstain"] = True
    return result


rag_chain = (
    RunnableLambda(add_docs)
    | RunnableParallel(
        docs=RunnableLambda(lambda x: x["docs"]),
        answer=RunnableLambda(answer_or_abstain),
        abstain=RunnableLambda(lambda x: x["abstain"]),
    )
    | RunnableLambda(finalize_abstain)
)


def ask(question: str) -> dict:
    # Ausgabe: answer, abstain, docs, sources_str
    res = rag_chain.invoke({"question": question})
    sources_str = summarize_sources(res["docs"])
    return {
        "answer": res["answer"],
        "abstain": bool(res["abstain"]),
        "docs": res["docs"],
        "sources_str": sources_str,
    }
