import csv
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from query import llm, OUT_DIR_PATH, rag_chain, summarize_sources
from dotenv import load_dotenv

load_dotenv()

# Baseline: No-RAG
no_rag_prompt = ChatPromptTemplate.from_template(
    """
Du bist ein wissenschaftlich vorsichtiger Assistent.
Antworte in ein bis zwei kurzen, präzisen Sätzen und auf Deutsch.

Frage:
{question}

Antwort:
"""
)
no_rag_chain = no_rag_prompt | llm | StrOutputParser()


# Fragen (Mini Testset)
QUESTIONS = [
    "Welche gesundheitlichen Effekte hat regelmäßige körperliche Aktivität?",
    "Was ist der Schlüssel in der Prävention und Behandlung von Typ-2-Diabetes?",
    "Schädigt Kreatin die Nieren?",
    "Welche Rolle spielt Ernährung bei der Prävention kardiovaskulärer Erkrankungen?",
    "Wie beeinflusst Schlaf die psychische Gesundheit?",
    "Welche Effekte haben Ausdauer- und Krafttraining auf die Körperzusammensetzung?",
    "Sind Low-Carb-Diäten mit der Sterblichkeit assoziiert?",
    "Welche Faktoren beeinflussen die Adhärenz an gesundheitsfördernde Verhaltensweisen?",
    "Was sind häufige Missverständnisse bei Protein-Supplementen?",
    "Welche Bedeutung hat Lebensstilmodifikation in der kardiovaskulären Prävention?",
]


def run():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR_PATH / f"mini_eval_{ts}.csv"

    rows = []
    rag_abstains = 0
    no_rag_abstains = 0

    for q in QUESTIONS:
        rag_res = rag_chain.invoke({"question": q})
        rag_answer = rag_res["answer"].strip()
        rag_sources = summarize_sources(rag_res["docs"])
        rag_abstain = bool(rag_res["abstain"])

        if rag_abstain:
            rag_abstains += 1

        no_rag_answer = no_rag_chain.invoke({"question": q}).strip()
        if no_rag_answer == "Ich weiß es nicht.":
            no_rag_abstains += 1

        rows.append(
            {
                "question": q,
                "rag_answer": rag_answer,
                "rag_abstain": rag_abstain,
                "rag_sources": rag_sources,
                "no_rag_answer": no_rag_answer,
                "no_rag_abstain": (no_rag_answer == "Ich weiß es nicht."),
            }
        )

    with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "rag_answer",
                "rag_abstain",
                "rag_sources",
                "no_rag_answer",
                "no_rag_abstain",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV geschrieben: {out_file.as_posix()}")
    print(f"RAG Abstentions: {rag_abstains}/{len(QUESTIONS)}")
    print(f"No-RAG Abstentions: {no_rag_abstains}/{len(QUESTIONS)}")


if __name__ == "__main__":
    run()
