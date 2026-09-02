"""Create the small sample PDF shipped with the project."""

from pathlib import Path

import pymupdf


OUTPUT = Path(__file__).resolve().parents[1] / "sample_documents" / "ai_at_work.pdf"

PAGES = [
    (
        "Artificial Intelligence at Work",
        "Artificial intelligence systems can support employees by searching large document "
        "collections, classifying requests and drafting answers. A useful system should not "
        "only generate fluent text. It should also make its evidence visible so that users can "
        "verify important claims. Retrieval augmented generation, usually called RAG, combines "
        "information retrieval with a language model. The retrieval component first selects "
        "relevant passages. The language model then receives those passages as context.",
    ),
    (
        "Benefits and Risks",
        "RAG can reduce hallucinations because the answer is grounded in selected source text. "
        "It does not eliminate errors. Poor document quality, unsuitable chunk sizes or an "
        "ambiguous question can still produce weak results. Organizations should evaluate both "
        "retrieval quality and answer quality. Sensitive documents also require appropriate data "
        "protection. Local embedding models keep the source text on the user's computer, while "
        "cloud-based language models may transmit retrieved passages to an external provider.",
    ),
    (
        "Evaluation",
        "A small evaluation set should contain representative questions and expected evidence. "
        "For retrieval, teams can measure whether a relevant passage appears among the top "
        "results. For generated answers, reviewers can assess correctness, completeness and "
        "whether citations actually support each claim. Comparing multiple chunk sizes and "
        "retrieval depths helps identify a reasonable configuration for the document collection.",
    ),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for title, body in PAGES:
        page = document.new_page()
        page.insert_text((72, 72), title, fontsize=18)
        text_rect = pymupdf.Rect(72, 110, 523, 760)
        page.insert_textbox(text_rect, body, fontsize=11, lineheight=1.4)
    document.save(OUTPUT)
    document.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
