"""Tests for the synthetic evaluation retriever factory (Search & Ranking).

The factory is a SMOKE-TEST tool: these tests check wiring, corpus construction and
determinism only. They never assert ranking quality (FakeEmbedder is not semantic).
"""

import asyncio
from pathlib import Path

import pytest

from contracts.evaluation import EvalCase
from contracts.retrieval import Filters, Retriever, SearchResult
from eval.golden_set import load_golden_set
from kc.retrieval.evaluation import SyntheticEvaluationRetriever, create_evaluation_retriever

KC_HELP = "https://knowledgecenter.gcora.genpact.com/help/"
SAML = KC_HELP + "configure-sequence-for-saml-20-authentication"
KAFKA = KC_HELP + "kafka-configuration"
JES_SEQUENCE = KC_HELP + "job-execution-service-jes-overview"
JES_ORCHESTRATION = (
    KC_HELP + "installation-deployment-and-configuration-job-execution-service-jes-overview"
)
EMAIL = KC_HELP + "email-delivery-issues"


def _case(question: str, urls: list[str], product: str | None = "sequence") -> EvalCase:
    return EvalCase(question=question, expected_urls=urls, product=product, reference_answer=None)


def _build(cases: list[EvalCase]) -> SyntheticEvaluationRetriever:
    retriever = asyncio.run(create_evaluation_retriever(cases))
    assert isinstance(retriever, SyntheticEvaluationRetriever)
    return retriever


def _search(
    retriever: Retriever, query: str, filters: Filters | None = None, k: int = 5
) -> list[SearchResult]:
    return asyncio.run(retriever.search(query, filters, k))


# --- interface and labelling ---------------------------------------------------------------


def test_factory_returns_a_synthetic_retriever() -> None:
    retriever = _build([_case("How do I configure SAML?", [SAML])])
    assert retriever.is_synthetic is True


def test_supports_the_retriever_search_interface() -> None:
    retriever: Retriever = _build([_case("How do I configure SAML?", [SAML])])  # mypy check
    results = _search(retriever, "configure SAML", filters=Filters(products=["sequence"]), k=3)
    assert all(isinstance(r, SearchResult) for r in results)
    assert len(results) <= 3
    with pytest.raises(ValueError, match="k must be >= 1"):  # same validation as Retriever
        _search(retriever, "configure SAML", k=0)


def test_factory_can_be_awaited_inside_a_running_event_loop() -> None:
    async def runner() -> list[SearchResult]:
        retriever = await create_evaluation_retriever([_case("How do I configure SAML?", [SAML])])
        return await retriever.search("How do I configure SAML?")

    assert [r.url for r in asyncio.run(runner())] == [SAML]


# --- synthetic corpus ----------------------------------------------------------------------


def test_positive_case_produces_searchable_content() -> None:
    question = "How do I configure SAML 2.0 in SeQuence?"
    retriever = _build([_case(question, [SAML])])

    (document,) = retriever._documents
    assert document.url == SAML
    assert document.chunk_texts == (question,)
    assert document.doc_type == "other"
    assert document.version is None

    results = _search(retriever, question)
    assert [r.url for r in results] == [SAML]
    assert results[0].text == question


def test_repeated_url_shares_one_document_with_one_chunk_per_case() -> None:
    cases = [
        _case("How do I configure SAML?", [SAML]),
        _case("How to set up Kafka?", [KAFKA]),
        _case("איך מגדירים SAML?", [SAML]),
    ]
    retriever = _build(cases)

    assert [d.url for d in retriever._documents] == [SAML, KAFKA]  # first appearance
    saml = retriever._documents[0]
    assert saml.chunk_texts == ("How do I configure SAML?", "איך מגדירים SAML?")
    assert len(set(saml.chunk_ids)) == 2

    # Every indexed chunk is retrievable with a k larger than the corpus (no ranking claim).
    expected_ids = {chunk_id for d in retriever._documents for chunk_id in d.chunk_ids}
    assert len(expected_ids) == 3
    found = {r.chunk_id for r in _search(retriever, "איך מגדירים SAML?", k=10)}
    assert found == expected_ids


def test_url_listed_twice_in_one_case_gives_one_chunk() -> None:
    retriever = _build([_case("How do I configure SAML?", [SAML, SAML])])
    (document,) = retriever._documents
    assert document.chunk_texts == ("How do I configure SAML?",)


def test_case_with_several_urls_adds_its_question_to_each_document() -> None:
    question = "What is the Job Execution Service?"
    retriever = _build([_case(question, [JES_SEQUENCE, JES_ORCHESTRATION], product=None)])
    assert [d.url for d in retriever._documents] == [JES_SEQUENCE, JES_ORCHESTRATION]
    assert all(d.chunk_texts == (question,) for d in retriever._documents)


def test_negative_cases_create_no_documents() -> None:
    retriever = _build(
        [
            _case("How do I bake a cake?", [], product=None),
            _case("How do I configure SAML?", [SAML]),
            _case("Who won the match?", []),
        ]
    )
    assert [d.url for d in retriever._documents] == [SAML]


@pytest.mark.parametrize("cases", [[], [_case("How do I bake a cake?", [], product=None)]])
def test_no_positive_cases_produce_a_valid_empty_retriever(cases: list[EvalCase]) -> None:
    retriever = _build(cases)
    assert retriever.is_synthetic is True
    assert retriever._documents == ()
    assert _search(retriever, "How do I configure SAML?") == []


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_or_whitespace_question_is_accepted(question: str) -> None:
    # Current behaviour, documented: the chunk is created and indexed. Text search cannot
    # match it (no tokens), but vector search still returns it.
    retriever = _build([_case(question, [SAML])])
    (document,) = retriever._documents
    assert document.chunk_texts == (question,)
    assert [r.chunk_id for r in _search(retriever, "anything")] == list(document.chunk_ids)


def test_environment_and_dot_env_do_not_affect_the_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Invalid values: if the factory read them, RetrievalSettings would raise.
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_S", "abc")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "0")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "RETRIEVAL_TIMEOUT_S=-1\nRETRIEVAL_CANDIDATE_K=abc\nRETRIEVAL_RELEASE_NOTES_WEIGHT=5\n",
        encoding="utf-8",
    )

    retriever = _build([_case("How do I configure SAML?", [SAML])])

    assert [r.url for r in _search(retriever, "How do I configure SAML?")] == [SAML]


# --- product resolution --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [(JES_SEQUENCE, "sequence"), (JES_ORCHESTRATION, "orchestration"), (EMAIL, "sequence")],
)
def test_explicit_mapping_is_used_when_no_case_has_a_product(url: str, expected: str) -> None:
    case = _case("Product-agnostic question", [url], product=None)
    retriever = _build([case])

    (document,) = retriever._documents
    assert document.product == expected
    assert {r.product for r in _search(retriever, "Product-agnostic question")} == {expected}
    assert case.product is None  # the EvalCase itself is not changed


def test_product_from_the_cases_is_used_and_none_is_ignored() -> None:
    retriever = _build(
        [
            _case("Email not delivered?", [EMAIL], product="sequence"),
            _case("Emails are not sent", [EMAIL], product=None),
            _case("Configure Kafka", [KAFKA], product="orchestration"),
        ]
    )
    assert [(d.url, d.product) for d in retriever._documents] == [
        (EMAIL, "sequence"),
        (KAFKA, "orchestration"),
    ]


def test_case_product_takes_precedence_over_the_fallback_mapping() -> None:
    # The fallback mapping says "sequence" for EMAIL; a product given by a case wins.
    retriever = _build([_case("Email not delivered?", [EMAIL], product="orchestration")])
    (document,) = retriever._documents
    assert document.product == "orchestration"


def test_conflicting_products_for_one_url_raise_value_error() -> None:
    cases = [
        _case("SAML in SeQuence", [SAML], product="sequence"),
        _case("SAML in Orchestration", [SAML], product="orchestration"),
    ]
    with pytest.raises(ValueError, match="conflicting products"):
        _build(cases)


def test_url_without_a_resolvable_product_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no product for"):
        _build([_case("Product-agnostic question", [KAFKA], product=None)])


@pytest.mark.parametrize("product", ["Sequence", "seQuence 9", "", "cora"])
def test_invalid_product_raises_value_error(product: str) -> None:
    with pytest.raises(ValueError, match="invalid product"):
        _build([_case("How do I configure SAML?", [SAML], product=product)])


def test_one_valid_and_one_invalid_product_for_a_url_raises_value_error() -> None:
    cases = [
        _case("SAML in SeQuence", [SAML], product="sequence"),
        _case("SAML again", [SAML], product="Sequence"),
    ]
    with pytest.raises(ValueError, match="invalid product 'Sequence'"):
        _build(cases)


def test_url_that_is_not_a_kc_article_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Knowledge Center article"):
        _build([_case("Question", ["https://example.com/help/saml"])])


# --- determinism ---------------------------------------------------------------------------


def test_repeated_construction_is_deterministic() -> None:
    cases = [
        _case("How do I configure SAML?", [SAML]),
        _case("Configure Kafka", [KAFKA], product="orchestration"),
        _case("What is the Job Execution Service?", [JES_SEQUENCE, JES_ORCHESTRATION], None),
        _case("Emails are not sent", [EMAIL], product=None),
    ]
    queries = ["configure", "Job Execution Service", "emails", "שאלה בעברית"]

    first, second = _build(cases), _build(cases)

    assert first._documents == second._documents
    assert [d.chunk_ids for d in first._documents] == [d.chunk_ids for d in second._documents]
    for query in queries:
        assert _search(first, query, k=10) == _search(second, query, k=10)


# --- the real Golden Set (data owned by Evaluation; read-only) -----------------------------


def test_builds_from_the_real_golden_set() -> None:
    cases = load_golden_set()
    try:
        retriever = _build(cases)
    except ValueError as error:
        pytest.fail(
            "The synthetic evaluation retriever cannot be built from eval/golden_set.jsonl: "
            f"{error}\nIf a URL has no resolvable product, either add it to "
            "_FALLBACK_PRODUCTS in src/kc/retrieval/evaluation.py (Search & Ranking) or give "
            "a case that expects it a product in the Golden Set (Evaluation)."
        )

    expected = list(dict.fromkeys(url for case in cases for url in case.expected_urls))
    assert [d.url for d in retriever._documents] == expected, (
        "synthetic documents must be one per unique Golden Set URL, in first-seen order"
    )
    assert all(d.doc_type == "other" for d in retriever._documents)
