from rag_engine import Passage, Retriever, chunk_pages, paper_id, preview_answer, source_context, extract_pdf
import pytest


def test_chunk_page_metadata_and_overlap():
    chunks = chunk_pages([' '.join(f'word{i}' for i in range(100))], 'a', 'Demo', 40, 10)
    assert len(chunks) == 3
    assert all(p.page == 1 and p.paper_id == 'a' for p in chunks)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_retrieval_respects_selected_papers():
    chunks = [Passage('a', 'Alpha', 2, 'neural network convolution classification images' * 5),
              Passage('b', 'Beta', 4, 'graph neural network molecular property prediction' * 5)]
    r = Retriever(chunks)
    assert r.search('molecular prediction', ['b'])[0].paper_id == 'b'
    assert r.search('molecular prediction', ['a']) == []
    assert not r.search('', ['a'])


def test_sources_and_empty_answers():
    p = Passage('abc', 'Study', 3, 'Important findings.', .5)
    assert '[S1]' in source_context([p]) and 'page 3' in source_context([p])
    assert 'Insufficient evidence' in preview_answer([])
    assert 'retrieval-only' in preview_answer([p])


def test_validation():
    with pytest.raises(ValueError, match='Not a PDF'):
        extract_pdf(b'not a pdf')
    assert paper_id(b'same') == paper_id(b'same')
    with pytest.raises(ValueError):
        chunk_pages(['hello'], 'a', 'b', 10, 10)

from rag_engine import extractive_answer, followup_search_query


def test_followup_retrieval_keeps_prior_question():
    turns = [{'role': 'user', 'content': 'Explain the molecular predictor'},
             {'role': 'assistant', 'content': 'It is described in the paper.'}]
    assert followup_search_query('How was it evaluated?', turns) == 'Explain the molecular predictor How was it evaluated?'
    assert followup_search_query('What is it?', []) == 'What is it?'


def test_extract_answers_cite_original_sentences():
    p = Passage('a', 'Paper', 2, 'The experiments compare model predictions against a held out test set of molecules. ' * 3, .9)
    answer = extractive_answer('How are models evaluated?', [p])
    assert '[S1]' in answer
    assert 'The experiments compare model predictions' in answer
    assert 'verbatim' in answer
