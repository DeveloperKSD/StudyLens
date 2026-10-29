"""StudyLens — lightweight conversational research-paper RAG."""
from pathlib import Path
import streamlit as st
from rag_engine import (extract_pdf, paper_id, chunk_pages, Retriever, llm_answer,
                        extractive_answer, followup_search_query, compare_retrieval)
from providers import PROVIDERS, CLOUD_PROVIDERS, default_cloud_provider  # also loads .env

st.set_page_config(page_title='StudyLens | Research Paper RAG', page_icon='📚', layout='wide')
st.markdown('''<style>
.block-container{max-width:1080px;padding-top:1.6rem}
.stButton>button{border-radius:9px} h1,h2,h3{letter-spacing:-.02em}
[data-testid="stChatMessage"]{border-radius:12px}
</style>''', unsafe_allow_html=True)
for key, default in [('library', {}), ('chat', []), ('compare_result', ''), ('compare_sources', [])]:
    if key not in st.session_state:
        st.session_state[key] = default


def add_pdf(name: str, contents: bytes):
    pid = paper_id(contents)
    if pid in st.session_state.library:
        return False, f'{name}: already in the library.'
    pages = extract_pdf(contents)
    title = Path(name).stem.replace('_', ' ').replace('-', ' ')
    chunks = chunk_pages(pages, pid, title)
    if not chunks:
        raise ValueError('No usable text chunks in this PDF.')
    st.session_state.library[pid] = {'id': pid, 'name': name, 'title': title,
                                     'pages': len(pages), 'chunks': chunks}
    return True, f'Added {title} ({len(pages)} pages).'


def selectors(label: str, key: str, minimum=1):
    options = {p['id']: p['title'] for p in st.session_state.library.values()}
    ids = st.multiselect(label, list(options), format_func=lambda x: options.get(x, x),
                         default=list(options), key=key)
    if len(ids) < minimum:
        st.info(f'Select at least {minimum} paper(s).')
    return ids


def sources_view(passages):
    st.caption('Source IDs correspond to the excerpts used for this response. Page numbers are PDF page positions.')
    for i, passage in enumerate(passages, 1):
        with st.expander(f'[S{i}] {passage.paper_title} — page {passage.page} · similarity {passage.score:.3f}'):
            st.write(passage.text)


def answer_for(question, evidence, history=None, comparison=False):
    if mode == 'No-key: source extracts':
        return extractive_answer(question, evidence)
    return llm_answer(question, evidence, api_key=api_key, model=model,
                      compare=comparison, provider=provider, history=history)


with st.sidebar:
    st.title('📚 StudyLens')
    st.caption('Understand and compare research papers')
    st.divider()
    st.subheader('Answer settings')
    mode = st.selectbox('Answer mode', ['No-key: source extracts', 'Local AI: Ollama', 'Cloud AI'],
                        help='No-key is ready immediately. Local AI requires one-time Ollama installation, not an API key. '
                             'Cloud AI lets you pick OpenAI, Anthropic, Gemini and more.')
    provider = 'ollama' if mode == 'Local AI: Ollama' else 'openai'
    api_key, model = '', ''

    def model_picker(prov, label):
        """Dropdown of known models, plus a free-text option because model names change often."""
        models = prov.resolved_models()
        choice = st.selectbox(label, models + ['Other (type a model name)'] if models else ['Other (type a model name)'],
                              key=f'model_{prov.id}')
        if choice.startswith('Other'):
            return st.text_input('Model name', key=f'custom_model_{prov.id}',
                                 help='Exact model ID from the provider’s documentation.').strip()
        return choice

    if mode == 'Local AI: Ollama':
        prov = PROVIDERS['ollama']
        model = model_picker(prov, 'Local model')
        st.caption('Needs Ollama running locally. No API key, but model download and computer resources are required. '
                   'Install a model first with `ollama pull MODEL_NAME`.')
    elif mode == 'Cloud AI':
        ids = [p.id for p in CLOUD_PROVIDERS]
        provider = st.selectbox('Provider', ids, index=ids.index(default_cloud_provider()),
                                format_func=lambda i: PROVIDERS[i].label + (' ✓' if PROVIDERS[i].has_env_key() else ''),
                                help='✓ means a key for this provider was found in your .env file.')
        prov = PROVIDERS[provider]
        model = model_picker(prov, 'Cloud model')
        if prov.has_env_key():
            st.caption(f'🔑 Using `{prov.env_key}` from your .env file.')
            if st.checkbox('Use a different key for this session', key=f'override_{provider}'):
                api_key = st.text_input(f'{prov.label} API key', type='password', key=f'key_{provider}')
        elif prov.needs_key:
            st.warning(f'No `{prov.env_key}` found in .env. Add it there (then restart), or paste a key below for this session only.')
            api_key = st.text_input(f'{prov.label} API key', type='password', key=f'key_{provider}')
        else:
            api_key = st.text_input(f'{prov.label} API key (optional)', type='password', key=f'key_{provider}')
        st.caption(f'API calls can incur charges; relevant paper excerpts are sent to {prov.label}.')
    else:
        st.caption('No credentials or model downloads. Shows selected original sentences, not generated explanations.')
    st.divider()
    st.caption('PDF library and conversation are kept for this app session only. No Docker or database.')

st.title('Research Paper Study & Comparison')
st.write('Add PDFs, ask questions naturally, and check every answer against the original evidence.')
t1, t2, t3 = st.tabs(['📁 Paper library', '💬 Ask & follow up', '📊 Compare papers'])

with t1:
    st.subheader('Your research papers')
    uploads = st.file_uploader('Upload text-based PDF papers', type=['pdf'], accept_multiple_files=True,
                               help='Maximum 20 MB and 350 PDF pages per file. Image-only PDFs need OCR.')
    if st.button('Add PDFs', type='primary', disabled=not uploads):
        for upload in uploads:
            try:
                added, note = add_pdf(upload.name, upload.getvalue())
                (st.success if added else st.info)(note)
            except ValueError as e:
                st.error(f'{upload.name}: {e}')
    with st.expander('Try two included example papers'):
        if st.button('Load samples'):
            for path in sorted((Path(__file__).parent / 'sample_papers').glob('*.pdf')):
                try:
                    _, note = add_pdf(path.name, path.read_bytes())
                    st.info(note)
                except ValueError as e:
                    st.error(str(e))
    library = list(st.session_state.library.values())
    a, b, c = st.columns(3)
    a.metric('Papers', len(library))
    b.metric('Pages', sum(p['pages'] for p in library))
    c.metric('Searchable chunks', sum(len(p['chunks']) for p in library))
    if not library:
        st.info('Add at least one paper to get started.')
    for paper in library:
        left, right = st.columns([6, 1])
        left.markdown(f'**{paper["title"]}** · {paper["pages"]} pages · {len(paper["chunks"])} indexed chunks')
        if right.button('Remove', key='remove_' + paper['id']):
            del st.session_state.library[paper['id']]
            st.session_state.chat = []
            st.session_state.compare_result = ''
            st.rerun()

with t2:
    st.subheader('Chat with your papers')
    selected = selectors('Select papers for this conversation', 'chat_papers')
    with st.expander('Search options'):
        top_k = st.slider('Evidence excerpts per question', 3, 10, 6)
    if st.button('Clear conversation', disabled=not st.session_state.chat):
        st.session_state.chat = []
        st.rerun()
    if not st.session_state.chat:
        st.info('Example: “What problem does this paper solve?” Then ask “How was that method evaluated?”')
    for entry in st.session_state.chat:
        with st.chat_message('user'):
            st.markdown(entry['question'])
        with st.chat_message('assistant'):
            st.markdown(entry['answer'])
            with st.expander(f'Check sources ({len(entry["evidence"])})'):
                sources_view(entry['evidence'])
    with st.form('qa_form', clear_on_submit=True):
        question = st.text_input('Ask a question or follow up', placeholder='What limitations did the authors mention?')
        submitted = st.form_submit_button('Send question', type='primary', disabled=not selected)
    if submitted and question.strip():
        # Preserve the topic of follow-ups in the TF-IDF query and conversational context for LLMs.
        recent = []
        for entry in st.session_state.chat[-3:]:
            recent.extend([{'role': 'user', 'content': entry['question']},
                           {'role': 'assistant', 'content': entry['answer']}])
        retrieval_query = followup_search_query(question, recent)
        # Rebuild the index from the currently selected papers to avoid stale selections.
        chunks = [p for pid in selected for p in st.session_state.library[pid]['chunks']]
        with st.spinner('Finding relevant evidence…'):
            evidence = Retriever(chunks).search(retrieval_query, selected, k=top_k)
            try:
                response = answer_for(question, evidence, history=recent)
                st.session_state.chat.append({'question': question, 'answer': response, 'evidence': evidence})
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
                st.info('Switch to No-key: source extracts or confirm your model is running.')

with t3:
    st.subheader('Compare papers')
    chosen = selectors('Choose 2–5 papers', 'compare_papers', minimum=2)
    focus = st.text_input('What should the comparison focus on?',
                          placeholder='Methods, datasets, results, evaluation metrics, limitations…')
    if st.button('Compare selected papers', type='primary', disabled=not 2 <= len(chosen) <= 5):
        chunks = [p for pid in chosen for p in st.session_state.library[pid]['chunks']]
        retriever = Retriever(chunks)
        evidence = compare_retrieval(retriever, chosen)
        if focus.strip():
            evidence += [p for pid in chosen for p in retriever.search(focus, [pid], k=3)]
        # Separate evidence budgets for each paper, avoiding accidental overrepresentation.
        balanced, seen = [], set()
        for pid in chosen:
            count = 0
            for p in evidence:
                signature = (p.paper_id, p.page, p.text)
                if p.paper_id == pid and signature not in seen and count < 5:
                    balanced.append(p)
                    seen.add(signature)
                    count += 1
        with st.spinner('Comparing evidence…'):
            try:
                if mode == 'No-key: source extracts':
                    sections = ['### Retrieved evidence by paper',
                                '*This is an extractive comparison, not an AI-generated interpretation.*']
                    for i, p in enumerate(balanced, 1):
                        sections.append(f'**{p.paper_title} · PDF p. {p.page} [S{i}]**\n\n'
                                        + extractive_answer(focus or 'methods evaluation results limitations', [p], max_sentences=2)
                                          .split('*These are verbatim')[0].replace('[S1]', f'[S{i}]'))
                    response = '\n\n'.join(sections)
                else:
                    response = answer_for('Compare the selected papers' + (f', focusing on {focus}' if focus else '') + '.',
                                          balanced, comparison=True)
                st.session_state.compare_result = response
                st.session_state.compare_sources = balanced
            except RuntimeError as e:
                st.error(str(e))
    if st.session_state.compare_result:
        st.markdown(st.session_state.compare_result)
        st.download_button('Export comparison (.md)', st.session_state.compare_result,
                           file_name='studylens_comparison.md', mime='text/markdown')
        with st.expander('Check comparison sources'):
            sources_view(st.session_state.compare_sources)
        st.caption('Evidence may be incomplete; verify important scientific claims in the full papers.')
