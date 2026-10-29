import pytest
import requests
import providers
from providers import PROVIDERS, build_request, parse_response, get_provider, default_cloud_provider
from rag_engine import Passage, llm_answer

MSGS = [{'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': 'a1'},
        {'role': 'user', 'content': 'q2'}]
PASSAGE = [Passage('a', 'Paper', 1, 'Some evidence text ' * 10, .5)]


def test_openai_style_request():
    url, headers, body = build_request(PROVIDERS['openai'], 'sk-x', 'gpt-4o-mini', MSGS)
    assert url == 'https://api.openai.com/v1/chat/completions'
    assert headers['Authorization'] == 'Bearer sk-x' and body['messages'] == MSGS


def test_anthropic_request_separates_system_and_alternates():
    url, headers, body = build_request(PROVIDERS['anthropic'], 'k', 'claude-sonnet-5-5', MSGS)
    assert url.endswith('/v1/messages') and headers['x-api-key'] == 'k'
    assert body['system'] == 'SYS'
    assert [m['role'] for m in body['messages']] == ['user', 'assistant', 'user']


def test_gemini_request_maps_roles_and_uses_header_key():
    url, headers, body = build_request(PROVIDERS['gemini'], 'g', 'gemini-2.5-flash', MSGS)
    assert url.endswith('/models/gemini-2.5-flash:generateContent') and 'key=' not in url
    assert headers['x-goog-api-key'] == 'g'
    assert [c['role'] for c in body['contents']] == ['user', 'model', 'user']
    assert body['systemInstruction']['parts'][0]['text'] == 'SYS'


def test_same_role_turns_are_merged():
    msgs = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'a'}, {'role': 'user', 'content': 'b'}]
    _, _, body = build_request(PROVIDERS['anthropic'], 'k', 'm', msgs)
    assert len(body['messages']) == 1 and 'a' in body['messages'][0]['content'] and 'b' in body['messages'][0]['content']


def test_parse_each_style():
    assert parse_response(PROVIDERS['openai'], {'choices': [{'message': {'content': 'hi'}}]}) == 'hi'
    assert parse_response(PROVIDERS['anthropic'], {'content': [{'type': 'text', 'text': 'hi'}]}) == 'hi'
    assert parse_response(PROVIDERS['gemini'], {'candidates': [{'content': {'parts': [{'text': 'hi'}]}}]}) == 'hi'
    assert parse_response(PROVIDERS['ollama'], {'message': {'content': 'hi'}}) == 'hi'


def test_env_key_model_and_url_overrides(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', ' secret ')
    monkeypatch.setenv('OPENAI_MODELS', 'a, b')
    monkeypatch.setenv('CUSTOM_BASE_URL', 'http://example.test/v1/')
    assert PROVIDERS['anthropic'].env_api_key() == 'secret'
    assert PROVIDERS['openai'].resolved_models() == ['a', 'b']
    assert build_request(PROVIDERS['custom'], '', 'm', MSGS)[0] == 'http://example.test/v1/chat/completions'


def test_default_provider(monkeypatch):
    for p in PROVIDERS.values():
        if p.env_key:
            monkeypatch.delenv(p.env_key, raising=False)
    monkeypatch.delenv('DEFAULT_PROVIDER', raising=False)
    assert default_cloud_provider() == 'openai'
    monkeypatch.setenv('GEMINI_API_KEY', 'x')
    assert default_cloud_provider() == 'gemini'
    monkeypatch.setenv('DEFAULT_PROVIDER', 'anthropic')
    assert default_cloud_provider() == 'anthropic'


def test_unknown_provider():
    with pytest.raises(ValueError):
        get_provider('nope')


def test_llm_answer_requires_key_and_never_calls_network_without_it(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='ANTHROPIC_API_KEY'):
        llm_answer('q', PASSAGE, provider='anthropic', model='claude-sonnet-5-5')


def test_llm_answer_uses_env_key_and_parses(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'env-key')
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {'content': [{'type': 'text', 'text': ' answer [S1] '}]}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers)
        return R()
    monkeypatch.setattr(requests, 'post', fake_post)
    assert llm_answer('q', PASSAGE, provider='anthropic', model='m') == 'answer [S1]'
    assert seen['headers']['x-api-key'] == 'env-key'


def test_http_error_message_is_readable(monkeypatch):
    class R:
        def raise_for_status(self): raise requests.HTTPError('401')
        def json(self): return {'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}
    monkeypatch.setattr(requests, 'post', lambda *a, **k: R())
    with pytest.raises(RuntimeError, match='invalid x-api-key'):
        llm_answer('q', PASSAGE, api_key='bad', provider='anthropic', model='m')
