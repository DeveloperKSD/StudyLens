"""LLM provider registry: keys come from environment variables / a .env file, never from source code.

Supported out of the box: OpenAI, Anthropic, Google Gemini, Groq, OpenRouter, Mistral, DeepSeek,
xAI, Together AI, a local Ollama server, and any custom OpenAI-compatible endpoint (LM Studio, vLLM...).
Only `requests` is used, so no provider SDKs need to be installed.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

try:  # python-dotenv is optional at import time so the tests/engine work without it
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

API_STYLES = ('openai', 'anthropic', 'gemini', 'ollama')


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    style: str                       # one of API_STYLES
    base_url: str
    env_key: str                     # environment variable holding the API key ('' = none needed)
    models: tuple[str, ...] = ()
    needs_key: bool = True
    env_base_url: str = ''           # optional env var that overrides base_url
    env_models: str = ''             # optional env var: comma-separated model list override
    extra: dict = field(default_factory=dict)

    def resolved_base_url(self) -> str:
        return (os.getenv(self.env_base_url, '') if self.env_base_url else '').strip() or self.base_url

    def resolved_models(self) -> list[str]:
        override = os.getenv(self.env_models, '') if self.env_models else ''
        listed = [m.strip() for m in override.split(',') if m.strip()]
        return listed or list(self.models)

    def env_api_key(self) -> str:
        return os.getenv(self.env_key, '').strip() if self.env_key else ''

    def has_env_key(self) -> bool:
        return bool(self.env_api_key())


# Model names change often. Edit these lists, or set e.g. OPENAI_MODELS="gpt-4o,gpt-4.1" in .env.
PROVIDERS: dict[str, Provider] = {p.id: p for p in [
    Provider('openai', 'OpenAI', 'openai', 'https://api.openai.com/v1', 'OPENAI_API_KEY',
             ('gpt-4o-mini', 'gpt-4o'), env_base_url='OPENAI_BASE_URL', env_models='OPENAI_MODELS'),
    Provider('anthropic', 'Anthropic (Claude)', 'anthropic', 'https://api.anthropic.com/v1', 'ANTHROPIC_API_KEY',
             ('claude-haiku-5-5', 'claude-sonnet-5-5', 'claude-opus-5-5'), env_models='ANTHROPIC_MODELS'),
    Provider('gemini', 'Google Gemini', 'gemini', 'https://generativelanguage.googleapis.com/v1beta', 'GEMINI_API_KEY',
             ('gemini-2.5-flash', 'gemini-2.5-pro'), env_models='GEMINI_MODELS'),
    Provider('groq', 'Groq', 'openai', 'https://api.groq.com/openai/v1', 'GROQ_API_KEY',
             ('llama-3.3-70b-versatile', 'llama-3.1-8b-instant'), env_models='GROQ_MODELS'),
    Provider('openrouter', 'OpenRouter', 'openai', 'https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY',
             ('openai/gpt-4o-mini', 'anthropic/claude-sonnet-5-5', 'google/gemini-2.5-flash'),
             env_models='OPENROUTER_MODELS'),
    Provider('mistral', 'Mistral', 'openai', 'https://api.mistral.ai/v1', 'MISTRAL_API_KEY',
             ('mistral-small-latest', 'mistral-large-latest'), env_models='MISTRAL_MODELS'),
    Provider('deepseek', 'DeepSeek', 'openai', 'https://api.deepseek.com/v1', 'DEEPSEEK_API_KEY',
             ('deepseek-chat',), env_models='DEEPSEEK_MODELS'),
    Provider('xai', 'xAI (Grok)', 'openai', 'https://api.x.ai/v1', 'XAI_API_KEY',
             ('grok-4',), env_models='XAI_MODELS'),
    Provider('together', 'Together AI', 'openai', 'https://api.together.xyz/v1', 'TOGETHER_API_KEY',
             ('meta-llama/Llama-3.3-70B-Instruct-Turbo',), env_models='TOGETHER_MODELS'),
    Provider('custom', 'Custom (OpenAI-compatible)', 'openai', 'http://127.0.0.1:1234/v1', 'CUSTOM_API_KEY',
             (), needs_key=False, env_base_url='CUSTOM_BASE_URL', env_models='CUSTOM_MODELS'),
    Provider('ollama', 'Ollama (local)', 'ollama', 'http://127.0.0.1:11434', '',
             ('llama3.2:3b', 'qwen2.5:3b', 'phi3:mini'), needs_key=False,
             env_base_url='OLLAMA_HOST', env_models='OLLAMA_MODELS'),
]}

CLOUD_PROVIDERS = [p for p in PROVIDERS.values() if p.id != 'ollama']


def get_provider(provider_id: str) -> Provider:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise ValueError(f'Unknown model provider: {provider_id!r}') from None


def default_cloud_provider() -> str:
    """DEFAULT_PROVIDER from .env if valid, else the first provider that has a key, else OpenAI."""
    wanted = os.getenv('DEFAULT_PROVIDER', '').strip().lower()
    if wanted in PROVIDERS and wanted != 'ollama':
        return wanted
    return next((p.id for p in CLOUD_PROVIDERS if p.has_env_key()), 'openai')


# ---------------------------------------------------------------- request building (pure, testable)

def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
    return system, [m for m in messages if m['role'] != 'system']


def _merge_turns(turns: list[dict]) -> list[dict]:
    """Anthropic/Gemini need strictly alternating roles starting with 'user'; merge same-role neighbours."""
    merged: list[dict] = []
    for turn in turns:
        if merged and merged[-1]['role'] == turn['role']:
            merged[-1] = {'role': turn['role'], 'content': merged[-1]['content'] + '\n\n' + turn['content']}
        else:
            merged.append(dict(turn))
    while merged and merged[0]['role'] != 'user':
        merged.pop(0)
    return merged


def build_request(provider: Provider, api_key: str, model: str, messages: list[dict],
                  max_tokens: int = 1500) -> tuple[str, dict, dict]:
    """Return (url, headers, json_body) for the provider's chat endpoint."""
    base = provider.resolved_base_url().rstrip('/')
    if provider.style == 'openai':
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        body = {'model': model, 'temperature': 0, 'max_tokens': max_tokens, 'messages': messages}
        return f'{base}/chat/completions', headers, body
    if provider.style == 'anthropic':
        system, turns = split_system(messages)
        headers = {'x-api-key': api_key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'}
        body = {'model': model, 'max_tokens': max_tokens, 'temperature': 0, 'system': system,
                'messages': _merge_turns(turns)}
        return f'{base}/messages', headers, body
    if provider.style == 'gemini':
        system, turns = split_system(messages)
        contents = [{'role': 'model' if t['role'] == 'assistant' else 'user', 'parts': [{'text': t['content']}]}
                    for t in _merge_turns(turns)]
        headers = {'x-goog-api-key': api_key, 'Content-Type': 'application/json'}
        # Gemini 2.5 "thinking" tokens count toward the output limit, so leave generous headroom.
        body = {'systemInstruction': {'parts': [{'text': system}]}, 'contents': contents,
                'generationConfig': {'temperature': 0, 'maxOutputTokens': max(max_tokens, 4096)}}
        return f'{base}/models/{model}:generateContent', headers, body
    if provider.style == 'ollama':
        body = {'model': model, 'messages': messages, 'stream': False, 'options': {'temperature': 0}}
        return f'{base}/api/chat', {'Content-Type': 'application/json'}, body
    raise ValueError(f'Unsupported API style: {provider.style}')


def parse_response(provider: Provider, data: dict) -> str:
    if provider.style == 'openai':
        return data['choices'][0]['message']['content'] or ''
    if provider.style == 'anthropic':
        return ''.join(b.get('text', '') for b in data['content'] if b.get('type') == 'text')
    if provider.style == 'gemini':
        parts = data['candidates'][0]['content']['parts']
        return ''.join(p.get('text', '') for p in parts)
    if provider.style == 'ollama':
        return data['message']['content']
    raise ValueError(f'Unsupported API style: {provider.style}')


def error_detail(data: object, fallback: str) -> str:
    """Pull a readable message out of the different provider error formats."""
    if isinstance(data, dict):
        err = data.get('error', data.get('message'))
        if isinstance(err, dict):
            return str(err.get('message') or err)
        if err:
            return str(err)
    return fallback
