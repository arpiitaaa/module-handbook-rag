"""
llm.py - one function, `chat()`, that talks to whichever LLM you choose.

Backends
    ollama    : local and free. Needs Ollama running (https://ollama.com) and a pulled model,
                e.g.  ollama pull llama3.2:3b
    anthropic : Claude via API. Needs ANTHROPIC_API_KEY (put it in a .env file in the
                project root:  ANTHROPIC_API_KEY=sk-ant-...   - .env is git-ignored)
    mock      : fake answers, only for testing the pipeline without any model

Override default models with env vars OLLAMA_MODEL / ANTHROPIC_MODEL or the --model flag.
"""
import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
    "mock": "mock",
}


def chat(backend: str, system: str, user: str, model: str | None = None, max_tokens: int = 700) -> str:
    model = model or DEFAULT_MODELS[backend]

    if backend == "ollama":
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama ({e}). Is it running? Try: ollama serve  "
                f"and make sure the model exists: ollama pull {model}") from e

    if backend == "anthropic":
        import anthropic  # imported here so Ollama users do not need the package
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=0, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    if backend == "mock":
        low = system.lower()
        if "faithfulness" in low:
            return "SUPPORTED"
        if "strict grader" in low:
            return "CORRECT"
        return "Mock answer based on the first passage [1]."

    raise ValueError(f"unknown backend: {backend}")