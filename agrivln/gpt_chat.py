# Drop-in replacement for ollama.chat that calls the OpenAI API.
import base64, os, mimetypes
from openai import OpenAI

_client = OpenAI()
# Paper setup: GPT-4.1 for STL (LLM), GPT-4.1-mini for decisions (VLM)
MODEL_MAP = {"deepseek-r1:32b": "gpt-4.1", "qwen2.5vl:32b": "gpt-4.1-mini"}

class _Obj(dict):
    __getattr__ = dict.__getitem__   # allows response.message.content and response["message"]["content"]

def _img_part(img):
    if isinstance(img, (bytes, bytearray)):
        b64, mime = base64.b64encode(img).decode(), "image/jpeg"
    elif isinstance(img, str) and os.path.exists(img):
        mime = mimetypes.guess_type(img)[0] or "image/jpeg"
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    else:  # already a base64 string
        b64, mime = img, "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

def chat(model, messages, **kwargs):
    converted = []
    for m in messages:
        if m.get("images"):
            parts = [{"type": "text", "text": m.get("content", "")}]
            parts += [_img_part(i) for i in m["images"]]
            converted.append({"role": m["role"], "content": parts})
        else:
            converted.append({"role": m["role"], "content": m.get("content", "")})
    r = _client.chat.completions.create(
        model=MODEL_MAP.get(model, model), messages=converted, temperature=0)
    return _Obj(
        message=_Obj(role="assistant", content=r.choices[0].message.content),
        prompt_eval_count=r.usage.prompt_tokens,   # used by the -t token option
        eval_count=r.usage.completion_tokens,
    )
