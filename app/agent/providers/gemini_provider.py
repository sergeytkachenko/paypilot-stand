import json

import httpx

from app import config
from app.agent.providers.base import ModelResponse, Provider


API_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/{model}:generateContent"
)

DEFAULT_MODEL = "gemini-flash-latest"


def _to_gemini(messages: list[dict]) -> list[dict]:
    out = []

    for m in messages:
        if m["role"] == "user":
            out.append({
                "role": "user",
                "parts": [
                    {"text": m["content"]}
                ],
            })

        elif m["role"] == "assistant":
            parts = []

            if m.get("content"):
                parts.append({
                    "text": m["content"]
                })

            for tc in m.get("tool_calls") or []:
                function_call = {
                    "name": tc["name"],
                    "args": tc["arguments"],
                }

                parts.append({
                    "functionCall": function_call,
                    **(
                        {"thoughtSignature": tc["thought_signature"]}
                        if tc.get("thought_signature")
                        else {}
                    ),
                })

            out.append({
                "role": "model",
                "parts": parts,
            })

        elif m["role"] == "tool":
            content = m["content"]

            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {"result": content}

            out.append({
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": m["name"],
                            "response": content,
                        }
                    }
                ],
            })

    return out


def _to_gemini_tools(tools: list[dict]) -> list[dict]:
    if not tools:
        return []

    return [{
        "functionDeclarations": [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
            for tool in tools
        ]
    }]


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self.model = config.LLM_MODEL or DEFAULT_MODEL

    def complete(self, system, messages, tools):
        payload = {
            "systemInstruction": {
                "parts": [
                    {"text": system}
                ]
            },
            "contents": _to_gemini(messages),
        }

        gemini_tools = _to_gemini_tools(tools)

        if gemini_tools:
            payload["tools"] = gemini_tools

        url = API_URL.format(model=self.model)

        resp = httpx.post(
            url,
            json=payload,
            timeout=60,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": config.GEMINI_API_KEY,
            },
        )
        if resp.status_code == 429:
            raise RuntimeError(
                "Gemini quota exceeded. "
                "Please wait and retry later or check Gemini API billing/quota."
            )

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gemini {resp.status_code}: {resp.text[:1000]}"
            )

        data = resp.json()

        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])

        text_parts = []
        tool_calls = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])

            elif "functionCall" in part:
                function_call = part["functionCall"]

                thought_signature = (
                    part.get("thoughtSignature")
                    or function_call.get("thought_signature")
                )

                tool_calls.append({
                    "id": function_call["name"],
                    "name": function_call["name"],
                    "arguments": function_call.get("args", {}),
                    "thought_signature": thought_signature,
                })

        usage = data.get("usageMetadata", {})

        return ModelResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            model=self.model,
        )
