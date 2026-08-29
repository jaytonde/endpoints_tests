import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://localhost:30000/v1"
MODEL = "gemma-4-31B-it"
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
IMAGE_PATH = Path(__file__).with_name("5_image.webp")


def env_flag(name, default=True):
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


TEST_IMAGE_CAPABILITY = "true"
TEST_TOOL_CAPABILITY = "true"
TEST_REASONING_CAPABILITY = "true"


def chat(payload):
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():

    common = {
        "model": MODEL,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 1024,
    }

    response = chat({
        **common,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    })
    print(json.dumps(response, indent=2))
    content = response["choices"][0]["message"].get("content") or ""

    print(f"\nsimple chat response : {content}")
    chat_ok = "4" in content
    print("PASS basic chat" if chat_ok else "FAIL basic chat")

    tool_ok = True
    if TEST_TOOL_CAPABILITY:
        tool_response = chat({
            **common,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "What is the weather in London?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }],
            "tool_choice": "required",
        })

        message = tool_response["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        print(f"\nTEST_TOOL_CAPABILITY response : message -> {message}, calls -> {calls} ")
        tool_ok = bool(calls)
        if tool_ok:
            function = calls[0].get("function", {})
            try:
                arguments = json.loads(function.get("arguments", ""))
                tool_ok = function.get("name") == "get_weather" and isinstance(arguments, dict)
            except json.JSONDecodeError:
                tool_ok = False

        print("PASS tool calling" if tool_ok else "FAIL tool calling")
        if not tool_ok:
            print(json.dumps(tool_response, indent=2))

        no_tool_response = chat({
            **common,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }],
            "tool_choice": "auto",
        })
        message = no_tool_response["choices"][0]["message"]
        print(f"\nTEST_NO_TOOL_CAPABILITY response : message -> {message}")

        no_tool_ok = not message.get("tool_calls") and "4" in (message.get("content") or "")

        print("PASS no unnecessary tool call" if no_tool_ok else "FAIL no unnecessary tool call")
        if not no_tool_ok:
            print(json.dumps(no_tool_response, indent=2))
        tool_ok = tool_ok and no_tool_ok
    else:
        print("SKIP tool calling")

    reasoning_ok = True
    if TEST_REASONING_CAPABILITY:
        reasoning_response = chat({
            **common,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{
                "role": "user",
                "content": "Which is larger: 9.11 or 9.8? Explain your reasoning.",
            }],
        })
        message = reasoning_response["choices"][0]["message"]

        print(f"\nTEST_REASONING_CAPABILITY response : message -> {message}")

        reasoning_ok = isinstance(message.get("reasoning"), str) and bool(message["reasoning"].strip())
        content_ok = isinstance(message.get("content"), str) and bool(message["content"].strip())

        print("PASS reasoning field" if reasoning_ok else "FAIL reasoning field")
        print("PASS final content" if content_ok else "FAIL final content")
        if not reasoning_ok or not content_ok:
            print(json.dumps(reasoning_response, indent=2))
        reasoning_ok = reasoning_ok and content_ok
    else:
        print("SKIP reasoning field")

    image_ok = True
    if TEST_IMAGE_CAPABILITY:
        try:
            image_data = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
            image_response = chat({
                **common,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "What single digit is shown in this image? Reply with only the digit.",
                        },
                    ],
                }],
            })
            image_content = image_response["choices"][0]["message"].get("content") or ""
            print(f"TEST_IMAGE_CAPABILITY response : message -> {image_content}")
            image_ok = re.findall(r"\d+", image_content) == ["5"]
            print("PASS image understanding" if image_ok else "FAIL image understanding")
            if not image_ok:
                print(json.dumps(image_response, indent=2))
        except Exception as error:
            image_ok = False
            print(f"FAIL image understanding: {error}")
    else:
        print("SKIP image understanding")

    raise SystemExit(0 if chat_ok and tool_ok and reasoning_ok and image_ok else 1)


if __name__ == "__main__":
    main()
