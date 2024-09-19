import random
from typing import List, Optional

from openai import OpenAI
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
def sota_model(
    history: List[List[Optional[str]]],
    temperature: float = 0.6,
    top_p: float = 0.7,
    max_output_tokens: int = 2048,
):
    # client = OpenAI()
    client = OpenAI(
            api_key=os.environ.get("SOTA_API_KEY"),
            base_url=os.environ.get("SOTA_API_BASE")
        )
    messages = []
    for human, ai in history:
        if human:
            messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})

    stream = client.chat.completions.create(
        model=os.environ.get("SOTA_API_MODEL"),
        messages=messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def model1(
    history: List[List[Optional[str]]],
    temperature: float = 0.8,
    top_p: float = 0.7,
    max_output_tokens: int = 2048,
):
    # client = OpenAI()
    client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE")
        )
    messages = []
    for human, ai in history:
        if human:
            messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})

    stream = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model=os.environ.get("API_MODEL_1"),
        messages=messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def model2(
    history: List[List[Optional[str]]],
    temperature: float = 0.8,
    top_p: float = 0.7,
    max_output_tokens: int = 2048,
):
    # client = OpenAI()
    client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)
    messages = []
    for human, ai in history:
        if human:
            messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})

    stream = client.chat.completions.create(
        # model="gpt-4-turbo",
        model=os.environ.get("API_MODEL_2"),
        messages=messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def model3(
    history: List[List[Optional[str]]],
    temperature: float = 0.8,
    top_p: float = 0.7,
    max_output_tokens: int = 2048,
):
    # client = OpenAI()
    client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)
    messages = []
    for human, ai in history:
        if human:
            messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})

    stream = client.chat.completions.create(
        # model="gpt-4-turbo",
        model=os.environ.get("API_MODEL_3"),
        messages=messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def get_all_models():
    return [
        {
            # "name": os.environ.get("SOTA_API_MODEL"),
            "name": 'Cherokee Language Preserve Model',
            "model": sota_model,
        },
        {
            "name":os.environ.get("API_MODEL_1"),
            "model": model1,
        },
        {
            "name": os.environ.get("API_MODEL_2"),
            "model": model2,
        },
        {
            "name": os.environ.get("API_MODEL_3"),
            "model": model3,
        },
    ]


# def get_random_models(number: int = 4):
#     return random.sample(get_all_models(), number)
def get_random_models():
    return get_all_models()
