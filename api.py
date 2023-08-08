import argparse
import json
from typing import List

import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request

from client import Client

OPEN_CROSS_DOMAIN = False


async def stream(
        request: Request,
        prompt: str = Body(..., description="Question", example="你是谁呀"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "你好",
                    "你好！很高兴为你提供帮助。有什么我可以为你做的？",
                ]
            ],
        ),
):

    async def event_generator(chat_history):
        last_print_len = 0
        for resp in cli.stream_chat(
                query=prompt, chat_history=chat_history, streaming=True
        ):
            data = resp["result"][last_print_len:]
            if '�' == data or ' �' == data or '� ' == data:
                continue
            last_print_len = len(resp["result"])
            if await request.is_disconnected():
                break
            yield {
                "data": json.dumps({"content": data}, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator(history))


def main():
    global app
    global cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = FastAPI()
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
    )
    app.post("/chat-docs/stream")(stream)

    cli = Client()
    cli.init_cfg()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()