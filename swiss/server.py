"""OpenAI-compatible HTTP endpoints over an Engine or an Embedder."""

import json
import socketserver
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core.sampling import SamplingParams

UNSUPPORTED = (
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "tools",
    "functions",
    "suffix",
    "response_format",
)
ROLES = ("system", "developer", "user", "assistant")


class RequestError(Exception):
    def __init__(self, message, status=HTTPStatus.BAD_REQUEST, param=None):
        super().__init__(message)
        self.status, self.param = HTTPStatus(status), param

    def payload(self):
        kind = "invalid_request_error" if self.status < 500 else "server_error"
        return {
            "error": {
                "message": str(self),
                "type": kind,
                "param": self.param,
                "code": self.status.phrase.lower().replace(" ", "_"),
            },
        }


def sampling_params(body, capacity):
    for name in UNSUPPORTED:
        if body.get(name) not in (None, False, [], {}):
            raise RequestError(f"unsupported parameter: {name}", param=name)
    if body.get("n", 1) != 1 or body.get("best_of", 1) != 1:
        raise RequestError("only n=1 is supported", param="n")
    if body.get("echo"):
        raise RequestError("echo is not supported", param="echo")
    stop = body.get("stop") or ()
    stop = (stop,) if isinstance(stop, str) else tuple(stop)
    limit = body.get("max_completion_tokens", body.get("max_tokens"))
    try:
        return SamplingParams(
            max_tokens=capacity if limit is None else limit,
            temperature=float(
                1.0 if body.get("temperature") is None else body["temperature"],
            ),
            top_p=float(body.get("top_p", 1.0)),
            top_k=body.get("top_k", 0),
            min_p=float(body.get("min_p", 0.0)),
            seed=body.get("seed"),
            stop=stop,
            stop_token_ids=tuple(body.get("stop_token_ids") or ()),
        )
    except (TypeError, ValueError) as error:
        raise RequestError(str(error)) from None


def prompt_tokens(engine, prompt):
    if isinstance(prompt, list) and len(prompt) == 1 and not isinstance(prompt[0], int):
        prompt = prompt[0]
    if isinstance(prompt, str):
        return engine.encode(prompt)
    if (
        isinstance(prompt, list)
        and prompt
        and all(type(token) is int for token in prompt)
    ):
        return prompt
    raise RequestError("prompt must be a string or a list of token IDs", param="prompt")


def chat_messages(messages):
    """Normalize messages to role and content pairs for a chat template."""
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a nonempty list", param="messages")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            raise RequestError(
                "each message needs a system, developer, user or assistant role",
                param="messages",
            )
        content = message.get("content")
        if isinstance(content, list):
            if any(
                not isinstance(part, dict) or part.get("type") != "text"
                for part in content
            ):
                raise RequestError(
                    "only text content parts are supported",
                    param="messages",
                )
            content = "".join(part.get("text", "") for part in content)
        elif not isinstance(content, str):
            raise RequestError("message content must be text", param="messages")
        normalized.append(
            {
                "role": "system" if message["role"] == "developer" else message["role"],
                "content": content,
            },
        )
    return normalized


def completion(identifier, name, text, reason, usage, chat):
    choice = {"index": 0, "finish_reason": reason}
    if chat:
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice.update(text=text, logprobs=None)
    return {
        "id": identifier,
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "model": name,
        "choices": [choice],
        "usage": usage,
    }


def chunk(
    identifier,
    name,
    created,
    chat,
    *,
    text=None,
    role=False,
    reason=None,
    usage=None,
):
    """One streamed object."""
    payload = {
        "id": identifier,
        "object": "chat.completion.chunk" if chat else "text_completion",
        "created": created,
        "model": name,
        "choices": [],
    }
    if usage is not None:
        payload["usage"] = usage
        return payload
    if chat:
        delta = (
            {"role": "assistant", "content": ""}
            if role
            else ({"content": text} if text else {})
        )
        payload["choices"].append({"index": 0, "delta": delta, "finish_reason": reason})
    else:
        payload["choices"].append(
            {"index": 0, "text": text or "", "logprobs": None, "finish_reason": reason},
        )
    return payload


def embedding_inputs(body):
    """The texts or token lists of an embeddings request."""

    def single(item):
        return (
            isinstance(item, str)
            and item
            or isinstance(item, list)
            and item
            and all(type(t) is int for t in item)
        )

    value = body.get("input")
    if (
        isinstance(value, str)
        or isinstance(value, list)
        and value
        and all(type(t) is int for t in value)
    ):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or not all(single(item) for item in value)
    ):
        raise RequestError(
            "input must be a nonempty string, token list, or list of them",
            param="input",
        )
    if body.get("encoding_format", "float") != "float":
        raise RequestError("only float encoding is supported", param="encoding_format")
    dimensions = body.get("dimensions")
    if dimensions is not None and (type(dimensions) is not int or dimensions < 1):
        raise RequestError("dimensions must be a positive integer", param="dimensions")
    return value, dimensions


def model_card(name, created, capacity):
    return {
        "id": name,
        "object": "model",
        "created": created,
        "owned_by": "swiss",
        "max_model_len": capacity,
    }


MAX_BODY = 8 << 20


class Handler(BaseHTTPRequestHandler):
    server_version = "swiss"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        if self.server.verbose:
            super().log_message(format, *args)

    def send_json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_request_error(self, error):
        self.send_json(error.status, error.payload())

    def read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise RequestError(
                "request body too large",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            body = json.loads(self.rfile.read(length) or b"null")
        except json.JSONDecodeError:
            raise RequestError("request body must be JSON") from None
        if not isinstance(body, dict):
            raise RequestError("request body must be a JSON object")
        model = body.get("model")
        if model not in (None, self.server.engine.name):
            raise RequestError(
                f"model '{model}' is not served here",
                HTTPStatus.NOT_FOUND,
                "model",
            )
        return body

    def do_GET(self):
        engine = self.server.engine
        card = model_card(engine.name, self.server.started, engine.capacity)
        try:
            if self.path == "/health":
                self.send_json(HTTPStatus.OK, {"status": "ok", "busy": engine.busy})
            elif self.path == "/v1/models":
                self.send_json(HTTPStatus.OK, {"object": "list", "data": [card]})
            elif self.path == f"/v1/models/{engine.name}":
                self.send_json(HTTPStatus.OK, card)
            else:
                raise RequestError(
                    f"no route for GET {self.path}",
                    HTTPStatus.NOT_FOUND,
                )
        except RequestError as error:
            self.send_request_error(error)

    def do_POST(self):
        try:
            engine = self.server.engine
            routes = (
                ("/v1/embeddings",)
                if hasattr(engine, "embed")
                else ("/v1/completions", "/v1/chat/completions")
            )
            if self.path not in routes:
                raise RequestError(
                    f"no route for POST {self.path}",
                    HTTPStatus.NOT_FOUND,
                )
            body = self.read_body()
            if self.path == "/v1/embeddings":
                inputs, dimensions = embedding_inputs(body)
                vectors, tokens = engine.embed(inputs, dimensions)
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "model": engine.name,
                        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                        "data": [
                            {
                                "object": "embedding",
                                "index": index,
                                "embedding": vector.tolist(),
                            }
                            for index, vector in enumerate(vectors)
                        ],
                    },
                )
                return
            chat = self.path == "/v1/chat/completions"
            if chat:
                messages = chat_messages(body.get("messages"))
                try:
                    tokens = engine.encode_chat(messages)
                except Exception as error:
                    raise RequestError(f"chat template failed: {error}") from None
            else:
                tokens = prompt_tokens(engine, body.get("prompt"))
            params = sampling_params(body, engine.capacity)
            if len(tokens) > engine.capacity:
                raise RequestError(
                    f"prompt has {len(tokens)} tokens; the context holds "
                    f"{engine.capacity}",
                    param="messages" if chat else "prompt",
                )
            identifier = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
            if body.get("stream"):
                include_usage = bool(
                    (body.get("stream_options") or {}).get("include_usage"),
                )
                self.stream(engine, tokens, params, identifier, chat, include_usage)
            else:
                text, reason, usage = engine.complete(tokens, params)
                self.send_json(
                    HTTPStatus.OK,
                    completion(identifier, engine.name, text, reason, usage, chat),
                )
        except RequestError as error:
            self.send_request_error(error)
        except ValueError as error:
            self.send_request_error(RequestError(str(error)))
        except Exception as error:
            self.send_request_error(
                RequestError(
                    f"{type(error).__name__}: {error}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
            )

    def stream(self, engine, tokens, params, identifier, chat, include_usage):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        created, broken = int(time.time()), []

        def send(payload):
            try:
                self.wfile.write(
                    b"data: "
                    + (
                        payload
                        if isinstance(payload, bytes)
                        else json.dumps(payload).encode()
                    )
                    + b"\n\n",
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                broken.append(True)

        def piece(**fields):
            return chunk(identifier, engine.name, created, chat, **fields)

        if chat:
            send(piece(role=True))
        events = engine.generate(tokens, params, cancelled=lambda: bool(broken))
        try:
            for event in events:
                if broken:
                    break
                if event.finish_reason:
                    send(piece(reason=event.finish_reason))
                    if include_usage:
                        send(piece(usage=event.usage))
                elif event.text:
                    send(piece(text=event.text))
        except Exception as error:
            category = (
                "invalid_request_error"
                if isinstance(error, ValueError)
                else "server_error"
            )
            send({"error": {"message": str(error), "type": category}})
        finally:
            events.close()
        if not broken:
            send(b"[DONE]")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine, *, verbose=False):
        super().__init__(address, Handler)
        self.engine, self.verbose, self.started = engine, verbose, int(time.time())

    def server_bind(self):
        # Skip HTTPServer's getfqdn(host), which can stall on MagicDNS hosts.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
