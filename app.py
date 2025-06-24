import json
import os.path
import time
import uuid

from bottle import Bottle, request, response, run

import torch
from utils import init_env, get_model_config
from llm_model import LlmModel
from llm_trainer import TrainerTools, streaming_generate
from constant import system_prompt_content

init_env()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

model = LlmModel(get_model_config()).to(device)
model.load_state_dict(torch.load('./last_checkpoint.bin', weights_only=True))
model.eval()

app = Bottle()

with open('./static/index.html', 'r') as f:
    html = f.read()


def fmt_msg(event, data):
    return f"{json.dumps({'event': event, 'data':data})}\n\n"


def generate_user_uuid() -> str:
    unique_id = uuid.uuid4()
    return unique_id.hex

@app.get('/')
def index():
    new_uuid = generate_user_uuid()
    return html.replace('{{__USER_UUID_PLACEHOLDER__}}', new_uuid)


@app.hook('after_request')
def enable_cors():
    """Add CORS headers to all responses to allow cross-origin requests."""
    response.set_header('Access-Control-Allow-Origin', '*')
    response.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    response.set_header('Access-Control-Allow-Headers', 'Content-Type')


@app.route('/api/chat', method=['OPTIONS'])
def options_handler():
    """Handle CORS pre-flight requests."""
    return


@app.route('/api/chat', method=['POST'])
def sse_chat():
    """
    Handles chat requests and returns a streaming response via SSE.
    """
    response.content_type = 'text/event-stream'
    response.set_header('Cache-Control', 'no-cache')
    response.set_header('Connection', 'keep-alive')

    try:
        payload = request.json
        chat_history = payload.get('history')
        thinking = payload.get('thinking')
        user_uuid = payload.get('uuid')
        temperature = payload.get('temperature')
        top_p = payload.get('top_p')
        think_budget_enable = payload.get('think_budget_enable')

        if not think_budget_enable:
            reasoning_budget = None
        else:
            reasoning_budget = payload.get('think_budget')

        if not thinking:
            chat_history = chat_history[-5:]
        else:
            chat_history = chat_history[-1:]

        if not chat_history:
            yield fmt_msg('error', 'Chat history cannot be empty')
            return
    except (json.JSONDecodeError, AttributeError):
        yield fmt_msg('error', 'Invalid JSON payload')
        return

    try:
        chat_history = [{'role':'system', 'content':system_prompt_content}, *chat_history]
        chat_template = TrainerTools().tokenizer.apply_chat_template(chat_history, tokenizer=False)

        if not thinking:
            chat_template = f'{chat_template}<assistant><reasoning> </reasoning>'
        else:
            chat_template = f'{chat_template}<assistant><reasoning>'

        generator = streaming_generate(
            model=model,
            prompt=chat_template,
            max_position_embeddings=1024,
            max_new_tokens=1024,
            temperature=temperature,
            k=None,
            p=top_p,
            device=device,
            reasoning_budget=reasoning_budget
        )

        full_response = ''
        type = 'thinking_chunk' if thinking else 'answer_chunk'
        think_token_count = 0
        for chunk in generator:
            full_response = f'{full_response}{chunk}'
            if chunk == '</s>': break
            if chunk == '<assistant>' or chunk == '</assistant>': continue
            if chunk == '</reasoning>' or chunk == '</answer>': continue
            if chunk == '<reasoning>':
                type = 'thinking_chunk'
                think_token_count = 0
                continue
            elif chunk == '<answer>':
                type = 'answer_chunk'
                continue
            if type == 'thinking_chunk':
                think_token_count += 1

            yield fmt_msg(type, chunk)

        if user_uuid:
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            if not os.path.exists('./server_log/'):
                os.mkdir('./server_log/')

            with open(f'./server_log/{user_uuid}.txt', 'a') as f:
                f.write(f"[{cur_time}] [temperature={temperature}, top_p={top_p}] {chat_template}{full_response}\n")
    except Exception as e:
        print(f"Error during model generation: {e}")
        yield fmt_msg('error', f'Internal server error: {e}')


if __name__ == '__main__':
    # Use 'waitress' for a production-ready WSGI server on Windows/Linux
    try:
        from waitress import serve

        serve(app, host='0.0.0.0', port=8080)
    except ImportError:
        print("Waitress not found, falling back to Bottle's default server.")
        # Fallback for environments without waitress
        run(app, host='0.0.0.0', port=8080, server='paste')
    print("Bottle server started at http://0.0.0.0:8080/")
