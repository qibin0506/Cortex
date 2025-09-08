import json
import os
import uuid

from bottle import Bottle, request, response, run

import torch
from utils import init_env, get_model_config
from llm_model import LlmModel
from llm_trainer import TrainerTools, streaming_generate

init_env()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
model_name = 'dpo.bin'

if not os.path.exists(f'./{model_name}'):
    from modelscope import snapshot_download
    snapshot_download(
        f'qibin0506/Cortex-V2',
        allow_file_pattern=[model_name],
        local_dir='./'
    )

model = LlmModel(get_model_config(long_context=True)).to(device=device, dtype=dtype)
model.load_state_dict(torch.load('./dpo.bin', weights_only=True))
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
        think_budget_enable = thinking and payload.get('think_budget_enable')
        think_budget = payload.get('think_budget')
        if not think_budget:
            think_budget_enable = False

        # 仅保留两轮对话
        chat_history = chat_history[-3:]

        # if not thinking:
        #     chat_history = chat_history[-5:]
        # else:
        #     chat_history = chat_history[-1:]

        if not chat_history:
            yield fmt_msg('error', 'Chat history cannot be empty')
            return
    except (json.JSONDecodeError, AttributeError):
        yield fmt_msg('error', 'Invalid JSON payload')
        return

    try:
        chat_history = [{'role':'system', 'content':'你是由QB开发的聊天助手Cortex'}, *chat_history]
        chat_template = TrainerTools().tokenizer.apply_chat_template(chat_history, tokenizer=False)

        if not thinking:
            chat_template = f'{chat_template}<assistant><think> </think>'
        else:
            chat_template = f'{chat_template}<assistant><think>'

        prompt_token = TrainerTools().tokenizer.encode(chat_template, unsqueeze=True)
        output_token_count = max(2048 - prompt_token.shape[-1], 0)

        if think_budget_enable:
            think_budget_content = '。考虑到用户的时间限制，我现在必须根据思考直接给出解决方案\n'
            think_budget_encoded = TrainerTools().tokenizer.encode(f'{think_budget_content}</think>')
            think_budget = think_budget - len(think_budget_encoded)
            output_token_count = min(think_budget, output_token_count)

            generator = streaming_generate(
                model=model,
                prompt=prompt_token,
                max_position_embeddings=2048,
                max_new_tokens=output_token_count,
                temperature=temperature,
                k=None,
                p=top_p,
                device=device
            )

            think_content = ''
            for chunk in generator:
                think_content += chunk
                if chunk == '</think>': break
                yield fmt_msg('thinking_chunk', chunk)

            if '</think>' not in think_content:
                think_content += f'{think_budget_content}</think>'
                yield fmt_msg('thinking_chunk', think_budget_content)

            prompt_token = torch.concat([prompt_token, TrainerTools().tokenizer.encode(think_content, unsqueeze=True)], dim=-1)
            output_token_count = max(2048 - prompt_token.shape[-1], 0)

        generator = streaming_generate(
            model=model,
            prompt=prompt_token,
            max_position_embeddings=2048,
            max_new_tokens=output_token_count,
            temperature=temperature,
            k=None,
            p=top_p,
            device=device
        )

        type = 'thinking_chunk' if thinking else 'answer_chunk'
        for chunk in generator:
            if chunk == '</s>': break
            if chunk == '<assistant>' or chunk == '</assistant>': continue
            if chunk == '</think>' or chunk == '</answer>': continue
            if chunk == '<think>':
                type = 'thinking_chunk'
                continue
            elif chunk == '<answer>':
                type = 'answer_chunk'
                continue

            yield fmt_msg(type, chunk)
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
