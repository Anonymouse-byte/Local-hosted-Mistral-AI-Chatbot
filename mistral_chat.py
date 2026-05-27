import ollama
import gradio as gr

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and clear in your responses."

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULTS = {
    "model": "mistral-nemo:latest",
    "temperature":    0.7,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.1,
    "num_predict":    1024,
    "seed":           -1,
}


def get_installed_models():
    try:
        result = ollama.list()
        models = getattr(result, "models", []) or []
        names = []
        for m in models:
            name = m.model if hasattr(m, "model") else (m.get("model") or m.get("name") or str(m))
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def chat(
    user_input,
    history,          # list of {"role": ..., "content": ...} dicts (Gradio format)
    system_prompt,
    model,
    temperature,
    top_p,
    top_k,
    repeat_penalty,
    num_predict,
    seed,
):
    """Stream a response from the local Ollama model."""

    options = {
        "temperature":    float(temperature),
        "top_p":          float(top_p),
        "top_k":          int(top_k),
        "repeat_penalty": float(repeat_penalty),
        "num_predict":    int(num_predict),
        "num_gpu":        99,   # use all available GPU layers
    }
    if int(seed) != -1:
        options["seed"] = int(seed)

    # Build message list: system + prior turns + new user message
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_input})

    # Stream tokens
    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True,
            options=options,
        )
    except ollama.ResponseError as e:
        yield (
            f"[Error] {e}. Make sure the selected Ollama model is installed and available. "
            "Run `ollama list` to verify installed models and `ollama pull <model>` to install one."
        )
        return

    partial = ""
    try:
        for chunk in stream:
            token = chunk["message"]["content"]
            partial += token
            yield partial          # Gradio streams each partial reply
    except ollama.ResponseError as e:
        error_msg = f"[Error] {e}. Make sure the selected Ollama model is installed and available."
        yield error_msg
        return


def build_ui():
    available_models = get_installed_models()
    model_choices = available_models
    model_note = (
        "**Installed Ollama models:** " + ", ".join(f"`{m}`" for m in available_models)
        if available_models
        else "**No installed Ollama models were detected.** Install one using `ollama pull <model>` and refresh, or run `ollama list` to verify."
    )

    with gr.Blocks(
        title="Mistral Chat",
    ) as demo:

        gr.Markdown("## _AI_ — Mistral local chat")
        gr.Markdown(model_note)

        with gr.Row():
            # ── Left: chat ────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=480,
                    show_label=False,
                    avatar_images=(None, None),
                )
                with gr.Row():
                    user_box = gr.Textbox(
                        placeholder="Type a message…",
                        show_label=False,
                        scale=9,
                        container=False,
                        autofocus=True,
                        interactive=bool(available_models),
                    )
                    send_btn = gr.Button(
                        "Send",
                        scale=1,
                        variant="primary",
                        interactive=bool(available_models),
                    )

            # ── Right: parameters ─────────────────────────────────────────
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Parameters")

                model_dd = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    label="Model",
                    info=(
                        "Select one of the installed Ollama models."
                        if available_models
                        else "No installed models available."
                    ),
                    interactive=bool(available_models),
                )
                temperature_sl = gr.Slider(0, 2,    value=DEFAULTS["temperature"],    step=0.01, label="temperature")
                top_p_sl       = gr.Slider(0, 1,    value=DEFAULTS["top_p"],          step=0.01, label="top_p")
                top_k_sl       = gr.Slider(1, 100,  value=DEFAULTS["top_k"],          step=1,    label="top_k")
                rep_sl         = gr.Slider(1, 2,    value=DEFAULTS["repeat_penalty"], step=0.01, label="repeat_penalty")
                pred_sl        = gr.Slider(64, 4096,value=DEFAULTS["num_predict"],    step=64,   label="num_predict")
                seed_sl        = gr.Slider(-1, 999, value=DEFAULTS["seed"],           step=1,    label="seed  (−1 = random)")

                gr.Markdown("### System prompt")
                system_box = gr.Textbox(
                    value=SYSTEM_PROMPT,
                    lines=5,
                    max_lines=12,
                    label="",
                    show_label=False,
                    placeholder="Enter system/internal instructions…",
                )

                clear_btn = gr.Button("Clear chat", variant="secondary")

        # ── State ─────────────────────────────────────────────────────────
        history_state = gr.State([])   # list of {"role":…,"content":…}

        # ── Helpers ───────────────────────────────────────────────────────
        def submit(user_input, history, system_prompt, model,
                   temperature, top_p, top_k, repeat_penalty, num_predict, seed):
            if not user_input.strip():
                yield history, history, ""
                return

            if not available_models:
                history = history + [{"role": "user", "content": user_input}]
                history = history + [{
                    "role": "assistant",
                    "content": (
                        "[Error] No installed Ollama models were detected. "
                        "Install one with `ollama pull <model>` and restart the app."
                    ),
                }]
                display = [
                    {"role": ("user" if m["role"] == "user" else "assistant"),
                     "content": m["content"],
                     "metadata": {"title": "User" if m["role"] == "user" else "_AI_"}}
                    for m in history
                ]
                yield display, history, ""
                return

            if model not in available_models:
                history = history + [{"role": "user", "content": user_input}]
                history = history + [{
                    "role": "assistant",
                    "content": (
                        f"[Error] Model '{model}' is not installed. "
                        f"Installed models: {', '.join(available_models)}."
                    ),
                }]
                display = [
                    {"role": ("user" if m["role"] == "user" else "assistant"),
                     "content": m["content"],
                     "metadata": {"title": "User" if m["role"] == "user" else "_AI_"}}
                    for m in history
                ]
                yield display, history, ""
                return

            history = history + [{"role": "user", "content": user_input}]
            history = history + [{"role": "assistant", "content": ""}]

            gen = chat(
                user_input, history[:-1], system_prompt, model,
                temperature, top_p, top_k, repeat_penalty, num_predict, seed,
            )
            for partial in gen:
                history[-1]["content"] = partial
                # Rename roles for display: user → "User", assistant → "_AI_"
                display = [
                    {"role": ("user" if m["role"] == "user" else "assistant"),
                     "content": m["content"],
                     "metadata": {"title": "User" if m["role"] == "user" else "_AI_"}}
                    for m in history
                ]
                yield display, history, ""

        def clear(_):
            return [], [], ""

        # ── Wire up ───────────────────────────────────────────────────────
        param_inputs = [
            system_box, model_dd,
            temperature_sl, top_p_sl, top_k_sl,
            rep_sl, pred_sl, seed_sl,
        ]

        send_btn.click(
            submit,
            inputs=[user_box, history_state] + param_inputs,
            outputs=[chatbot, history_state, user_box],
        )
        user_box.submit(
            submit,
            inputs=[user_box, history_state] + param_inputs,
            outputs=[chatbot, history_state, user_box],
        )
        clear_btn.click(clear, inputs=[history_state],
                        outputs=[chatbot, history_state, user_box])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        theme=gr.themes.Default(
            primary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("IBM Plex Mono"),
        ),
        css="""
        #chatbot .message.user  { background: #1e293b; color: #e2e8f0; }
        #chatbot .message.bot   { background: #0f172a; color: #94a3b8; }
        .label-user { font-weight: 700; letter-spacing: .05em; }
        """,
    )          # opens http://127.0.0.1:7860 in your browser