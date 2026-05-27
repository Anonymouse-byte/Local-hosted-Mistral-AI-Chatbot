# Mistral Local Chat

A lightweight, locally-hosted AI chatbot UI built with [Gradio](https://gradio.app/) and [Ollama](https://ollama.com/). The key feature is the ability to **adjust model parameters in real time**, mid-conversation — no restart required.

---

## Why it was made

Most chat interfaces lock you into a fixed configuration. This app was built specifically so you can experiment with how a model behaves *while you're talking to it*. Want to make responses more creative? Crank up the temperature. Getting repetitive output? Adjust the repeat penalty. All controls are live on the sidebar and take effect on the very next message.

---

## Features

- **Live parameter controls** — temperature, top-p, top-k, repeat penalty, token limit, and seed are all adjustable mid-conversation
- **Model switcher** — swap between any locally installed Ollama model without restarting
- **Editable system prompt** — change the assistant's persona or instructions at any time
- **Streaming responses** — tokens appear as they are generated
- **Fully local** — no data leaves your machine; no API keys required

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- At least one Ollama model pulled (e.g. `mistral-nemo:latest`)

---

## Setup

### 1. Install Ollama

Follow the instructions at [https://ollama.com/download](https://ollama.com/download) for your platform, then start the Ollama service:

```bash
ollama serve
```

### 2. Pull a model

```bash
ollama pull mistral-nemo
```

You can pull any supported model. Run `ollama list` to see what's installed.

### 3. Clone the repo and install dependencies

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install ollama gradio
```

### 4. Run the app

```bash
python app.py
```

The UI will be available at [http://127.0.0.1:7860](http://127.0.0.1:7860).

---

## Parameter reference

| Parameter | What it does | Default |
|---|---|---|
| **temperature** | Controls randomness. Higher = more creative, lower = more deterministic. | `0.7` |
| **top_p** | Nucleus sampling threshold. Lower values make output more focused. | `0.9` |
| **top_k** | Limits the pool of tokens considered at each step. | `40` |
| **repeat_penalty** | Penalises repeated tokens. Higher values reduce repetition. | `1.1` |
| **num_predict** | Maximum number of tokens to generate per response. | `1024` |
| **seed** | Fixed seed for reproducible outputs. Set to `-1` for random. | `-1` |

---

## Notes

- If no models are detected, the chat input will be disabled. Make sure `ollama serve` is running before launching the app.
- The system prompt can be changed at any time; it takes effect from the next message onwards.
- GPU layers are automatically maximised (`num_gpu: 99`); Ollama will use however many layers your hardware supports.
