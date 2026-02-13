
import requests
import json
import os
import re
from collections import Counter

# Path to the API key file (placed next to this node's folder)
API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mancer_api_key.txt")

# Directory for persisted chat histories
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history")

# Hidden system prompt injected at the start of every request to sanitize output
_HIDDEN_SYSTEM_PROMPT = (
    "You are a raw text completion engine. Output only the requested text and nothing else. "
    "Do not use Markdown, do not use bold text, do not use headers, and do not use code blocks. "
    "Do not include introductory filler like \"Here is the story.\" Just output the raw string."
)

# Rough token budget — leave room for the response (max_tokens is separate,
# but we cap the *input* context to avoid overflowing the model's window).
_MAX_CONTEXT_CHARS = 4096 * 4  # ~4096 tokens at ~4 chars/token

# Memory retrieval constants
_LONG_TERM_COUNT = 3   # number of keyword-relevant past exchanges to include
_SHORT_TERM_COUNT = 5  # number of most-recent exchanges to include

# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #

def _strip_markdown(text):
    """Remove accidental markdown formatting from the response."""
    # Remove fenced code blocks (```lang ... ```)
    text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).strip('`').strip(), text)
    # Remove inline code backticks
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # Remove any remaining stray backticks
    text = text.replace('`', '')
    # Remove markdown headers (# ## ### etc.)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Remove italic (*text* or _text_) — careful not to strip underscores in words
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)
    # Remove strikethrough (~~text~~)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    # Remove horizontal rules (---, ***, ___)
    text = re.sub(r'^(\s*[-*_]){3,}\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _load_api_key():
    """Load the Mancer API key from mancer_api_key.txt."""
    if os.path.isfile(API_KEY_FILE):
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    return None


def _get_key_status():
    """Return a display string showing whether a key is loaded."""
    key = _load_api_key()
    if key:
        # Show first 4 and last 4 chars, mask the rest
        if len(key) > 8:
            return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"
        else:
            return '*' * len(key)
    return "(no key found — add mancer_api_key.txt)"


# --------------------------------------------------------------------------- #
#  Chat-history persistence
# --------------------------------------------------------------------------- #

def _sanitize_filename(name):
    """Turn an arbitrary conversation_id into a safe filename."""
    name = re.sub(r'[^\w\-. ]', '_', name.strip())
    return name if name else "default"


def _history_path(conversation_id):
    """Return the full path to the JSON history file for a conversation."""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    return os.path.join(CHAT_HISTORY_DIR, f"{_sanitize_filename(conversation_id)}.json")


def _load_history(conversation_id):
    """Load the exchange list from disk. Returns [] if the file doesn't exist."""
    path = _history_path(conversation_id)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, OSError):
            pass  # corrupted — start fresh
    return []


def _save_history(conversation_id, history):
    """Persist the exchange list to disk."""
    path = _history_path(conversation_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
#  Pseudo-vector relevance search (lightweight keyword matching)
# --------------------------------------------------------------------------- #

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into about between through during before after "
    "and but or nor not so yet i me my we our you your he him his she "
    "her it its they them their this that these those".split()
)


def _tokenize(text):
    """Lowercase split, remove stop-words and very short tokens."""
    return [w for w in re.findall(r'[a-z0-9]+', text.lower())
            if w not in _STOP_WORDS and len(w) > 1]


def _keyword_relevance(query_tokens, exchange):
    """Score an exchange against the query using token overlap (Jaccard-ish)."""
    exchange_text = exchange.get("user", "") + " " + exchange.get("assistant", "")
    exchange_tokens = set(_tokenize(exchange_text))
    if not exchange_tokens or not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    overlap = query_set & exchange_tokens
    # Weight by how many query terms matched, normalised by query length
    return len(overlap) / len(query_set) if query_set else 0.0


def _find_relevant_exchanges(history, current_input, short_term_indices, count=_LONG_TERM_COUNT):
    """Return the `count` most keyword-relevant exchanges, excluding short-term ones."""
    query_tokens = _tokenize(current_input)
    if not query_tokens:
        return []

    scored = []
    for idx, ex in enumerate(history):
        if idx in short_term_indices:
            continue  # skip entries already in short-term memory
        score = _keyword_relevance(query_tokens, ex)
        if score > 0:
            scored.append((score, idx, ex))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [item[2] for item in scored[:count]]


# --------------------------------------------------------------------------- #
#  Token-budget helper
# --------------------------------------------------------------------------- #

def _estimate_tokens(text):
    """Rough token count: ~4 characters per token."""
    return max(1, len(text) // 4)


def _messages_char_len(messages):
    """Total character length of all message contents."""
    return sum(len(m.get("content", "")) for m in messages)


# --------------------------------------------------------------------------- #
#  ComfyUI Node
# --------------------------------------------------------------------------- #

class MancerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key_status": ("STRING", {"multiline": False, "default": _get_key_status(), "tooltip": "API key is loaded from mancer_api_key.txt in the node folder. This field is display-only."}),
                "conversation_id": ("STRING", {"multiline": False, "default": "default", "tooltip": "A unique name for this conversation. History is saved to chat_history/<name>.json."}),
                "character_card_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Paste a character card definition here. This will be combined with the formatting rules into the system prompt."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"default": "mytholite"}),
                "max_tokens": ("INT", {"default": 500, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "dynatemp_mode": ("INT", {"default": 0, "min": 0, "max": 1}),
                "dynatemp_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "dynatemp_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "dynatemp_exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_a": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tfs": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smoothing_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "xtc_probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "xtc_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dry_multiplier": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "dry_base": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 3.0, "step": 0.01}),
                "dry_allowed_length": ("INT", {"default": 2, "min": 0, "max": 100}),
                "dry_range": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "Mancer API"
    OUTPUT_NODE = True  # has side-effects (writes history)

    # ------------------------------------------------------------------ #
    #  Context assembler
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_messages(combined_system, history, current_input):
        """
        Assemble the messages list within the token budget:
          1. System prompt  (formatting rules + system prompt + character card)
          2. Long-term memory  (up to 3 keyword-relevant past exchanges)
          3. Short-term memory (last 5 exchanges)
          4. Current user input
        """
        messages = []
        budget = _MAX_CONTEXT_CHARS

        # --- 1. System prompt (always included) ---
        system_msg = {"role": "system", "content": combined_system}
        messages.append(system_msg)
        budget -= len(combined_system)

        # --- Reserve space for the current user input (always included) ---
        current_msg = {"role": "user", "content": current_input}
        budget -= len(current_input)

        # --- 3. Short-term memory (last N exchanges) ---
        short_term_start = max(0, len(history) - _SHORT_TERM_COUNT)
        short_term_indices = set(range(short_term_start, len(history)))
        short_term = history[short_term_start:]

        # --- 2. Long-term memory (most relevant, excluding short-term) ---
        long_term = _find_relevant_exchanges(
            history, current_input, short_term_indices, count=_LONG_TERM_COUNT
        )

        # Add long-term memory messages (older relevant context)
        if long_term:
            ltm_header = {"role": "system", "content": "[Relevant earlier context]"}
            header_len = len(ltm_header["content"])
            if budget > header_len:
                messages.append(ltm_header)
                budget -= header_len

            for ex in long_term:
                user_text = ex.get("user", "")
                asst_text = ex.get("assistant", "")
                pair_len = len(user_text) + len(asst_text)
                if budget - pair_len < 0:
                    break  # no room
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": asst_text})
                budget -= pair_len

        # Add short-term memory messages (recent conversation flow)
        if short_term:
            stm_header = {"role": "system", "content": "[Recent conversation]"}
            header_len = len(stm_header["content"])
            if budget > header_len:
                messages.append(stm_header)
                budget -= header_len

            for ex in short_term:
                user_text = ex.get("user", "")
                asst_text = ex.get("assistant", "")
                pair_len = len(user_text) + len(asst_text)
                if budget - pair_len < 0:
                    break  # no room
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": asst_text})
                budget -= pair_len

        # --- 4. Current user input (always last) ---
        messages.append(current_msg)
        return messages

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #
    def generate_content(self, api_key_status, conversation_id, character_card_text,
                         system_prompt, user_prompt, model, max_tokens, temperature,
                         dynatemp_mode, dynatemp_min, dynatemp_max, dynatemp_exponent,
                         repetition_penalty, presence_penalty, frequency_penalty,
                         top_k, min_p, top_a, top_p, typical_p, tfs,
                         smoothing_factor, xtc_probability, xtc_threshold,
                         dry_multiplier, dry_base, dry_allowed_length, dry_range):

        # Load the real API key from file (ignore the widget value)
        api_key = _load_api_key()
        if not api_key:
            return ("Error: No API key found. Please create a file called 'mancer_api_key.txt' in the ComfyuiMancer node folder and paste your Mancer API key into it.",)

        url = "https://neuro.mancer.tech/oai/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Build the combined system message: formatting rules > system prompt > character card
        combined_system = _HIDDEN_SYSTEM_PROMPT
        if system_prompt.strip():
            combined_system += "\n\n" + system_prompt.strip()
        if character_card_text.strip():
            combined_system += "\n\n" + character_card_text.strip()

        # Load conversation history
        history = _load_history(conversation_id)

        # Assemble context-aware messages
        messages = self._build_messages(combined_system, history, user_prompt)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "dynatemp_mode": dynatemp_mode,
            "dynatemp_min": dynatemp_min,
            "dynatemp_max": dynatemp_max,
            "dynatemp_exponent": dynatemp_exponent,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "top_k": top_k,
            "min_p": min_p,
            "top_a": top_a,
            "top_p": top_p,
            "typical_p": typical_p,
            "tfs": tfs,
            "smoothing_factor": smoothing_factor,
            "xtc_probability": xtc_probability,
            "xtc_threshold": xtc_threshold,
            "dry_multiplier": dry_multiplier,
            "dry_base": dry_base,
            "dry_allowed_length": dry_allowed_length,
            "dry_range": dry_range,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                result = data["choices"][0]["message"]["content"]
                result = _strip_markdown(result)

                # Persist the new exchange to history
                history.append({
                    "user": user_prompt,
                    "assistant": result
                })
                _save_history(conversation_id, history)

                return (result,)
            else:
                return (f"Error: Unexpected response format: {json.dumps(data)}",)

        except requests.exceptions.RequestException as e:
            return (f"Error: {str(e)}",)
