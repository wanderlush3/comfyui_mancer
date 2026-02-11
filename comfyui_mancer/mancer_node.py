
import requests
import json
import os
import re

# Path to the API key file (placed next to this node's folder)
API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mancer_api_key.txt")

# Hidden system prompt injected at the start of every request to sanitize output
_HIDDEN_SYSTEM_PROMPT = (
    "You are a raw text completion engine. Output only the requested text and nothing else. "
    "Do not use Markdown, do not use bold text, do not use headers, and do not use code blocks. "
    "Do not include introductory filler like \"Here is the story.\" Just output the raw string."
)

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

class MancerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key_status": ("STRING", {"multiline": False, "default": _get_key_status(), "tooltip": "API key is loaded from mancer_api_key.txt in the node folder. This field is display-only."}),
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

    def generate_content(self, api_key_status, system_prompt, user_prompt, model, max_tokens, temperature, 
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
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _HIDDEN_SYSTEM_PROMPT},
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
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
                return (result,)
            else:
                return (f"Error: Unexpected response format: {json.dumps(data)}",)
                
        except requests.exceptions.RequestException as e:
            return (f"Error: {str(e)}",)
