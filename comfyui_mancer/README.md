
# ComfyUI-Mancer

A custom node for ComfyUI that allows you to generate text using the Mancer API.

## Installation

1. Copy this folder into your ComfyUI `custom_nodes` directory.
2. Ensure you have the `requests` library installed in your Python environment.
   ```bash
   pip install requests
   ```

## Usage

- **api_key**: Your Mancer API key.
- **system_prompt**: The instructions for the AI (e.g., "You are a helpful assistant").
- **user_prompt**: The actual query or text you want to process.
- **Other Parameters**: These correspond to the advanced sampling settings provided by the Mancer API.

## Features

- Supports all major Mancer parameters (Temperature, Min P, DRY, XTC, etc.).
- Simple string output for easy chaining with other nodes.
