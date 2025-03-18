#!/usr/bin/env python3
import argparse
import json
import logging
import os
from openai import OpenAI  # Updated import for OpenAI SDK v1.0.0+
import sys
import markdown  # Required for converting Markdown to HTML

# Dictionary of available models with updated pricing and descriptions.
AVAILABLE_MODELS = {
    "gpt-4.5-preview": {
        "price": "Input: $75.00, Cached: $37.50, Output: $150.00 per 1M tokens",
        "description": "Latest preview of GPT-4.5 with highest quality outputs."
    },
    "gpt-4o": {
        "price": "Input: $2.50, Cached: $1.25, Output: $10.00 per 1M tokens",
        "description": "Standard GPT-4o model with competitive pricing."
    },
    "gpt-4o-audio-preview": {
        "price": "Input: $2.50, Output: $10.00 per 1M tokens",
        "description": "Audio-specific preview model for GPT-4o."
    },
    "gpt-4o-realtime-preview": {
        "price": "Input: $5.00, Cached: $2.50, Output: $20.00 per 1M tokens",
        "description": "Real-time GPT-4o preview model for rapid responses."
    },
    "gpt-4o-mini": {
        "price": "Input: $0.15, Cached: $0.075, Output: $0.60 per 1M tokens",
        "description": "Smaller, cost-effective variant of GPT-4o."
    },
    "gpt-4o-mini-audio-preview": {
        "price": "Input: $0.15, Output: $0.60 per 1M tokens",
        "description": "Audio preview model for GPT-4o-mini."
    },
    "gpt-4o-mini-realtime-preview": {
        "price": "Input: $0.60, Cached: $0.30, Output: $2.40 per 1M tokens",
        "description": "Real-time variant of GPT-4o-mini for dynamic tasks."
    },
    "o1": {
        "price": "Input: $15.00, Cached: $7.50, Output: $60.00 per 1M tokens",
        "description": "High-quality o1 model for advanced tasks."
    },
    "o3-mini": {
        "price": "Input: $1.10, Cached: $0.55, Output: $4.40 per 1M tokens",
        "description": "Compact and efficient o3-mini model."
    },
    "o1-mini": {
        "price": "Input: $1.10, Cached: $0.55, Output: $4.40 per 1M tokens",
        "description": "Mini version of o1 for budget-conscious usage."
    },
    "gpt-4o-mini-search-preview": {
        "price": "Input: $0.15, Output: $0.60 per 1M tokens",
        "description": "Search-optimized GPT-4o-mini preview model."
    },
    "gpt-4o-search-preview": {
        "price": "Input: $2.50, Output: $10.00 per 1M tokens",
        "description": "Search-optimized GPT-4o preview model."
    },
    "computer-use-preview": {
        "price": "Input: $3.00, Output: $12.00 per 1M tokens",
        "description": "Optimized for tasks involving computer use."
    },
    "grok-v2": {
        "price": "Migrated via xAI settings (see Grok-2-latest)",
        "description": "Grok language model v2 using xAI migration (via OpenAI SDK)."
    }
}

def print_available_models():
    print("Available Models and Rates:")
    for model, info in AVAILABLE_MODELS.items():
        print(f"  - {model}: {info['price']} - {info['description']}")
    print()

def load_api_keys(api_key_file: str):
    """
    Load API keys from a JSON file.
    Expected JSON structure:
      {
        "openai_api_key": "...",
        "grok_api_key": "..."
      }
    At least one key must be provided.
    """
    try:
        with open(api_key_file, "r", encoding="utf-8") as f:
            keys = json.load(f)
        openai_key = keys.get("openai_api_key", "").strip()
        grok_key = keys.get("grok_api_key", "").strip()
        if not openai_key and not grok_key:
            raise ValueError("At least one of 'openai_api_key' or 'grok_api_key' must be provided in the API key file.")
        return openai_key, grok_key
    except Exception as e:
        logging.error(f"Error loading API keys from {api_key_file}: {e}")
        exit(1)

def generate_summary_and_title(transcript_text, speaker_hints=None, model="grok-v2", grok_api_key=None):
    """
    Build an LLM prompt that incorporates speaker hints and the transcript.
    Call the chosen LLM API to generate a summary and title.
    """
    prompt = (
    "You are an expert conversation analyst and summarizer. You are provided with a transcript of a conversation that includes timestamps and "
    "speaker labels formatted as 'Speaker SPEAKER_##'. The transcript may cover any topic and does not include explicit speaker names or roles; however, "
    "contextual clues (and any provided speaker hints) can help you deduce these details.\n\n"
    
    "Your tasks are as follows:\n"
    "1. Deduce Speaker Mapping: Identify each unique speaker ID and deduce their likely identity, role, or affiliation based on the context. "
    "Present this mapping in a clear and organized format (for example, as a bullet list or table).\n\n"
    
    "2. Generate a Comprehensive Summary: Write a detailed, structured summary of the conversation that captures all major themes, topics, "
    "questions, decisions, and outcomes discussed. Ensure your summary is organized, thorough, and omits timestamps, focusing solely on the "
    "substance of the dialogue.\n\n"
    
    "3. Title Creation: Generate a descriptive title that encapsulates the overall theme or main outcome of the conversation.\n\n"
    
    "Formatting Requirements:\n"
    "- Start with a line: 'Title: <Your Recommended Title>'\n"
    "- Next, include a section titled 'Speaker Mapping:' where you list each speaker ID along with your deduced details.\n"
    "- Next, include a section titled 'Outline summary:' where you list the key topics and a few key bulleted points about each.\n"
    "- Finally, include a section titled 'Detailed Summary:' that contains your comprehensive summary of the conversation, "
    "  elaborating greatly on the outline structure, with one labeled section per topic. For each topic, write a paragraph "
    "  expanding on all relevant details from the bullet in the outline structure.\n\n"
    
    "Transcript:\n"
    )
    if speaker_hints:
        prompt += f"Additional speaker hints: {speaker_hints}. "
    prompt += "\n\nTranscript:\n" + transcript_text

    if model == "grok-v2":
        logging.info("Using Grok v2 API via xAI migration settings.")
        client = OpenAI(
            api_key=grok_api_key,
            base_url="https://api.x.ai/v1"
        )
        try:
            response = client.chat.completions.create(
                model="grok-2-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Grok API call failed: {e}")
            exit(1)
    else:
        logging.info(f"Using OpenAI model: {model}")
        client = OpenAI(api_key=OpenAI.api_key)  # Use the globally set OpenAI key
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            exit(1)

def main():
    epilog_text = "Available Models and Rates:\n"
    for model, info in AVAILABLE_MODELS.items():
        epilog_text += f"  - {model}: {info['price']} - {info['description']}\n"
    
    parser = argparse.ArgumentParser(
        description=(
            "Generate a detailed summary and deduce speaker roles from a conversation transcript using an LLM API.\n\n"
            "Usage:\n"
            "  summarize_transcript.py [options] transcript_file\n\n"
            "Options:\n"
            "  --speaker-hints   Provide hints about speaker identities.\n"
            "  --model           Specify the LLM model to use (see available models below).\n"
            "  --api-key-file    Path to a JSON file containing API keys for OpenAI and/or Grok v2.\n"
            "  --output-file     File path to store the generated summary.\n"
            "  --output-format   Format the output as 'json', 'markdown', or 'html'.\n"
            "  --list-models     List available models and their rates and exit.\n"
        ),
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("transcript_file", nargs='?', help="Path to the transcript file")
    parser.add_argument("--speaker-hints", type=str, default=None,
                        help=('Hints about speaker identities. Example: "Eric, Nacu, Chris are known to be speakers. '
                              "Eric is the one asking questions about billing deadlines." ))
    parser.add_argument("--model", type=str, default="grok-v2",
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="LLM model to use (see available models below)")
    parser.add_argument("--api-key-file", type=str, default="api_keys.json",
                        help="Path to the JSON file containing API keys for OpenAI and/or Grok v2.")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and their rates and exit")
    parser.add_argument("--output-file", type=str,
                        help="File path to store the generated summary")
    parser.add_argument("--output-format", type=str, choices=["json", "markdown", "html"], default="markdown",
                        help="Format of the output summary (json, markdown, or html)")
    
    # Show help if no arguments are provided.
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        sys.exit(0)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    transcript_path = args.transcript_file
    if not transcript_path:
        logging.error("No transcript file provided.")
        parser.print_help()
        sys.exit(1)
        
    if not os.path.exists(transcript_path):
        logging.error(f"Transcript file {transcript_path} does not exist.")
        sys.exit(1)

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
    except Exception as e:
        logging.error(f"Failed to read transcript file: {e}")
        sys.exit(1)

    # Load API keys.
    openai_key, grok_key = load_api_keys(args.api_key_file)
    # Check model-specific key requirements.
    if args.model == "grok-v2":
        if not grok_key:
            logging.error("Grok model selected but grok_api_key is not provided in the API key file.")
            sys.exit(1)
    else:
        if not openai_key:
            logging.error("Non-Grok model selected but openai_api_key is not provided in the API key file.")
            sys.exit(1)
    # For non-Grok usage, set the OpenAI API key.
    OpenAI.api_key = openai_key

    logging.info("Generating summary and title...")
    result = generate_summary_and_title(
        transcript_text,
        speaker_hints=args.speaker_hints,
        model=args.model,
        grok_api_key=grok_key
    )

    # Format output according to requested format.
    if args.output_format == "json":
        output_data = {"summary": result}
        formatted_output = json.dumps(output_data, indent=2)
    elif args.output_format == "html":
        # Convert markdown to HTML.
        html_content = markdown.markdown(result)
        formatted_output = f"<html><body>{html_content}</body></html>"
    else:  # markdown
        formatted_output = result

    # Either print the output or store it in a file.
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            logging.info(f"Summary saved to {args.output_file}")
        except Exception as e:
            logging.error(f"Failed to write summary to {args.output_file}: {e}")
            sys.exit(1)
    else:
        print(formatted_output)

if __name__ == "__main__":
    main()
