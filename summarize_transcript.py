#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import time
import openai
import subprocess
import hashlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="docx.styles.styles")

# Set up colored logging
try:
    import colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
        log_colors={
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
        }
    ))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.warning("Install 'colorlog' for colored output: pip install colorlog")

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
    try:
        with open(api_key_file, "r", encoding="utf-8") as f:
            keys = json.load(f)
        openai_key = keys.get("openai_api_key", "").strip()
        grok_key = keys.get("grok_api_key", "").strip()
        if not openai_key and not grok_key:
            raise ValueError("At least one of 'openai_api_key' or 'grok_api_key' must be provided.")
        return openai_key, grok_key
    except Exception as e:
        logging.error(f"Error loading API keys from {api_key_file}: {e}")
        exit(1)

def generate_context_id(transcript_text):
    return hashlib.sha256(transcript_text.encode('utf-8')).hexdigest()

def generate_summary_and_title(client, transcript_text, transcript_filename, speaker_hints=None, model="grok-v2", grok_api_key=None, custom_prompt=None, context_id=None):
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = (
            "You are an expert conversation analyst and summarizer. You are provided with a transcript of a conversation that includes "
            "timestamps and speaker labels formatted as 'Speaker SPEAKER_##'. Although explicit speaker names or roles are not given, contextual "
            "clues (and any provided speaker hints) can help deduce these details.\n\n"
            "Your tasks are as follows:\n"
            "1. Deduce Speaker Mapping: Identify each unique speaker ID and deduce their likely identity, role, or affiliation based on the context. "
            "Present this mapping as a Markdown table with three columns: 'Name' (the speaker ID, e.g., 'SPEAKER_00'), 'Identity/Role/Affiliation' "
            "(their deduced identity, role, or affiliation), and 'Contribution' (a paragraph describing their specific contributions to the conversation).\n\n"
            "2. Generate a Comprehensive Summary: Write a detailed, structured summary of the conversation that captures all major themes, topics, "
            "questions, decisions, and outcomes. The summary should be organized into an outline of key topics with bullet points for each, "
            "followed by a detailed explanation.\n\n"
            "3. Title Creation: Generate a descriptive title that encapsulates the overall theme or main outcome of the conversation.-n"
            " - If possible, include the most prominent speaker identities and organization names in the title.\n"
            " - If dicernable, indicate in the title the conversation type. Examples include, Meeting, Voicemail, Phone Call, Interview, Conversation.\n"
            " - The name of the transcript file should further be used to infer context. It may contain dates, topics, or names. It may indicate that it is a voicemail, or just a recording.\n"
            " - Example: Voicemail from Tom Froman at ACS Technical Products to Eric about Polymer Exemption Status.\n"
            " - Example: 1-on-1 converation between Tim and Minday about Dinner Plans.\n"
            " - Example: Meeting between 3M, ADM, and Iowa State University about recycling project issues.\n"
            "Formatting Requirements (use Markdown):\n"
            "- Start with a bolded title line: '**Title: <Your Recommended Title>**'. \n"
            "  Ensure a blank line before and after this line.\n"
            "- Next, include a bolded section header: '**Speaker Mapping:**'. Follow this with a Markdown table with headers 'Name', 'Identity/Role/Affiliation', and 'Contribution'.\n"
            "  Ensure a blank line before and after this section header.\n"
            "- Next, include a bolded section header: '**Outline Summary:**'. Provide key topics with a few bullet points per topic.\n"
            "  Ensure a blank line before and after this section header.\n"
            "- Finally, include a bolded section header: '**Detailed Summary:**'. Elaborate on the outline in depth.\n"
            "  Ensure a blank line before and after this section header.\n\n"
            "Example output format:\n"
            "```"
            "**Title: Collaborative Meeting on Project X**\n"
            "\n"
            "**Speaker Mapping:**\n\n"
            "| Name         | Identity/Role/Affiliation       | Contribution                                                                 |\n"
            "|--------------|---------------------------------|------------------------------------------------------------------------------|\n"
            "| SPEAKER_00   | John Doe, Project Manager at Company A | Introduced the project timeline and led discussions on resource allocation. |\n"
            "| SPEAKER_01   | Jane Smith, Engineer at Company B     | Provided technical insights on implementation challenges and proposed solutions. |\n"
            "\n"
            "**Outline Summary:**\n\n"
            "- Topic 1\n"
            "  - Point 1\n"
            "  - Point 2\n"
            "- Topic 2\n"
            "  - Point 3\n"
            "\n"
            "**Detailed Summary:**\n\n"
            "The meeting focused on Project X, where John Doe outlined the timeline...\n"
            "```\n\n"
            "Ignore all timestamps in the transcript; focus solely on the dialogue content."
        )
        if transcript_filename:
            prompt += f"\n\nTranscript filename: {os.path.basename(transcript_filename)}"
        if speaker_hints:
            prompt += f"\n\nAdditional speaker hints: {speaker_hints}"
        prompt += "\n\nTranscript contents:\n" + transcript_text

    extra_headers = {"x-context-id": context_id} if context_id else {}

    if model == "grok-v2":
        logging.info("Using Grok v2 API via xAI migration settings.")
        grok_client = openai.OpenAI(
            api_key=grok_api_key,
            base_url="https://api.x.ai/v1"
        )
        try:
            response = grok_client.chat.completions.create(
                model="grok-2-latest",
                messages=[{"role": "user", "content": prompt}],
                extra_headers=extra_headers
            )
        except Exception as e:
            logging.error(f"Grok API call failed: {e}")
            exit(1)
    else:
        logging.info(f"Using OpenAI model: {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_headers=extra_headers
            )
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            exit(1)
    result = response.choices[0].message.content.strip()
    logging.info(f"Raw LLM output:\n{result}")
    return result, prompt

def convert_to_docx(markdown_text, output_file):
    """Convert Markdown text to DOCX using Pandoc."""
    try:
        temp_md_file = "temp_summary.md"
        with open(temp_md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        cmd = ['pandoc', '-s', temp_md_file, '-o', output_file, '--wrap=none']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Pandoc conversion failed: {result.stderr}")
            sys.exit(1)
        os.remove(temp_md_file)
        logging.info(f"Converted Markdown to DOCX at {output_file}")
    except FileNotFoundError:
        logging.error("Pandoc is not installed or not found in PATH. Install it via 'pip install pandoc' or system package manager.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during Pandoc conversion: {e}")
        sys.exit(1)

def main():
    logging.info("Entering main() function")
    
    epilog_text = "Available Models and Rates:\n"
    for model, info in AVAILABLE_MODELS.items():
        epilog_text += f"  - {model}: {info['price']} - {info['description']}\n"
    
    parser = argparse.ArgumentParser(
        description=(
            "Generate a detailed summary and deduce speaker roles from a conversation transcript using an LLM API, "
            "or continue a conversation from a context file.\n\n"
            "Usage:\n"
            "  summarize_transcript.py [options] transcript_file\n\n"
            "Options:\n"
            "  --speaker-hints       Provide hints about speaker identities (for new transcripts).\n"
            "  --prompt              Custom prompt for follow-up questions (required if input is a context file).\n"
            "  --model               Specify the LLM model to use (see available models below).\n"
            "  --api-key-file        Path to a JSON file containing API keys for OpenAI and/or Grok v2.\n"
            "  --output-file         File path to store the generated summary (overrides default naming unless --rename-from-context is used).\n"
            "  --output-format       Format the output as 'json', 'markdown', 'html', 'docx', or 'pdf'.\n"
            "  --rename-from-context Use transcript timestamp and LLM title as basename for all output files.\n"
            "  --list-models         List available models and their rates and exit.\n"
        ),
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("transcript_file", nargs='?', help="Path to the transcript or context file")
    parser.add_argument("--speaker-hints", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for follow-up questions when using a context file")
    parser.add_argument("--model", type=str, default="grok-v2", choices=list(AVAILABLE_MODELS.keys()))
    parser.add_argument("--api-key-file", type=str, default="api_keys.json")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--output-format", type=str, choices=["json", "markdown", "html", "docx", "pdf"], default="markdown")
    parser.add_argument("--rename-from-context", action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        sys.exit(0)

    output_extensions = {
        "json": ".json",
        "markdown": ".md",
        "html": ".html",
        "docx": ".docx",
        "pdf": ".pdf"
    }

    transcript_path = args.transcript_file
    if not transcript_path:
        logging.error("No transcript or context file provided.")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(transcript_path):
        logging.error(f"File {transcript_path} does not exist.")
        sys.exit(1)

    is_context_file = transcript_path.endswith("-context.json")
    context_id = None
    if is_context_file:
        if not args.prompt:
            logging.error("When using a context file (ending in '-context.json'), a --prompt is required for follow-up questions.")
            sys.exit(1)
        with open(transcript_path, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
        transcript_text = context_data.get("transcript_file_content", "")
        context_id = context_data.get("context_id")
        if not transcript_text:
            original_transcript = context_data.get("transcript_file")
            if original_transcript and os.path.exists(original_transcript):
                with open(original_transcript, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
            else:
                logging.error("Context file missing transcript content and original transcript not found.")
                sys.exit(1)
        speaker_hints = context_data.get("speaker_hints")
        previous_summary = context_data.get("summary", "")
        custom_prompt = f"Previous summary:\n{previous_summary}\n\nNew question:\n{args.prompt}\n\nTranscript:\n{transcript_text}"
    else:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        context_id = generate_context_id(transcript_text)
        custom_prompt = None

    openai_key, grok_key = load_api_keys(args.api_key_file)
    if args.model == "grok-v2":
        if not grok_key:
            logging.error("Grok model selected but grok_api_key is not provided.")
            sys.exit(1)
        client = None
    else:
        if not openai_key:
            logging.error("Non-Grok model selected but openai_api_key is not provided.")
            sys.exit(1)
        client = openai.OpenAI(api_key=openai_key)

    logging.info("Generating summary and title..." if not is_context_file else "Processing follow-up question...")
    result, prompt_used = generate_summary_and_title(
        client,
        transcript_text,
        transcript_path,
        speaker_hints=args.speaker_hints if not is_context_file else speaker_hints,
        model=args.model,
        grok_api_key=grok_key,
        custom_prompt=custom_prompt,
        context_id=context_id
    )

    output_folder = os.path.dirname(os.path.abspath(transcript_path))
    basename = os.path.splitext(os.path.basename(transcript_path))[0]

    # Define output file paths based on --rename-from-context
    if args.rename_from_context:
        # Extract timestamp from transcript if available
        timestamp = None
        for line in transcript_text.splitlines():
            if line.startswith("Date and Time:"):
                try:
                    timestamp_str = line[len("Date and Time:"):].strip()
                    timestamp = time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    timestamp = time.strftime("%Y%m%d-%H%M%S", timestamp)
                    logging.info(f"Using timestamp from transcript: {timestamp_str}")
                    break
                except ValueError:
                    logging.warning(f"Invalid timestamp format in transcript: {timestamp_str}")
        if not timestamp:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(transcript_path)))
            logging.info(f"No timestamp in transcript; using file modified time: {timestamp}")
        title_line = result.splitlines()[0] if result else ""
        if title_line.startswith("**Title:"):
            title = title_line[len("**Title:"):].rstrip("**").strip()
        elif title_line.startswith("Title:"):
            title = title_line[len("Title:"):].strip()
        else:
            title = "summary"
        title = re.sub(r'[\\/*?:"<>|]', '', title)
        basename = f"{timestamp}-{title}"
        default_output_file = os.path.join(output_folder, f"{basename}{output_extensions[args.output_format]}")
        context_filepath = os.path.join(output_folder, f"{basename}-context.json")
    else:
        default_output_file = args.output_file or os.path.join(output_folder, f"{basename} Summary{output_extensions[args.output_format]}")
        context_filepath = os.path.join(output_folder, f"{basename}-context.json")

    # Check for existing files
    if os.path.exists(default_output_file):
        logging.warning(f"Output file {default_output_file} already exists and will be overwritten.")
    if os.path.exists(context_filepath):
        logging.warning(f"Context file {context_filepath} already exists and will be overwritten.")

    # Write the output files
    formatted_output = None
    if args.output_format == "json":
        output_data = {"summary": result}
        formatted_output = json.dumps(output_data, indent=2)
        with open(default_output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        logging.info(f"Summary saved to {default_output_file}")
    elif args.output_format == "html":
        try:
            import markdown
            from xhtml2pdf import pisa
            html_content = markdown.markdown(result, extensions=['extra', 'tables'])
            formatted_output = f"<html><body>{html_content}</body></html>"
            with open(default_output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            logging.info(f"Summary saved to {default_output_file}")
        except ImportError:
            logging.error("markdown and xhtml2pdf are required for HTML output. Install via 'pip install markdown xhtml2pdf'.")
            sys.exit(1)
    elif args.output_format == "docx":
        convert_to_docx(result, default_output_file)
    elif args.output_format == "pdf":
        try:
            import markdown
            from xhtml2pdf import pisa
            html_content = markdown.markdown(result, extensions=['extra', 'tables'])
            formatted_output = f"<html><body>{html_content}</body></html>"
            with open(default_output_file, "w+b") as pdf_file:
                pisa_status = pisa.CreatePDF(formatted_output, dest=pdf_file)
            if pisa_status.err:
                logging.error("Error occurred while generating PDF")
                sys.exit(1)
            logging.info(f"Summary saved as PDF to {default_output_file}")
        except ImportError:
            logging.error("markdown and xhtml2pdf are required for PDF output. Install via 'pip install markdown xhtml2pdf'.")
            sys.exit(1)
    else:  # markdown
        formatted_output = result
        with open(default_output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        logging.info(f"Summary saved to {default_output_file}")

    # Save context file
    context = {
        "context_id": context_id,
        "prompt": prompt_used,
        "summary": result,
        "transcript_file": transcript_path if not is_context_file else context_data["transcript_file"],
        "transcript_file_content": transcript_text,
        "model": args.model,
        "speaker_hints": args.speaker_hints if not is_context_file else speaker_hints
    }
    with open(context_filepath, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2)
    logging.info(f"Context saved to {context_filepath} with context_id: {context_id}")

if __name__ == "__main__":
    main()