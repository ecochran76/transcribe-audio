# summarize_transcript.py

## Overview

`summarize_transcript.py` is a Python script that generates structured summaries from conversation transcripts using a large language model (LLM). It processes text transcripts to produce a Markdown-formatted output containing a title, a speaker mapping table, an outline summary, and a detailed summary. The script supports multiple LLM backends (e.g., OpenAI’s GPT models, xAI’s Grok v2) and can output summaries in various formats, including Markdown, JSON, HTML, DOCX, and PDF. It also offers an optional feature to rename output files based on a timestamp and the generated title.

This script is ideal for summarizing meeting notes, interviews, voicemails, or any recorded conversation where speaker roles and key points need to be extracted and organized.

## Usage

Basic usage:
```bash
python summarize_transcript.py transcript.txt --model gpt-4o-mini --output-format markdown
```

With renaming and custom output:
```bash
python summarize_transcript.py transcript.txt --model grok-v2 --rename-from-context --output-format docx --api-key-file custom_api_keys.json
```

For follow-up questions with a context file:
```bash
python summarize_transcript.py previous-context.json --prompt "What were the key decisions made?" --model gpt-4o-mini
```

## Arguments

| Argument              | Type    | Description                                                                                   | Default          | Required |
|-----------------------|---------|-----------------------------------------------------------------------------------------------|------------------|----------|
| `transcript_file`     | str     | Path to the transcript file or context JSON to summarize.                                     | None             | Yes      |
| `--speaker-hints`     | str     | Optional hints about speaker identities (e.g., "John is SPEAKER_00, a manager").             | None             | No       |
| `--prompt`            | str     | Custom prompt for follow-up questions (required if input is a context file).                  | None             | No/Yes*  |
| `--model`             | str     | LLM model to use (e.g., `gpt-4o-mini`, `grok-v2`). See available models below.                | `grok-v2`        | No       |
| `--api-key-file`      | str     | Path to JSON file with API keys (`openai_api_key`, `grok_api_key`).                           | `api_keys.json`  | No       |
| `--list-models`       | flag    | List available models and their rates, then exit.                                             | False            | No       |
| `--output-file`       | str     | Custom path for the summary output file (overrides default unless `--rename-from-context`).   | None             | No       |
| `--output-format`     | str     | Output format: `json`, `markdown`, `html`, `docx`, `pdf`.                                     | `markdown`       | No       |
| `--rename-from-context`| flag    | Rename output files using a timestamp from the transcript and the LLM-generated title.        | False            | No       |

*Required if `transcript_file` is a context JSON file (ends with `-context.json`).

### Available Models
Run `python summarize_transcript.py --list-models` to see the full list. Examples:
- `gpt-4o-mini`: Cost-effective OpenAI model ($0.15 input, $0.60 output per 1M tokens).
- `grok-v2`: xAI’s Grok v2, migrated via OpenAI SDK (pricing via xAI settings).

## Configuration

### API Keys
The script requires API keys for the selected LLM, stored in a JSON file (default: `api_keys.json`):
```json
{
  "openai_api_key": "sk-...",
  "grok_api_key": "xai-..."
}
```
- At least one key is required, depending on the `--model` chosen.
- Use `--api-key-file` to specify a custom path.

## Output Format

### Markdown (Default)
When `--output-format` is `markdown` or unspecified:
```
**Title: Collaborative Meeting on Bio-Based Acrylic Acid Production**

**Speaker Mapping:**
| Name         | Identity/Role/Affiliation       | Contribution                                                                 |
|--------------|---------------------------------|------------------------------------------------------------------------------|
| SPEAKER_00   | Colleen Tahan, Accountant at Sherwin-Williams | Discussed budget constraints and funding options for the project.           |
| SPEAKER_01   | Christian Charnay, Senior Scientist   | Presented research findings on microbial fermentation and proposed next steps. |

**Outline Summary:**
- Funding Discussion
  - Budget constraints identified
  - Funding options proposed
- Technical Planning
  - Fermentation research shared

**Detailed Summary:**
The meeting, held on March 3, 2025, focused on developing a bio-based approach to acrylic acid production...
```

### Other Formats
- **JSON**: `{"summary": "<markdown_content>"}`
- **HTML**: Converted from Markdown using `markdown` library.
- **DOCX**: Converted using Pandoc.
- **PDF**: Converted from HTML using `xhtml2pdf`.

### File Naming with `--rename-from-context`
- Uses the "Date and Time:" line from the transcript (e.g., `2025-03-03 14:30:00` becomes `20250303-143000`).
- Appends the LLM-generated title (e.g., `Collaborative-Meeting-on-Bio-Based-Acrylic-Acid-Production`).
- Example: `20250303-143000-Collaborative-Meeting-on-Bio-Based-Acrylic-Acid-Production.docx`
- Falls back to transcript file modified time if no timestamp is found.

## Dependencies

- **Required**: `openai` (for OpenAI models), `colorlog` (optional, for colored logs).
- **Optional (by output format)**:
  - `markdown`: For HTML and PDF conversion.
  - `xhtml2pdf`: For PDF output.
  - Pandoc: For DOCX output (install separately, see [Pandoc.org](https://pandoc.org/)).
- Install via:
  ```bash
  pip install openai colorlog markdown xhtml2pdf
  ```

## Workflow

1. **Input**: Provide a transcript file generated by `transcribe_audio.py` or a previous context JSON file.
2. **Processing**:
   - The script reads the transcript and optional speaker hints.
   - It constructs a prompt for the LLM, requesting a title, speaker mapping table, outline, and detailed summary.
   - The LLM processes the transcript and returns a Markdown-formatted response.
3. **Output**:
   - Saves the summary in the specified format.
   - Generates a context JSON file (e.g., `<basename>-context.json`) for follow-up questions.

## Key Features

- **Speaker Mapping Table**: Identifies speakers (e.g., `SPEAKER_00`), deduces their roles/affiliations, and describes their contributions in a three-column Markdown table.
- **Flexible Title Generation**: Incorporates prominent names, organizations, and transcript filename context (e.g., dates, topics).
- **Timestamp Integration**: Extracts "Date and Time:" from the transcript for renaming, ensuring consistency with audio metadata.
- **Multi-Model Support**: Compatible with OpenAI and xAI models via a unified API interface.

## Troubleshooting

- **No Output Files**: Ensure the transcript file exists and is readable. Check logs for errors (e.g., API key issues).
- **Invalid Timestamp**: Verify the transcript starts with a "Date and Time:" line in `YYYY-MM-DD HH:MM:SS` format. If missing, it uses the file’s modified time.
- **Title Not Renamed**: Confirm the LLM output starts with `**Title:` or `Title:`. Check the `Raw LLM output` log.
- **API Errors**: Validate `api_keys.json` contains the correct key for the chosen model. Use `--api-key-file` if custom.
- **DOCX/PDF Fails**: Install Pandoc (`pandoc`) for DOCX and `markdown`/`xhtml2pdf` for PDF (`pip install markdown xhtml2pdf`).

## Example Workflow

1. Transcribe an audio file:
   ```bash
   python transcribe_audio.py meeting.mp3 --method whisper --speakers -o meeting.txt
   ```
   Output: `meeting.txt` with "Date and Time: 2025-03-03 14:30:00".

2. Summarize the transcript:
   ```bash
   python summarize_transcript.py meeting.txt --model gpt-4o-mini --rename-from-context --output-format docx
   ```
   Output: `20250303-143000-Collaborative-Meeting.docx` and `20250303-143000-Collaborative-Meeting-context.json`.

3. Ask a follow-up question:
   ```bash
   python summarize_transcript.py 20250303-143000-Collaborative-Meeting-context.json --prompt "What decisions were made?" --model gpt-4o-mini
   ```

## Notes

- **Performance**: Larger models (e.g., `gpt-4o`) may provide better summaries but increase processing time and cost.
- **Customization**: Modify the prompt in `generate_summary_and_title` for different summary structures or styles.
- **Logging**: Detailed logs are output to the console (colored with `colorlog`) for debugging.

## See Also

- [transcribe_audio.py](transcribe_audio.md)
- [auto_transcribe_audio.py](auto_transcribe_audio.md)
