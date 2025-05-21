# Audio Transcription and Summarization Pipeline

This project provides a suite of Python scripts to transcribe audio files, generate detailed summaries from transcripts, and automate the process with folder watching capabilities. It leverages state-of-the-art AI models for transcription (e.g., Whisper, AssemblyAI) and summarization (e.g., GPT-4o-mini, Grok v2), making it suitable for processing meeting recordings, interviews, voicemails, and other audio conversations.

## Features

- **Transcription**: Convert audio files to text with optional speaker identification using Whisper or AssemblyAI.
- **Summarization**: Generate structured summaries with speaker mappings, outlines, and detailed explanations using LLMs.
- **Automation**: Monitor a folder for new audio files, transcribe them, and summarize them automatically.
- **Flexible Output**: Save transcripts and summaries in multiple formats (text, JSON, Markdown, HTML, DOCX, PDF).
- **Contextual Renaming**: Optionally rename output files using a timestamp from the audio metadata and the LLM-generated title.
- **Speaker Mapping**: Detailed tables identifying speakers, their roles, and contributions in the conversation.

## Project Structure

- **`transcribe_audio.py`**: Transcribes audio files using Whisper or AssemblyAI, with support for speaker diarization.
- **`summarize_transcript.py`**: Summarizes transcripts using an LLM, producing a structured output with a title, speaker mapping, outline, and detailed summary.
- **`auto_transcribe_audio.py`**: Automates the transcription and summarization process by watching a folder for new audio files.

## Prerequisites

- **Python**: 3.8 or higher
- **FFmpeg**: Required for audio processing with `pydub`. Install via:
  - Windows: `choco install ffmpeg` (via Chocolatey) or download from [FFmpeg.org](https://ffmpeg.org/)
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
- **Pandoc**: Required for DOCX output in `summarize_transcript.py`. Install via:
  - Windows: `choco install pandoc` or download from [Pandoc.org](https://pandoc.org/)
  - Linux: `sudo apt-get install pandoc`
  - macOS: `brew install pandoc`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv transcribe_env
   source transcribe_env/bin/activate  # Linux/macOS
   transcribe_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` with:
   ```
   openai-whisper
   assemblyai
   soundfile
   tqdm
   torch
   pyannote-audio
   pydub
   mutagen
   colorlog
   markdown
   xhtml2pdf
   openai
   ```

4. **Configure API Keys**:
   - Create an `api_keys.json` file in the project root:
     ```json
     {
       "openai_api_key": "your-openai-api-key",
       "grok_api_key": "your-xai-grok-api-key"
     }
     ```
   - For `pyannote-audio` (speaker diarization), get a Hugging Face token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept model conditions at:
     - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
     - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Usage

### 1. Transcribe an Audio File
```bash
python transcribe_audio.py audio.mp3 --method whisper --speakers --model small -o output.txt
```
- `--method`: `whisper` or `assemblyai` (default: config-based)
- `--speakers`: Enable speaker identification (Whisper only)
- `--model`: Whisper model (e.g., `base`, `small`, `medium`)
- `-o`: Output file (defaults to `<audio_file>.txt` if not specified)
- `--json`: Output as JSON instead of text

### 2. Summarize a Transcript
```bash
python summarize_transcript.py transcript.txt --model gpt-4o-mini --rename-from-context --output-format docx
```
- `--model`: LLM model (e.g., `gpt-4o-mini`, `grok-v2`)
- `--rename-from-context`: Rename output files using transcript timestamp and LLM title
- `--output-format`: `json`, `markdown`, `html`, `docx`, `pdf` (default: `markdown`)
- `--speaker-hints`: Optional hints for speaker identities

### 3. Automate Transcription and Summarization
```bash
python auto_transcribe_audio.py /path/to/audio_folder --output-folder /path/to/output --move --process-existing
```
- `--output-folder`: Directory for output files
- `--move`: Move audio, transcript, summary, and context files to output folder
- `--process-existing`: Process existing audio files at startup
- Configurable via `auto_transcribe_config.ini` (see below)

## Configuration

### `auto_transcribe_config.ini`
Created automatically if missing:
```ini
[Transcription]
method = whisper
speakers = False
model = base
temp_dir = 
transcript_output_format = txt

[Summarization]
model = gpt-4o-mini
api_key_file = api_keys.json
summary_output_format = docx
speaker_hints = 
rename_from_context = False
```
- Edit to customize transcription and summarization settings.

## Output Format

### Transcript (from `transcribe_audio.py`)
```
Date and Time: 2025-03-03 14:30:00

[0.00s - 5.23s] Speaker SPEAKER_00: Hello, this is a test.
[5.23s - 10.45s] Speaker SPEAKER_01: Great to hear you!
```

### Summary (from `summarize_transcript.py` with `--rename-from-context`)
- Filename: `20250303-143000-Collaborative-Meeting.docx`
- Content:
  ```
  **Title: Collaborative Meeting on Bio-Based Acrylic Acid Production**

  **Speaker Mapping:**
  | Name         | Identity/Role/Affiliation       | Contribution                                                                 |
  |--------------|---------------------------------|------------------------------------------------------------------------------|
  | SPEAKER_00   | Colleen Tahan, Accountant at Sherwin-Williams | Discussed budget constraints and funding options for the project.           |
  | SPEAKER_01   | Christian Charnay, Senior Scientist   | Presented research findings on microbial fermentation and proposed next steps. |

  **Outline Summary:**
  - Topic 1
    - Point 1
    - Point 2
  - Topic 2
    - Point 3

  **Detailed Summary:**
  The meeting focused on developing a bio-based approach to acrylic acid production...
  ```

## Detailed Documentation

For in-depth information on each script, see:
- [transcribe_audio.py](docs/transcribe_audio.md)
- [summarize_transcript.py](docs/summarize_transcript.md)
- [auto_transcribe_audio.py](docs/auto_transcribe_audio.md)

## Troubleshooting

- **FFmpeg Errors**: Ensure FFmpeg is installed and added to your PATH.
- **API Key Issues**: Verify `api_keys.json` contains valid keys for OpenAI or xAI.
- **Speaker Diarization Fails**: Check Hugging Face token and model permissions in `transcription_config.ini`.
- **Pandoc Not Found**: Install Pandoc for DOCX output.
- **Output Files Not Renamed**: Ensure `--rename-from-context` is set and the transcript contains a "Date and Time:" line.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for transcription.
- [AssemblyAI](https://www.assemblyai.com/) for alternative transcription.
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization.
- [xAI](https://x.ai/) for Grok v2 summarization capabilities.
