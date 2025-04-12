My LLM Engineering Lab
======================

Used to quickly prototype ideas before turning them into applications in python using Anaconda and JupyterLab.

# Data Science Environment Setup

## Anaconda and JupyterLab

1. Clone the repo
    git clone https://github.com/jstoops/llm-engineer-lab.git
2. Download and install Anaconda: https://www.anaconda.com/download
3. Run Anaconda PowerShell Prompt
4. Nav to project directory and create an environment using a setup file:

    conda env create -f environment.yml
5. Download and install Ollama for open-source LLMs: https://ollama.com/
6. Create a .env file in project root with key OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, HF_TOKEN, etc.
7. Create closed-source LLM keys and set to secret key values in .env
    - OpenAI API for GPT4o: https://platform.openai.com/settings/organization/api-keys
    - Google AI for Gemini API: https://ai.google.dev/gemini-api/docs
    - Anthropic for Claude Sonnet: https://console.anthropic.com/settings/keys
    - DeepSeek: https://platform.deepseek.com/api_keys
    - HuggingFace: https://huggingface.co/settings/tokens
8. Activate environment:

    conda activate llms-eng
9. Verify correct python version is being used, e.g. 3.11.11:

    python --version
10. Open Jupyter environment:

    jupyter lab

Start environment after inital setup:
1. Run Anaconda PowerShell Prompt
2. Nav to project directory
3. Activate environment:

    conda activate llms-eng
4. Open Jupyter environment:

    jupyter lab

## Audio Setup

1. Download FFmpeg from the official website: https://ffmpeg.org/download.html
2. Extract the downloaded files to a location on your computer (e.g., C:\ffmpeg)
3. Add the FFmpeg bin folder to your system PATH:
    1. Right-click on 'This PC' or 'My Computer' and select 'Properties'
    2. Click on 'Advanced system settings'
    3. Click on 'Environment Variables'
    4. Under 'System variables', find and edit 'Path'
    5. Add a new entry with the path to your FFmpeg bin folder (e.g., C:\ffmpeg\bin)
4. Restart your command prompt, and within Jupyter Lab do Kernel -> Restart kernel, to pick up the changes
5. Open a new command prompt and run this to make sure it's installed OK ffmpeg -version

Check all required packages installed in JupyterLab:

    !ffmpeg -version
    !ffprobe -version
    !ffplay -version

## Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click File->New notebook in Drive
3. From top right select downarrow next to Connect->Change runtime type, e.g. Python 3 on a CPU with High CPU RAM toggled off for cheapest option
4. Select Connect then once connected click on RAM & Disk to view Resources
5. Select folder icon on left to open file browser on local disk (temporary and wiped once finished using box)
6. Select key icon from left menu to open secrets for environment variables and toggle on the keys associated with the notebook
7. Click Share to share the notebook on Google Drive

# Lab Projects

- [Website Summarizer](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/site-summary-require-js.ipynb)
- [AI-Powered Marketing Brochures](projects/brochure-multi-prompt-spanish.ipynb)
- [Tech Question AI Assistant](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/tech-questions.ipynb)
- [4 AIs Having a Conversation](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/4-way-AI-conversation.ipynb)
- [Airline AI Assistant](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/airline-ai-assistant.ipynb)
- [Meeting Minutes From Audio File](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/meeting-minutes.ipynb)

# HugglingFace Library Experiments

- [Pipelines](https://github.com/jstoops/llm-engineer-lab/blob/main/hf-libs/pipelines.ipynb)
- [Tokenizers](https://github.com/jstoops/llm-engineer-lab/blob/main/hf-libs/tokenizers.ipynb)
- [Models](https://github.com/jstoops/llm-engineer-lab/blob/main/hf-libs/models.ipynb)

