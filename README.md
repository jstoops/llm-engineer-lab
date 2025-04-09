My LLM Engineering Lab
======================

Used to quickly prototype ideas before turning them into applications in python using Anaconda and JupyterLab.

# Data Science Environment Setup

1. Clone the repo
    git clone https://github.com/jstoops/llm-engineer-lab.git
2. Download and install Anaconda: https://www.anaconda.com/download
3. Run Anaconda PowerShell Prompt
4. Nav to project directory and create an environment using a setup file:

    conda env create -f environment.yml
5. Download and install Ollama for open-source LLMs: https://ollama.com/
6. Create a .env file in project root with key OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY and DEEPSEEK_API_KEY
7. Create closed-source LLM keys and set to secret key values in .env
    - OpenAI API for GPT4o: https://platform.openai.com/settings/organization/api-keys
    - Google AI for Gemini API: https://ai.google.dev/gemini-api/docs
    - Anthropic for Claude Sonnet: https://console.anthropic.com/settings/keys
    - DeepSeek: https://platform.deepseek.com/api_keys
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

# Lab Projects

- [Website Summarizer](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/site-summary-require-js.ipynb)
- [AI-Powered Marketing Brochures](projects/brochure-multi-prompt-spanish.ipynb)
- [Tech Question AI Assistant](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/tech-questions.ipynb)
