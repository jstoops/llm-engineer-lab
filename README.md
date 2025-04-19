My LLM Engineering Lab
======================

Used to quickly prototype ideas before turning them into applications in python using Anaconda, JupyterLab, and Google Colab.

Inspiration from these recommended courses and books:
- [LLM Engineering: Master AI, Large Language Models & Agents by Ed Donner](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/)
- [AI-Agents: Automation & Business with LangChain & LLM Apps by Arnold Oberleiter](https://www.udemy.com/course/ai-agents-automation-business-with-langchain-llm-apps/)
- [LLM Engineer's Handbook by Paul Iusztin and Maxime Labonne](https://www.packtpub.com/en-us/product/llm-engineers-handbook-9781836200062)
- [The Machine Learning Solutions Architect Handbook](https://www.packtpub.com/en-us/product/the-machine-learning-solutions-architect-handbook-9781805124825)

**Table of content**
- [Data Science Environment Setup](#setup)
- [Patterns](#patterns)
- [Lab Projects](#lab-projects)
- [HugglingFace Library Experiments](#hf-lib-exp)
- [Tools](#tools)
- [Skills Developed](#skills)

<a id="setup"></a>
# Data Science Environment Setup

## Anaconda and JupyterLab

1. Clone the repo
    git clone https://github.com/jstoops/llm-engineer-lab.git
2. Download and install Anaconda: https://www.anaconda.com/download
3. Run Anaconda PowerShell Prompt
4. Nav to project directory and create an environment using a setup file:

    conda env create -f environment.yml
5. Download and install Ollama for open-source LLMs: https://ollama.com/
6. Create a .env file in project root with keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, HF_TOKEN, LLAMA_CLOUD_API_KEY, etc.
7. Create keys/tokens and set to secret key values in .env
    - OpenAI API for GPT4o: https://platform.openai.com/settings/organization/api-keys
    - Google AI for Gemini API: https://ai.google.dev/gemini-api/docs
    - Anthropic for Claude Sonnet: https://console.anthropic.com/settings/keys
    - DeepSeek: https://platform.deepseek.com/api_keys
    - HuggingFace: https://huggingface.co/settings/tokens
    - LlamaCloud: Go to https://cloud.llamaindex.ai/ then API Keys
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

## Google Colab Setup

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click File->New notebook in Drive
3. From top right select downarrow next to Connect->Change runtime type, e.g. Python 3 on a CPU with High CPU RAM toggled off for cheapest option
4. Select Connect then once connected click on RAM & Disk to view Resources
5. Select folder icon on left to open file browser on local disk (temporary and wiped once finished using box)
6. Select key icon from left menu to open secrets for environment variables and toggle on the keys associated with the notebook
7. Click Share to share the notebook on Google Drive

## Clang Setup

1. Download Visual Studio 2022 Community edition: https://visualstudio.microsoft.com/downloads/
2. Run VisualStudioSetup.exe and select Individual components
3. Under _Compilers, build tools, and runtimes_ check `C++ Clang Compiler for Windows`
4. Click install
5. Add the Clang bin folder to your system PATH:
    1. Right-click on 'This PC' or 'My Computer' and select 'Properties'
    2. Click on 'Advanced system settings'
    3. Click on 'Environment Variables'
    4. Under 'System variables', find and edit 'Path'
    5. Add a new entry with the path to your Clang bin folder (e.g., C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\bin)
6. Restart your command prompt, and within Jupyter Lab do Kernel -> Restart kernel, to pick up the changes
5. Open a new command prompt and run this to make sure it's installed OK

    clang --version

<a id="patterns"></a>
# Patterns

## FTI Design Pattern

The feature/training/inference (FTI) architecture is the ML system pattern used as the core architecure in the LLM pipeline design. The FTI pipelines act as logical layers and this high-level architecture is language-, framework-, platform-, and inftrastructure agnostic.

The FTI pattern is followed to compute the features, train the model, and make predictions using 3 or more pipelines that each have a clearly defined scope and interface.

The data and feature pipelines scales horizontally based on CPU and RAM load, the training pipeline scales vertically by adding more GPUs, and the inference pipeline scales horizontally based on the number of client requests.

<img src="./content/FTI-Pipelines-Architecture.jpg" alt="Feature/training/inference (FTI) architecture" />

<a id="lab-projects"></a>
# Lab Projects

- [Website Summarizer](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/site-summary-require-js.ipynb): Give it a URL, and it will respond with a summary.
- [AI-Powered Marketing Brochures](projects/brochure-multi-prompt-spanish.ipynb): a product that builds a Brochure for a company to be used for prospective clients, investors and potential recruits when provided a company name and their primary website.
- [AIs Having a Chat](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/4-way-AI-conversation.ipynb): an adversarial conversation between Chatbots.
- [Store Chatbot](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/chatbot.ipynb): a conversational AI with multi-shot prompting.
- [Airline AI Assistant](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/airline-ai-assistant.ipynb): an AI Customer Support assistant for an Airline.
- [Meeting Minutes Program](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/meeting-minutes.ipynb): generate meeting minutes from an audio recording of a meeting on your Google Drive.
- [Expert Knowledge Worker](https://github.com/jstoops/llm-engineer-lab/blob/main/projects/rag-knowledge-worker.ipynb): embeds documents in a vector datastore and uses RAG (Retrieval Augmented Generation) to ensure question/answering assistant is highly accuracy.

<a id="hf-lib-exp"></a>
# HugglingFace Library Experiments

- [Pipelines](https://github.com/jstoops/llm-engineer-lab/blob/main/hf-libs/pipelines.ipynb): exploring the HuggingFace High Level API.
- [Tokenizers](https://github.com/jstoops/llm-engineer-lab/blob/main/hf-libs/tokenizers.ipynb): using different Tokenizers.
- [Models](https://github.com/jstoops/llm-engineer-lab/blob/main/hf-libs/models.ipynb): exploring the heart of the transformers library.

<a id="tools"></a>
# Tools

- [Tech Question AI Assistant](https://github.com/jstoops/llm-engineer-lab/blob/main/tools/tech-questions.ipynb): takes a technical question, and responds with an explanation. Optimized for LLM and pythin code expertise.
- [Data Preparation for Training RAG Agent](https://github.com/jstoops/llm-engineer-lab/blob/main/tools/llama-parse.ipynb): uses LlamaParse to convert PDFs and other document types to markdown.
- [Improve Code Performance](https://github.com/jstoops/llm-engineer-lab/blob/main/tools/code-converter.ipynb): uses Frontier and open-source models to generate high performance C++ code from Python code.
- [Get GPU Info](https://github.com/jstoops/llm-engineer-lab/blob/main/tools/gpu-info.ipynb): code to display information about the GPUs that are currently running on a Notebook in Colab.
- [Image Generator](https://github.com/jstoops/llm-engineer-lab/blob/main/tools/image-generator.ipynb): uses Dall-E 2 or 3 to generate an image based on a user prompt.

<a id="skills"></a>
# Skills Developed

- Confidently use the OpenAI & Ollama API including streaming with markdown and JSON generation
- Use the API for OpenAPI's GPT, Anthropic's Claude and Google's Gemini
- Constrast and contrast the leading Frontier LLMs
- Write code that interacts between multiple Frontier LLMs
- Describe transformers, tokens, context windows, API costs, etc
- Confidently code with APIs for Frontier Models GPT, Claude and Gemini

## Multi-Model AI Chatbot Assistant Development

Build multi-modal AI Chatbot Assistants with UI, Tools, and Agents for enhanced expertise:
- Implement customer support assistants with Chat UIs
- Create data science UIs in Gradio
- Provide context in a prompt including multi-shot prompting
- Use Agents to carry out sequential activities
- Create Function-calling Tools
- Implement multi-modal AI Assistants with Agents and Tools including an interactive UI

## HuggingFace Libraries

Navigate the HuggingPlace platform, run code in Colab and use HuggingFace pipelines, tokenizers and models:
- Find Models, Datasets and Spaces on the HuggingFace platform
- Use Google Colab to code on a high spec GPU runtime
- Use HuggingFace pipelines for a wide variety of inference tasks
- Use pipelines to generate text, images and audio
- Create tokenizers for models
- Translate between text and tokens
- Understand special tokens and chat templates
- Work with HuggingFace lower level APIs
- Use HuggingFace models to generate text
- Compare the results across 5 open source models
- Confidently work with tokenizers and models
- Run inference on open-source models
- Implement an LLM solution combining Frontier and Open-source models
- Build solutions with open-source LLMs with HuggingFace Transformers

## Comparing Open and Closed Source Models

Compare LLMs to identify the right one for the task at hand:
- Navigate the most useful leaderboards and arenas to evaluate LLMs
- Compare LLMs based on their basic attributes and benchmarks
- Give real-world use cases of LLMs solving commercial problems
- Confidently choose the right LLM for projects, backed by metrics

## Leveraging Frontier Models for High-Performance Code Generation in C++

Build a product that converts Python code to C++ for performance:
- Assess Frontier and Open-Source models for coding ability
- Use Frontier and open-source models to generate code
- Implement solutions that use Frontier and Open-source LLMs to generate code
- Use HuggingFace inference endpoints to deploy models on AWS, Azure, and GCP

## Evaluating LLM Code Generation Performance

Evaluating LLM performance by looking at Model-Centric vs Business-Centric metrics:
- Compare performance of open-source and closed source models
- Describe different commercial use cases for code generation
- Build solutions that use code generation for diverse tasks

## Retrieval Augmented Generation (RAG)

- Explain the idea behind RAG
- Walk through the high level flow for adding expertise to queries
- Implement a version of RAG without vector databases
- Explain how RAG uses vector embeddings and vector datastores to add context to prompts, define LangChain and read / split Documents
- Describe the LangChain framework, with benefits and limitations
- Create and populate a Vector Database with the contents of a Knowledge Base
- Use LangChain to read in a Knowledge Base of documents
- Use LangChain to divide up documents into overlaping chunks
- Convert chunks of text into Vectors using OpenAIEmbeddings
- Store the Vectors in Chroma, a popular open-source Vector datastore
- Visualize and explore Vectors in a Chroma Vector Datastore in 2D and 3D
- Create a Conversation Chain in LangChain for a chat conversation with retrieval
- Ask questions and receive answers demonstrating expert knowledge
- Build a Knowledge Worker assistant with chat UI
- Create a RAG Knowledge Worker using LangChain and Chroma
- Familar with LangChain's declarative language LCEL
- Understand how LangChain works behind the scenes
- Debug and fix common issues with RAG
