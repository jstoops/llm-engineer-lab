import gradio as gr
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os

# It's a best practice to load secrets from environment variables.
# os.environ["OPENAI_API_KEY"] = "sk-..." 

# --- 1. Initialize Non-Async Components ---
client = MultiServerMCPClient(
    {
        "MongoDB": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "mongodb-mcp-server",
                "--connectionString",
                # IMPORTANT: Replace this with your actual MongoDB connection string.
                "<conection string>"
            ]
        }
    }
)
model = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# --- 2. Define the Core Asynchronous Logic ---
async def chat_function(message: str, history: list):
    tools = await client.get_tools()
    agent = create_react_agent(model=model, tools=tools)
    response_dict = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    final_response = response_dict['messages'][-1].content
    return final_response

# --- 3. Create and Launch the Gradio UI ---
iface = gr.ChatInterface(
    fn=chat_function,
    title="LangGraph MongoDB Agent",
    description="Ask questions about your MongoDB database using the Model Context Protocol.",
    examples=[
        ["Tell me all mongo collections"], 
        ["List all databases"],
        ["How many documents are in the 'users' collection?"]
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    iface.launch()