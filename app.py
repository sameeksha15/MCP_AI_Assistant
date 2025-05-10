from dotenv import load_dotenv
from langchain_groq import ChatGroq

from mcp_use import MCPAgent, MCPClient
import os

async def run_memory_chat():
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
    config_file = "browser_mcp.json"
    print("Initializing MCP client...")
    
    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model_name="qwen-qwq-32b")
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,
    )
    
    print("Starting memory chat...")
    print("Type 'exit' to quit.")
    print("Type 'clear' to clear memory.")
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break
            elif user_input.lower() == "clear":
                agent.clear_memory()
                print("Memory cleared.")
                continue
            print("Assistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"An error occurred: {e}")
    finally:
        if client and client.sessions:
            await client.close_all_sessions()
