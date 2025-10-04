import os
from mem0 import Memory
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Configure mem0 with Azure OpenAI
config = {
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": AZURE_OPENAI_DEPLOYMENT,
            "temperature": 0.1,
            "max_tokens": 2000,
            "azure_kwargs": {
                "azure_deployment": AZURE_OPENAI_DEPLOYMENT,
                "api_version": AZURE_OPENAI_API_VERSION,
                "azure_endpoint": AZURE_OPENAI_ENDPOINT,
                "api_key": AZURE_OPENAI_API_KEY,
            },
        },
    },
    "embedder": {
        "provider": "azure_openai",
        "config": {
            "model": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            "embedding_dims": 1536,
            "azure_kwargs": {
                "api_version": AZURE_OPENAI_API_VERSION,
                "azure_deployment": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                "azure_endpoint": AZURE_OPENAI_ENDPOINT,
                "api_key": AZURE_OPENAI_API_KEY,
            },
        },
    },
    "version": "v1.1",
}

# Initialize memory with Azure OpenAI config
try:
    memory = Memory.from_config(config)
    print("✅ mem0 client initialized successfully with Azure OpenAI.")
except Exception as e:
    print(f"❌ Error initializing mem0 client: {e}")
    exit()

def run_chatbot():
    """
    Starts a conversational loop with memory context.
    """
    user_id = "user-1234"
    print("\n🤖 Chatbot is ready! (Type 'exit' to end)")
    print("---------------------------------------------------------")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("🤖 Goodbye!")
            break
        
        try:
            # Search for relevant memories
            relevant_memories = memory.search(user_input, user_id=user_id)
            
            # Build context from memories
            context = ""
            if relevant_memories and 'results' in relevant_memories:
                context = "\n".join([mem.get('memory', '') for mem in relevant_memories['results'][:3]])
                print(f"[Found {len(relevant_memories['results'])} relevant memories]")
            
            # Here you'd integrate with Azure OpenAI for chat response
            # For now, store the interaction
            messages = [{"role": "user", "content": user_input}]
            memory.add(messages, user_id=user_id)
            
            print(f"Bot: Memory stored. Context: {context[:100]}..." if context else "Bot: Memory stored.")
            
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    run_chatbot()
