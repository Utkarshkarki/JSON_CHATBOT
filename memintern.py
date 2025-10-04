import os
from mem0 import Memory

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize memory (no user_id here)
memory = Memory()

def run_chatbot():
    user_id = "user-1234"
    conversation_history = []
    
    print("\n Chatbot is ready! (Type 'exit' to end)")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print(" Goodbye!")
            break
        
        # Search for relevant memories
        relevant_memories = memory.search(user_input, user_id=user_id)
        
        # Build context from memories
        context = "\n".join([mem['memory'] for mem in relevant_memories])
        
        # Send to your LLM (e.g., OpenAI) with context
        # response = your_llm_call(user_input, context, conversation_history)
        
        # Store the conversation in memory
        messages = [
            {"role": "user", "content": user_input}
            # {"role": "assistant", "content": response}
        ]
        memory.add(messages, user_id=user_id)
        
if __name__ == "__main__":
    run_chatbot()
