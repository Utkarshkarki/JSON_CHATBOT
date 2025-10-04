import os
from mem0 import Memory

# 1. Initialize the mem0 client
# A unique ID for the user to keep their memories separate
USER_ID = "interactive-user-001" 
try:
    # Set a default model that is good for chat and instructions
    # This configuration is optional but can improve performance
    config = {
        "vector_store": {
            "provider": "qdrant",
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o",
            }
        }
    }
    memory = Memory(user_id=USER_ID)
    print("✅ mem0 client initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing mem0 client: {e}")
    print("Please ensure your MEM0_API_KEY environment variable is set.")
    exit()

def run_smart_chatbot():
    """
    Runs an interactive chatbot that combines short-term and long-term memory.
    """
    print("\n🤖 Smart Chatbot is ready!")
    print("---------------------------------------------------------")
    print("You can chat normally with me.")
    print("To save a fact, start your message with 'remember:'.")
    print("For example: 'remember: My favorite city is Paris.'")
    print("Type 'exit' to end the conversation.")
    print("---------------------------------------------------------")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("🤖 Goodbye!")
            break

        try:
            # --- LONG-TERM MEMORY LOGIC ---

            # 2. Check if the user wants to store a memory
            if user_input.lower().startswith("remember:"):
                fact_to_remember = user_input[len("remember:"):].strip()
                if fact_to_remember:
                    memory.add(fact_to_remember, metadata={'source': 'user_command'})
                    print("Bot: OK, I've stored that for you.")
                else:
                    print("Bot: Please provide a fact to remember after 'remember:'.")
                continue # Skip to the next loop iteration

            # 3. If not storing, first search long-term memory for relevant facts
            # We set a high relevance threshold to avoid pulling up incorrect facts.
            search_results = memory.search(query=user_input, limit=1, min_score=0.9)

            if search_results:
                retrieved_memory = search_results[0]['text']
                # Respond using the highly relevant long-term memory
                response = memory.chat(
                    f"Based on what I know ('{retrieved_memory}'), answer this question: {user_input}"
                )
                print(f"Bot: {response}")

            # --- SHORT-TERM MEMORY LOGIC ---
            else:
                # 4. If no relevant long-term memories are found, use conversational chat
                response = memory.chat(user_input)
                print(f"Bot: {response}")

        except Exception as e:
            print(f"❌ An error occurred: {e}")

# --- Main execution ---
if __name__ == "__main__":
    run_smart_chatbot()