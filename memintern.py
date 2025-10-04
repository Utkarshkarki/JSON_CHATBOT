import os
from mem0 import Memory

# 1. Initialize the mem0 client
# The client automatically uses the MEM0_API_KEY from your environment variables.
# You can optionally specify a user ID to manage separate memories for different users.
try:
    memory = Memory(user_id="user-1234")
    print("✅ mem0 client initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing mem0 client: {e}")
    print("Please ensure your MEM0_API_KEY is set correctly.")
    exit()

# 2. Define the main chatbot function
def run_chatbot():
    """
    Starts a conversational loop with the user.
    """
    print("\n🤖 Chatbot is ready! (Type 'exit' to end the conversation)")
    print("---------------------------------------------------------")

    while True:
        # Get user input from the console
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("🤖 Goodbye!")
            break

        # 3. Use mem0 to get a response
        # The `chat` method sends the user's message and gets a response.
        # mem0 automatically handles the conversation history (short-term memory).
        try:
            response = memory.chat(user_input)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"❌ An error occurred while communicating with mem0: {e}")


# 4. Run the chatbot
if __name__ == "__main__":
    run_chatbot()




    #now we will write code for longterm memory
    