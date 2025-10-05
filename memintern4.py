import os
from mem0 import Memory

# This class will represent a single chat session for a user.
class ChatbotSession:
    def __init__(self, user_id):
        """
        Initializes the chatbot for a specific user.
        """
        self.user_id = user_id
        try:
            # Initialize mem0 client with the user's unique ID
            self.memory = Memory(user_id=self.user_id)
            print(f"\n✅ Chatbot session started for user {self.user_id}")
        except Exception as e:
            print(f"Error initializing mem0 client {e}")
            self.memory = None

    def start_chat(self):
        """
        Starts an interactive chat loop for the session.
        """
        if not self.memory:
            return

        print("---------------------------------------------------------")
        print("Type 'exit' to end this session.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("🤖 Ending this session. Goodbye")
                break
            
            try:
                # The chat method automatically uses the user's entire history
                response = self.memory.chat(user_input)
                print(f"Bot: {response}")
            except Exception as e:
                print(f"❌ An error occurred: {e}")


# --- Main execution to simulate multiple sessions ---
if __name__ == "__main__":
    # A unique identifier for our user. In a real app, this would
    # come from your user authentication system (e.g., email, database ID).
    USER_ID = "utkarsh-12345"

    # --- SESSION 1: The user logs in for the first time ---
    print("--- 🚀 Simulating First User Session ---")
    # In this session, tell the chatbot your name and a hobby.
    first_session = ChatbotSession(USER_ID)
    first_session.start_chat()

    # --- SESSION 2: The user logs out and comes back later ---
    print("\n\n--- 🔄 Imagine the user logs out and returns the next day ---")
    print("--- Simulating Second User Session ---")
    # Now, ask a question that relies on information from the first session.
    second_session = ChatbotSession(USER_ID)
    second_session.start_chat()