import os
from mem0 import Memory

# 1. Initialize the mem0 client
# We'll use a specific user_id to associate memories with this user.
USER_ID = "user-alex-789"
try:
    memory = Memory(user_id=USER_ID)
    print("✅ mem0 client initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing mem0 client: {e}")
    exit()


def store_user_profile():
    """
    Stores specific facts about a user as long-term memories.
    """
    print(f"\n🧠 Storing long-term memories for user: {USER_ID}")
    
    # List of facts to store
    user_facts = [
        "My name is Alex.",
        "I live in London.",
        "I enjoy hiking and playing the guitar.",
        "My favorite food is pizza."
    ]

    # Use memory.add() to store each fact individually
    for fact in user_facts:
        memory.add(fact, metadata={'source': 'user_profile'})
        print(f"   📝 Added: '{fact}'")


def answer_questions_from_memory():
    """
    Asks questions and uses memory.search() to find the answers.
    """
    print(f"\n🔍 Retrieving memories for user: {USER_ID}")
    
    questions = [
        "What is my name?",
        "Where do I live?",
        "What are my hobbies?",
        "Do I like pasta?" # This tests for information that isn't stored
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        
        # Use memory.search() to find relevant stored information
        # It performs a semantic search, not just a keyword search.
        search_results = memory.search(query=question, limit=1)
        
        if search_results:
            # The most relevant memory is typically the first result
            best_match = search_results[0]['text']
            print(f"   ✅ Answer from memory: {best_match}")
        else:
            print("   ❌ Answer from memory: I don't have that information.")


# --- Main execution ---
if __name__ == "__main__":
    # Step 1: Store the user's profile information
    store_user_profile()

    print("\n---------------------------------------------------------")

    # Step 2: Ask questions to retrieve the stored information
    answer_questions_from_memory()