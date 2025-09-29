import json
import os
from typing import TypedDict
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

# --- 0. Setup ---




# Initialize the LLM
# This model will be our "brain" for making decisions.
llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0)


# --- 1. Define the State ---
# We add `user_response` to store the user's latest message.
class AgentState(TypedDict):
    current_node_id: str
    customer_name: str
    user_response: str # New field to hold the user's natural language input

# --- 2. Load Workflow ---
with open('kuralynx-workflow.json', 'r') as f:
    workflow_data = json.load(f)['workflow_json']
nodes_map = {node['id']: node for node in workflow_data['nodes']}


# --- 3. Create Node Functions ---
# These functions now get real user input.
def create_node_action(node_id: str):
    """Factory to create functions for each node in our graph."""
    def node_action(state: AgentState) -> AgentState:
        node_info = nodes_map[node_id]
        instruction = node_info['data']['instruction'].replace(
            "{{customer.name}}", state["customer_name"]
        ) 

        print("-" * 20)
        print(f"Executing Node: {node_info['data']['name']}")
        print(f"🤖 Anna: {instruction}")

        # If it's an end node or has no edges, the conversation is over.
        if node_info['type'] == 'end' or not node_info['data'].get('edges'):
            return {"user_response": "Conversation ended."}
        
        # Get REAL user input
        user_input = input("👤 You: ")
        return {
            "current_node_id": node_id,
            "user_response": user_input
        }

    return node_action


# --- 4. Define the Intelligent Router ---
# This is the new, LLM-powered router.
def intelligent_router(state: AgentState):
    """
    Uses an LLM to decide the next node based on the user's response.
    """
    print("🧠 LLM Router is thinking...")
    current_node_info = nodes_map[state['current_node_id']]
    edges = current_node_info['data'].get('edges', [])
    
    if not edges:
        return END

    # Build a prompt for the LLM to make a choice
    prompt = f"""You are an intelligent router for a healthcare chatbot.
    The user has just said: "{state['user_response']}"

    Based on their response, which of the following paths should the conversation take?
    Please choose the best option from the list below.

    Options:
    """
    for edge in edges:
        prompt += f"- `{edge['destination_node_id']}`: {edge['condition']}\n"
    
    prompt += "\nRespond with ONLY the destination node ID (e.g., 'eligibility-1') and nothing else."

    # Ask the LLM to make a decision
    response = llm.invoke(prompt)
    next_node_id = response.content.strip().replace("`", "")

    # Basic validation
    if next_node_id not in [edge['destination_node_id'] for edge in edges]:
        print("⚠️ LLM returned an invalid node ID. Defaulting to end.")
        return END
        
    print(f"🧠 Router decided path: {next_node_id}")
    return next_node_id


# --- 5. Build the Graph ---
workflow = StateGraph(AgentState)

start_node_id = next(node['id'] for node in workflow_data['nodes'] if node['type'] == 'conversation-start')

# Add all nodes to the graph
for node_id, node_data in nodes_map.items():
    workflow.add_node(node_id, create_node_action(node_id))
    # Add a conditional edge from each node (that's not an end node) to the router
    if node_data['type'] != 'end':
        destination_map = {edge['destination_node_id']: edge['destination_node_id'] for edge in node_data['data'].get('edges', [])}
        destination_map[END] = END # Allow router to end conversation
        workflow.add_conditional_edges(node_id, intelligent_router, destination_map)

workflow.set_entry_point(start_node_id)
app = workflow.compile()


# --- 6. Run the Chatbot ---
print("--- LangGraph Chatbot Session Started (with Natural Language) ---")
initial_state = {
    "customer_name": "Maria",
    "user_response": "",
    "current_node_id": start_node_id # Start at the beginning
}

# The `stream` method executes the graph. The input is passed to the entrypoint node.
for event in app.stream(initial_state, {"recursion_limit": 25}):
    pass # The printing happens inside the nodes and router now

print("\n--- LangGraph Chatbot Session Ended ---")