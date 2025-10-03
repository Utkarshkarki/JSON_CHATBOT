import json
from typing import Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()


# --- 0. Setup LLM ---
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-05-01-preview",
    temperature=1
)


# --- 1. Load Workflow Data ---
with open("kuralynx-workflow.json", "r") as f:
    workflow_data = json.load(f)["workflow_json"]

nodes_map = {node["id"]: node for node in workflow_data["nodes"]}


# Global state to track conversation flow
conversation_state = {
    "current_node_id": None,
    "customer_name": "Maria",
    "conversation_history": [],
    "chat_messages": [],
    "session_id": None,
    "session_start_time": None,
    "workflow_ended": False,  # NEW: Explicit end flag
}


# --- CONVERSATION HISTORY STORAGE FUNCTIONS ---

def save_conversation_to_file(filename: str = None):
    """Save the complete conversation history to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_history_{conversation_state['customer_name']}_{timestamp}.json"
    
    history_data = {
        "session_id": conversation_state["session_id"],
        "customer_name": conversation_state["customer_name"],
        "session_start_time": conversation_state["session_start_time"],
        "session_end_time": datetime.now().isoformat(),
        "current_node_id": conversation_state["current_node_id"],
        "workflow_steps": conversation_state["conversation_history"],
        "chat_messages": [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            }
            for msg in conversation_state["chat_messages"]
        ],
        "total_messages": len(conversation_state["chat_messages"]),
        "total_workflow_steps": len(conversation_state["conversation_history"])
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Conversation history saved to: {filename}")
    return filename


def add_chat_message(role: str, content: str):
    """Add a message to the chat history with timestamp."""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    conversation_state["chat_messages"].append(message)


# --- 2. Create Workflow Tools ---

@tool
def get_current_node_info(node_id: str = None) -> Dict[str, Any]:
    """Get information about the current or specified workflow node."""
    if not node_id:
        node_id = conversation_state["current_node_id"]

    if node_id not in nodes_map:
        return {"error": "Invalid node ID"}

    node_info = nodes_map[node_id]
    return {
        "id": node_id,
        "name": node_info["data"]["name"],
        "instruction": node_info["data"]["instruction"],
        "type": node_info["type"],
        "edges": node_info["data"].get("edges", []),
    }


@tool
def execute_node(node_id: str, customer_name: str = "Maria") -> str:
    """Execute a specific workflow node and return its instruction."""
    if node_id not in nodes_map:
        return f"Error: Node {node_id} not found"

    node_info = nodes_map[node_id]
    instruction = node_info["data"]["instruction"].replace(
        "{{customer.name}}", customer_name
    )

    # Update global state
    conversation_state["current_node_id"] = node_id
    conversation_state["conversation_history"].append(
        {
            "node_id": node_id,
            "instruction": instruction,
            "node_name": node_info["data"]["name"],
            "timestamp": datetime.now().isoformat()
        }
    )
    
    add_chat_message("assistant", instruction)
    
    # Check if this is an end node
    if node_info["type"] == "end":
        conversation_state["workflow_ended"] = True
        return f"{instruction}\n\n[WORKFLOW_END]"
    
    return instruction


@tool
def get_user_response(prompt: str) -> str:
    """Display a message to the user and get their response."""
    print(f"\n🤖 Anna: {prompt}")
    try:
        user_input = input("👤 You: ")
    except (EOFError, KeyboardInterrupt):
        user_input = "exit"
        conversation_state["workflow_ended"] = True
    
    add_chat_message("user", user_input)
    return user_input


@tool
def route_to_next_node(user_response: str, current_node_id: str = None) -> str:
    """Determine the next node based on user response and current workflow state."""
    if not current_node_id:
        current_node_id = conversation_state["current_node_id"]

    if current_node_id not in nodes_map:
        conversation_state["workflow_ended"] = True
        return "END"

    current_node_info = nodes_map[current_node_id]
    edges = current_node_info["data"].get("edges", [])

    # Check if current node is an end node or has no edges
    if not edges or current_node_info["type"] == "end":
        conversation_state["workflow_ended"] = True
        return "END"

    # Build routing prompt for LLM
    routing_prompt = f"""You are an intelligent router for a healthcare chatbot workflow.

Current Node: {current_node_info['data']['name']}
User Response: "{user_response}"

Based on the user's response, choose the most appropriate next step from these options:

Available Routes:
"""
    for edge in edges:
        routing_prompt += f"- {edge['destination_node_id']}: {edge['condition']}\n"

    routing_prompt += "\nRespond with ONLY the destination node ID (example: eligibility-1) and nothing else."

    # Get LLM decision
    response = llm.invoke(routing_prompt)
    next_node_id = (
        getattr(response, "content", str(response))
        .strip()
        .replace("`", "")
        .replace("'", "")
        .replace('"', "")
    )

    # Validate the decision
    valid_destinations = [edge["destination_node_id"] for edge in edges]
    if next_node_id not in valid_destinations:
        print(f"⚠️ Invalid routing decision: {next_node_id}. Using first available route.")
        next_node_id = valid_destinations[0] if valid_destinations else "END"
        
    if next_node_id == "END":
        conversation_state["workflow_ended"] = True
        return "END"

    print(f"🧠 Router decided: {current_node_id} → {next_node_id}")
    
    conversation_state["conversation_history"].append({
        "action": "routing",
        "from_node": current_node_id,
        "to_node": next_node_id,
        "user_response": user_response,
        "timestamp": datetime.now().isoformat()
    })
    
    return next_node_id


@tool
def start_workflow(customer_name: str = "Maria") -> str:
    """Initialize the healthcare workflow conversation."""
    start_node = next(
        (node for node in workflow_data["nodes"] if node["type"] == "conversation-start"),
        None,
    )
    if not start_node:
        return "Error: No start node found in workflow"

    # Initialize session
    conversation_state["customer_name"] = customer_name
    conversation_state["current_node_id"] = start_node["id"]
    conversation_state["conversation_history"] = []
    conversation_state["chat_messages"] = []
    conversation_state["workflow_ended"] = False
    conversation_state["session_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    conversation_state["session_start_time"] = datetime.now().isoformat()
    
    add_chat_message("system", f"Healthcare workflow started for {customer_name}")

    return f"Workflow initialized. Start node ID: {start_node['id']}"


@tool
def check_if_workflow_ended() -> Dict[str, Any]:
    """Check if the workflow has reached its end state."""
    current_node_id = conversation_state["current_node_id"]
    
    # Check multiple end conditions
    is_end_node = False
    if current_node_id and current_node_id in nodes_map:
        node_info = nodes_map[current_node_id]
        is_end_node = node_info["type"] == "end"
        has_no_edges = not node_info["data"].get("edges", [])
    else:
        has_no_edges = True
    
    ended = conversation_state["workflow_ended"] or is_end_node or has_no_edges
    
    return {
        "workflow_ended": ended,
        "current_node": current_node_id,
        "is_end_node": is_end_node,
        "steps_completed": len(conversation_state["conversation_history"]),
    }


# --- 3. Create Agent with Tools ---

tools = [
    get_current_node_info,
    execute_node,
    get_user_response,
    route_to_next_node,
    start_workflow,
    check_if_workflow_ended,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Anna, a healthcare workflow assistant guiding users through a structured conversation.

WORKFLOW PROCESS:
1. Use start_workflow to initialize (if not already started)
2. Use execute_node to deliver the current node's message
3. Use get_user_response to get user input
4. Use route_to_next_node to determine the next step
5. IMPORTANT: Use check_if_workflow_ended after routing to verify if workflow should continue
6. If workflow has ended, stop immediately

CRITICAL RULES:
- Execute ONE workflow step at a time
- After routing, ALWAYS check if workflow has ended
- When check_if_workflow_ended returns workflow_ended=True, STOP and say "Workflow complete"
- Never continue after reaching an end node
- Be conversational and helpful

Current customer: {customer_name}
""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,  # Reduced from 50
    early_stopping_method="generate",  # Changed from default
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)


# --- 4. Simplified Workflow Controller ---

class HealthcareWorkflowController:
    """Controller to manage the agent + tools conversation loop."""

    def __init__(self, agent_executor: AgentExecutor, customer_name: str = "Maria"):
        self.agent_executor = agent_executor
        self.customer_name = customer_name
        self.conversation_file = None
        self.max_steps = 50  # Safety limit

    def run_workflow(self):
        print("--- LangChain Healthcare Workflow Started ---")
        print(f"Customer: {self.customer_name}")
        print("-" * 50)

        try:
            # Initialize workflow
            print("\n🚀 Initializing workflow...")
            init_result = self.agent_executor.invoke(
                {
                    "input": f"Start the healthcare workflow for {self.customer_name}",
                    "customer_name": self.customer_name,
                }
            )

            step_count = 0
            
            # Main workflow loop with explicit exit conditions
            while step_count < self.max_steps:
                step_count += 1
                
                # Check if workflow should end
                if conversation_state["workflow_ended"]:
                    print("\n✅ Workflow ended by state flag")
                    break
                
                # Check current node status
                current_node = conversation_state.get("current_node_id")
                if current_node and current_node in nodes_map:
                    node_type = nodes_map[current_node]["type"]
                    if node_type == "end":
                        print(f"\n✅ Reached end node: {current_node}")
                        break
                
                # Execute one workflow step
                print(f"\n--- Step {step_count} ---")
                result = self.agent_executor.invoke(
                    {
                        "input": "Execute the next workflow step: get current node, execute it, get user response, route to next node, then check if workflow ended",
                        "customer_name": self.customer_name,
                    }
                )
                
                # Check result for end signals
                output = str(result.get("output", "")).upper()
                if any(keyword in output for keyword in ["WORKFLOW COMPLETE", "WORKFLOW END", "CONVERSATION ENDED"]):
                    print("\n✅ Workflow completed (detected in output)")
                    break
                
                # Safety check for no meaningful output
                if not result.get("output"):
                    print("\n⚠️ No output from agent, ending workflow")
                    break

        except KeyboardInterrupt:
            print("\n\n⚠️ Workflow interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in workflow: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "-" * 50)
            print("--- Healthcare Workflow Session Ended ---")
            
            # Save conversation
            self.conversation_file = save_conversation_to_file()
            self.print_summary()

    def print_summary(self):
        print(f"\n📊 Workflow Summary for {self.customer_name}:")
        print(f"Session ID: {conversation_state['session_id']}")
        print(f"Total messages: {len(conversation_state['chat_messages'])}")
        print(f"Workflow steps: {len(conversation_state['conversation_history'])}")
        
        print("\n📝 Workflow Steps:")
        for i, step in enumerate(conversation_state["conversation_history"], 1):
            if "node_name" in step:
                print(f"  {i}. {step['node_name']} (ID: {step['node_id']})")
            elif "action" in step:
                print(f"  {i}. Routing: {step['from_node']} → {step['to_node']}")
        
        if self.conversation_file:
            print(f"\n✅ Full history saved to: {self.conversation_file}")


# --- 5. Run ---

if __name__ == "__main__":
    controller = HealthcareWorkflowController(agent_executor, "Maria")
    controller.run_workflow()
