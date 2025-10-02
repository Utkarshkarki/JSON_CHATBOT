import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# --- 0. Setup ---
from dotenv import load_dotenv
load_dotenv()  

llm = ChatOpenAI(model="gpt-4o-mini", temperature=20)

# --- 1. Load Workflow Data ---
with open("kuralynx-workflow.json", "r") as f:
    workflow_data = json.load(f)["workflow_json"]

nodes_map = {node["id"]: node for node in workflow_data["nodes"]}

# Global state to track conversation flow
conversation_state = {
    "current_node_id": None,
    "customer_name": "Maria",
    "conversation_history": [],
}

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
        }
    )
    return instruction


@tool
def get_user_response(prompt: str) -> str:
    """Display a message to the user and get their response."""
    print(f"\n🤖 Anna: {prompt}")
    try:
        user_input = input("👤 You: ")
    except EOFError:
        user_input = ""
    return user_input


@tool
def route_to_next_node(user_response: str, current_node_id: str = None) -> str:
    """Determine the next node based on user response and current workflow state."""
    if not current_node_id:
        current_node_id = conversation_state["current_node_id"]

    if current_node_id not in nodes_map:
        return "END"

    current_node_info = nodes_map[current_node_id]
    edges = current_node_info["data"].get("edges", [])

    if not edges or current_node_info["type"] == "end":
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

    # Get LLM decision (ChatOpenAI.invoke can take a plain string)
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
        print(f"⚠️ Invalid routing decision: {next_node_id}. Ending conversation.")
        return "END"

    print(f"🧠 Router decided: {current_node_id} → {next_node_id}")
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

    conversation_state["customer_name"] = customer_name
    conversation_state["current_node_id"] = start_node["id"]
    conversation_state["conversation_history"] = []

    return f"Healthcare workflow started for {customer_name}. Beginning conversation..."


@tool
def check_conversation_status() -> Dict[str, Any]:
    """Get the current status of the conversation workflow."""
    return {
        "current_node": conversation_state["current_node_id"],
        "customer_name": conversation_state["customer_name"],
        "steps_completed": len(conversation_state["conversation_history"]),
        "is_active": conversation_state["current_node_id"] is not None,
    }

# --- 3. Create Agent with Tools ---

tools = [
    get_current_node_info,
    execute_node,
    get_user_response,
    route_to_next_node,
    start_workflow,
    check_conversation_status,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Anna, a healthcare workflow assistant. Your job is to guide users through a structured conversation workflow using the available tools.

WORKFLOW PROCESS:
1. Start with start_workflow tool to initialize
2. Use execute_node to get the current step's message
3. Use get_user_response to interact with the user
4. Use route_to_next_node to determine where to go next
5. Repeat steps 2-4 until workflow ends

IMPORTANT GUIDELINES:
- Always execute nodes in sequence according to the workflow
- Get real user input for each step using get_user_response
- Use the routing logic to determine next steps
- Be conversational and helpful
- Keep the workflow moving forward
- End gracefully when the workflow completes

Current customer: {customer_name}
""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create agent and executor
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=50,
    return_intermediate_steps=True,
)

# --- 4. Agent-Driven Workflow Controller ---

class HealthcareWorkflowController:
    """Controller to manage the agent + tools conversation loop."""

    def __init__(self, agent_executor: AgentExecutor, customer_name: str = "Maria"):
        self.agent_executor = agent_executor
        self.customer_name = customer_name
        self.workflow_active = True

    def run_workflow(self):
        print("--- LangChain Tool-Calling Healthcare Workflow Started ---")
        print(f"Customer: {self.customer_name}")
        print("-" * 50)

        try:
            # Initialize the workflow
            _ = self.agent_executor.invoke(
                {
                    "input": f"Start the healthcare workflow for customer {self.customer_name}",
                    "customer_name": self.customer_name,
                }
            )

            # Continue the workflow loop
            while self.workflow_active:
                status_result = self.agent_executor.invoke(
                    {
                        "input": "Check the current conversation status and continue the workflow if active",
                        "customer_name": self.customer_name,
                    }
                )

                output_text = str(status_result.get("output", "")).upper()
                if (
                    "END" in output_text
                    or "CONVERSATION ENDED" in output_text
                    or "WORKFLOW COMPLETE" in output_text
                ):
                    self.workflow_active = False
                    break

                continue_result = self.agent_executor.invoke(
                    {
                        "input": "Execute the next step in the workflow: get current node, execute it, get user response, and route to next node",
                        "customer_name": self.customer_name,
                    }
                )

                # Safety check - if no meaningful progress, break
                if not continue_result.get("output"):
                    break

        except Exception as e:
            print(f"Error in workflow: {e}")
        finally:
            print("\n" + "-" * 50)
            print("--- Healthcare Workflow Session Ended ---")
            self.print_summary()

    def print_summary(self):
        print(f"\nWorkflow Summary for {self.customer_name}:")
        print(f"Steps completed: {len(conversation_state['conversation_history'])}")
        for i, step in enumerate(conversation_state["conversation_history"], 1):
            print(f"{i}. {step['node_name']} (ID: {step['node_id']})")

# --- 5. Run ---

if __name__ == "__main__":
    controller = HealthcareWorkflowController(agent_executor, "Maria")
    controller.run_workflow()
