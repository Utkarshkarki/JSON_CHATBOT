import json

def run_simple_chatbot(workflow_file: str, customer_name: str = "John Doe"):
    """
    Runs a simple, terminal-based chatbot from a JSON workflow file.
    """
    # 1. Load and parse the workflow
    with open(workflow_file, 'r') as f:
        workflow = json.load(f)['workflow_json']

    # Create a mapping of node IDs to their data for easy access
    nodes_map = {node['id']: node for node in workflow['nodes']}

    # 2. Find the starting point
    current_node_id = None
    for node in workflow['nodes']:
        if node['type'] == 'conversation-start':
            current_node_id = node['id']
            break

    if not current_node_id:
        print("Error: Could not find a 'conversation-start' node.")
        return

    print("--- Chatbot Session Started ---")

    # 3. Main conversation loop
    while current_node_id:
        node = nodes_map[current_node_id]['data']
        node_type = nodes_map[current_node_id]['type']

        print(f"\n[NODE: {node['name']}]")

        # Format the instruction with the customer's name
        instruction = node['instruction'].replace("{{customer.name}}", customer_name)
        print(f"🤖 Anna: {instruction}")

        # Check if this is the end of the conversation
        if node_type == 'end' or not node.get('edges'):
            break

        # 4. Get user's choice to determine the next step
        print("\nPaths you can take:")
        edges = node['edges']
        for i, edge in enumerate(edges):
            print(f"  {i + 1}: {edge['condition']}")

        choice = -1
        while choice < 1 or choice > len(edges):
            try:
                user_input = input(f"Enter your choice (1-{len(edges)}): ")
                choice = int(user_input)
                if not (1 <= choice <= len(edges)):
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # 5. Move to the next node based on the choice
        current_node_id = edges[choice - 1]['destination_node_id']

    print("\n--- Chatbot Session Ended ---")


if __name__ == "__main__":
    # Make sure 'kuralynx-workflow.json' is in the same directory
    run_simple_chatbot('kuralynx-workflow.json', customer_name="Utkarsh")