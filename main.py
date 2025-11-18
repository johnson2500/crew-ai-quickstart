from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger

client = LlamaStackClient(
  base_url="http://localhost:8321"
)

def agent_1():
    agent = Agent(
        client,
        model="ollama/llama3.2:3b",
        instructions="You are a helpful assistant that can use tools to answer questions.",
        tools=[{"type": "web_search"}],
    )
    session_id = agent.create_session(session_name="My conversation")

    turn_response = agent.create_turn(
        session_id=session_id,
        messages=[{"role": "user", "content": "Tell me about Llama models"}],
    )
    AgentEventLogger().log(turn_response)

def register_vector_store():
    current_vector_stores = client.vector_stores.list()
    print(current_vector_stores)
    if not current_vector_stores or len(current_vector_stores) == 0:
        client.vectorstores.register(
            vectorstore_id="my_new_vector_store",
            provider_id="sqlite_vec",
            args={"path": "my_vector_store.db"},
        )
        return client.vectorstores.list()[0].id
    else:
        return current_vector_stores[0].id

def agent_2():

    # vector_store_id = register_vector_store()

    agent = Agent(
        client,
        # Check with `llama-stack-client models list`
        model="ollama/llama3.2:3b",
        instructions="You are a helpful assistant",
        # Enable RAG using file_search tool
        # Note: vector_store_ids should match your actual vector store IDs
        tools=[
            {
                "type": "file_search",
                # If you do not have a vector store yet, you can create one using the Llama Stack API or CLI,
                # then reference its ID here. For now, use a placeholder and update it later:
                "vector_store_ids": ["vs_89e1bfb9-4c21-4e86-a04c-acfbf2cf7d2d"],  # Create this vector store and set its actual ID
            },
            # Code interpreter is not a built-in tool type in the responses API
            # You may need to register it as a function tool or use toolgroups
        ],
    )
    session_id = agent.create_session(session_name="My 2nd conversation")

    turn_response = agent.create_turn(
        session_id=session_id,
        messages=[{"role": "user", "content": "Tell me about Llama models"}],
    )
    accumulator = []
    for log in AgentEventLogger().log(turn_response):
        accumulator.append(log)

    print("".join(accumulator))


def main():
    agent_2()

if __name__ == "__main__":
    main()