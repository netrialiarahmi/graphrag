import logging
import asyncio
from utils.langgraph_agent import create_agent

async def main():
    agent = create_agent()
    final_state = {}
    config = {"configurable": {"thread_id": "test"}}
    start_state = {
        "query": "apa itu definisi bangunan",
        "trace_id": "test_script",
        "logs": [],
        "narratives": [],
        "primary_doc_ids": []
    }
    for event in agent.stream(start_state):
        for k, v in event.items():
            print(f"Node: {k}, keys: {list(v.keys())}")
            final_state.update(v)
            if "final_chunks" in v:
                print("FOUND final_chunks WITH LEN:", len(v["final_chunks"]))
                
    print("FINISHED.")
    print("FINAL CHUNKS in final_state:", len(final_state.get('final_chunks', [])))

if __name__ == "__main__":
    asyncio.run(main())