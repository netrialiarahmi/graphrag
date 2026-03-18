import os

app_path = r"C:\Users\MyBook Hype AMD\Documents\GitHub\graphrag\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_block = """            # ════════════════════════════════════════════════════════════
            # LANGGRAPH AGENTIC ROUTER PIPELINE (WITH NARRATIVE UI)
            # ════════════════════════════════════════════════════════════
            from utils.langgraph_agent import create_agent
            
            with progress_container.status("🤖 **Memutar Strategi Penelusuran Hukum...**", expanded=True) as status:
                agent = create_agent()
                final_state = {"logs": [], "narratives": [], "primary_doc_ids": [], "context_docs": {}, "answer": ""}
                seen_narratives = 0
                
                # Execute agent and stream state updates live
                for event in agent.stream({"query": query, "logs": [], "narratives": [], "primary_doc_ids": []}):
                    for node_name, state_update in event.items():
                        # Track state dynamically
                        final_state.update(state_update)
                        
                        # Print legal 'thoughts' live to user if there are new ones
                        curr_narr = state_update.get("narratives", [])
                        if len(curr_narr) > seen_narratives:
                            for nar in curr_narr[seen_narratives:]:
                                st.markdown(f"💭 *{nar}*")
                            seen_narratives = len(curr_narr)
                            
                status.update(label="✅ **Analisis Hukum Selesai**", state="complete", expanded=False)
                
                # Update session state for UI compatibility
                st.session_state.search_doc_ids = final_state.get("primary_doc_ids", [])
                st.session_state.search_context_docs = final_state.get("context_docs", {})
                st.session_state.search_answer = final_state.get("answer", "")
                st.session_state.search_edges = {"edges": []}
            
            # Show the raw debug logs beneath the query window!
            with st.expander("⚙️ System Debug Logs"):
                for log in final_state.get("logs", []):
                    st.code(log, language="bash")
"""

# Find the start and end of the block to replace
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "            # ════════════════════════════════════════════════════════════" in line:
        if start_idx == -1:
            start_idx = i
    if "        except Exception as e:" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]
    with open(app_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Patch V2 successful.")
else:
    print("Failed to find replacement indices.")
