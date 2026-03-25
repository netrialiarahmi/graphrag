import os

app_path = r"C:\Users\MyBook Hype AMD\Documents\GitHub\graphrag\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "    # Display answer" in line:
        # Comment out the next 4 lines
        lines[i] = "    # Display answer (Now handled inside the try block for streaming)\n"
        if i+1 < len(lines) and "if st.session_state.search_answer:" in lines[i+1]:
            lines[i+1] = "    # if st.session_state.search_answer:\n"
            lines[i+2] = "    #     section_divider(\"Jawaban\")\n"
            lines[i+3] = "    #     st.markdown(st.session_state.search_answer)\n"
        break

new_block = """            # ════════════════════════════════════════════════════════════
            # LANGGRAPH AGENTIC ROUTER PIPELINE (WITH NARRATIVE UI & STREAM)
            # ════════════════════════════════════════════════════════════
            from utils.langgraph_agent import create_agent
            from utils.llm_stance import ask_about_documents_stream
            
            with progress_container.status("🤖 **Memutar Strategi Penelusuran Hukum...**", expanded=True) as status:
                agent = create_agent()
                final_state = {"logs": [], "narratives": [], "primary_doc_ids": [], "context_docs": {}, "answer": "", "final_chunks": []}
                seen_narratives = 0
                
                # Execute agent and stream state updates live
                for event in agent.stream({"query": query, "logs": [], "narratives": [], "primary_doc_ids": []}):
                    for node_name, state_update in event.items():
                        final_state.update(state_update)
                        
                        curr_narr = state_update.get("narratives", [])
                        if len(curr_narr) > seen_narratives:
                            for nar in curr_narr[seen_narratives:]:
                                st.markdown(f"💭 {nar}")
                            seen_narratives = len(curr_narr)
                            
                status.update(label="✅ **Penelusuran Selesai**", state="complete", expanded=False)
                
                st.session_state.search_doc_ids = final_state.get("primary_doc_ids", [])
                st.session_state.search_context_docs = final_state.get("context_docs", {})
                st.session_state.search_edges = {"edges": []}
            
            with st.expander("⚙️ System Debug Logs"):
                for log in final_state.get("logs", []):
                    st.code(log, language="bash")
                    
            section_divider("⚖️ Analisis Hukum")
            chunks = final_state.get("final_chunks", [])
            rel_context = final_state.get("relationship_context", "")
            
            # LIVE STREAMING THE ANSWER
            gen = ask_about_documents_stream(query, chunks, rel_context)
            full_ans = st.write_stream(gen)
            
            st.session_state.search_answer = full_ans

"""

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "            # ════════════════════════════════════════════════════════════" in line:
        if start_idx == -1:
            start_idx = i
    if "        except Exception as e:" in line:
        if start_idx != -1:
            end_idx = i
            break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]
    with open(app_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Patch V3 successful.")
else:
    print("Failed to patch.")
