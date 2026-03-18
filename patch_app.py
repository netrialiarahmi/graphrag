import os

app_path = r"C:\Users\MyBook Hype AMD\Documents\GitHub\graphrag\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# The target block to replace is inside the `if query and search_btn:` -> `try:`
# We know the try block starts at 786.
# 787 is the comment # ════════════════════════════...
# 1136 is the line just before except Exception as e:

new_block = """            # ════════════════════════════════════════════════════════════
            # LANGGRAPH AGENTIC ROUTER PIPELINE
            # ════════════════════════════════════════════════════════════
            from utils.langgraph_agent import create_agent
            
            with progress_container.status("🤖 **Memproses pertanyaan dengan Agent...**", expanded=True) as status:
                agent = create_agent()
                
                # Execute agent
                final_state = agent.invoke({"query": query, "logs": [], "primary_doc_ids": []})
                
                # Show logs in the expander
                for log in final_state.get("logs", []):
                    st.markdown(log)
                    
                status.update(label="✅ **Jawaban selesai dirumuskan**", state="complete", expanded=False)
                
                # Update session state for UI compatibility
                st.session_state.search_doc_ids = final_state.get("primary_doc_ids", [])
                st.session_state.search_context_docs = final_state.get("context_docs", {})
                st.session_state.search_answer = final_state.get("answer", "")
                st.session_state.search_edges = {"edges": []}
"""

new_lines = lines[:786] + [new_block] + lines[1136:]

with open(app_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Patching app.py successful.")
