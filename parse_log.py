import json

with open("output/logs/debug.log", "r", encoding="utf-8") as f:
    for line in f:
        if "4cd9f827-5f55-46e6-9173-284588a384ea" not in line:
            continue
        data = json.loads(line)
        msg = data.get("message", "")
        print("=======", msg, "=======")
        if "payload" in data:
            payload = data["payload"]
            if "retrieval_items" in payload:
                items = payload["retrieval_items"]
                print(f"COUNT retrieval_items: {len(items)}")
            if "chunks" in payload:
                chunks = payload["chunks"]
                print(f"COUNT chunks: {len(chunks)}")
                if len(chunks) > 0:
                    print("First chunks doc IDs:", [(c.get("doc_id"), c.get("content")[:20].strip() + ("..." if len(c.get("content", "")) > 20 else "")) for c in chunks[:10]])
                    for c in chunks:
                        cont = c.get("content", "").lower().strip()
                        if "cukup jelas" in cont or "kosong" in cont:
                            print(f"WARNING: Noisy chunk found! doc_id: {c.get('doc_id')}, content: {cont}")
            if "answer" in payload:
                print("Answer preview:", payload["answer"][:500])
            if "expanded_queries" in payload:
                print("Expanded queries:", len(payload["expanded_queries"]))
            
            if msg == "final_answer prompt input":
                sys_p = payload.get("system_prompt", "")
                usr_p = payload.get("user_prompt", "")
                if "Konteks Dokumen yang Ditemukan:" in usr_p:
                    print("USER PROMPT LENGTH:", len(usr_p))
                    idx = usr_p.find("Konteks Dokumen yang Ditemukan:")
                    print("Context section:", usr_p[idx:idx+200] + "...")
