from rag_pipeline.ingest import extract_text, chunk_text
from rag_pipeline.embed_store import embed_chunks, store_chunks
from rag_pipeline.retrieve import retrieve
from rag_pipeline.generate import generate_answer

def main():
    print("=== Finsight RAG ===")
    pdf = input("Enter PDF path: ")

    print("\n[1] Extracting text...")
    text = extract_text(pdf)

    print("[2] Chunking...")
    chunks = chunk_text(text)

    print("[3] Embedding + Storing...")
    vectors = embed_chunks(chunks)
    store_chunks(chunks, vectors)

    print("[✓] Ingestion complete.")
    print("\nAsk questions about the report (type 'exit' to quit, 'clear' to reset conversation)\n")

    conversation_history = []
    
    while True:
        q = input("> ")
        if q.lower() == "exit":
            break
        if q.lower() == "clear":
            conversation_history = []
            print("[✓] Conversation history cleared.\n")
            continue

        # Build conversation context string for better retrieval
        conv_context = ""
        if conversation_history:
            # Include recent questions to help retrieval understand follow-ups
            recent_questions = " ".join([prev_q for prev_q, _ in conversation_history[-2:]])
            conv_context = recent_questions
        
        context = retrieve(q, conversation_context=conv_context)
        answer = generate_answer(q, context, conversation_history)
        
        # Add to conversation history
        conversation_history.append((q, answer))
        
        # Keep only last 5 exchanges to avoid prompt bloat
        if len(conversation_history) > 5:
            conversation_history = conversation_history[-5:]

        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")

if __name__ == "__main__":
    main()
