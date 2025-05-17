# Agentic RAG Implementation
LLM used: llama-3.3-70b-versatile via Groq API and gemini/gemini-1.5-flash for CrewAI Agent<br>
Embeddings: sentence-transformers/all-mpnet-base-v2 <br>
Vector DB: FAISS <br>
Important Libraries: HuggingFace, LangChain, CrewAI <br>
Demo Deployment: https://agentic-rag5471.streamlit.app/

# Flow
![Agentic RAG](agentic_rag.png)
1. Retrieval (Optional)<br>
Just like Traditional RAG, documents are chunked and embedded into a Vector DB.
A query is embedded and compared to these chunks for similarity.

2. Decision Making <br>
LLM determines if the local context (vector DB) is sufficient to answer the query.<br>
If YES → use local context from the vector DB.<br>
If NO → trigger CrewAI Agent to perform web search via Serper and web scraping to extract content. <br>
This is the key upgrade from traditional RAG — it allows adaptive behavior based on the query type.

3. Augmentation <br>
If local retrieval is used: query + retrieved chunks → augmented prompt.<br>
If web info is used: query + web-extracted content → augmented prompt.

4. Generation<br>
The final LLM receives the full augmented input (from local or external source) and return grounded response.

# Difference between traditional RAG and Agentic RAG
<br>

**Traditional RAG flow**
![RAG Flow](rag.png)
1. Flexibility <br>
   Traditional RAG always retrieves from Vector DB but Agentic RAG retrieves from Vector DB only when needed.
2. Extra Searches <br>
   Traditional RAG searches in local content unless setup multi source RAG. Agentic RAG searches from web based on LLM decision. Therefore, Agentic RAG provides dynamic knowledge base.
3. Query Handling <br>
   Agentic RAG excels in handling complex queries by coordinating multiple sub-tasks through agents. This orchestration allows for thorough exploration of topics, resulting in well-rounded answers.


