__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from crewai_tools import SerperDevTool,ScrapeWebsiteTool 
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, LLM
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI = st.secrets["GEMINI"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)

crew_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GEMINI,
    max_tokens=500,
    temperature=0.7
)

def check_local_knowledge(query,context):
    # Simple LLM-based agentic decision
    decision_prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer. 
Output Format:
    - Answer: Yes/No
Study the below examples and based on that, respond to the last question. 
Examples:
    Input: 
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes
    Input: 
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = decision_prompt.format(text=context, query=query)
    print("Decision Prompt: ",formatted_prompt)
    response = llm.invoke(formatted_prompt)
    print("Output: ",response)
    return response.content.strip().lower() == "yes"

def setup_web_scraping_agent():
    """Setup the web scraping agent and related components"""
    search_tool = SerperDevTool()  # Tool for performing web searches
    scrape_website = ScrapeWebsiteTool()  # Tool for extracting data from websites
    
    # Define the web search agent
    web_search_agent = Agent(
        role="Expert Web Search Agent",
        goal="Identify and retrieve relevant web data for user queries",
        backstory="An expert in identifying valuable web sources for the user's needs",
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )
    
    # Define the web scraping agent
    web_scraper_agent = Agent(
        role="Expert Web Scraper Agent",
        goal="Extract and analyze content from specific web pages identified by the search agent",
        backstory="A highly skilled web scraper, capable of analyzing and summarizing website content accurately",
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )
    
    # Define the web search task
    search_task = Task(
    description=(
        "Search for multiple web pages or articles that are highly relevant to the topic: '{topic}'. "
        "Return a list of the top 3 to 5 web pages that contain valuable, factual, and recent information. "
        "Each result must include the title and a direct link to the source."
    ),
    expected_output=(
        "A list of dictionaries. Each dictionary must contain two fields:\n"
        "- 'title': The title of the web page\n"
        "- 'url': A direct link to the source\n\n"
        "Return at least 3 results."
    ),
    tools=[search_tool],
    agent=web_search_agent,
)

    # Define the web scraping task
    scraping_task = Task(
        description=(
             "Extract and analyze data from the given web page or website related to the topic: '{topic}'. "
        "Use all available tools to retrieve the content. Focus on the most relevant sections that provide insights. "
        "\n\nIMPORTANT: Return your result as a Python dictionary with exactly two keys:\n"
        "'source': the full URL of the page you scraped, and\n"
        "'content': a plain-text summary of the key findings, max 1500 characters.\n\n"
        "This format is required for proper source citation in the final answer."
        ),
        expected_output=(
        "A dictionary in the form:\n"
        "{'source': '<url>', 'content': '<concise summary>'}\n\n"
        "The summary must capture the most relevant findings related to the topic, "
        "and the 'source' must be the original web page URL."
        ),
        tools=[scrape_website],
        agent=web_scraper_agent,
    )
    
    # Define the crew to manage agents and tasks
    crew = Crew(
        agents=[web_search_agent, web_scraper_agent],
        tasks=[search_task, scraping_task],
        verbose=1,
        memory=False,
    )
    return crew

def get_web_content(query):
    """Get content from web scraping"""
    crew = setup_web_scraping_agent()
    result = crew.kickoff(inputs={"topic": query})
    if isinstance(result.raw, dict):
        url = result.raw.get ("source") or result.raw.get("url") or "Unknown URL"
        content = result.raw.get("content", "")
        return f"[Source: {url}]\n{content}"
    else:
        return str(result.raw)

def setup_vector_db(uploaded_file):
    """Setup vector database from PDF"""
    # Load and chunk PDF
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load_and_split()

    for i, page in enumerate(pages):
        page.metadata["source"] = uploaded_file
        page.metadata["page"] = i + 1  # Human-readable page
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)
    
    # Create vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    return vector_db

def get_local_content(vector_db, query):
    """Get content from vector database"""
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    content_with_citations = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        content_with_citations += f"[{i+1}] (Source: {source}, Page: {page})\n{doc.page_content}\n\n"
    
    return content_with_citations


def generate_final_answer(context, query):
    """Generate final answer using LLM"""
    messages = [
        (
            "system",
            '''You are an useful assistant. Answer the user's question accurately based on the provided context.
            The {context} need to include citation index such as [1], [2], etc. Try to use every indexed content to answer the question. 
            Make a citation using index after each content. After that, make a reference listing that contains the source, page number or URL and index of the content.''',
        ),
        ("system", f"Context: {context}"),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content


def process_query(query, vector_db, local_context):
    """Main function to process user query"""
    print(f"Processing query: {query}")
    
    # Step 1: Check if we can answer from local knowledge
    can_answer_locally = check_local_knowledge(query, local_context)
    print(f"Can answer locally: {can_answer_locally}")
    
    # Step 2: Get context either from local DB or web
    if can_answer_locally:
        context = get_local_content(vector_db, query)
        decision = "Can answer locally. Retrieved context from local documents"
    else:
        context = get_web_content(query)
        decision = "Cannot answer locally. Retrieved context from web scraping"
    
    # Step 3: Generate final answer
    answer = generate_final_answer(context, query)
    return decision, answer


def main():
    # Setup
    st.title("Agentic RAG Demo")
    st.write("This is a demo of the Agentic RAG system.")
    st.write("Ask a question as query and check the answer.")
    st.write("Example: What is Agentic RAG?")
    query = st.text_input("Enter your query:")
    upload_file = st.file_uploader("Upload your knowledge base in PDF file:", type=["pdf"])

    if upload_file is not None and query.strip():
        with open("uploaded_file.pdf", "wb") as f:
            f.write(upload_file.read())
        pdf_path = "uploaded_file.pdf"  # Path to your PDF file
        vector_db = setup_vector_db(pdf_path)
    # Process query
        if st.button("Submit"):
            # Get initial context from PDF for routing
            local_context = get_local_content(vector_db,query)
            # Example usage
            decision, result = process_query(query, vector_db, local_context)
            st.write(f"Decision from Agent: {decision}")
            st.write("\n")
            st.write("Answer:")
            st.write(result)
    else:
        if not upload_file:
            st.warning("Please upload a PDF file to create the knowledge base.")
        if not query.strip():
            st.warning("Please enter a query.")

      
    
if __name__ == "__main__":
    main()
