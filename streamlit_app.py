import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# Assuming 'retriever' is correctly imported from your local vector module
from vector import retriever 

# 1. Page Configuration
st.set_page_config(page_title="Genetic Variation RAG Assistant", page_icon="🧬")
st.title("🧬 Genetic Variation RAG Assistant")
st.write("Ask questions about genetic variations based on your retrieved data.")

# 2. Initialize Model and Chain (Cached so it doesn't rebuild on every rerun)
@st.cache_resource
def init_rag_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash", 
        google_api_key=...
    )

    template = """You are a helpful assistant for answering questions about genetic variations.
Use the following context to answer the question.
You must provide the source of the information in your answer, and you must use all the information provided in the context to answer the question.
If the context does not contain enough information to answer the question, say "I don't know".

Context:
{reviewed_information}

Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = init_rag_chain()

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about the genetic variation documentation."}
    ]

# 4. Display Past Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle New User Input
if question := st.chat_input("Enter your question here..."):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching context and generating answer..."):
            try:
                # Retrieve relevant documents
                docs = retriever.invoke(question)
                
                # Format context strings
                reviewed_information = "\n\n".join(
                    [f"[Source {doc.metadata['source'].split('/')[-1].split(".")[0]}] {doc.page_content}" for doc in docs]
                )
                
                # Run the chain
                results = chain.invoke({
                    "reviewed_information": reviewed_information, 
                    "question": question
                })
                
                response_text = results.text
                
                # Render the final response
                st.markdown(response_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")