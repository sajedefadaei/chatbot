from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

api_key = ...
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

template = """You are a helpful assistant for answering questions about genetic variations.
Use the following context to answer the question.
You must provide the source of the information in your answer, and you must use all the information provided in the context to answer the question.
If the context does not contain enough information to answer the question, say "I don't know".

Context:
{reviewed_information}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n ------------------------------------------------ \n\n")
    question = input("Enter your question (or 'quit' to exit): ")
    print("\n\n ------------------------------------------------ \n\n")
    
    if question.lower() == "quit":
        break

    docs = retriever.invoke(question)

    reviewed_information = "\n\n".join([f"[Source {doc.metadata}] {doc.page_content}" for doc in docs])
    results = chain.invoke({"reviewed_information": reviewed_information, "question": question})
    print(results.text)
