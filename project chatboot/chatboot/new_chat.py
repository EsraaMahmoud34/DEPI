import os
import traceback
# We swap HuggingFace Endpoint for Google's LangChain integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Configuration ---
# !!! PASTE YOUR NEW GOOGLE API KEY HERE !!!
# Do not use the one you leaked online.
GOOGLE_API_KEY = "AIzaSyCF-epOd0cnEUNaOR0VkidIJZAls3K-sjw"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# We keep the same Index and Embedding model so we can read your existing files
INDEX_NAME = "faiss_index_136k"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Using the model you requested
MODEL_NAME = "gemini-2.0-flash"

print(f"--- [Chatbot]: Initializing with {MODEL_NAME}... ---")

# --- 2. Load Embeddings ---
# NOTE: We MUST use HuggingFace embeddings here because your FAISS index
# was created with them. If we switch to Google embeddings, the index won't work.
try:
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
except Exception as e:
    print("\nCRITICAL ERROR: Could not load embeddings")
    print(f"Error Message: {e}")
    exit()

# --- 3. Load the Vector Store ---
try:
    vector_store = FAISS.load_local(
        INDEX_NAME,
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"\nError loading index: {e}")
    print("Make sure the folder 'faiss_index_136k' exists in this directory.")
    exit()

# --- 4. Setup the LLM (Google Gemini) ---
try:
    # We use ChatGoogleGenerativeAI which fits perfectly into LangChain
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.6,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True # Helpful for some Gemini versions
    )
    print("✅ Successfully connected to Google Gemini.")

except Exception as e:
    print(f"\n❌ ERROR: Could not connect to Google Gemini")
    print(f"Error: {e}")
    exit()

# --- 5. Setup the Chat Chain ---
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
chat_history = []

# Gemini works great with this structure

# --- ENHANCED PROMPT ---
# We give the AI specific instructions on how to handle the data.
system_instruction = """
You are an advanced Industrial AI Maintenance Assistant.

Your role:
- Explain machine behavior, sensor readings, malfunction indicators, and product defect results in a simple, friendly, human-like way.
- You understand natural human language (not only data language).

Context Awareness:
1. You have access to a database of machine sensor records (CSV data) and feature definitions (PDF data). The PDF contains human-friendly meanings of features like motor readings, pressures, temperatures, vibrations, pumps, and more.
2. When the user mentions a specific machine ID or product, extract exact numerical values from the context.

Prediction Interpretation:
If the user provides the output of the machine-failure model (e.g., failure_probability, prediction_result):
- Interpret the prediction in simple language.
- Explain what the probability means.
- Suggest practical actions and next maintenance steps.
- Warn about risks only if supported by the context.

Image Defect Model:
If the user provides image-model results (e.g., defect type, defect location):
- Explain the defect in plain English.
- Suggest what to do, how dangerous it is, and whether production should stop.

Explain Features:
If the user asks “What is HDF, TWF, RNF…?”
- Use the PDF context definitions.
- Explain in human language, not technical jargon.

Conversation Style:
- Friendly, helpful, calm tone.
- Use bullet points for clarity.
- Highlight important numbers using **bold**.
- Keep answers short and concise.
- Never guess numbers not found in the context. If missing, say:
  “I cannot find that specific information in my current database.”

User Assistance:
You can:
- Analyze machine readings the user manually enters.
- Comment on predictions from the UI.
- Give maintenance advice.
- Recommend preventive actions.
- Compare readings with typical ranges (only if in context).
- Provide step-by-step instructions.

Limitations:
If information is missing or unclear, explicitly say so.

Context:
{context}
"""




prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", "Question: {question}"),
])



def answer_question(question):
    global chat_history
    
    print("   [Debug] Searching for documents...")
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    # LangChain fills in the variables...
    chain = prompt | llm
    
    print(f"   [Debug] Sending to {MODEL_NAME}...")
    
    try:
        # Invoke Gemini
        response_message = chain.invoke({"context": context, "question": question})
        response_content = response_message.content
        
    except Exception as e:
        print(f"\n❌ ERROR during Gemini generation: {e}")
        raise e
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response_content))
    
    return response_content, docs

print("--- 🤖 Chatbot is Ready! (Type 'exit' to quit) ---")

# --- 6. Main Loop ---
while True:
    try:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break

        ans, docs = answer_question(q)
        print(f"Bot: {ans}\n")
        
    except Exception as e:
        print("\n" + "="*40)
        print(f"❌ CRASH REPORT")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        traceback.print_exc()
        print("="*40 + "\n")