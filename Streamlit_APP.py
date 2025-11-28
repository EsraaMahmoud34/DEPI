import os
import streamlit as st
import pandas as pd
import joblib

# --- Imports for Machine Learning & Chatbot ---
# Google Gemini Integration
from langchain_google_genai import ChatGoogleGenerativeAI
# Hugging Face Embeddings (Must match how FAISS was created)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Page Config ---
st.set_page_config(page_title="AI Project Hub", page_icon="⚙️🤖", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ About the App")
    st.write("""
    This project includes two tools:
    1. **Machine Failure Predictor**: Predict equipment failure using sensor data.
    2. **RAG Chatbot**: Ask questions about your documents using **Google Gemini**.
    """)
    st.markdown("---")
    st.caption("Developed by Esraa Mahmoud & Team 🧠")

# --- Tabs ---
tab1, tab2 = st.tabs(["⚙️ Machine Failure Predictor", "🤖 Gemini Chatbot"])

# ============================
# Tab 1: Machine Failure Predictor (Unchanged)
# ============================
with tab1:
    # Load Model
    # Ensure this path is correct relative to where you run the command
    model_path = "DEPI Project\\First Machine_falilure_model\\machine_failure_model.pkl"
    
    try:
        model = joblib.load(model_path)
        model_loaded = True
    except FileNotFoundError:
        st.error(f"⚠️ Model file not found at: {model_path}")
        model_loaded = False

    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>⚙️ AI-Powered Machine Failure Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Predict potential equipment failure using AI & IoT sensor data.</p>", unsafe_allow_html=True)
    st.write("---")

    if model_loaded:
        st.subheader("🧩 Machine & Process Details")
        col1, col2 = st.columns(2)
        with col1:
            product_id = st.text_input("🆔 Product ID", "")
            machine_id = st.text_input("🏭 Machine ID", "")
        with col2:
            type_option = st.selectbox("⚙️ Type", ["L", "M", "H"])

        st.write("---")
        st.subheader("🌡️ Sensor Measurements")
        col1, col2, col3 = st.columns(3)
        with col1:
            air_temp = st.text_input("Air temperature [K]", "300")
        with col2:
            process_temp = st.text_input("Process temperature [K]", "310")
        with col3:
            rot_speed = st.text_input("Rotational speed [rpm]", "1500")

        col4, col5 = st.columns(2)
        with col4:
            torque = st.text_input("Torque [Nm]", "40")
        with col5:
            tool_wear = st.text_input("Tool wear [min]", "120")

        st.write("---")
        st.subheader("⚠️ Failure Type Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        twf = col1.radio("TWF", ["No", "Yes"])
        hdf = col2.radio("HDF", ["No", "Yes"])
        pwf = col3.radio("PWF", ["No", "Yes"])
        osf = col4.radio("OSF", ["No", "Yes"])
        rnf = col5.radio("RNF", ["No", "Yes"])

        # Encode inputs
        encode = lambda x: 1 if x == "Yes" else 0
        twf, hdf, pwf, osf, rnf = map(encode, [twf, hdf, pwf, osf, rnf])
        type_map = {'L': 0, 'M': 1, 'H': 2}
        type_encoded = type_map.get(type_option, 0)

        input_data = pd.DataFrame([[type_encoded, float(air_temp), float(process_temp),
                                    float(rot_speed), float(torque), float(tool_wear),
                                    twf, hdf, pwf, osf, rnf]],
                                columns=['Type','Air temperature [K]','Process temperature [K]',
                                         'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]',
                                         'TWF','HDF','PWF','OSF','RNF'])

        if st.button("🚀 Predict Machine Failure", use_container_width=True):
            try:
                prediction = model.predict(input_data)[0]
                probability = float(model.predict_proba(input_data)[0][1] * 100)
                progress_value = min(probability / 100, 1.0)
                if prediction == 1:
                    st.error(f"⚠️ **Machine Failure Predicted!** \n\nFailure Probability: **{probability:.2f}%**")
                    st.progress(progress_value)
                else:
                    st.success(f"✅ **No Failure Expected.** \n\nFailure Probability: **{probability:.2f}%**")
                    st.progress(progress_value)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================
# Tab 2: Google Gemini Chatbot
# ============================
with tab2:
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🤖 Industrial AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Ask questions about your documents using <b>Gemini 2.0 Flash</b>.</p>", unsafe_allow_html=True)
    st.write("---")

    # --- Configuration ---
    # !!! PASTE YOUR KEY HERE !!!
    GOOGLE_API_KEY = "AIzaSyCF-epOd0cnEUNaOR0VkidIJZAls3K-sjw"
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    # Path Configuration
    # Updated absolute path as requested
    INDEX_NAME = r"project chatboot\chatboot\faiss_index_136k"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    MODEL_NAME = "gemini-2.0-flash"

    # --- Resource Loading (Cached) ---
    @st.cache_resource
    def load_resources():
        print("--- Loading Chatbot Resources... ---")
        try:
            # 1. Load Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME, 
                model_kwargs={'device': 'cpu'}
            )
            
            # 2. Load Vector Store
            vector_store = FAISS.load_local(
                INDEX_NAME, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            
            # 3. Setup Gemini
            llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                temperature=0.6,
                google_api_key=GOOGLE_API_KEY,
                convert_system_message_to_human=True
            )
            return retriever, llm
        except Exception as e:
            st.error(f"Failed to load resources: {e}")
            return None, None

    retriever, llm = load_resources()

    # --- Chat UI ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- THE ENHANCED PROMPT ---
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

    # Handle Input
    if user_input := st.chat_input("Ask about machine status, definitions, or maintenance..."):
        # 1. User Message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 2. Assistant Response
        if retriever and llm:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("Thinking...")
                
                try:
                    # Retrieve
                    docs = retriever.invoke(user_input)
                    context_text = "\n\n".join([d.page_content for d in docs])
                    
                    # Create Chain with Enhanced Prompt
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_instruction),
                        ("human", "Question: {question}"),
                    ])
                    
                    chain = prompt_template | llm
                    
                    # Generate
                    response_message = chain.invoke({
                        "context": context_text, 
                        "question": user_input
                    })
                    response_text = response_message.content
                    
                    # Display
                    placeholder.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    # Sources
                    with st.expander("View Retrieved Data"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Source {i+1} ({doc.metadata.get('source', 'Unknown')}):**")
                            st.caption(doc.page_content[:300] + "...")
                            
                except Exception as e:
                    placeholder.error(f"Error: {e}")
                    traceback.print_exc()
        else:
            st.error("Chatbot is offline. Please check API Key and Index Path.")