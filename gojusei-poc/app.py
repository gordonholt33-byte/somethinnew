import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import tempfile
import os

# Initialize the local AI
# We use llama3.2 for the PoC as it's fast and lightweight for laptops
llm = Ollama(model="llama3.2", temperature=0.0)

st.set_page_config(page_title="Gojusei Bid Evaluator", layout="wide")
st.title("Gojusei Bid Evaluator (Offline PoC)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Bid Requirements")
    itt_input = st.text_area("Paste Mandatory ITT Requirements:", height=200, 
                             placeholder="e.g. Must have 5 years Agile experience. Must have active Security Clearance.")
    
    st.header("2. Candidate CV")
    uploaded_file = st.file_uploader("Upload CV (PDF)", type="pdf")

with col2:
    st.header("3. Compliance Audit")
    if st.button("Run Offline Audit", type="primary") and uploaded_file and itt_input:
        with st.spinner("Agent evaluating locally... Please wait."):
            
            # Save uploaded file to a temporary location to read it
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            # Parse the PDF
            loader = PyPDFLoader(path)
            pages = loader.load()
            cv_text = "\n".join([page.page_content for page in pages])
            
            # The AI Prompt
            prompt = PromptTemplate.from_template("""
            You are a strict UK Public Sector Bid Assessor. 
            
            ITT REQUIREMENTS:
            {itt}
            
            CANDIDATE CV:
            {cv}
            
            Evaluate the CV against the ITT requirements. Output a structured RAG matrix (Red/Amber/Green).
            For every requirement, you MUST quote the exact text from the CV that proves your score. If it is missing, say "NOT FOUND".
            """)
            
            # Run the AI
            chain = prompt | llm
            result = chain.invoke({"itt": itt_input, "cv": cv_text})
            
            # Display results
            st.markdown("### AI Output:")
            st.write(result)