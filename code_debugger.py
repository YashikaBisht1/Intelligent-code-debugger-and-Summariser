import ast
import re
from collections import defaultdict
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import os
import google.generativeai as genai

# Set up GenAI with your Google API key
GOOGLE_API_KEY = os.getenv("gemini_api")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to perform syntax parsing using AST
def analyze_code(code):
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return str(e)

# Function to find unused variables using AST
def find_unused_variables(code):
    tree = ast.parse(code)
    declared_vars = set()
    used_vars = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                declared_vars.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)

    unused_vars = declared_vars - used_vars
    return unused_vars

# Function to detect cyclic dependencies using graph traversal
def detect_cyclic_dependencies(edges):
    graph = defaultdict(list)
    for src, dest in edges:
        graph[src].append(dest)

    visited = set()
    recursion_stack = set()

    def dfs(node):
        if node in recursion_stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        recursion_stack.add(node)
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True

        recursion_stack.remove(node)
        return False

    for node in graph:
        if dfs(node):
            return True

    return False

# Function to clean and tokenize input code for the ML model
def preprocess_code(code):
    clean_code = re.sub(r'\s+', ' ', code)  # Remove excess whitespaces
    return clean_code

# Function to summarize code and find bugs using Google GenAI
def summarize_and_find_bugs(code, task):
    try:
        model = genai.GenerativeModel("gemini-1.0-pro")
        response = model.generate_content(f"{task}: {code}")
        bot_response = response.candidates[0].content.parts[0].text
        return bot_response
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize CodeT5 or similar model for code debugging (Assumes preloaded model)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
    return tokenizer, model

def generate_fix(model, tokenizer, buggy_code):
    input_ids = tokenizer.encode(f"fix: {buggy_code}", return_tensors="pt")
    outputs = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)
    fixed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return fixed_code

# Render Python code with syntax highlighting
def render_code(code):
    formatter = HtmlFormatter(style="friendly", full=True)
    highlighted_code = highlight(code, PythonLexer(), formatter)
    return highlighted_code

# Main Streamlit Application
st.title("Intelligent Code Debugger and Summarizer")
st.write("Upload your buggy Python code and let the model suggest fixes and summaries!")

uploaded_file = st.file_uploader("Upload your Python file", type=".py")
if uploaded_file is not None:
    # Read and display the input code
    buggy_code = uploaded_file.read().decode("utf-8")
    st.subheader("Uploaded Code:")
    st.components.v1.html(render_code(buggy_code), height=400)

    # Analyze code for syntax errors
    syntax_error = analyze_code(buggy_code)
    unused_vars = find_unused_variables(buggy_code)

    if syntax_error:
        st.warning(f"Syntax Error Detected: {syntax_error}")

    if unused_vars:
        st.warning(f"Unused Variables Detected: {', '.join(unused_vars)}")

    # Debugging Process
    st.write("**Suggesting Fixes...**")
    tokenizer, model = load_model()

    preprocessed_code = preprocess_code(buggy_code)
    try:
        fixed_code = generate_fix(model, tokenizer, preprocessed_code)
        st.subheader("Suggested Fix:")
        st.components.v1.html(render_code(fixed_code), height=400)
    except Exception as e:
        st.error(f"An error occurred while generating fixes: {e}")

    # Code summarization using GenAI
    if st.button("Summarize Code"):
        with st.spinner('Summarizing code...'):
            summary_result = summarize_and_find_bugs(buggy_code, "Summarize the given code")
        st.subheader("Code Summary:")
        st.write(summary_result)

# Debugging History Feature
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file is not None:
    st.session_state.history.append({"buggy": buggy_code, "fixed": fixed_code})

if st.button("Show Debugging History"):
    st.write("### Debugging History")
    for i, entry in enumerate(st.session_state.history):
        st.write(f"#### Debug Session {i+1}")
        st.write("**Buggy Code:")
        st.components.v1.html(render_code(entry["buggy"]), height=300)
        st.write("**Fixed Code:")
        st.components.v1.html(render_code(entry["fixed"]), height=300)
