import streamlit as st
from agent_graph import run_agent
from load_documents import load_documents

st.set_page_config(page_title="NHHT HR Agent", layout="wide")

st.title("🤝 Te Hoa o te Kaitātaki – NHHT HR Agent")

user_input = st.text_input("Ask your HR question:")

if st.button("Submit"):
    if user_input:
        output = run_agent(user_input)
        st.success(output)
    else:
        st.warning("Please enter a question.")

st.sidebar.title("Load HR Documents (Admin Only)")
if st.sidebar.button("Load Documents"):
    load_documents()
    st.sidebar.success("Documents loaded successfully.")
