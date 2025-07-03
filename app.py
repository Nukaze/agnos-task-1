import streamlit as st
from ollama_client import OllamaClient
import time

import utils



def main():
    
    st.set_page_config(
        page_title="Agnos Health LLM by Nukaze",
        page_icon="⚕️",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    USER = "user"
    ASSISTANT = "assistant"
    
    st.title("Agnos Health LLM by Nukaze")
    
    
    if "ollama_client" not in st.session_state:
        # Initialize the Ollama client
        st.session_state.ollama_client = OllamaClient()
    
    if "history_messages" not in st.session_state:
        st.session_state.history_messages = []
    
    if "count" not in st.session_state:
        st.session_state.count = 0
        
        
        
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        
        # get available llm models from ollama
        ollama_models = st.session_state.ollama_client.get_available_models()
        
        
        template_models = ["Ollama Model 1", "Ollama Model 2", "Ollama Model 3"]
        # display llm model from ollama
        selected_model = st.selectbox(
            "Select LLM Model",
            options=[template_models, ollama_models][len(ollama_models) > 0],
            index=0
        )
        
        temparature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="""Controls the creativity and randomness of the AI's answers. [Lower value] temperatures mean more focused, deterministic, and factual responses. [Higher value] temperatures lead to more random, diverse, and creative (sometimes less coherent) outputs."""
        )
        
        system_prompt = st.text_area(
            "System Prompt",
            value=utils.get_system_prompt(),
            height=200,
            max_chars=500,
            help="A system prompt to guide the AI's behavior. This is a good place to set the tone and style of the responses."
        )
        
        # Option to reset the chat history
        if st.button("Reset Chat History"):
            st.session_state.history_messages = []
            st.session_state.count = 0
            st.success("Chat history has been reset.")
        
    
    # main chat interface
    for message in st.session_state.history_messages:    
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input("Enter your message"):
        
        st.chat_message(USER).markdown(user_prompt)
        
        st.session_state.history_messages.append({"role": USER, "content": user_prompt})
        
        if count := st.session_state.count:
            st.session_state.count += 1

        # loading indicator
        with st.spinner("Thinking..."):
            
            response_placeholder = st.empty()
            
            if delay := st.session_state.get("delay", 1.5):
                time.sleep(delay)
                
            with st.chat_message(ASSISTANT):

                # response = "This is a placeholder response from the LLM."  # Replace with actual LLM call
                response = f"Placeholder reponse from LLM {count}"  # Simulating a response for demonstration
        
                st.markdown(response)
                st.session_state.history_messages.append({"role": ASSISTANT, "content": response})
        
        
    
    

if __name__ == "__main__":
    main()