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
        st.session_state.ollama_client = OllamaClient()
    
    if "history_messages" not in st.session_state:
        st.session_state.history_messages = []
    
    if "count" not in st.session_state:
        st.session_state.count = 0
        
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Get available LLM models from Ollama
        ollama_models = st.session_state.ollama_client.get_available_models()
        
        # Display LLM model from Ollama
        if ollama_models:
            model_names = [model.get("name", "Unknown") for model in ollama_models]
            selected_model = st.selectbox(
                "Select LLM Model",
                options=model_names,
                index=3
            )
        else:
            selected_model = "llama2"  # Default fallback model
            st.warning("No models found. Using default model: llama2")
        
        # Debug info
        st.sidebar.write(f"Selected model: {selected_model}")
        st.sidebar.write(f"Available models: {len(ollama_models) if ollama_models else 0}")
        
        temperature = st.slider(
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
        if st.button("Reset Chat History", type="secondary"):
            st.session_state.history_messages = []
            st.session_state.count = 0
            st.success("Chat history has been reset.")
        
        # Debug: Show conversation history length
        st.sidebar.write(f"Chat history length: {len(st.session_state.history_messages)}")
        
        # Add option to use simple prompt vs conversation
        use_conversation = st.sidebar.checkbox("Use Conversation History", value=True, help="Enable to use full conversation history, disable for single prompt")
    
    # main chat interface
    for message in st.session_state.history_messages:    
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input("Enter your message"):
        # Display user message
        st.chat_message(USER).markdown(user_prompt)
        st.session_state.history_messages.append({"role": USER, "content": user_prompt})
        
        # Loading indicator and streaming response
        with st.spinner("Thinking..."):
            response_placeholder = st.empty()
            
            # Create assistant message container
            with st.chat_message(ASSISTANT):
                message_placeholder = st.empty()
                
                # Get streaming response from Ollama
                full_response = ""
                try:
                    # Debug: Show what we're sending
                    st.sidebar.write(f"Debug: Model: {selected_model}")
                    st.sidebar.write(f"Debug: Use conversation: {use_conversation}")
                    
                    if use_conversation and len(st.session_state.history_messages) > 0:
                        # Use conversation history mode
                        try:
                            # Prepare conversation history for the model
                            # Use last 20 messages to avoid token limits
                            recent_messages_limit = 20
                            conversation_history = st.session_state.history_messages[-recent_messages_limit:]
                            
                            # Add current user message to history
                            conversation_history.append({"role": "user", "content": user_prompt})
                            
                            # Debug: Show what we're sending
                            st.sidebar.write(f"Debug: Sending {len(conversation_history)} messages")
                            
                            # Generate streaming response with full conversation history
                            # Pass conversation_history as a list - generate_response will detect it automatically
                            response_stream = st.session_state.ollama_client.generate_response(
                                model=selected_model,
                                prompt=conversation_history,  # Pass as list for conversation mode
                                system_prompt=system_prompt,
                                temperature=temperature,
                                stream=True
                            )
                            
                            # Display streaming response
                            response_received = False
                            for chunk in response_stream:
                                response_received = True
                                if chunk.startswith("Error:"):
                                    st.error(chunk)
                                    st.sidebar.error(f"Ollama Error: {chunk}")
                                    break
                                else:
                                    full_response += chunk
                                    # Update the message in real-time
                                    message_placeholder.markdown(full_response + "▌")
                            
                            if not response_received:
                                st.sidebar.warning("Conversation mode failed, trying simple prompt...")
                                raise Exception("No response from conversation mode")
                                
                        except Exception as conv_error:
                            st.sidebar.warning(f"Conversation mode failed: {str(conv_error)}")
                            # Fallback to simple prompt mode
                            use_conversation = False
                    
                    if not use_conversation or len(st.session_state.history_messages) == 0:
                        # Use simple prompt mode (fallback or first message)
                        st.sidebar.write("Debug: Using simple prompt mode")
                        
                        # Build simple prompt with system prompt
                        simple_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant: "
                        
                        # Generate streaming response with simple prompt
                        # Pass simple_prompt as a string - generate_response will detect it automatically
                        response_stream = st.session_state.ollama_client.generate_response(
                            model=selected_model,
                            prompt=simple_prompt,  # Pass as string for simple prompt mode
                            system_prompt=None,  # Already included in prompt
                            temperature=temperature,
                            stream=True
                        )
                        
                        # Display streaming response
                        response_received = False
                        for chunk in response_stream:
                            response_received = True
                            if chunk.startswith("Error:"):
                                st.error(chunk)
                                st.sidebar.error(f"Ollama Error: {chunk}")
                                break
                            else:
                                full_response += chunk
                                # Update the message in real-time
                                message_placeholder.markdown(full_response + "▌")
                        
                        if not response_received:
                            st.error("No response received from Ollama")
                            st.sidebar.error("No response received from Ollama")
                            full_response = "Error: No response received from Ollama"
                    
                    # Final update without cursor
                    message_placeholder.markdown(full_response)
                    
                    # Add to chat history
                    st.session_state.history_messages.append({"role": ASSISTANT, "content": full_response})
                    
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.sidebar.error(f"Exception: {error_message}")
                    st.session_state.history_messages.append({"role": ASSISTANT, "content": error_message})

if __name__ == "__main__":
    main()