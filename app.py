import os
import streamlit as st
from ollama_client import OllamaClient
import time
import utils
from utils import PineconeManager, TextEmbedder


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
    
    if "is_dev" not in st.session_state:
        st.session_state.is_dev = st.secrets.get("is_dev", False)
        
    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = OllamaClient()
        
    if "use_single_prompt_mode" not in st.session_state:
        st.session_state.use_single_prompt_mode = st.secrets.get("USE_SINGLE_PROMPT_MODE", False)
    
    print("use single prompt:", st.session_state.use_single_prompt_mode)
    
    # Initialize RAG service for vector search
    if "rag_service" not in st.session_state:
        try:
            # Initialize Pinecone RAG service
            print("Initializing TextEmbedder for RAG service...")
            embedder = TextEmbedder()
            print("TextEmbedder initialized successfully")
            
            print("Initializing PineconeManager...")
            st.session_state.rag_service = PineconeManager(
                api_key=st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY"),
                cloud=st.secrets.get("PINECONE_ENVIRONMENT_CLOUD") or os.getenv("PINECONE_ENVIRONMENT_CLOUD") or "aws",
                region=st.secrets.get("PINECONE_ENVIRONMENT_REGION") or os.getenv("PINECONE_ENVIRONMENT_REGION"),
                index_name=st.secrets.get("PINECONE_INDEX_NAME"),
                embedder=embedder,
                dimension=1024,
            )
            print("RAG service initialized successfully")
            
            if st.session_state.is_dev:
                st.sidebar.success("RAG service initialized successfully")
        except Exception as e:
            print(f"Failed to initialize RAG service: {str(e)}")
            st.session_state.rag_service = None
            if st.session_state.is_dev:
                st.sidebar.error(f"Failed to initialize RAG service: {str(e)}")
    
    if "history_messages" not in st.session_state:
        st.session_state.history_messages = []

        
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
            selected_model = "gemma3"  # Default fallback model
            st.warning("No models found. Using default model: gemma3")
        
        if st.session_state.is_dev:
            # Debug info
            st.sidebar.write(f"is dev: {st.session_state.is_dev}")
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
        
        if st.session_state.is_dev:
            # Debug: Show conversation history length
            st.sidebar.write(f"Chat history length: {len(st.session_state.history_messages)}")
        
        # RAG Settings
        st.sidebar.header("RAG Settings")
        use_rag = st.sidebar.checkbox("Enable RAG (Vector Search)", value=True, help="Enable to search vector database for relevant context")
        rag_k = st.sidebar.slider("Number of context documents", min_value=1, max_value=10, value=5, help="Number of relevant documents to retrieve")
        
        # Show RAG status
        if st.session_state.rag_service:
            st.sidebar.success("✅ RAG Service: Connected")
        else:
            st.sidebar.error("❌ RAG Service: Not Available")
            use_rag = False
    
    # Helper function to retrieve relevant context from vector database
    def get_relevant_context(query: str, k: int = 5) -> str:
        """Retrieve relevant context from vector database for RAG"""
        if not st.session_state.rag_service:
            if st.session_state.is_dev:
                st.sidebar.warning("RAG service not available - no context will be retrieved")
            return ""
        
        try:
            # Retrieve relevant documents
            print(f"Retrieving context for query: {query[:50]}...")
            results = st.session_state.rag_service.retrieve(query, k=k)
            
            print(f"RAG retrieved {len(results) if results else 0} results")
            
            if not results:
                if st.session_state.is_dev:
                    st.sidebar.info("No relevant documents found")
                return ""
            
            # Format context from retrieved documents - include content and sources
            context_parts = []
            for i, result in enumerate(results, 1):
                # Extract source URL and metadata
                source = result.metadata.get('source', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
                timestamp = result.metadata.get('forum_posted_timestamp', '') if hasattr(result, 'metadata') else ''
                reply_count = result.metadata.get('forum_reply_count', '') if hasattr(result, 'metadata') else ''
                
                # Try to get content from different possible locations
                content = ""
                if hasattr(result, 'metadata') and result.metadata:
                    # First try to get content from metadata (where we store it)
                    content = result.metadata.get('content', '')
                elif hasattr(result, 'page_content'):
                    content = result.page_content
                elif hasattr(result, 'content'):
                    content = result.content
                
                print(f"\n[DEBUG] Content for result {i}: {content[:200]}...\n")
                print(f"[source]: [{source}]")
                
                # Create context part with content and metadata
                context_part = f"[{i}] Source: {source}"
                
                # Add metadata info
                if timestamp:
                    context_part += f" (Posted Date: {timestamp})"
                if reply_count:
                    context_part += f" (Replies: {reply_count})"

                
                context_parts.append(context_part)
            
            context = "\n\n".join(context_parts)
            
            if st.session_state.is_dev:
                st.sidebar.write(f"Debug: Retrieved {len(results)} relevant documents")
                st.sidebar.write(f"Debug: Context length: {len(context)} characters")
                # Log only sources for debugging
                sources_only = [result.metadata.get('source', 'Unknown') if hasattr(result, 'metadata') else 'Unknown' for result in results]
                st.sidebar.write(f"Debug: Sources found: {sources_only}")
            
            return context
            
        except Exception as e:
            print(f"Error in get_relevant_context: {str(e)}")
            import traceback
            traceback.print_exc()
            if st.session_state.is_dev:
                st.sidebar.error(f"Error retrieving context: {str(e)}")
            return ""

    # Helper function to handle streaming response display
    def display_streaming_response(response_stream, message_placeholder):
        """Display streaming response and return full response text"""
        full_response = ""
        response_received = False
        
        for chunk in response_stream:
            response_received = True
            if chunk.startswith("Error:"):
                st.error(chunk)
                st.sidebar.error(f"Ollama Error: {chunk}")
                return chunk  # Return error message
            else:
                full_response += chunk
                # Update the message in real-time
                message_placeholder.markdown(full_response + "▌")
        
        if not response_received:
            error_msg = "No response received from Ollama"
            st.error(error_msg)
            st.sidebar.error(error_msg)
            return f"Error: {error_msg}"
        
        # Final update without cursor
        message_placeholder.markdown(full_response)
        return full_response

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
            # Create assistant message container
            with st.chat_message(ASSISTANT):
                message_placeholder = st.empty()
                
                # Get streaming response from Ollama
                try:
                    # Debug: Show what we're sending
                    if st.session_state.is_dev:
                        st.sidebar.write(f"Debug: Model: {selected_model}")
                    
                    # Retrieve relevant context using RAG if enabled
                    context = ""
                    if use_rag and st.session_state.rag_service:
                        if st.session_state.is_dev:
                            st.sidebar.write("Debug: Retrieving relevant context...")
                        context = get_relevant_context(user_prompt, k=rag_k)
                        
                    # Use conversation history mode
                    try:
                        # Prepare conversation history for the model (last 20 messages)
                        recent_messages_limit = 20
                        conversation_history = st.session_state.history_messages[-recent_messages_limit:]
                        
                        # Add current user message to history
                        conversation_history.append({"role": "user", "content": user_prompt})
                        
                        if st.session_state.is_dev:
                            st.sidebar.write(f"Debug: Sending {len(conversation_history)} messages")

                        
                        # Add context to the system prompt if RAG is enabled and context is found
                        enhanced_system_prompt = system_prompt
                        if context:
                            enhanced_system_prompt = f"{system_prompt}\n\n**Relevant Context from Agnos Medical Database:**\n{context}\n\nPlease use this context to provide source url accurate, evidence-based responses."
                            if st.session_state.is_dev:
                                st.sidebar.write(f"Debug: Enhanced system prompt with {len(context)} chars of context")
                    
                    
                        is_single_prompt_mode = st.session_state.use_single_prompt_mode.lower() != False
                        print("single prompt mode", type(is_single_prompt_mode))
                        
                        final_prompt = [conversation_history, user_prompt][is_single_prompt_mode]
                        print("type of prompt", type(final_prompt))
                        print('\n')
                        
                        # Generate streaming response with conversation history
                        response_stream = st.session_state.ollama_client.generate_response(
                            model=selected_model,
                            prompt=final_prompt,  # Pass as list for conversation mode
                            system_prompt=enhanced_system_prompt,
                            temperature=temperature,
                            stream=True
                        )
                        
                        full_response = display_streaming_response(response_stream, message_placeholder)
                        
                        
                    except Exception as conv_error:
                        if st.session_state.is_dev:
                            st.sidebar.warning(f"Conversation mode failed: {str(conv_error)}")
                        # Fallback to simple prompt mode
                        raise Exception("Conversation mode failed, using simple prompt")
                    
                    # Add to chat history
                    st.session_state.history_messages.append({"role": ASSISTANT, "content": full_response})
                    
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    if st.session_state.is_dev:
                        st.sidebar.error(f"Exception: {error_message}")
                    st.session_state.history_messages.append({"role": ASSISTANT, "content": error_message})

if __name__ == "__main__":
    main()