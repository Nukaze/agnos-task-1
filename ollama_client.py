import requests
import json
import os
import streamlit as st
from dotenv import load_dotenv

class OllamaClient:
    def __init__(self, base_url=None, username=None, password=None):
        # Try to get base_url in following order:
        # 1. Passed parameter
        # 2. Environment variable (for local development)
        # 3. Default localhost
        
        # Load environment variables from .env file (local development)
        load_dotenv()
        
        # Set debug flag based on development mode
        self.debug_mode = st.secrets.get("is_dev", False) or os.getenv("is_dev")
        
        self.base_url = (
            base_url 
            or st.secrets.get("OLLAMA_SERVICE_URL") 
            or os.getenv("OLLAMA_SERVICE_URL") 
            or "http://localhost:11434"
        )
        self.username = (
            username 
            or st.secrets.get("NGROK_BASIC_AUTH_USERNAME") 
            or os.getenv("NGROK_BASIC_AUTH_USERNAME")
        )
        self.password = (
            password 
            or st.secrets.get("NGROK_BASIC_AUTH_PASSWORD") 
            or os.getenv("NGROK_BASIC_AUTH_PASSWORD")
        )
        
        self.auth = (self.username, self.password) if self.username and self.password else None
            
        if not self.base_url:
            self.base_url = "http://localhost:11434"
            
        self.models = self.get_available_models()


    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            # http://localhost:11434/api/tags
            response = requests.get(f"{self.base_url}/api/tags", auth=self.auth)
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception as e:
            if self.debug_mode:
                print(f"Error fetching models: {e}")
            return []


    def generate_response(self, model, prompt, system_prompt=None, temperature=0.5, stream=True):
        """Generate a response from the specified model with streaming support
        Supports both single prompts (str) and conversation history (list)"""
        try:
            if self.debug_mode:
                print(f"Debug: generate_response called with model: {model}")
            
            # Check if prompt is a list (conversation history) or string (single prompt)
            if isinstance(prompt, list):
                # Handle conversation history
                if self.debug_mode:
                    print(f"Debug: Detected conversation history (length: {len(prompt)})")
                conversation_prompt = ""
                
                # Add conversation history
                for i, msg in enumerate(prompt):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_prompt += f"{role}: {msg['content']}\n"
                    if self.debug_mode:
                        print(f"Debug: Added message {i+1}: {role} (length: {len(msg['content'])})")
                
                # Add the final prompt for the assistant to respond
                conversation_prompt += "Assistant: "
                if self.debug_mode:
                    print(f"Debug: Conversation prompt length: {len(conversation_prompt)}")
                
                # Use conversation prompt and pass system prompt separately
                final_prompt = conversation_prompt
                
            else:
                # Handle single prompt
                if self.debug_mode:
                    print(f"Debug: Detected single prompt (length: {len(prompt)})")
                final_prompt = prompt
            
            if self.debug_mode:
                print(f"Debug: Final prompt length: {len(final_prompt)}")
                print(f"Debug: System prompt length: {len(system_prompt) if system_prompt else 0}")
            
            payload = {
                "model": model,
                "prompt": final_prompt,
                "stream": stream,
                "temperature": temperature,
                "options": {
                    "num_gpu": 99,  # Use all available GPUs
                    "num_thread": 8  # Fallback to CPU threads if no GPU
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                if self.debug_mode:
                    print(f"Debug: Added system prompt to payload")

            if self.debug_mode:
                print(f"Debug: Sending request to {self.base_url}/api/generate")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                auth=self.auth
            )
            
            if self.debug_mode:
                print(f"Debug: Response status code: {response.status_code}")
            
            if not stream:
                if response.status_code == 200:
                    return response.json().get("response", "Error: No response generated")
                return f"Error: {response.status_code} - {response.text}"
            
            # Handle streaming response
            if response.status_code == 200:
                chunk_count = 0
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if "response" in json_response:
                                chunk_count += 1
                                if chunk_count == 1 and self.debug_mode:
                                    print(f"Debug: First chunk received: {json_response['response'][:50]}...")
                                yield json_response["response"]
                        except json.JSONDecodeError:
                            continue
                if self.debug_mode:
                    print(f"Debug: Total chunks received: {chunk_count}")
                    if chunk_count == 0:
                        print("Debug: No chunks received from streaming response")
            else:
                if self.debug_mode:
                    print(f"Debug: Error response: {response.text}")
                yield f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Exception in generate_response: {str(e)}")
            yield f"Error: {str(e)}" 