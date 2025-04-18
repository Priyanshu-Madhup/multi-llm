import streamlit as st
from openai import OpenAI
import base64
import os
from io import BytesIO

# Streamlit page configuration (must be first Streamlit command)
st.set_page_config(page_title="Multi LLM Chat", page_icon="💬", layout="centered")

# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = "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"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "deepseek/deepseek-r1:free"
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "DeepSeek R1"

# Define available models
model_options = {
    "DeepSeek R1": "deepseek/deepseek-r1:free",
    "DeepSeek V3": "deepseek/deepseek-chat-v3-0324:free",
    "Llama 4 Mavrick": "meta-llama/llama-4-maverick:free",
    "Llama 4 Scout": "meta-llama/llama-4-scout:free",
    "Llama 3.3": "meta-llama/llama-3.3-70b-instruct:free",
    "Gemini 2.5 pro": "google/gemini-2.5-pro-exp-03-25:free",
    "Gemma 3": "google/gemma-3-27b-it:free"
}

# Define model logos (local file paths)
model_logos = {
    "DeepSeek R1": "deepseeklogo.png",
    "DeepSeek V3": "deepseeklogo.png",
    "Llama 4 Mavrick": "llamalogo.png",
    "Llama 4 Scout": "llamalogo.png",
    "Llama 3.3": "llamalogo.png",
    "Gemini 2.5 pro": "gemini logo.png",
    "Gemma 3": "gemmalogo.jpeg"
}

# Initialize OpenAI client
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.session_state.api_key,
    )
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {str(e)}")
    client = None

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")

    # Model selection
    selected_model_name = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(st.session_state.selected_model_name)
    )
    st.session_state.selected_model = model_options[selected_model_name]
    st.session_state.selected_model_name = selected_model_name

    # Display current model info
    st.info(f"Currently using: {selected_model_name}")
    st.caption(f"Model ID: {st.session_state.selected_model}")

    # API Key input
    api_key_input = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=st.session_state.api_key
    )

    # Save API Key button
    if st.button("Save API Key"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("API Key saved!")
            st.rerun()
        else:
            st.error("Please enter a valid API key.")

    # Display API key status
    if st.session_state.api_key:
        st.success("API Key is set")
    else:
        st.error("API Key not set")

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Title with dynamic logo
try:
    image_path = model_logos[selected_model_name]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Logo file not found: {image_path}")
    
    # Convert the local image to base64
    with open(image_path, "rb") as img_file:
        local_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Determine image MIME type
    image_mime = "image/jpeg" if image_path.endswith(".jpeg") else "image/png"
    
    # Create a unified title with logo shifted left and doubled size
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 10px;">
            <h1 style="margin: 0;">Multi LLM Chat</h1>
            <img src="data:{image_mime};base64,{local_img_base64}" height="100" style="margin-left: 10px;">
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")
    st.title("Multi LLM Chat")  # Fallback
st.write(f"Powered by {st.session_state.selected_model_name} via OpenRouter API")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user" and "image" in message:
            st.image(message["image"], width=200)  # Display uploaded image
            st.markdown(message["content"])
        else:
            st.markdown(message["content"])


# Ensure image is processed only when the send button is clicked
prompt = st.chat_input("Type your message here...", key="chat_input_unique")
uploaded_image = None
with st.sidebar:
    if selected_model_name in ["Llama 4 Mavrick", "Llama 4 Scout", "Llama 3.3", "Gemma 3", "Gemini 2.5 pro"]:
        st.header("Attach Image")
        uploaded_image = st.file_uploader("Attach Image", type=["png", "jpg", "jpeg"], key="image_upload_unique")

# Process input only when prompt or uploaded_image is provided and send button is clicked
if prompt or (uploaded_image and st.button("Send")):
    if not client:
        st.error("OpenAI client not initialized. Please check API key.")
    else:
        # Prepare user message
        user_message = {"role": "user", "content": prompt or "What is in this image?"}

        # Handle image if uploaded
        image_data_url = None
        if uploaded_image:
            # Convert uploaded image to base64 data URL
            image_bytes = uploaded_image.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_mime = "image/png" if uploaded_image.type == "image/png" else "image/jpeg"
            image_data_url = f"data:{image_mime};base64,{image_base64}"
            user_message["image"] = BytesIO(image_bytes)  # Store for display

        # Add user message to session state
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            if image_data_url:
                st.image(user_message["image"], width=200)
            st.markdown(user_message["content"])

        # Prepare API message payload
        api_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message["content"]}
                ]
            }
        ]
        if image_data_url:
            api_messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url}
                }
            )

        # Send request to OpenRouter API
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501",  # Replace with your site URL
                    "X-Title": "Streamlit Chat App",  # Replace with your site title
                },
                extra_body={},
                model=st.session_state.selected_model,
                messages=api_messages
            )
            response = completion.choices[0].message.content.strip()

            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)

        except Exception as e:
            st.error(f"API Error: {str(e)}")



