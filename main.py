# import streamlit as st
# from openai import OpenAI
# import base64
# import os
# from io import BytesIO

# # Streamlit page configuration (must be first Streamlit command)
# st.set_page_config(page_title="Multi LLM Chat", page_icon="💬", layout="centered")

# # Initialize session state
# if "api_key" not in st.session_state:
#     st.session_state.api_key = ""
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "selected_model" not in st.session_state:
#     st.session_state.selected_model = "deepseek/deepseek-r1:free"
# if "selected_model_name" not in st.session_state:
#     st.session_state.selected_model_name = "DeepSeek R1"

# # Define available models
# model_options = {
#     "DeepSeek R1": "deepseek/deepseek-r1:free",
#     "DeepSeek V3": "deepseek/deepseek-chat-v3-0324:free",
#     "Llama 4 Mavrick": "meta-llama/llama-4-maverick:free",
#     "Llama 4 Scout": "meta-llama/llama-4-scout:free",
#     "Llama 3.3": "meta-llama/llama-3.3-70b-instruct:free",
#     "Gemini 2.5 pro": "google/gemini-2.5-pro-exp-03-25:free",
#     "Gemma 3": "google/gemma-3-27b-it:free"
# }

# # Define model logos (local file paths)
# model_logos = {
#     "DeepSeek R1": "deepseeklogo.png",
#     "DeepSeek V3": "deepseeklogo.png",
#     "Llama 4 Mavrick": "llamalogo.png",
#     "Llama 4 Scout": "llamalogo.png",
#     "Llama 3.3": "llamalogo.png",
#     "Gemini 2.5 pro": "gemini logo.png",
#     "Gemma 3": "gemmalogo.jpeg"
# }

# # Initialize OpenAI client
# try:
#     client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=st.session_state.api_key,
#     )
# except Exception as e:
#     st.error(f"Failed to initialize OpenAI client: {str(e)}")
#     client = None

# # Sidebar for configuration
# with st.sidebar:
#     st.header("Settings")

#     # Model selection
#     selected_model_name = st.selectbox(
#         "Select Model",
#         options=list(model_options.keys()),
#         index=list(model_options.keys()).index(st.session_state.selected_model_name)
#     )
#     st.session_state.selected_model = model_options[selected_model_name]
#     st.session_state.selected_model_name = selected_model_name

#     # Display current model info
#     st.info(f"Currently using: {selected_model_name}")
#     st.caption(f"Model ID: {st.session_state.selected_model}")

#     # API Key input
#     api_key_input = st.text_input(
#         "OpenRouter API Key",
#         type="password",
#         value=st.session_state.api_key
#     )

#     # Save API Key button
#     if st.button("Save API Key"):
#         if api_key_input:
#             st.session_state.api_key = api_key_input
#             try:
#                 # Reinitialize OpenAI client with the new API key
#                 client = OpenAI(
#                     base_url="https://openrouter.ai/api/v1",
#                     api_key=st.session_state.api_key,
#                 )
#                 st.success("API Key saved and client reinitialized!")
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Failed to reinitialize OpenAI client: {str(e)}")
#         else:
#             st.error("Please enter a valid API key.")

#     # Display API key status
#     if st.session_state.api_key:
#         st.success("API Key is set")
#     else:
#         st.error("API Key not set")

#     # Clear chat button
#     if st.button("Clear Chat History"):
#         st.session_state.messages = []
#         st.rerun()

# # Title with dynamic logo
# try:
#     image_path = model_logos[selected_model_name]
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Logo file not found: {image_path}")
    
#     # Convert the local image to base64
#     with open(image_path, "rb") as img_file:
#         local_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
#     # Determine image MIME type
#     image_mime = "image/jpeg" if image_path.endswith(".jpeg") else "image/png"
    
#     # Create a unified title with logo shifted left and doubled size
#     st.markdown(
#         f"""
#         <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 10px;">
#             <h1 style="margin: 0;">Multi LLM Chat</h1>
#             <img src="data:{image_mime};base64,{local_img_base64}" height="100" style="margin-left: 10px;">
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
# except Exception as e:
#     st.error(f"Error loading logo: {str(e)}")
#     st.title("Multi LLM Chat")  # Fallback
# st.write(f"Powered by {st.session_state.selected_model_name} via OpenRouter API")

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         if message["role"] == "user" and "image" in message:
#             st.image(message["image"], width=200)  # Display uploaded image
#             st.markdown(message["content"])
#         else:
#             st.markdown(message["content"])


# # Ensure image is processed only when the send button is clicked
# prompt = st.chat_input("Type your message here...", key="chat_input_unique")
# uploaded_image = None
# with st.sidebar:
#     if selected_model_name in ["Llama 4 Mavrick", "Llama 4 Scout", "Llama 3.3", "Gemma 3", "Gemini 2.5 pro"]:
#         st.header("Attach Image")
#         uploaded_image = st.file_uploader("Attach Image", type=["png", "jpg", "jpeg"], key="image_upload_unique")

# # Process input only when prompt or uploaded_image is provided and send button is clicked
# if prompt or (uploaded_image and st.button("Send")):
#     if not client:
#         st.error("OpenAI client not initialized. Please check API key.")
#     else:
#         # Prepare user message
#         user_message = {"role": "user", "content": prompt or "What is in this image?"}

#         # Handle image if uploaded
#         image_data_url = None
#         if uploaded_image:
#             # Convert uploaded image to base64 data URL
#             image_bytes = uploaded_image.read()
#             image_base64 = base64.b64encode(image_bytes).decode('utf-8')
#             image_mime = "image/png" if uploaded_image.type == "image/png" else "image/jpeg"
#             image_data_url = f"data:{image_mime};base64,{image_base64}"
#             user_message["image"] = BytesIO(image_bytes)  # Store for display

#         # Add user message to session state
#         st.session_state.messages.append(user_message)

#         # Display user message
#         with st.chat_message("user"):
#             if image_data_url:
#                 st.image(user_message["image"], width=200)
#             st.markdown(user_message["content"])

#         # Prepare API message payload
#         api_messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": user_message["content"]}
#                 ]
#             }
#         ]
#         if image_data_url:
#             api_messages[0]["content"].append(
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": image_data_url}
#                 }
#             )

#         # Send request to OpenRouter API
#         try:
#             completion = client.chat.completions.create(
#                 extra_headers={
#                     "HTTP-Referer": "http://localhost:8501",  # Replace with your site URL
#                     "X-Title": "Streamlit Chat App",  # Replace with your site title
#                 },
#                 extra_body={},
#                 model=st.session_state.selected_model,
#                 messages=api_messages
#             )
#             response = completion.choices[0].message.content.strip()

#             # Add assistant response to session state
#             st.session_state.messages.append({"role": "assistant", "content": response})

#             # Display assistant response
#             with st.chat_message("assistant"):
#                 st.markdown(response)

#         except Exception as e:
#             st.error(f"API Error: {str(e)}")



# import streamlit as st
# from groq import Groq
# from openai import OpenAI
# import base64
# import os
# from io import BytesIO

# # Streamlit page config
# st.set_page_config(page_title="LLM Chat (Groq + OpenRouter)", page_icon="💬", layout="centered")

# # Session state initialization
# if "api_key" not in st.session_state:
#     st.session_state.api_key = ""
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "provider" not in st.session_state:
#     st.session_state.provider = "Groq"
# if "selected_model_name_groq" not in st.session_state:
#     st.session_state.selected_model_name_groq = "LLaMA 3.3 70B"
# if "selected_model_name_openrouter" not in st.session_state:
#     st.session_state.selected_model_name_openrouter = "DeepSeek R1"

# # Model options per provider
# model_options = {
#     "Groq": {
#         "LLaMA 3.3 70B": "llama-3.3-70b-versatile"
#     },
#     "OpenRouter": {
#         "DeepSeek R1": "deepseek/deepseek-r1:free",
#         "Gemini 2.5 pro": "google/gemini-2.5-pro-exp-03-25:free"
#     }
# }

# model_logos = {
#     "LLaMA 3.3 70B": "llamalogo.png",
#     "DeepSeek R1": "deepseeklogo.png",
#     "Gemini 2.5 pro": "gemini logo.png"
# }

# # Sidebar settings
# with st.sidebar:
#     st.header("Settings")

#     provider = st.selectbox("Select Provider", ["Groq", "OpenRouter"], index=["Groq", "OpenRouter"].index(st.session_state.provider))
#     st.session_state.provider = provider

#     if provider == "Groq":
#         model_names = list(model_options["Groq"].keys())
#         selected_model_name = st.selectbox("Select Groq Model", options=model_names, index=model_names.index(st.session_state.selected_model_name_groq))
#         st.session_state.selected_model_name_groq = selected_model_name
#     else:
#         model_names = list(model_options["OpenRouter"].keys())
#         selected_model_name = st.selectbox("Select OpenRouter Model", options=model_names, index=model_names.index(st.session_state.selected_model_name_openrouter))
#         st.session_state.selected_model_name_openrouter = selected_model_name

#     # Unified selection
#     st.session_state.selected_model_name = selected_model_name
#     st.session_state.selected_model = model_options[provider][selected_model_name]

#     st.info(f"Using {provider} / {selected_model_name}")
#     st.caption(f"Model ID: {st.session_state.selected_model}")

#     api_key_input = st.text_input(f"{provider} API Key", type="password", value=st.session_state.api_key)
#     if st.button("Save API Key"):
#         if api_key_input:
#             st.session_state.api_key = api_key_input
#             st.success("API Key saved!")
#             st.rerun()
#         else:
#             st.error("Please enter a valid API key.")

#     if st.session_state.api_key:
#         st.success("API Key is set")
#     else:
#         st.error("API Key not set")

#     if st.button("Clear Chat History"):
#         st.session_state.messages = []
#         st.rerun()

# # Title with logo
# try:
#     image_path = model_logos.get(st.session_state.selected_model_name, None)
#     if image_path and os.path.exists(image_path):
#         with open(image_path, "rb") as img_file:
#             local_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
#         image_mime = "image/jpeg" if image_path.endswith(".jpeg") else "image/png"
#         st.markdown(
#             f"""
#             <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 10px;">
#                 <h1 style="margin: 0;">Multi LLM Chat</h1>
#                 <img src="data:{image_mime};base64,{local_img_base64}" height="100" style="margin-left: 10px;">
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     else:
#         st.title("Multi LLM Chat")
# except Exception as e:
#     st.error(f"Error loading logo: {str(e)}")
#     st.title("Multi LLM Chat")

# st.write(f"Powered by {st.session_state.selected_model_name} via {st.session_state.provider}")

# # Display previous messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Input prompt
# prompt = st.chat_input("Type your message here...")

# # Process input
# if prompt:
#     user_message = {"role": "user", "content": prompt}
#     st.session_state.messages.append(user_message)

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     try:
#         if st.session_state.provider == "Groq":
#             client = Groq(api_key=st.session_state.api_key)
#         else:
#             client = OpenAI(
#                 base_url="https://openrouter.ai/api/v1",
#                 api_key=st.session_state.api_key,
#             )

#         system_msg = {"role": "system", "content": "You are a helpful assistant."}
#         full_messages = [system_msg] + st.session_state.messages

#         if st.session_state.provider == "Groq":
#             chat_completion = client.chat.completions.create(
#                 messages=full_messages,
#                 model=st.session_state.selected_model,
#             )
#             response = chat_completion.choices[0].message.content.strip()
#         else:
#             chat_completion = client.chat.completions.create(
#                 model=st.session_state.selected_model,
#                 messages=full_messages,
#                 extra_headers={
#                     "HTTP-Referer": "http://localhost:8501",
#                     "X-Title": "Streamlit Chat App",
#                 },
#             )
#             response = chat_completion.choices[0].message.content.strip()

#         assistant_message = {"role": "assistant", "content": response}
#         st.session_state.messages.append(assistant_message)

#         with st.chat_message("assistant"):
#             st.markdown(response)

#     except Exception as e:
#         st.error(f"API Error: {str(e)}")

import streamlit as st
from groq import Groq
from openai import OpenAI
import base64
import os
from io import BytesIO

# Streamlit page config
st.set_page_config(page_title="LLM Chat (Groq + OpenRouter)", page_icon="💬", layout="centered")

# Session state initialization
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state.provider = "Groq"
if "selected_model_name_groq" not in st.session_state:
    st.session_state.selected_model_name_groq = "LLaMA 3.3 70B"
if "selected_model_name_openrouter" not in st.session_state:
    st.session_state.selected_model_name_openrouter = "DeepSeek R1"

# Model options per provider
model_options = {
    "Groq": {
        "LLaMA 3.3 70B": "llama-3.3-70b-versatile",
        "DeepSeek R1": "deepseek-r1-distill-llama-70b",
        "Llama 4 Mavrick" : "meta-llama/llama-4-maverick-17b-128e-instruct",
        "Llama 4 Scout" : "meta-llama/llama-4-scout-17b-16e-instruct",
        "Mistral Saba" : "mistral-saba-24b"
    },
    "OpenRouter": {
        "DeepSeek R1": "deepseek/deepseek-r1:free",
        "DeepSeek V3": "deepseek/deepseek-chat-v3-0324:free",
        "Llama 4 Mavrick": "meta-llama/llama-4-maverick:free",
        "Llama 4 Scout": "meta-llama/llama-4-scout:free",
        "Llama 3.3": "meta-llama/llama-3.3-70b-instruct:free",
        #"Gemini 2.5 pro": "google/gemini-2.5-pro-exp-03-25:free",
        "Gemma 3": "google/gemma-3-27b-it:free"
    }
}

image_capable_models = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
    #"google/gemini-2.5-pro-exp-03-25:free",
    "google/gemma-3-27b-it:free"
    "mistral-saba-24b"
]

model_logos = {
    "DeepSeek R1": "deepseeklogo.png",
    "DeepSeek V3": "deepseeklogo.png",
    "Llama 4 Mavrick": "llamalogo.png",
    "Llama 4 Scout": "llamalogo.png",
    "Llama 3.3": "llamalogo.png",
    "Gemini 2.5 pro": "gemini logo.png",
    "Gemma 3": "gemmalogo.jpeg"
}

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("Select Provider", ["Groq", "OpenRouter"], index=["Groq", "OpenRouter"].index(st.session_state.provider))
    st.session_state.provider = provider

    if provider == "Groq":
        model_names = list(model_options["Groq"].keys())
        selected_model_name = st.selectbox("Select Groq Model", options=model_names, index=model_names.index(st.session_state.selected_model_name_groq))
        st.session_state.selected_model_name_groq = selected_model_name
    else:
        model_names = list(model_options["OpenRouter"].keys())
        selected_model_name = st.selectbox("Select OpenRouter Model", options=model_names, index=model_names.index(st.session_state.selected_model_name_openrouter))
        st.session_state.selected_model_name_openrouter = selected_model_name

    # Unified selection
    st.session_state.selected_model_name = selected_model_name
    st.session_state.selected_model = model_options[provider][selected_model_name]

    st.info(f"Using {provider} / {selected_model_name}")
    st.caption(f"Model ID: {st.session_state.selected_model}")

    api_key_input = st.text_input(f"{provider} API Key", type="password", value=st.session_state.api_key)
    if st.button("Save API Key"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("API Key saved!")
            st.rerun()
        else:
            st.error("Please enter a valid API key.")

    if st.session_state.api_key:
        st.success("API Key is set")
    else:
        st.error("API Key not set")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Title with logo
try:
    image_path = model_logos.get(st.session_state.selected_model_name, None)
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            local_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        image_mime = "image/jpeg" if image_path.endswith(".jpeg") else "image/png"
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 10px;">
                <h1 style="margin: 0;">Multi LLM Chat</h1>
                <img src="data:{image_mime};base64,{local_img_base64}" height="100" style="margin-left: 10px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.title("Multi LLM Chat")
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")
    st.title("Multi LLM Chat")

st.write(f"Powered by {st.session_state.selected_model_name} via {st.session_state.provider}")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image"):
            st.image(message["image"], width=200)
        st.markdown(message["content"])

# Input prompt
prompt = st.chat_input("Type your message here...")

# Optional image uploader (only for image-capable models)
uploaded_image = None
if st.session_state.selected_model in image_capable_models:
    with st.sidebar:
        st.header("Attach Image")
        uploaded_image = st.file_uploader("Attach Image", type=["png", "jpg", "jpeg"], key="image_upload_unique")

# Process input
if prompt or (uploaded_image and st.button("Send")):
    user_message = {"role": "user", "content": prompt or "What is in this image?"}
    image_data_url = None

    if uploaded_image:
        image_bytes = uploaded_image.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_mime = "image/png" if uploaded_image.type == "image/png" else "image/jpeg"
        image_data_url = f"data:{image_mime};base64,{image_base64}"
        user_message["image"] = BytesIO(image_bytes)

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        if image_data_url:
            st.image(user_message["image"], width=200)
        st.markdown(user_message["content"])

    try:
        if st.session_state.provider == "Groq":
            client = Groq(api_key=st.session_state.api_key)
        else:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=st.session_state.api_key,
            )

        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        api_messages = [system_msg]

        for msg in st.session_state.messages:
            content_block = [{"type": "text", "text": msg["content"]}]
            if msg.get("image") and st.session_state.selected_model in image_capable_models:
                content_block.append({
                    "type": "image_url",
                    "image_url": {"url": image_data_url}
                })
            api_messages.append({"role": msg["role"], "content": content_block if image_data_url else msg["content"]})

        if st.session_state.provider == "Groq":
            chat_completion = client.chat.completions.create(
                messages=api_messages,
                model=st.session_state.selected_model,
            )
        else:
            chat_completion = client.chat.completions.create(
                model=st.session_state.selected_model,
                messages=api_messages,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Streamlit Chat App",
                },
            )

        response = chat_completion.choices[0].message.content.strip()
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)

        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        st.error(f"API Error: {str(e)}")