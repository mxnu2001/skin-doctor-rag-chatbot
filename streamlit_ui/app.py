import streamlit as st
import requests

# Replace with your ngrok / deployed API URL
API_URL = "https://8ffd18aa5194.ngrok-free.app/ask"

st.set_page_config(page_title="Skin Doctor Chatbot", page_icon="💬")
st.title("💬 Skin Doctor Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.text_input("Ask a skin-related question:")

if st.button("Send") and user_input:
    # Append user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call API safely
    try:
        response = requests.post(API_URL, json={"question": user_input})
        response.raise_for_status()  # raises error for bad HTTP status
        data = response.json()
        raw_answer = data.get("answer", "Sorry, no answer received.")

        # Clean the response: remove 'Patient:' / 'Doctor:' prefixes for UI
        chat_lines = raw_answer.splitlines()
        answer = []
        for line in chat_lines:
            if line.startswith("Doctor:"):
                # Strip the prefix for nicer display
                answer.append(line.split(":", 1)[1].strip())
        answer_text = "\n".join(answer) if answer else "Sorry, no answer found."

    except Exception as e:
        answer_text = f"Error: {e}"

    # Append bot response
    st.session_state.messages.append({"role": "bot", "content": answer_text})

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Doctor:** {msg['content']}")
