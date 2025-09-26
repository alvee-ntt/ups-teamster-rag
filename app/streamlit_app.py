import streamlit as st
from watsonx_client import WatsonXClient


def main():
    st.set_page_config(
        page_title="UPS Agreements Assistant",
        page_icon="💬",
        layout="centered"
    )

    st.title("💬 UPS Agreements Assistant")

    # Initialize client
    try:
        client = WatsonXClient()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.info("""
        Please set the following environment variables:
        - `WATSONX_APIKEY`: Your IBM WatsonX API key
        - `WATSONX_DEPLOYMENT_ID`: Your deployment ID
        - `WATSONX_REGION`: IBM Cloud region (optional, defaults to 'us-south')
        """)
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about UPS agreements..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.call_nonstream(prompt)

                    if isinstance(response, dict) and "choices" in response:
                        assistant_response = response["choices"][0]["message"]["content"]
                    else:
                        assistant_response = "I apologize, but I couldn't process your request. Please try again."

                    st.markdown(assistant_response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()