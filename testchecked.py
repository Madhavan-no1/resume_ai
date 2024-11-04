
import google.generativeai as genai 
from pathlib import Path
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure GenAI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to initialize the generative model for Resume Reviewer
def initialize_resume_model():
    generation_config = {"temperature": 0.9}
    return genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Function to generate content based on prompts
def generate_content(model, prompts):
    results = []
    for prompt_text in prompts:
        response = model.generate_content([prompt_text])
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_part = candidate.content.parts[0]
                if text_part.text:
                    results.append(f"**Prompt:** {prompt_text}\n**Description:**\n{text_part.text}\n")
                else:
                    results.append(f"**Prompt:** {prompt_text}\n**Description:** No valid content generated.\n")
            else:
                results.append(f"**Prompt:** {prompt_text}\n**Description:** No content parts found.\n")
        else:
            results.append(f"**Prompt:** {prompt_text}\n**Description:** No candidates found.\n")
    
    return results

# Streamlit app
def main():
    st.set_page_config(page_title="AI Assistant Platform", layout="wide")
    st.title("AI Assistant Platform")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "uploaded_resume" not in st.session_state:
        st.session_state.uploaded_resume = None
    if "resume_results" not in st.session_state:
        st.session_state.resume_results = []

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Choose an Application", ["Job Chat Assistant", "Resume Reviewer", "History"])

    if tab == "Job Chat Assistant":
        st.header("Job Chat Assistant")

        # Initialize the chatbot model
        model = genai.GenerativeModel('gemini-pro')
        
        # Chat interface
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                if entry["role"] == "user":
                    st.markdown(f"**You:** {entry['message']}")
                else:
                    st.markdown(f"**Bot:** {entry['message']}")

        # Input box for user question
        user_input = st.text_input("Ask a question related to jobs, resumes, or skills", key="chat_input")

        if st.button("Send", key="send_chat"):
            if user_input.strip() == "":
                st.warning("Please enter a question.")
            else:
                # Combine user input with context for the model
                combined_prompt = f"COnsider yourself as resume assistant where you help the user with resume jobs and job skill related question about companies and many more similiar to a job seeking person now give output if its related or partially related o carrer and job orelse say im not trained on this data: '{user_input}'. Please respond accordingly."
                
                # Send combined prompt to the model
                response = model.generate_content([combined_prompt])
                bot_response = response.candidates[0].content.parts[0].text if response.candidates else "No response."
                
                # Store the conversation
                st.session_state.chat_history.append({"role": "user", "message": user_input})
                st.session_state.chat_history.append({"role": "bot", "message": bot_response})

                # Display the new message
                st.markdown(f"**You:** {user_input}")
                st.markdown(f"**Bot:** {bot_response}")

                # Save to overall history
                st.session_state.history.append({
                    "type": "chat",
                    "content": {"user": user_input, "bot": bot_response}
                })

    elif tab == "Resume Reviewer":
        st.header("Resume Reviewer")

        # Upload an image file for resume review
        uploaded_file = st.file_uploader("Upload your resume as an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.session_state.uploaded_resume = uploaded_file
            # Save the uploaded file temporarily
            with open("temp_resume.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize the model
            resume_model = initialize_resume_model()
            
            # Input for multiple prompts
            st.write("Enter prompts (one per line):")
            prompts = st.text_area("Prompts", height=150)
            
            # Button to generate content
            if st.button("Generate Description", key="generate_resume"):
                if prompts.strip():
                    prompt_list = [prompt.strip() for prompt in prompts.split('\n') if prompt.strip()]
                    st.session_state.resume_results = generate_content(resume_model, prompt_list)

                    # Save to overall history
                    st.session_state.history.append({
                        "type": "resume",
                        "content": {
                            "image": uploaded_file,
                            "descriptions": st.session_state.resume_results
                        }
                    })
                else:
                    st.warning("Please enter at least one prompt.")
            
            # Optionally remove the temporary file
            Path("temp_resume.jpg").unlink()
        
        # Display the uploaded resume image and generated results
        if st.session_state.uploaded_resume and st.session_state.resume_results:
            st.image(st.session_state.uploaded_resume, caption='Uploaded Resume Image.', use_column_width=True)
            st.subheader("Generated Descriptions:")
            for description in st.session_state.resume_results:
                st.markdown(description)

    elif tab == "History":
        st.header("History of Interactions")

        if st.session_state.history:
            for idx, entry in enumerate(st.session_state.history, 1):
                st.markdown(f"### Entry {idx}")
                if entry["type"] == "chat":
                    st.markdown(f"**You:** {entry['content']['user']}")
                    st.markdown(f"**Bot:** {entry['content']['bot']}")
                elif entry["type"] == "resume":
                    st.image(entry["content"]["image"], caption=f'Resume Image {idx}', use_column_width=True)
                    for desc in entry["content"]["descriptions"]:
                        st.markdown(desc)
                st.markdown("---")
        else:
            st.write("No history available yet.")

if __name__ == "__main__":
    main()
