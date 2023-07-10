import os
import openai
import streamlit as st

from tempfile import NamedTemporaryFile
from dotenv import load_dotenv, find_dotenv
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

model_name = "gpt-3.5-turbo"

tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key=openai.api_key,
    temperature=0,
    model_name=model_name
)

agent_name = "chat-conversational-react-description"

agent = initialize_agent(
    agent=agent_name,
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

# set title
st.title('Ask a question to an image')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question about your image:')

    with NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name

        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
                st.write(response)
            follow_up_question = st.text_input('Ask a follow up question:')
            if follow_up_question and follow_up_question != "":
                prompt = '{}\n{}\n'.format(response, follow_up_question)
                print(prompt)
                follow_up_response = agent.run(prompt)
                st.write(follow_up_response)




