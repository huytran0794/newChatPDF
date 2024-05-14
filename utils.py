from sentence_transformers import SentenceTransformer
import pinecone
from openai import OpenAI, AuthenticationError
import streamlit as st
indexName = 'langchain-chatbot-pdf-demo'

model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(
    api_key="d2f46e6d-e5af-494e-9645-5d5f5912125f",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)
index = pinecone.Index('langchain-chatbot-pdf-demo')


def checkValidOpenAPI(api_key):
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
    except AuthenticationError:
        return False
    else:
        return True


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']


def query_refiner(client, conversation, query):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"},
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    result = (response.choices)[0].message.content
    return result


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):

        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + \
            st.session_state['responses'][i+1] + "\n"
    return conversation_string
