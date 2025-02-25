import os
import time
import chainlit as cl
from dotenv import load_dotenv 
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

load_dotenv()

VECTOR_INDEX_NAME = 'wix-upgrade'
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ADA_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_ADA_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION= os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    
embeddings = AzureOpenAIEmbeddings(
    deployment=AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_ADA_DEPLOYMENT_VERSION
)


contextualize_q_system_prompt = """
Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


qa_system_prompt = """
Act like a WiX (WixToolset) development expert and help me with related WiX development questions. 

Some **strict rules** you have to follow NO MATTER WHAT:
Only answer related questions about WiX development.
If you don't know the answer, say:
I don't have such information, please refer to WiX offical documentation for the information. https://docs.firegiant.com/

Please Use the following pieces of context to answer the user's question. 
----------------
{context}
"""
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


chat_history_store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="The Major Updates in Wix5",
            message="What's the major updates in Wix5?",
            icon="/public/images/specifications.svg",
            ),
        cl.Starter(
            label="Migrate Project from Wix4 to Wix5",
            message="How to mirage the wix4 project to wix5?",
            icon="/public/images/jigsaw.png",
            ),
        cl.Starter(
            label="Build Burn Bootstrapper on Wix5",
            message="How to build a burn bootstrapper on wix5?",
            icon="/public/images/bundle.svg",
            )
        ]

@cl.on_chat_start
async def start():
    docs_vector_store = FAISS.load_local(
        folder_path="./vector_stores", 
        embeddings=embeddings, 
        index_name=VECTOR_INDEX_NAME,
        allow_dangerous_deserialization=True)
    
    llm=AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
        openai_api_type="azure",
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        streaming=True,
        verbose=True,
    )
    
    retriever = docs_vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    cl.user_session.set("chain", conversational_rag_chain)


@cl.on_message
async def main(message: cl.Message):
    try:
        chain = cl.user_session.get("chain")
        msg = cl.Message(content="")
        res = chain.invoke(
            {"input": message.content}, 
            config={
                "configurable": {"session_id": cl.user_session.get("id")}
            }
        )
        response = res["answer"]

        stream_size = 20
        for i in range(int(round(len(response) / stream_size)) + 1):
            msg.content = response[0 : (i + 1) * stream_size]
            await msg.send()
            time.sleep(0.05)

    except Exception as e:
        print(f"An error occurred: {e}")
        if e.code == "content_filter":
            await cl.ErrorMessage(
                """
                The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. 
                To learn more about our content filtering policies please read our documentation: 
                https://go.microsoft.com/fwlink/?linkid=2198766
                """
                ).send()