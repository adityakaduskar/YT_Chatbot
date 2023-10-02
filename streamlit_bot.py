import gradio as gr
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document


from llama_hub.youtube_transcript import YoutubeTranscriptReader

from llama_index import VectorStoreIndex

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index.llm_predictor import LLMPredictor
from langchain.llms import LlamaCpp


# Use Sentence Transformers from hugginface to generate embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(
    model_name=model_name
)

# Load the LLAMA-2-18 8 bit quantised model in GGUF format as LlamaCPP
llm = LlamaCpp(
   
    model_path="/Users/adity/.cache/lm-studio/models/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GGUF/codeup-llama-2-13b-chat-hf.Q8_0.gguf",
    n_gpu_layers=-1,
    n_batch=512,
    temperature=0.1,
    max_tokens=256,
    top_p=1,
    verbose=True, 
    f16_kv=True,
    n_ctx=4096,
    use_mlock=True,n_threads=4,
    stop=["Human:","User:"]

)

# Create a service context object, allowing us to use the sentence embeddings and Llama2 model
llm_pred=LLMPredictor(llm=llm)
embedding_model = LangchainEmbedding(hf) 
service_context = ServiceContext.from_defaults(embed_model=embedding_model,llm_predictor=llm_pred)
index=None




# The "load_video" function takes in youtube_url as input and indexes the youtube video transcript

def load_video(youtube_url):
    print("In Load Data")

    if youtube_url.strip()=="":
        st.error("Enter A youtube URL")
        return None
    else:
        try:
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(ytlinks=[youtube_url])
    
        
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            return index
        except:
            print("Enter a valid youtube URL")
            st.error("Enter a valid youtube URL")
            return None

# The user will enter the youtube_url and press submit => which loads the index
index=None
chat_engine=None

# 
# Clicked: Set to True when Submit button is clicked
# Index: The vector index is stored in this session object. The Index will be persistent until a new URL is entered.

if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'index' not in st.session_state:
    st.session_state.index=None

### click_button changes to True when button is clicked
def click_button():
    st.session_state.clicked = True
with st.sidebar:
    st.title("Youtube Q&A ChatBot powered by Llama-2")
             
    youtube_url = st.sidebar.text_input('Enter Youtube URL', '')
    submit_btn=st.sidebar.button('Submit',on_click=click_button)
    ## When the submit button is clicked, load the data and set the index session_state to the loaded index
    if st.session_state.clicked: 
        print("Going to Load Data")
        index=load_data(youtube_url)
        st.session_state.index=index
        print("Index ",index)
        
        st.session_state.clicked=False # set it to false , so that load_data function is not called for every single user message



#print("Index",index)

print("Index State ",st.session_state.index)
#create the chat_engine object if the index has been loaded 
if st.session_state.index!=None:
    chat_engine=st.session_state.index.as_chat_engine(verbose=True,chat_mode="context",service_context=service_context)
    print("Chat engine",chat_engine) 
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    full_response = ''
    with st.chat_message("assistant"):
        with st.spinner("Thinking of an answer..."):
            print("Calling Chat Engine")
            if chat_engine!=None:
                response = chat_engine.stream_chat(prompt)
                placeholder = st.empty()
                
                for item in response.response_gen:
                    full_response += item
                    placeholder.markdown(full_response.strip("Assistant:"))
                placeholder.markdown(full_response)
    if full_response!="":
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)