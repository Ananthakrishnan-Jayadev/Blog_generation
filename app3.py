import streamlit as st
from langchain import PromptTemplate
from lmstudio.client import Client

def get_llama_response(input_text, no_words, blog_style):
    # Connect to LM Studio server
    client = Client(server_url="http://localhost:1234/v1/models")

    # Specify the model name you want to use on LM Studio
    model_name = "TheBloke/Llama-2-7B-Chat-GGUF"

    # Load the model from LM Studio
    llm = client.llm.load(model_name)

    # Prompt Template
    template = """
    Write a blog for {blog_style} job profile on the topic '{input_text}'
    within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)

    # Generate the response from the LLama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    
    return response

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## creating two more columns for additional 2 fields

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

## Final response
if submit:
    st.write(get_llama_response(input_text, no_words, blog_style))
