import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

## Function to get response from LLama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    # Load the LLM
    def load_llm(openai_api_key):
        # Point to the local server, using OpenAI wrapper for compatibility
        llm = OpenAI(model="TheBloke/Llama-2-7B-Chat-GGUF", base_url="http://localhost:1234/v1", api_key=openai_api_key)
        return llm
    
    # Load the LLM model with the provided OpenAI API key
    llm = load_llm(openai_api_key='lm-studio')
    
    # Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)
    
    # Format the prompt with the given variables
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    
    # Generate the response from the LLM
    response = llm(formatted_prompt)
    
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## creating two more columns for additional 2 fields

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
    
