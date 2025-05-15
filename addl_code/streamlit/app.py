import streamlit as st
from milvus_query import *
st.set_page_config(layout="wide")
import re
import json 
sections_dict= json.load(open('section_dict.json','r'))

def calculate_height(text, min_height=10, max_height=400, line_height=20):
    # Estimate the height based on text length
    lines = text.count('\n') + 1
    height = min_height + lines * line_height
    return min(max_height, height)



st.title("⚖️ Legal LLM APP")
with st.form("my_form"):
    search_text = st.text_area("Please enter the relevant sections or case details:", "Section 132-A  Municipal corporation Act")
    submitted = st.form_submit_button("Submit")
if submitted:

    # rag_chain = get_subsections()
    # Regex pattern to capture the desired keys
    # Using a non-capturing group with (?: ...) instead of capturing group
    pattern = r'\b\d+(?:-[A-Za-z]+)?\b'

    # Find all matches in the text
    matches = re.findall(pattern, search_text)
    sections_result = []
    for i in matches:
        for j in list(sections_dict.keys()):
            if i in j:
                sections_result.append(sections_dict[j])
    print(sections_result)
    if len(sections_result) > 0:
        search_prayer_text = '\n'.join(sections_result)
    else:
        sections_result  = sections_search(search_text,20) # semantic search with
        search_prayer_text = '\n'.join(sections_result)
    prompt = f"""
                Objective: Create a concise summary of the provided legal document related to the Hyderabad Municipality Corporation.

                Document Details:
                - Document Context: {search_prayer_text}
                - Key Section: {search_text}

                Instructions: 
                - Summarize the main points and contextual information from the key section. total word count should be less than 200 words.
              """



    st.write_stream(llm.stream(prompt))
    #prayer_result = prayers_search(search_text+" "+search_prayer_text,6)
    #prayer = prayer_result[0] if prayer_result else ''
    search_judgeemnt_text = f'''section: {search_text}
    Sample prayer:{search_prayer_text}
    '''
    result_judgement = order_search(search_judgeemnt_text)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('Judgements')
        
    with col2:
        st.subheader('Summary of Judgements')

        

    # Assuming prayer_result and result_judgement are lists of equal length
    max_rows =  len(result_judgement)

    for row in range(max_rows):
    
        with col1:
            if row < len(result_judgement):
                item = f'''File_name: {result_judgement[row]['FileName']}
                {result_judgement[row]['orderText']}
                '''
                st.text_area("", item, height=200)

        with col2:
            if row < len(result_judgement):
                prompt = f"""
                Summarize the following legal judgment. The summary should include:
                1. Key case details (e.g., parties involved, court, and relevant dates).
                2. Final judgment or ruling.
                3. Key legal principles and reasoning used by the court.

                Ensure the summary is concise, clear, and under 200 words:

                {result_judgement[row]['orderText']}
                """
                # summary = llm(prompt)
                # for output in llm.stream(prompt):
                #     st.text_area("", value=st.write_stream(output), height=400)
                # st.text_area("", summary, height=400)
                output_container = st.empty()  # Placeholder for the streaming output
                complete_output = ""  # Accumulate the output here

                for i,output in enumerate(llm.stream(prompt)):  # Replace this with the actual streaming logic
                    complete_output += output  # Append the new chunk to the complete output
                    output_container.text_area("", value=complete_output, height=200, key=f"{row}_{i}")
