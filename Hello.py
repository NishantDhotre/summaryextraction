import streamlit as st

if __name__ == "__main__":
    st.title("Text Summary Extraction")
    st.markdown("<hr>", unsafe_allow_html=True)


    st.markdown("### Welcome ...")
    st.write('''Text extraction is the process of summarizing or extracting essential information from a larger body of text. 
             This website offers three distinct methods for text extraction:''')
    st.markdown("#### K-Means Clustering + Bigraph")
    st.write('''K-Means clustering groups sentences into clusters based on their similarity. 
             Bigraph analysis is then used to find the most representative sentences from each cluster, resulting in a concise summary.''')
    st.markdown("#### MMR (Maximal Marginal Relevance)")
    st.write('''MMR is a method that aims to select sentences for the summary that are both relevant to the main topic and diverse from one another. 
             It prevents redundancy and provides a more informative summary.''')
    st.markdown("#### TF-IDF + Cosine Similarity")
    st.write('''This method uses Term Frequency-Inverse Document Frequency (TF-IDF) to measure the importance of words in a document. 
             Cosine similarity is then employed to identify the similarity between sentences, allowing the extraction of a summary based on these measures.''')
    
