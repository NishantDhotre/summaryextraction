import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
from numpy.linalg import norm
from collections import Counter
import streamlit as st
from io import StringIO

# --------------------------------------------------------------
def cosine_similarity(x, y):
    return np.dot(x, y) / (norm(x) * norm(y))

# ------------------------------------Task 1`------------------------------
# removing punctuations
def preprocessing(sentences):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
                
    temp = []
    for sentence in sentences:
        res = ""
        for ele in sentence:
            if ele not in punctuations:
                res+=ele
        temp.append(res)
        
    sentences = temp
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    temp = []
    for sentence in sentences:#removing all unnecessary things from sentences
        sentence = sentence.lower()
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        temp.append(' '.join(words))
    return temp

# # --------------------------------------- Task 2 ----------------------------
def matrix_calculation(sentences):
    tf_matrix = []
    for sentence in sentences:
        word_count = Counter(sentence)
        tf_vector = {}
        for word, count in word_count.items():
            tf_vector[word] = count
        tf_matrix.append(tf_vector)

    unique_words = set(word for sentence in sentences for word in sentence)
    idf_vector = {}
    total_sentences = len(sentences)

    for word in unique_words:
        count = 0
        for sentence in sentences:
            if word in sentence:
                count +=1
        idf_vector[word] = math.log(total_sentences / (1 + count))

    temp = []
    for tf_vector in tf_matrix:
        tfidf_vector = {}
        for word, tf in tf_vector.items():
            tfidf_vector[word] = tf
        temp.append(tfidf_vector)

    sorted_unique_words = sorted(unique_words)
    tfidf_matrix = []
    for tfidf_vector in temp:
        row = [tfidf_vector.get(word, 0) for word in sorted_unique_words]
        tfidf_matrix.append(row)
    return tfidf_matrix, total_sentences
# # --------------------------------------- Task 3 -----------------------------------
def ranking(total_sentences, tfidf_matrix):
    G = nx.Graph()

    for i in range(total_sentences):
        G.add_node(i)

    for i in range(total_sentences):
        for j in range(i + 1, total_sentences):
            cosine = cosine_similarity(tfidf_matrix[i] , tfidf_matrix[j])
            G.add_edge(i, j, weight = cosine)

    return nx.pagerank(G, alpha= 0.9)
# --------------------------------------------- MMR ------------------------------
def mmr(sentences, tfidf_matrix, ranks, sorted_ranks):
    selected = []
    selected.append(sorted_ranks[0])
    rBs = sentences.copy()
    rBs.remove(sentences[sorted_ranks[0]])

    lamB = 0.7
    while len(rBs):
        new_ranks = {}
        for i in range(len(rBs)):
            temp = -10
            for j in selected:
                cosine = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
                temp = max(temp, cosine)
            new_ranks[i] = (lamB * ranks[i]) - (1 - lamB) * temp
        sorted_new_rank = sorted(new_ranks, key=ranks.get, reverse=True)[:1]
        selected.append(sorted_new_rank[0])
        rBs.remove(rBs[sorted_new_rank[0]])

    return  set(selected)
# ---------------------------------------------------------------------

# # `--------------------------------main-----------------------------------
if __name__ == "__main__":
    st.title("Text Summary Extraction")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Using Maximal Marginal Relevance")
    # User input options
    input_option = st.radio("Choose an input option:", ("Enter Text", "Upload Text File"))

    if input_option == "Enter Text":
        text = st.text_area("Enter your text here:",placeholder="Text.........")
        uploaded_file = None  # No file is uploaded in this case
    elif input_option == "Upload Text File":
        uploaded_file = st.file_uploader("Upload a text file (TXT)", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read()
        else:
            text = ""
            
    # -----------------------------------

    if st.button("Execute"):
        if not text and input_option == "Enter Text":
                    st.error("Please enter text or upload a text file to proceed.")
        elif not uploaded_file and input_option == "Upload Text File":
                    st.error("Please enter text or upload a text file to proceed.")
        else:
            if input_option == "Enter Text":
                # Split the input text into sentences
                sentences = nltk.sent_tokenize(text)
                
                # # Display the list of sentences
                # st.write("List of Sentences:")
                # for i, sentence in enumerate(sentences, start=1):
                #     st.write(f"{i}. {sentence}")

            elif input_option == "Upload Text File":
                sentences = []
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                while True:
                    line = stringio.readlines()
                    if not line:
                        break
                    sentences.append(line)
                sentences = sentences[0]
                uploaded_file.close()

        # --------------------------------------
            original_sentences = sentences
            
            sentences = preprocessing(sentences)#preprocessing sentences
            st.success("Text Pre-Processing done!")

            tfidf_matrix, total_sentences = matrix_calculation(sentences)#tfidf making
            ranks  = ranking(total_sentences, tfidf_matrix)#ranking all important sentences
            st.success("TF-IDF Vector Created!")


            sorted_ranks = []
            mmr_ranks = []

            sorted_ranks = sorted(ranks, key = ranks.get, reverse= True)
            mmr_ranks = mmr(sentences, tfidf_matrix, ranks, sorted_ranks)
                
            st.write(mmr_ranks)

            st.subheader("Extracted Summary:")

            for i , idx in enumerate(mmr_ranks):
                temp = str(i + 1)+ "]     "+ sentences[idx]
                temp += '\n'
                st.write(temp)








        