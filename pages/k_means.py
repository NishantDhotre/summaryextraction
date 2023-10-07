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
from random import sample
import random

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
# -------------------------------------------------------------- K-Means -----------
def k_mean(tfidf_matrix, k):

    centroids = sample(tfidf_matrix, k)

    # print(centroids)
    main_clusters = []

    while True:
        clusters = []
        for i in range(k):
            clusters.append([])
        for j in range(total_sentences):#cosine_similarity sentence to centroid
            similarity = []
            for idx, centroid in enumerate(centroids):
                cosine = cosine_similarity(tfidf_matrix[j], centroid)
                similarity.append(cosine)
            clusterIdx = np.argmax(similarity)
            clusters[clusterIdx].append(j)

        new_centroids = []

        for i in range(k):#updating centroids
            temp = []
            for idx in clusters[i]:
                temp.append(tfidf_matrix[idx])
            new_centroids.append(np.mean(temp, axis=0))

        if np.allclose(centroids, new_centroids):#breaking if there is no significant change in centroid
            break
        centroids = new_centroids
        main_clusters = clusters

    return main_clusters, centroids
#-------------------------------------- Bigram ----------
def get_bigrams(sentence):
    
    words = sentence.split()
    return [tuple((words[i], words[i+1])) for i in range(len(words) - 1)]

# ----------------------- closest to centroid [S1]------------
def get_closest_to_centroid(centroid, tfidf_matrix, cluster):
    max_distance = -1.0
    closest_sentence = None
    tie = []
    for sentence in cluster:
        cosine = cosine_similarity(tfidf_matrix[sentence], centroid)
        if cosine > max_distance:
            max_distance = cosine
            closest_sentence = sentence
        if cosine == max_distance:
            tie.append(sentence)

    if len(tie) > 1 :
        return(random.choice(tie))
    else:
        return(closest_sentence)
#--------------------------------- graph --------------------
def sentence_graph(s1, s2):
    G = {}
    if s2 is not None:
        s2_bigram = get_bigrams(sentences[s2])
    s1_bigram = get_bigrams(sentences[s1])
    start = "start"
    end = "end"

    G[start] = set()
    G[end] = set()
    for bigram in s1_bigram:
        G[bigram] = set()
        
    if s2 is not None:
        for bigram in s2_bigram:
            G[bigram] = set()

    G[start].add(s1_bigram[0])

    G[s1_bigram[-1]].add(end)

    for i in range(len(s1_bigram) - 1):
        G[s1_bigram[i]].add(s1_bigram[i + 1])

    if s2 is not None:
        for i in range(len(s2_bigram) - 1):
            G[s2_bigram[i]].add(s2_bigram[i + 1])
    return G

# ----------------------- sentence from graph --------------
def generate_sentence(G):
    final_sentence = ""
    current_node = "start"

    while current_node != "end":
        neighbors = list(G[current_node])
        next_node = random.choice(neighbors)
        if next_node[0] != 'e':
            final_sentence += (next_node[0] + " ")
        current_node = next_node
    return final_sentence

# -------------------------------------------------------------
def cluster_sentence_generation(main_clusters, centroid, tfidf_matrix):
    """
    Generate sentences for each cluster following the specified steps.
    """
    cluster_graph = {}
    selected_sentences = {}#for sorting

    for idx, cluster in enumerate(main_clusters):

        # Find S1 (closest to the centroid)
        s1 = get_closest_to_centroid(centroid[idx], tfidf_matrix, cluster)

        # Find S2 (with at least 3 bigrams in common with S1)
        s2 = None
        for sentence in cluster:
            if sentence != s1:
                curr_bigram = get_bigrams(sentences[sentence])
                t1 = set(get_bigrams(sentences[s1]))
                t2 = set(curr_bigram)
                common_bigrams = t1.intersection(t2)
                if len(common_bigrams) >= 3:
                    s2 = sentence
                    break

        # Construct sentence graph
        graph = sentence_graph(s1, s2)
        cluster_graph[idx] = graph

        # Generate a sentence
        generated_sentence = generate_sentence(graph)

        selected_sentences[s1] = generated_sentence

    return selected_sentences, cluster_graph

# --------------------------------printing it---------------
def print_it(selected_sentences):
    st.subheader("Extracted Summary:")

    myKeys = list(selected_sentences.keys())
    myKeys.sort()
    sorted_dict = {i: selected_sentences[i] for i in myKeys}

    for idx, sentence in sorted_dict.items():
        st.info(sentence)
# ---------------------------------------------------------------------

# # `--------------------------------main-----------------------------------
if __name__ == "__main__":
        
    st.title("Text Extraction Website")
    # User input options
    input_option = st.radio("Choose an input option:", ("Enter Text", "Upload Text File"))

    if input_option == "Enter Text":
        text = st.text_area("Enter your text here:", placeholder="Text.........")
        uploaded_file = None  # No file is uploaded in this case
        k = st.number_input("enter number of clusters:", value = 4)
    elif input_option == "Upload Text File":
        uploaded_file = st.file_uploader("Upload a text file (TXT)", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read()
        else:
            text = ""
        k = st.number_input("enter number of clusters:", value = 4)
        
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

        # --------------------------------
        original_sentences = sentences
        
        sentences = preprocessing(sentences)#preprocessing sentences
        st.success("Text Pre-Processing done!")

        tfidf_matrix, total_sentences = matrix_calculation(sentences)#tfidf making
        st.success("TF-IDF Vector Created!")
        
        clusters = []
        centroids = []

        # if st.button('Enter'):
        clusters, centroids = k_mean(tfidf_matrix, k)

        st.success("Clusters Formed Created!")
        st.write(clusters)

        st.success("Cluster sentence generated!")
        selected_sentences, cluster_graph = cluster_sentence_generation(clusters, centroids, tfidf_matrix)
        print_it(selected_sentences)
        

