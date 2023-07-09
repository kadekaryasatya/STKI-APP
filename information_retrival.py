#Import Library
import pandas as pd
import numpy as np
import streamlit as st
import math
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
stop_words_id = open('stopwordid.txt')
stop_words_id = set(stop_words_id.read().split())

stemmer = PorterStemmer()
sastrawi_stemmer = StemmerFactory().create_stemmer()
st.markdown("""
<style>
table td:nth-child(1) {
    display: none
}
table th:nth-child(1) {
    display: none
}
</style>
""", unsafe_allow_html=True)

# fungsi preprocessing, berisi lowercasing, stopword removal, dan lemmatization
def preprocess(text, is_using_stopword, stopword_lang, use_stem):
    
    # lowercase
    text = text.lower()
    
    # stopword removal
    if is_using_stopword == True:
        if stopword_lang == "Indonesia":
            text = ' '.join([word for word in text.split()
                            if word not in stop_words_id])
        else:
            text = ' '.join([word for word in text.split()
                            if word not in stop_words])
            
    #stemming
    if use_stem == "Stemming":
        if (stopword_lang == "Indonesia"):
            text = ' '.join([sastrawi_stemmer.stem(word)
                             for word in text.split()])
        else:
            text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

#Preprocessing
st.title("Preprocessing")
use_stem = "Stemming"
is_using_stopword = st.checkbox("Stopword Removal", value=True)
stopword_lang = st.selectbox("Stopwords Language", ("Indonesia", "English"))

def build_table(data):
        rows = []
        for key, val in data.items():
            row = [key, val]
            rows.append(row)
        return rows
    
def build_table_incidence_matrix(data, indexed_files):
        rows = []
        for key, val in data.items():
            row = [key]
            for file_id, file_name in indexed_files.items():
                if file_id in val:
                    row.append("1")
                else:
                    row.append("0")
            rows.append(row)
        return rows

def build_index(documents):
    idx = 1
    indexed_files = {}
    index = {}
    for text in documents:
        words = text.lower().split()
        indexed_files[idx] = f"dokumen{idx}"
        for word in words:
            if word not in index:
                index[word] = {}
            if idx not in index[word]:
                index[word][idx] = 1
            else:
                index[word][idx] += 1
        idx += 1
    return index, indexed_files

#pencarian query boolean
def search(query_words, index, indexed_files):
    connecting_words = []
    different_words = []
    for word in query_words:
        if word.lower() in ["and", "or", "not"]:
            connecting_words.append(word.lower())
        else:
            different_words.append(word.lower())
    if not different_words:
        st.write("Please enter query words")
        return []
    results = set(index[different_words[0]])
    for word in different_words[1:]:
        if word.lower() in index:
            results = set(index[word.lower()]) & results
        else:
            st.write(f"{word} not found in documents")
            return []
    for word in connecting_words:
        if word == "and":
            next_results = set(index[different_words[0]])
            for word in different_words[1:]:
                if word.lower() in index:
                    next_results = set(index[word.lower()]) & next_results
                else:
                    st.write(f"{word} not found in documents")
                    return []
            results = results & next_results
        elif word == "or":
            next_results = set(index[different_words[0]])
            for word in different_words[1:]:
                if word.lower() in index:
                    next_results = set(index[word.lower()]) | next_results
            results = results | next_results
        elif word == "not":
            not_results = set()
            for word in different_words[1:]:
                if word.lower() in index:
                    not_results = not_results | set(index[word.lower()])
            results = set(index[different_words[0]]) - not_results
    return results


#BOYER MOORE
def boyer_moore_search(documents, query):
    n = len(documents)
    m = len(query)
    if m == 0:
        return []

    # Preprocessing - Bad Character Heuristic
    bad_char = {}
    for i in range(m - 1):
        bad_char[query[i]] = max(1, m - i - 1)

    # Searching
    occurrences = []
    i = 0
    search_rows = []
    while i <= n - m:
        j = m - 1
        while j >= 0 and query[j] == documents[i + j]:
            j -= 1
        if j < 0:
            occurrences.append(i)
            i += m
        else:
            # Calculate the shift based on bad character heuristic
            if documents[i + j] in bad_char:
                shift = bad_char[documents[i + j]]
            else:
                shift = m
            i += shift

        # Add row to the search table
        text_row = list(documents)
        pattern_row = [' '] * i
        pattern_row.extend(list(query))
        pattern_row.extend([''] * (n - i - m))
        search_rows.append(text_row[:n])
        search_rows.append(pattern_row[:n])

        # Stop moving if characters match
        if j >= 0:
            pattern_row[-1] = query[j]
            j -= 1

    # Create the search table DataFrame
    search_table = pd.DataFrame(data=search_rows)
    search_table = search_table.replace('', np.nan)
    search_table = search_table.fillna('')
    st.table(search_table)

    return occurrences

#Input Document
documents = st.text_area("Dokumen", "").split("\n")
documents = [preprocess(doc, is_using_stopword, stopword_lang, use_stem)
    for doc in documents]

index, indexed_files = build_index(documents)

#Input Query
query = st.text_input("Query")
query = preprocess(query, is_using_stopword, stopword_lang, use_stem)
query_words = word_tokenize(query)




# tokenisasi
tokens = [query.split()] + [doc.lower().split() for doc in documents]
lexicon = []
for token in tokens:
    for word in token:
        if word not in lexicon:
            lexicon.append(word)

#TF-IDF
# menghitung df dan menghitung idf
df = {}
D = len(documents) + 1
for i in range(D):
    for token in set(tokens[i]):
        if i != 0:
            if token not in df:
                df[token] = 1
            else:
                df[token] += 1

# menghitung tf
tf = []
for i in range(D):
    tf.append({})
    for token in tokens[i]:
        if token not in tf[i]:
            tf[i][token] = 1
        else:
            tf[i][token] += 1

idf = {token: math.log10((D-1)/(df[token])) for token in df}

# menghitung bobot tf-idf
tfidf = []
for i in range(D):
    tfidf.append({})
    for token in tf[i]:
        tfidf[i][token] = tf[i][token] * (idf)[token]

tfidf1 = []
for i in range(D):
    tfidf1.append({})
    for token in tf[i]:
        tfidf1[i][token] = tf[i][token] * ((idf)[token] + 1)

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

# menyimpan hasil pada dataframe
df_result = pd.DataFrame(columns=['token'] + ['tf_Q'] + ['tf_d'+str(i) for i in range(1, D)] + [
                         'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_Q'] + ['weight_d'+str(i) for i in range(1, D)])
for token in lexicon:
    row = {'token': token}
    if token in tf[0]:
        row['tf_Q'] = tf[0][token]
    else:
        row['tf_Q'] = 0

    if token in tfidf[0]:
        row['weight_Q'] = tfidf1[0][token]
    else:
        row['weight_Q'] = 0
    
    for i in range(1, D):
        # tf_i
        if token in tf[i]:
            row['tf_d'+str(i)] = tf[i][token]
        else:
            row['tf_d'+str(i)] = 0
        # weight_i
        if token in tfidf[i]:
            row['weight_d'+str(i)] = tfidf1[i][token]
        else:
            row['weight_d'+str(i)] = 0
            
    # df
    if token in df:
        df_ = df[token]
    else:
        df_ = 0

    # D/df
    if df_ > 0:
        D_df = (D-1) / df_
    else:
        D_df = 0

    # IDF
    if token in idf:
        IDF = idf[token]
    else:
        IDF = 0

    # IDF+1
    IDF_1 = IDF + 1
    
    row['df'] = df_
    row['D/df'] = D_df
    row['IDF'] = IDF
    row['IDF+1'] = IDF_1

    df_result = pd.concat(
        [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)

# menampilkan output pada Streamlit
if query:
    st.title("Result")
    st.write(" ## Preprocessing Query:")
    df_query = pd.DataFrame({
        'Query': [query.split()]
    })
    st.table(df_query.round(2))

    st.write("Preprocessing Tiap Dokumen:")
    df_token = pd.DataFrame({
        'Dokumen': ['Query']+['Dokumen '+str(i) for i in range(1, D)],
        'Token': tokens
    })
    st.table(df_token)
  
#Boolean
    if query_words:
        inverted_index_table = build_table(index)

        results_files = []
        if query_words:
            files = search(query_words, index, indexed_files)
            results_files = [indexed_files[file_id] for file_id in files]

        st.write("## Inverted Index")
        st.table(inverted_index_table)

        st.write("## Incidence Matrix")
        incidence_matrix_table_header = [
            "Term"] + [file_name for file_name in indexed_files.values()]
        incidence_matrix_table = build_table_incidence_matrix(index, indexed_files)
        df_incidence_matrix_table = pd.DataFrame(
            incidence_matrix_table, columns=incidence_matrix_table_header)
        st.table(df_incidence_matrix_table)

        if not results_files:
            st.warning("No matching files")
        else:
            st.write("## Boolean Results")
            st.markdown(f"""
                     Dokumen yang relevan dengan query adalah:
                        **{', '.join(results_files)}**
                     """)            

#TF-IDF result
    st.write("## TF-IDF")
    st.table(df_result)
    
    # Menghitung total weight_d dari setiap dokumen
    num_docs = D - 1
    total_weight_d = [0] * num_docs
    for i in range(1, num_docs+1):
        col_name = 'weight_d' + str(i)
        total_weight_d[i-1] = df_result[col_name].sum()

    df_table = pd.DataFrame({'Rank': range(1, num_docs+1),
                         'Dokumen': ['D'+str(i) for i in range(1, num_docs+1)],
                         'Total Weight_d': total_weight_d})

    df_ranking = df_table.sort_values(by='Total Weight_d', ascending=False).reset_index(drop=True)
    df_ranking['Rank'] = df_ranking.index + 1

    st.write(" ## Rank TF-IDF")
    st.table(df_ranking)
 
    st.write("Hasil perhitungan jarak Dokumen dengan Query")
    df_distance = pd.DataFrame(columns=['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)])
    df_distance['Token'] = lexicon
    df_distance['Q' + chr(178)] = df_result['weight_Q'] ** 2
    for i in range(1, D):
        df_distance['D'+str(i) + chr(178)] = df_result['weight_d'+str(i)] ** 2
    st.table(df_distance)
    
    sqrt_q = round(math.sqrt(df_distance['Q' + chr(178)].sum()), 4)
    sqrt_d = []
    for i in range(1, D):
        sqrt_d.append(
            round(math.sqrt(df_distance['D'+str(i) + chr(178)].sum()), 4))
    
    sqrt_table = pd.DataFrame({'sqrt Q': [sqrt_q]})
    for i in range(1, D):
        sqrt_table['sqrt D'+str(i)] = [sqrt_d[i-1]]

    st.write("Hasil perhitungan akar kuadrat")
    st.table(sqrt_table)

    st.write("## Space Vector Model")
    df_space_vector = pd.DataFrame(
        columns=['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)] + ['Q*D'+str(i) for i in range(1, D)])
    df_space_vector['Token'] = lexicon
    df_space_vector['Q' + chr(178)] = df_result['weight_Q'] ** 2
    for i in range(1, D):
        df_space_vector['D'+str(i) + chr(178)
                        ] = df_result['weight_d'+str(i)] ** 2
    for i in range(1, D):
        for j in range(len(df_space_vector)):
            df_space_vector['Q*D'+str(i)][j] = df_space_vector['Q' + chr(178)][j] * df_space_vector['D'+str(i) + chr(178)][j]
    
    st.table(df_space_vector)
    
    qdotd_table = pd.DataFrame({'Q.D'+str(i): [round(df_space_vector['Q*D' + str(i)].sum(), 4)] for i in range(1, D)})
    st.write("Hasil perhitungan Q.D")
    st.table(qdotd_table)

    # for i in range(1, D):
    #     st.latex(
    #         r'''Q \cdot D''' + str(i) + r''' = ''' +
    #         str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + r''' '''
    #     )

    st.write("Perhitungan Cosine Similarity")
    df_cosine = pd.DataFrame(index=['Cosine'], columns=[
                             'D'+str(i) for i in range(1, D)])
    for i in range(1, D):
        st.latex(
            r'''Cosine\;\theta_{D''' + str(i) + r'''}=\frac{''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + '''}{''' + str(sqrt_q) + ''' * ''' + str(sqrt_d[i-1]) + '''}= ''' + str(round(df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1]), 4)) + r'''''')
        df_cosine['D'+str(i)] = df_space_vector['Q*D' +
                                                str(i)].sum() / (sqrt_q * sqrt_d[i-1])
    
    # for i in range(1, D):
    #     cosine_similarity = round(df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1]), 4)
    #     df_cosine['D'+str(i)] = cosine_similarity
   
    st.table(df_cosine)
   
    # Calculate cosine similarity
    df_cosine = df_cosine.transpose()
    df_cosine.columns = ['Cosine']
    df_cosine['Ranking'] = df_cosine['Cosine'].rank(ascending=False)
    df_cosine.sort_values(by='Ranking', inplace=True)

    # Create the ranking and document DataFrame
    df_ranking = pd.DataFrame({
        'Ranking': df_cosine['Ranking'].astype(int),
        'Document': df_cosine.index
    })

    st.write("## Rank VSM")
    
    st.table(df_ranking)


    if documents and query:
            st.title("Boyer-Moore Text Search App")
            all_occurrences = []
            for doc in documents:
                occurrences = boyer_moore_search(doc, query)
                all_occurrences.extend(occurrences)
            
            if all_occurrences:
                st.success(f"Pattern found at indices: {all_occurrences}")
            else:
                st.info("Pattern not found.")
    else:
            st.warning("Please enter documents and query.")

