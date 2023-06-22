import pandas as pd
import streamlit as st
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


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

# inisiasi stopword dan wordnetlemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# fungsi preprocessing, berisi lowercasing, stopword removal, dan lemmatization


def preprocess(text, use_stem_or_lem, is_using_stopword):
    # lowercase
    text = text.lower()
    # stopword removal
    if is_using_stopword == True:
        text = ' '.join([word for word in text.split()
                        if word not in stop_words])
    # lemmatization
    if use_stem_or_lem == "Lemmatization":
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    elif use_stem_or_lem == "Stemming":
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def build_index(documents):
    idx = 1
    indexed_files = {}
    index = {}
    for text in documents:
        words = text.lower().split()  # Lowercase and split the text into words
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


def search(query, index, indexed_files):
        connecting_words = []
        different_words = []
        for word in query:
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

st.title("Preprocessing")
use_stem_or_lem = st.selectbox(
    "Stemming/Lemmatization", ("Stemming", "Lemmatization"))
is_using_stopword = st.checkbox("Stopword Removal", value=True)
"---"

text_list1 = st.text_area("Enter your first document:")
text_list2 = st.text_area("Enter your second document:")

documents = [text_list1, text_list2]

documents = [preprocess(doc, use_stem_or_lem, is_using_stopword)
             for doc in documents]
query = st.text_input("Query")
query = preprocess(query, use_stem_or_lem, is_using_stopword)
index, indexed_files = build_index(documents)

# tokenisasi
tokens = [doc.lower().split() for doc in documents]

query_words = word_tokenize(query)

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
            st.write("## Results")
            st.markdown(f"""
                     Dokumen yang relevan dengan query adalah:
                        **{', '.join(results_files)}**
                     """)


# menghitung df dan menghitung idf
df = {}
D = len(documents)
for i in range(D):
    for token in set(tokens[i]):
        if token not in df:
            df[token] = 1
        else:
            df[token] += 1

idf = {token: math.log10(D/df[token]) for token in df}

# menghitung tf
tf = []
for i in range(D):
    tf.append({})
    for token in tokens[i]:
        if token not in tf[i]:
            tf[i][token] = 1
        else:
            tf[i][token] += 1


# menghitung bobot tf-idf
tfidf = []
for i in range(D):
    tfidf.append({})
    for token in tf[i]:
        tfidf[i][token] = tf[i][token] * idf[token]


# menyimpan hasil pada dataframe
df_result = pd.DataFrame(columns=['Q'] + ['tf_d'+str(i+1) for i in range(D)] + [
                         'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_d'+str(i+1) for i in range(D)])
for token in query.lower().split():
    row = {'Q': token}
    for i in range(D):
        # tf_i
        if token in tf[i]:
            row['tf_d'+str(i+1)] = tf[i][token]
        else:
            row['tf_d'+str(i+1)] = 0
        # weight_i
        if token in tfidf[i]:
            row['weight_d'+str(i+1)] = tfidf[i][token] + 1
        else:
            row['weight_d'+str(i+1)] = 0
    # df
    if token in df:
        df_ = df[token]
    else:
        df_ = 0

    # D/df
    if df_ > 0:
        D_df = D / df_
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

    # df_result = df_result.append(row, ignore_index=True)
    df_result = pd.concat([df_result, pd.DataFrame(row, index=[0])], ignore_index=True)


# menampilkan output pada Streamlit
if query:
    st.title("Result")
    st.write("Preprocessing Query:")
    df_query = pd.DataFrame({
        'Query': [query.split()]
    })
    st.table(df_query)
    st.write("Preprocessing Tiap Dokumen:")
    df_token = pd.DataFrame({
        'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
        'Token': tokens
    })
    st.table(df_token)
    st.write("TF-IDF Table query")
    st.table(df_result)

    st.write("Dokumen terurut berdasarkan bobot:")
    df_weight_sorted = pd.DataFrame({
        'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
        'Sum Weight': [sum([df_result['weight_d'+str(i+1)][j] for j in range(len(df_result))]) for i in range(D)]
    })
    st.dataframe(df_weight_sorted.sort_values(
        by=['Sum Weight'], ascending=False))