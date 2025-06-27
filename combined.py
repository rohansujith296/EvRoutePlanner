from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS       
from langchain_community.document_loaders import CSVLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def make_store1():
    loader = CSVLoader(file_path="/Users/rohansujith/Desktop/Python/EvroutePlanner/docs/detailed_ev_charging_stations.csv")
    docs = loader.load()
    docs = text_splitter.split_documents(docs)
    store = FAISS.from_documents(docs, embedding=embedding)
    store.save_local("ev_charging_stations")
    return store
   

def make_store2():
    loader = CSVLoader(file_path="/Users/rohansujith/Desktop/Python/EvroutePlanner/docs/electric_vehicles_spec_2025.csv.csv")
    docs = loader.load()
    docs = text_splitter.split_documents(docs)
    store2 = FAISS.from_documents(docs, embedding=embedding)
    store2.save_local("ev_specs")
    return store2

def make_store3():
    loader = CSVLoader(file_path="/Users/rohansujith/Desktop/Python/EvroutePlanner/docs/ev_charging_patterns.csv")
    docs = loader.load()
    docs = text_splitter.split_documents(docs)
    store3 = FAISS.from_documents(docs, embedding=embedding)
    store3.save_local("ev_charging_patterns")
    return store3
   

def make_store4():
    loader = CSVLoader(file_path="/Users/rohansujith/Desktop/Python/EvroutePlanner/docs/SyntheticTripsWestfield.csv")
    docs = loader.load()
    docs = text_splitter.split_documents(docs)
    store4 = FAISS.from_documents(docs, embedding=embedding)
    store4.save_local("ev_trips")
    return store4
    


def make_combined_store():
    vs = make_store1()
    vs2 = make_store2()
    vs3 = make_store3()
    vs4 = make_store4()

    vs.merge_from(vs2)
    vs.merge_from(vs3)
    vs.merge_from(vs4)

    vs.save_local("ev_combined")
    combined_store = FAISS.load_local("ev_combined", embedding , allow_dangerous_deserialization=True)
    return combined_store



