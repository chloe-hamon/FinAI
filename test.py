# TEST COMPLET DE LA STACK

# Test 1 - yfinance
import yfinance as yf
cac40 = yf.Ticker("^FCHI")
data = cac40.history(period="5d")
print(" yfinance fonctionne ")
print(data[["Close"]].tail(3))

# Test 2 - pandas
import pandas as pd
print("\npandas fonctionne ")

# Test 3 - chromadb
import chromadb
client = chromadb.Client()
print("\n ChromaDB fonctionne ")

# Test 4 - PyPDF2
from PyPDF2 import PdfReader
print("\n PyPDF2 fonctionne ")

print("\n Toute la stack est opérationnelle !")
