# streamlit_app.py
import streamlit as st
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now we can import from streamlit_app directory
from streamlit_app.app import main

if __name__ == "__main__":
    main()
