import streamlit as st
import sys
import os

# Add directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from streamlit_app.app import main

if __name__ == "__main__":
    main()
