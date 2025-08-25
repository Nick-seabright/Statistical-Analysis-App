# streamlit_app/pages/file_management.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to path if running this file directly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from streamlit_app.components.file_browser import file_browser

def show_file_management():
    st.markdown("<div class='subheader'>File Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>View and manage your saved files.</div>", unsafe_allow_html=True)
    
    # Check if save directory is set
    if 'save_directory' not in st.session_state:
        st.warning("Please set a save directory in the sidebar first.")
        return
    
    save_dir = st.session_state.save_directory
    
    # Create tabs for different file types
    tab1, tab2, tab3 = st.tabs(["Reports", "Models", "Data Files"])
    
    with tab1:
        st.subheader("Saved Reports")
        reports_dir = os.path.join(save_dir, "reports")
        
        # Make sure directory exists
        os.makedirs(reports_dir, exist_ok=True)
        
        # PDF reports
        st.markdown("#### PDF Reports")
        selected_pdf = file_browser(reports_dir, "pdf")
        
        if selected_pdf:
            st.write(f"Selected: {os.path.basename(selected_pdf)}")
            
            # Offer actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Open PDF", key="open_pdf"):
                    try:
                        import webbrowser
                        webbrowser.open(selected_pdf)
                    except Exception as e:
                        st.error(f"Could not open file: {str(e)}")
            
            with col2:
                try:
                    with open(selected_pdf, "rb") as f:
                        report_bytes = f.read()
                        
                    st.download_button(
                        "Download PDF",
                        report_bytes,
                        file_name=os.path.basename(selected_pdf),
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Could not read file: {str(e)}")
        
        # HTML reports
        st.markdown("#### HTML Reports")
        selected_html = file_browser(reports_dir, "html")
        
        if selected_html:
            st.write(f"Selected: {os.path.basename(selected_html)}")
            
            # Offer actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Open HTML", key="open_html"):
                    try:
                        import webbrowser
                        webbrowser.open(selected_html)
                    except Exception as e:
                        st.error(f"Could not open file: {str(e)}")
            
            with col2:
                try:
                    with open(selected_html, "rb") as f:
                        html_content = f.read()
                        
                    st.download_button(
                        "Download HTML",
                        html_content,
                        file_name=os.path.basename(selected_html),
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Could not read file: {str(e)}")
            
            with col3:
                if st.button("Preview", key="preview_html"):
                    try:
                        with open(selected_html, "r") as f:
                            html_content = f.read()
                        
                        st.components.v1.html(html_content, height=500, scrolling=True)
                    except Exception as e:
                        st.error(f"Could not preview file: {str(e)}")
    
    with tab2:
        st.subheader("Saved Models")
        models_dir = os.path.join(save_dir, "models")
        
        # Make sure directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        selected_model = file_browser(models_dir, "pkl")
        
        if selected_model:
            st.write(f"Selected: {os.path.basename(selected_model)}")
            
            # Offer actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Model", key="load_model_file"):
                    try:
                        import pickle
                        with open(selected_model, "rb") as f:
                            model = pickle.load(f)
                        
                        # Store in session state
                        model_name = os.path.basename(selected_model).split('.')[0]
                        if 'models' not in st.session_state:
                            st.session_state.models = {}
                        
                        st.session_state.models[model_name] = model
                        st.success(f"Model '{model_name}' loaded successfully!")
                    except Exception as e:
                        st.error(f"Could not load model: {str(e)}")
            
            with col2:
                try:
                    with open(selected_model, "rb") as f:
                        model_bytes = f.read()
                        
                    st.download_button(
                        "Download Model",
                        model_bytes,
                        file_name=os.path.basename(selected_model),
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"Could not read file: {str(e)}")
    
    with tab3:
        st.subheader("Data Files")
        data_dir = os.path.join(save_dir, "data")
        
        # Make sure directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # CSV files
        st.markdown("#### CSV Files")
        selected_csv = file_browser(data_dir, "csv")
        
        if selected_csv:
            st.write(f"Selected: {os.path.basename(selected_csv)}")
            
            # Offer actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Data", key="load_csv"):
                    try:
                        df = pd.read_csv(selected_csv)
                        st.session_state.data = df
                        st.success(f"Data loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns.")
                        st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"Could not load data: {str(e)}")
            
            with col2:
                try:
                    with open(selected_csv, "rb") as f:
                        csv_bytes = f.read()
                        
                    st.download_button(
                        "Download CSV",
                        csv_bytes,
                        file_name=os.path.basename(selected_csv),
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Could not read file: {str(e)}")

if __name__ == "__main__":
    show_file_management()