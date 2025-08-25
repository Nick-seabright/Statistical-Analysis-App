# streamlit_app/components/sidebar.py

import streamlit as st
import os

def create_sidebar():
    with st.sidebar:
        st.image("streamlit_app/assets/logo.png", width=150)
        st.title("Navigation")
        
        # File Path Configuration
        st.markdown("---")
        st.subheader("File Storage Settings")
        
        # Initialize the file path in session state if it doesn't exist
        if 'save_directory' not in st.session_state:
            # Default to user's Documents folder
            default_dir = os.path.join(os.path.expanduser("~"), "Documents", "StatisticalAnalysis")
            st.session_state.save_directory = default_dir
        
        # Allow user to specify a directory
        user_dir = st.text_input(
            "Save files to:",
            value=st.session_state.save_directory,
            help="Specify a folder where reports, models, and other files will be saved"
        )
        
        # Update session state if changed
        if user_dir != st.session_state.save_directory:
            st.session_state.save_directory = user_dir
        
        # Button to create the directory if it doesn't exist
        if st.button("Create Directory"):
            try:
                os.makedirs(st.session_state.save_directory, exist_ok=True)
                st.success(f"Directory created: {st.session_state.save_directory}")
            except Exception as e:
                st.error(f"Error creating directory: {str(e)}")
        
        # Navigation buttons (rest of your sidebar code)
        st.markdown("---")
        st.subheader("Navigation")
        
        if st.button("📥 Data Upload", key="nav_data"):
            st.session_state.current_section = "data_upload"
        
        if st.button("📊 Data Exploration", key="nav_explore"):
            st.session_state.current_section = "data_exploration"
        
        if st.button("📈 Statistical Analysis", key="nav_stats"):
            st.session_state.current_section = "statistical_analysis"
            
        if st.button("🎯 Threshold Analysis", key="nav_threshold"):
            st.session_state.current_section = "threshold_analysis"
            
        if st.button("🧠 Model Training", key="nav_models"):
            st.session_state.current_section = "model_training"
            
        if st.button("🔮 Make Predictions", key="nav_predict"):
            st.session_state.current_section = "predictions"
            
        if st.button("📝 Generate Report", key="nav_report"):
            st.session_state.current_section = "report_generation"

        if st.button("📁 File Management", key="nav_files"):
            st.session_state.current_section = "file_management"
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset All", key="nav_reset"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
        
        # Add version info
        st.markdown("---")
        st.caption("Statistical Analysis App v1.0.0")
        st.caption("© 2025")