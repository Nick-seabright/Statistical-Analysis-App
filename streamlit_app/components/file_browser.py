# streamlit_app/components/file_browser.py

import streamlit as st
import os
import pandas as pd
from datetime import datetime

def file_browser(directory_path, file_extension=None):
    """
    Display a simple file browser for the specified directory
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory to browse
    file_extension : str, optional
        Filter files by extension (e.g., 'pdf')
    
    Returns:
    --------
    str or None : Selected file path or None if no file selected
    """
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            st.warning(f"Directory does not exist: {directory_path}")
            return None
        
        # Get list of files
        files = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            # Filter by extension if specified
            if file_extension and not filename.endswith(f".{file_extension}"):
                continue
                
            # Get file info
            file_info = os.stat(file_path)
            files.append({
                'filename': filename,
                'path': file_path,
                'size_kb': file_info.st_size / 1024,
                'modified': datetime.fromtimestamp(file_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Create DataFrame
        if not files:
            st.info(f"No {file_extension or ''} files found in {directory_path}")
            return None
            
        files_df = pd.DataFrame(files)
        
        # Display files
        st.dataframe(files_df[['filename', 'size_kb', 'modified']])
        
        # Let user select a file
        selected_file = st.selectbox("Select a file:", files_df['filename'])
        
        if selected_file:
            selected_path = os.path.join(directory_path, selected_file)
            return selected_path
        
    except Exception as e:
        st.error(f"Error browsing files: {str(e)}")
        return None