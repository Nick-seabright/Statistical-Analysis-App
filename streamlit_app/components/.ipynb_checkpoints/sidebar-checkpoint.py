# streamlit_app/components/sidebar.py

import streamlit as st
import os

# In app.py or streamlit_app/components/sidebar.py:

def create_sidebar():
    with st.sidebar:
        try:
            # Try multiple paths to find the logo
            logo_paths = [
                "streamlit_app/assets/logo.png",
                os.path.join(os.path.dirname(__file__), "streamlit_app/assets/logo.png"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "streamlit_app/assets/logo.png")
            ]
            
            logo_loaded = False
            for path in logo_paths:
                try:
                    st.image(path, width=150)
                    logo_loaded = True
                    break
                except:
                    continue
        except Exception:
            pass  # Continue without logo
            
        st.title("Navigation")
        
        # Navigation buttons
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
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset All", key="nav_reset"):
            try:
                # Save important state variables
                save_directory = st.session_state.get('save_directory', None)
                
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key != 'save_directory':
                        del st.session_state[key]
                
                # Restore saved variables
                if save_directory:
                    st.session_state.save_directory = save_directory
                
                # Initialize essential session state
                st.session_state.current_section = "data_upload"
                st.session_state.data = None
                st.session_state.processed_data = None
                
                # Use the current rerun function
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting app: {e}")
                st.warning("Please refresh the browser page to reset the application.")
        
        # Display dataset info if data is loaded
        if st.session_state.data is not None:
            st.markdown("---")
            st.subheader("Dataset Info")
            st.write(f"Rows: {st.session_state.data.shape[0]}")
            st.write(f"Columns: {st.session_state.data.shape[1]}")
            
            # Display target variable info if processed data exists
            if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
                target_column = st.session_state.processed_data.get('target_column', None)
                if target_column:
                    st.write(f"Target: **{target_column}**")
                
                # Display target type if available
                if 'target_type' in st.session_state and st.session_state.target_type:
                    target_type = st.session_state.target_type.capitalize()
                    st.write(f"Target Type: {target_type}")
                
                # Display number of features
                if 'selected_features' in st.session_state.processed_data:
                    num_features = len(st.session_state.processed_data['selected_features'])
                    st.write(f"Features: {num_features}")
                    
                    # Option to show all selected features
                    if st.checkbox("Show selected features", value=False):
                        features = st.session_state.processed_data['selected_features']
                        st.write(", ".join(features[:5]) + ("..." if len(features) > 5 else ""))
                        if len(features) > 5:
                            st.expander("All features").write(", ".join(features))
        
        # Add version info
        st.markdown("---")
        st.caption("Statistical Analysis App v1.0.0")
        st.caption("© 2025")