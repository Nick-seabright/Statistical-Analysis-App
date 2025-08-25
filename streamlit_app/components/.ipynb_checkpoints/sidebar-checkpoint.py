# streamlit_app/components/sidebar.py
import streamlit as st

def create_sidebar():
    with st.sidebar:
        st.image("streamlit_app/assets/logo.png", width=150)
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
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
        
        # Add version info
        st.markdown("---")
        st.caption("Statistical Analysis App v1.0.0")
        st.caption("© 2025")