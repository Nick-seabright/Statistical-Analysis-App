# streamlit_app/components/model_cards.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def display_model_card(model_name, model_type, metrics, feature_importance=None):
    """Display a model card with key metrics and information"""
    
    with st.container():
        st.markdown(f"### {model_name}")
        st.markdown(f"**Model Type**: {model_type}")
        
        # Display metrics
        cols = st.columns(len(metrics))
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            cols[i].metric(metric_name, metric_value)
        
        # Display feature importance if available
        if feature_importance is not None:
            with st.expander("Feature Importance"):
                if isinstance(feature_importance, pd.DataFrame):
                    # Sort by importance
                    sorted_importance = feature_importance.sort_values(
                        'importance', ascending=False
                    ).reset_index(drop=True)
                    
                    # Display top 10 features
                    st.dataframe(sorted_importance.head(10))
                    
                    # Create feature importance plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sorted_importance.head(10).sort_values('importance').plot(
                        kind='barh', x='feature', y='importance', ax=ax
                    )
                    plt.title('Top 10 Features by Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.write("Feature importance not available in the expected format.")
        
        st.markdown("---")