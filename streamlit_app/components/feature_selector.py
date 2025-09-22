# streamlit_app/components/feature_selector.py

import streamlit as st
import pandas as pd
from typing import List, Optional

def feature_selector(
    available_features: List[str],
    key_prefix: str = "feature_select",
    default_all: bool = True,
    min_features: int = 1,
    max_default: int = 10,
    description: Optional[str] = None
) -> List[str]:
    """
    A reusable component for feature selection
    
    Parameters:
    -----------
    available_features : List[str]
        List of all available features to choose from
    key_prefix : str
        Prefix for Streamlit widget keys to avoid duplicates
    default_all : bool
        Whether to select all features by default
    min_features : int
        Minimum number of features required
    max_default : int
        Maximum number of features to select by default if not selecting all
    description : str, optional
        Optional description text to display
    
    Returns:
    --------
    List[str] : List of selected features
    """
    if description:
        st.markdown(f"<div class='info-text'>{description}</div>", unsafe_allow_html=True)
    
    # Allow selecting all features by default or choosing specific ones
    all_features = st.checkbox(
        "Use all available features", 
        value=default_all,
        key=f"{key_prefix}_all"
    )

    if all_features:
        selected = available_features
    else:
        # Let the user select specific features
        selected = st.multiselect(
            "Select features",
            options=available_features,
            default=available_features[:min(max_default, len(available_features))],
            key=f"{key_prefix}_multi"
        )
        
        # Show warning if fewer than minimum features selected
        if len(selected) < min_features:
            st.warning(f"Please select at least {min_features} feature(s).")
            return []

    # Display the number of selected features
    st.info(f"Using {len(selected)} features.")
    
    # Return the list of selected features
    return selected