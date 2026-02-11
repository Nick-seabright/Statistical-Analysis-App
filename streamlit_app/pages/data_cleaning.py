
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

def show_data_cleaning():
    """Data cleaning and preprocessing page"""
    
    st.markdown("<div class='subheader'>Data Cleaning & Filtering</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Clean, filter, and preprocess your data before analysis.</div>", unsafe_allow_html=True)
    
    # Check if data exists
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload or select data first.")
        if st.button("Go to Data Upload"):
            st.session_state.current_section = "data_upload"
            st.rerun()
        return
    
    # Get the data
    df = st.session_state.data.copy()
    
    # Initialize cleaned data if not exists
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = df.copy()
    
    # Display original data info
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Missing %", f"{missing_pct:.2f}%")
    
    # Create tabs for different cleaning operations
    cleaning_tabs = st.tabs([
        "🔍 Overview", 
        "🧹 Handle Missing Values", 
        "🎯 Filter Data", 
        "🔢 Handle Duplicates",
        "📊 Column Operations",
        "🔄 Transform Data"
    ])
    
    # Tab 1: Overview
    with cleaning_tabs[0]:
        show_data_overview(df)
    
    # Tab 2: Handle Missing Values
    with cleaning_tabs[1]:
        handle_missing_values(df)
    
    # Tab 3: Filter Data
    with cleaning_tabs[2]:
        filter_data(df)
    
    # Tab 4: Handle Duplicates
    with cleaning_tabs[3]:
        handle_duplicates(df)
    
    # Tab 5: Column Operations
    with cleaning_tabs[4]:
        column_operations(df)
    
    # Tab 6: Transform Data
    with cleaning_tabs[5]:
        transform_data(df)
    
    # Action buttons at the bottom
    st.markdown("---")
    st.markdown("<div class='subheader'>Actions</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Preview Cleaned Data", use_container_width=True):
            st.session_state.show_preview = True
    
    with col2:
        if st.button("💾 Apply Changes", use_container_width=True, type="primary"):
            st.session_state.data = st.session_state.cleaned_data.copy()
            # Clear processed data so it gets reprocessed with new data
            if 'processed_data' in st.session_state:
                del st.session_state.processed_data
            if 'data_types' in st.session_state:
                del st.session_state.data_types
            st.success("✅ Changes applied successfully! You can now proceed to configure and process your data.")
            st.rerun()
    
    with col3:
        if st.button("↩️ Reset to Original", use_container_width=True):
            st.session_state.cleaned_data = st.session_state.data.copy()
            st.success("Data reset to original state.")
            st.rerun()
    
    # Show preview if requested
    if st.session_state.get('show_preview', False):
        st.markdown("---")
        st.markdown("<div class='subheader'>Cleaned Data Preview</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Data:**")
            st.dataframe(df.head(10))
            st.write(f"Shape: {df.shape}")
        
        with col2:
            st.write("**Cleaned Data:**")
            st.dataframe(st.session_state.cleaned_data.head(10))
            st.write(f"Shape: {st.session_state.cleaned_data.shape}")

def show_data_overview(df):
    """Show overview of data quality"""
    st.markdown("### Data Quality Overview")
    
    # Missing values analysis
    st.markdown("#### Missing Values by Column")
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data Type': df.dtypes.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True)
        
        # Visualize missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_df.plot(x='Column', y='Missing %', kind='bar', ax=ax, color='coral')
        ax.set_title('Missing Values by Column')
        ax.set_ylabel('Missing %')
        ax.set_xlabel('Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.success("✅ No missing values detected!")
    
    # Data type distribution
    st.markdown("#### Data Types")
    dtype_counts = df.dtypes.value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(pd.DataFrame({
            'Data Type': dtype_counts.index.astype(str),
            'Count': dtype_counts.values
        }))
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(dtype_counts.values, labels=dtype_counts.index.astype(str), autopct='%1.1f%%')
        ax.set_title('Data Type Distribution')
        st.pyplot(fig)
    
    # Summary statistics
    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe(include='all').T, use_container_width=True)

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    st.markdown("### Handle Missing Values")
    
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if not missing_cols:
        st.success("✅ No missing values to handle!")
        return
    
    st.write(f"Found {len(missing_cols)} columns with missing values")
    
    # Strategy selection
    strategy = st.radio(
        "Select strategy:",
        ["Column-by-Column", "Global Strategy", "Drop Rows/Columns"]
    )
    
    if strategy == "Column-by-Column":
        selected_col = st.selectbox("Select column:", missing_cols)
        
        col_info = col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Missing:** {df[selected_col].isnull().sum()} ({df[selected_col].isnull().sum()/len(df)*100:.2f}%)")
            st.write(f"**Data Type:** {df[selected_col].dtype}")
        
        with col2:
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                fill_method = st.selectbox(
                    "Fill method:",
                    ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Custom Value", "Drop Rows"]
                )
            else:
                fill_method = st.selectbox(
                    "Fill method:",
                    ["Mode", "Forward Fill", "Backward Fill", "Custom Value", "Drop Rows"]
                )
        
        if st.button("Apply to Column", key="apply_col_missing"):
            df_cleaned = st.session_state.cleaned_data.copy()
            
            if fill_method == "Mean":
                df_cleaned[selected_col].fillna(df_cleaned[selected_col].mean(), inplace=True)
            elif fill_method == "Median":
                df_cleaned[selected_col].fillna(df_cleaned[selected_col].median(), inplace=True)
            elif fill_method == "Mode":
                df_cleaned[selected_col].fillna(df_cleaned[selected_col].mode()[0], inplace=True)
            elif fill_method == "Forward Fill":
                df_cleaned[selected_col].fillna(method='ffill', inplace=True)
            elif fill_method == "Backward Fill":
                df_cleaned[selected_col].fillna(method='bfill', inplace=True)
            elif fill_method == "Custom Value":
                custom_val = st.text_input("Enter custom value:")
                if custom_val:
                    df_cleaned[selected_col].fillna(custom_val, inplace=True)
            elif fill_method == "Drop Rows":
                df_cleaned.dropna(subset=[selected_col], inplace=True)
            
            st.session_state.cleaned_data = df_cleaned
            st.success(f"✅ Applied {fill_method} to {selected_col}")
            st.rerun()
    
    elif strategy == "Global Strategy":
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_strategy = st.selectbox(
                "Numeric columns strategy:",
                ["Mean", "Median", "Drop Rows"]
            )
        
        with col2:
            categorical_strategy = st.selectbox(
                "Categorical columns strategy:",
                ["Mode", "Custom Value", "Drop Rows"]
            )
        
        if st.button("Apply Global Strategy", key="apply_global_missing"):
            df_cleaned = st.session_state.cleaned_data.copy()
            
            # Handle numeric columns
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    if numeric_strategy == "Mean":
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    elif numeric_strategy == "Median":
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    elif numeric_strategy == "Drop Rows":
                        df_cleaned.dropna(subset=[col], inplace=True)
            
            # Handle categorical columns
            categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df_cleaned[col].isnull().any():
                    if categorical_strategy == "Mode":
                        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                    elif categorical_strategy == "Drop Rows":
                        df_cleaned.dropna(subset=[col], inplace=True)
            
            st.session_state.cleaned_data = df_cleaned
            st.success("✅ Global strategy applied successfully!")
            st.rerun()
    
    else:  # Drop Rows/Columns
        drop_option = st.radio("Drop:", ["Rows with any missing", "Rows with all missing", "Columns with missing > threshold"])
        
        if drop_option == "Columns with missing > threshold":
            threshold = st.slider("Missing % threshold:", 0, 100, 50)
            cols_to_drop = [col for col in df.columns if (df[col].isnull().sum() / len(df) * 100) > threshold]
            st.write(f"Columns to drop: {cols_to_drop}")
        
        if st.button("Apply Drop Strategy", key="apply_drop_missing"):
            df_cleaned = st.session_state.cleaned_data.copy()
            
            if drop_option == "Rows with any missing":
                df_cleaned.dropna(inplace=True)
            elif drop_option == "Rows with all missing":
                df_cleaned.dropna(how='all', inplace=True)
            else:
                df_cleaned.drop(columns=cols_to_drop, inplace=True)
            
            st.session_state.cleaned_data = df_cleaned
            st.success("✅ Drop strategy applied!")
            st.rerun()

def filter_data(df):
    """Filter data based on conditions"""
    st.markdown("### Filter Data")
    
    filter_type = st.radio("Filter by:", ["Column Values", "Row Range", "Sample Data"])
    
    if filter_type == "Column Values":
        selected_col = st.selectbox("Select column:", df.columns)
        
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            # Numeric filtering
            min_val, max_val = float(df[selected_col].min()), float(df[selected_col].max())
            
            filter_range = st.slider(
                f"Select range for {selected_col}:",
                min_val, max_val, (min_val, max_val)
            )
            
            if st.button("Apply Filter", key="apply_numeric_filter"):
                df_cleaned = st.session_state.cleaned_data.copy()
                df_cleaned = df_cleaned[(df_cleaned[selected_col] >= filter_range[0]) & 
                                       (df_cleaned[selected_col] <= filter_range[1])]
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Filtered {selected_col} to range [{filter_range[0]}, {filter_range[1]}]")
                st.rerun()
        else:
            # Categorical filtering
            unique_vals = df[selected_col].unique().tolist()
            selected_vals = st.multiselect(
                f"Select values to keep for {selected_col}:",
                unique_vals,
                default=unique_vals
            )
            
            if st.button("Apply Filter", key="apply_categorical_filter"):
                df_cleaned = st.session_state.cleaned_data.copy()
                df_cleaned = df_cleaned[df_cleaned[selected_col].isin(selected_vals)]
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Filtered {selected_col} to selected values")
                st.rerun()
    
    elif filter_type == "Row Range":
        total_rows = len(df)
        
        start_row = st.number_input("Start row:", min_value=0, max_value=total_rows-1, value=0)
        end_row = st.number_input("End row:", min_value=start_row+1, max_value=total_rows, value=min(total_rows, 1000))
        
        if st.button("Apply Row Filter", key="apply_row_filter"):
            df_cleaned = st.session_state.cleaned_data.copy()
            df_cleaned = df_cleaned.iloc[start_row:end_row]
            st.session_state.cleaned_data = df_cleaned
            st.success(f"✅ Filtered to rows {start_row} to {end_row}")
            st.rerun()
    
    else:  # Sample Data
        sample_method = st.radio("Sampling method:", ["Random Sample", "First N Rows", "Last N Rows"])
        
        if sample_method == "Random Sample":
            sample_size = st.slider("Sample size:", 100, len(df), min(len(df), 1000))
            random_seed = st.number_input("Random seed:", value=42)
            
            if st.button("Apply Sampling", key="apply_random_sample"):
                df_cleaned = df.sample(n=sample_size, random_state=random_seed)
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Random sample of {sample_size} rows created")
                st.rerun()
        
        elif sample_method == "First N Rows":
            n_rows = st.slider("Number of rows:", 100, len(df), min(len(df), 1000))
            
            if st.button("Apply Sampling", key="apply_first_n"):
                df_cleaned = df.head(n_rows)
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ First {n_rows} rows selected")
                st.rerun()
        
        else:  # Last N Rows
            n_rows = st.slider("Number of rows:", 100, len(df), min(len(df), 1000))
            
            if st.button("Apply Sampling", key="apply_last_n"):
                df_cleaned = df.tail(n_rows)
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Last {n_rows} rows selected")
                st.rerun()

def handle_duplicates(df):
    """Handle duplicate rows"""
    st.markdown("### Handle Duplicates")
    
    # Check for duplicates
    total_duplicates = df.duplicated().sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Duplicate Rows", f"{total_duplicates:,}")
    with col2:
        dup_pct = (total_duplicates / len(df) * 100) if len(df) > 0 else 0
        st.metric("Duplicate %", f"{dup_pct:.2f}%")
    
    if total_duplicates == 0:
        st.success("✅ No duplicate rows found!")
        return
    
    # Duplicate detection options
    duplicate_subset = st.multiselect(
        "Check duplicates based on columns (leave empty for all columns):",
        df.columns.tolist()
    )
    
    keep_option = st.radio(
        "Which duplicates to keep:",
        ["First occurrence", "Last occurrence", "Remove all duplicates"]
    )
    
    # Preview duplicates
    if st.button("Preview Duplicates", key="preview_duplicates"):
        if duplicate_subset:
            dup_rows = df[df.duplicated(subset=duplicate_subset, keep=False)]
        else:
            dup_rows = df[df.duplicated(keep=False)]
        
        st.write(f"Found {len(dup_rows)} duplicate rows:")
        st.dataframe(dup_rows.head(20))
    
    # Remove duplicates
    if st.button("Remove Duplicates", key="remove_duplicates", type="primary"):
        df_cleaned = st.session_state.cleaned_data.copy()
        
        keep_param = 'first' if keep_option == "First occurrence" else 'last' if keep_option == "Last occurrence" else False
        
        if duplicate_subset:
            df_cleaned.drop_duplicates(subset=duplicate_subset, keep=keep_param, inplace=True)
        else:
            df_cleaned.drop_duplicates(keep=keep_param, inplace=True)
        
        removed = len(st.session_state.cleaned_data) - len(df_cleaned)
        st.session_state.cleaned_data = df_cleaned
        st.success(f"✅ Removed {removed:,} duplicate rows!")
        st.rerun()

def column_operations(df):
    """Column-level operations"""
    st.markdown("### Column Operations")
    
    operation = st.selectbox(
        "Select operation:",
        ["Drop Columns", "Rename Columns", "Reorder Columns", "Create New Column"]
    )
    
    if operation == "Drop Columns":
        cols_to_drop = st.multiselect("Select columns to drop:", df.columns.tolist())
        
        if cols_to_drop and st.button("Drop Selected Columns", key="drop_cols"):
            df_cleaned = st.session_state.cleaned_data.copy()
            df_cleaned.drop(columns=cols_to_drop, inplace=True)
            st.session_state.cleaned_data = df_cleaned
            st.success(f"✅ Dropped {len(cols_to_drop)} columns")
            st.rerun()
    
    elif operation == "Rename Columns":
        col_to_rename = st.selectbox("Select column to rename:", df.columns.tolist())
        new_name = st.text_input("New column name:", value=col_to_rename)
        
        if new_name and new_name != col_to_rename and st.button("Rename Column", key="rename_col"):
            df_cleaned = st.session_state.cleaned_data.copy()
            df_cleaned.rename(columns={col_to_rename: new_name}, inplace=True)
            st.session_state.cleaned_data = df_cleaned
            st.success(f"✅ Renamed '{col_to_rename}' to '{new_name}'")
            st.rerun()
    
    elif operation == "Reorder Columns":
        st.write("Drag and drop to reorder (or use the input below):")
        
        # Simple text-based reordering
        current_order = df.columns.tolist()
        st.write("Current order:", current_order)
        
        new_order = st.multiselect(
            "Select columns in desired order:",
            current_order,
            default=current_order
        )
        
        if len(new_order) == len(current_order) and st.button("Apply New Order", key="reorder_cols"):
            df_cleaned = st.session_state.cleaned_data.copy()
            df_cleaned = df_cleaned[new_order]
            st.session_state.cleaned_data = df_cleaned
            st.success("✅ Column order updated")
            st.rerun()
    
    else:  # Create New Column
        st.markdown("Create a new column based on existing columns")
        
        new_col_name = st.text_input("New column name:")
        
        creation_method = st.radio("Creation method:", ["Arithmetic Operation", "Conditional Logic", "Copy Column"])
        
        if creation_method == "Arithmetic Operation":
            col1 = st.selectbox("Select first column:", df.select_dtypes(include=[np.number]).columns.tolist(), key="col1_arith")
            operation_type = st.selectbox("Operation:", ["+", "-", "*", "/"])
            col2 = st.selectbox("Select second column:", df.select_dtypes(include=[np.number]).columns.tolist(), key="col2_arith")
            
            if new_col_name and st.button("Create Column", key="create_arith_col"):
                df_cleaned = st.session_state.cleaned_data.copy()
                
                if operation_type == "+":
                    df_cleaned[new_col_name] = df_cleaned[col1] + df_cleaned[col2]
                elif operation_type == "-":
                    df_cleaned[new_col_name] = df_cleaned[col1] - df_cleaned[col2]
                elif operation_type == "*":
                    df_cleaned[new_col_name] = df_cleaned[col1] * df_cleaned[col2]
                elif operation_type == "/":
                    df_cleaned[new_col_name] = df_cleaned[col1] / df_cleaned[col2]
                
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Created column '{new_col_name}'")
                st.rerun()
        
        elif creation_method == "Copy Column":
            source_col = st.selectbox("Select column to copy:", df.columns.tolist())
            
            if new_col_name and st.button("Create Column", key="create_copy_col"):
                df_cleaned = st.session_state.cleaned_data.copy()
                df_cleaned[new_col_name] = df_cleaned[source_col].copy()
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Created column '{new_col_name}' as copy of '{source_col}'")
                st.rerun()

def transform_data(df):
    """Transform data values"""
    st.markdown("### Transform Data")
    
    selected_col = st.selectbox("Select column to transform:", df.columns.tolist())
    
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        transform_type = st.selectbox(
            "Transform type:",
            ["Normalize (0-1)", "Standardize (Z-score)", "Log Transform", "Square Root", "Box-Cox"]
        )
    else:
        transform_type = st.selectbox(
            "Transform type:",
            ["Lowercase", "Uppercase", "Title Case", "Strip Whitespace", "Replace Values"]
        )
    
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        # Numeric transformations
        if st.button("Apply Transform", key="apply_numeric_transform"):
            df_cleaned = st.session_state.cleaned_data.copy()
            
            try:
                if transform_type == "Normalize (0-1)":
                    min_val = df_cleaned[selected_col].min()
                    max_val = df_cleaned[selected_col].max()
                    df_cleaned[selected_col] = (df_cleaned[selected_col] - min_val) / (max_val - min_val)
                
                elif transform_type == "Standardize (Z-score)":
                    mean_val = df_cleaned[selected_col].mean()
                    std_val = df_cleaned[selected_col].std()
                    df_cleaned[selected_col] = (df_cleaned[selected_col] - mean_val) / std_val
                
                elif transform_type == "Log Transform":
                    df_cleaned[selected_col] = np.log1p(df_cleaned[selected_col])
                
                elif transform_type == "Square Root":
                    df_cleaned[selected_col] = np.sqrt(df_cleaned[selected_col])
                
                elif transform_type == "Box-Cox":
                    from scipy import stats
                    df_cleaned[selected_col], _ = stats.boxcox(df_cleaned[selected_col] + 1)
                
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Applied {transform_type} to {selected_col}")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error applying transform: {str(e)}")
    
    else:
        # String transformations
        if transform_type == "Replace Values":
            old_value = st.text_input("Value to replace:")
            new_value = st.text_input("Replace with:")
            
            if st.button("Apply Replace", key="apply_replace"):
                df_cleaned = st.session_state.cleaned_data.copy()
                df_cleaned[selected_col] = df_cleaned[selected_col].replace(old_value, new_value)
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Replaced '{old_value}' with '{new_value}' in {selected_col}")
                st.rerun()
        
        else:
            if st.button("Apply Transform", key="apply_string_transform"):
                df_cleaned = st.session_state.cleaned_data.copy()
                
                if transform_type == "Lowercase":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.lower()
                elif transform_type == "Uppercase":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.upper()
                elif transform_type == "Title Case":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.title()
                elif transform_type == "Strip Whitespace":
                    df_cleaned[selected_col] = df_cleaned[selected_col].astype(str).str.strip()
                
                st.session_state.cleaned_data = df_cleaned
                st.success(f"✅ Applied {transform_type} to {selected_col}")
                st.rerun()

if __name__ == "__main__":
    show_data_cleaning()
