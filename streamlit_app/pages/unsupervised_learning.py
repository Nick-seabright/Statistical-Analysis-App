import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import traceback

# Add the parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from edu_analytics.unsupervised_learning import (
    train_clustering_model,
    find_optimal_clusters,
    apply_dimensionality_reduction,
    calculate_optimal_components,
    detect_anomalies,
    compare_anomaly_methods,
    cluster_profiling,
    calculate_cluster_separation
)

def show_unsupervised_learning():
    """Main unsupervised learning page"""
    
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload or select data first.")
        if st.button("Go to Data Upload"):
            st.session_state.current_section = "data_upload"
            st.rerun()
        return
    
    # Check if data is processed
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("⚠️ Please process your data first.")
        if st.button("Go to Data Upload"):
            st.session_state.current_section = "data_upload"
            st.rerun()
        return
    
    st.markdown("<div class='subheader'>Unsupervised Learning</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Discover patterns, reduce dimensions, and detect anomalies in your data.</div>", unsafe_allow_html=True)
    
    # Get data
    X = st.session_state.processed_data['X']
    selected_features = st.session_state.processed_data['selected_features']
    
    # Create main tabs
    main_tabs = st.tabs(["🔵 Clustering", "📉 Dimensionality Reduction", "⚠️ Anomaly Detection", "📊 Results Summary"])
    
    with main_tabs[0]:
        show_clustering_tab(X, selected_features)
    
    with main_tabs[1]:
        show_dimensionality_reduction_tab(X, selected_features)
    
    with main_tabs[2]:
        show_anomaly_detection_tab(X, selected_features)
    
    with main_tabs[3]:
        show_results_summary()


def show_clustering_tab(X, selected_features):
    """Clustering analysis tab"""
    
    st.markdown("### Clustering Analysis")
    st.markdown("Group similar data points together to discover natural patterns in your data.")
    
    # Create sub-tabs for different clustering options
    cluster_tabs = st.tabs(["Configure & Train", "Find Optimal K", "Compare Methods"])
    
    with cluster_tabs[0]:
        show_clustering_training(X, selected_features)
    
    with cluster_tabs[1]:
        show_optimal_k_finder(X, selected_features)
    
    with cluster_tabs[2]:
        show_clustering_comparison(X, selected_features)


def show_clustering_training(X, selected_features):
    """Train a single clustering model"""
    
    st.markdown("#### Select Clustering Algorithm")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        algorithm = st.selectbox(
            "Algorithm",
            ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture", "Mean Shift", "Spectral"],
            help="Choose the clustering algorithm to use"
        )
        
        scale_data = st.checkbox("Scale data before clustering", value=True,
                                help="Recommended for most algorithms")
    
    with col2:
        st.markdown("**Algorithm Description:**")
        
        descriptions = {
            "K-Means": "🎯 **Partitioning Method** - Divides data into K spherical clusters by minimizing within-cluster variance. **Fast and scalable** for large datasets. **Pros:** Simple, efficient, works well when clusters are spherical and similar-sized. **Cons:** Needs K specified, sensitive to outliers. **Best for:** Customer segmentation, image compression, general exploratory analysis.",
            "DBSCAN": "🔍 **Density-Based** - Groups points that are closely packed together, marking outliers as noise. **Discovers arbitrary-shaped clusters**. **Pros:** No need to specify number of clusters, finds outliers automatically, handles non-spherical clusters. **Cons:** Struggles with varying densities, sensitive to parameters. **Best for:** Spatial data, anomaly detection, when cluster shapes are unknown.",
            "Hierarchical": "🌳 **Tree-Based** - Builds a dendrogram (tree) of nested clusters that can be cut at any level. **Pros:** No need to prespecify K, provides hierarchy visualization, deterministic results. **Cons:** Computationally expensive O(n³), doesn't scale well. **Best for:** Small-to-medium datasets, when you need cluster hierarchy, taxonomic analysis.",
            "Gaussian Mixture": "📊 **Probabilistic Model** - Assumes data comes from a mixture of Gaussian distributions. **Provides soft cluster assignments** (probabilities). **Pros:** Flexible cluster shapes (elliptical), probabilistic interpretation, handles overlapping clusters. **Cons:** Sensitive to initialization, assumes Gaussian distribution. **Best for:** When clusters overlap, need uncertainty estimates, data follows normal distributions.",
            "Mean Shift": "📍 **Mode-Seeking** - Finds clusters by locating centers of dense regions (modes). **Automatically determines cluster count**. **Pros:** No need to specify K, robust to outliers, no assumptions on cluster shape. **Cons:** Computationally expensive, sensitive to bandwidth parameter. **Best for:** Computer vision, tracking, when cluster number is unknown.",
            "Spectral": "🌈 **Graph-Based** - Uses eigenvalues of similarity matrix to reduce dimensions before clustering. **Excels with complex cluster shapes**. **Pros:** Handles non-convex clusters well, works when K-Means fails. **Cons:** Memory-intensive O(n²), needs K specified, scaling issues. **Best for:** Image segmentation, community detection in networks, non-convex clusters."
        }
        
        st.info(descriptions[algorithm])
    
    # Algorithm-specific parameters
    st.markdown("#### Configure Parameters")
    
    params = {}
    
    if algorithm == "K-Means":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_clusters'] = st.slider("Number of clusters", 2, 10, 3)
        with col2:
            params['init'] = st.selectbox("Initialization method", ['k-means++', 'random'])
        with col3:
            params['n_init'] = st.slider("Number of initializations", 1, 20, 10)
        params['max_iter'] = st.slider("Max iterations", 100, 1000, 300)
        
    elif algorithm == "DBSCAN":
        col1, col2 = st.columns(2)
        with col1:
            params['eps'] = st.slider("Epsilon (neighborhood size)", 0.1, 5.0, 0.5, 0.1,
                                     help="Maximum distance between two samples to be considered neighbors")
        with col2:
            params['min_samples'] = st.slider("Min samples", 2, 20, 5,
                                             help="Minimum number of samples in a neighborhood to form a cluster")
        params['metric'] = st.selectbox("Distance metric", ['euclidean', 'manhattan', 'cosine'])
        
    elif algorithm == "Hierarchical":
        col1, col2 = st.columns(2)
        with col1:
            params['n_clusters'] = st.slider("Number of clusters", 2, 10, 3)
        with col2:
            params['linkage'] = st.selectbox("Linkage method", ['ward', 'complete', 'average', 'single'])
        if params['linkage'] != 'ward':
            params['metric'] = st.selectbox("Distance metric", ['euclidean', 'manhattan', 'cosine'])
        else:
            params['metric'] = 'euclidean'
            
    elif algorithm == "Gaussian Mixture":
        col1, col2 = st.columns(2)
        with col1:
            params['n_components'] = st.slider("Number of components", 2, 10, 3)
        with col2:
            params['covariance_type'] = st.selectbox("Covariance type", ['full', 'tied', 'diag', 'spherical'])
        params['max_iter'] = st.slider("Max iterations", 50, 200, 100)
        
    elif algorithm == "Mean Shift":
        estimate_bandwidth = st.checkbox("Estimate bandwidth automatically", value=True)
        if not estimate_bandwidth:
            params['bandwidth'] = st.slider("Bandwidth", 0.5, 5.0, 2.0, 0.1)
        else:
            params['bandwidth'] = None
        params['bin_seeding'] = st.checkbox("Use bin seeding", value=False)
        
    elif algorithm == "Spectral":
        col1, col2 = st.columns(2)
        with col1:
            params['n_clusters'] = st.slider("Number of clusters", 2, 10, 3)
        with col2:
            params['affinity'] = st.selectbox("Affinity", ['rbf', 'nearest_neighbors'])
    
    params['random_state'] = 42
    
    # Train button
    if st.button("Train Clustering Model", type="primary"):
        try:
            with st.spinner(f"Training {algorithm} model..."):
                # Map algorithm name to model type
                model_map = {
                    "K-Means": "kmeans",
                    "DBSCAN": "dbscan",
                    "Hierarchical": "hierarchical",
                    "Gaussian Mixture": "gmm",
                    "Mean Shift": "meanshift",
                    "Spectral": "spectral"
                }
                
                model, labels, metrics = train_clustering_model(
                    X, model_map[algorithm], params, scale_data
                )
                
                # Store results
                if 'clustering_results' not in st.session_state:
                    st.session_state.clustering_results = {}
                
                st.session_state.clustering_results[algorithm] = {
                    'model': model,
                    'labels': labels,
                    'metrics': metrics,
                    'algorithm': algorithm,
                    'params': params,
                    'scale_data': scale_data
                }
                
                st.success(f"✅ Successfully trained {algorithm} model!")
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            st.code(traceback.format_exc())
    
    # Always display results if they exist for this algorithm
    if 'clustering_results' in st.session_state and algorithm in st.session_state.clustering_results:
        result = st.session_state.clustering_results[algorithm]
        st.markdown("---")
        st.markdown(f"### Current Results for {algorithm}")
        scale_data = result.get('scale_data', True)
        display_clustering_results(X, result['labels'], result['metrics'], algorithm, 
                                   selected_features, scale_data)


def display_clustering_results(X, labels, metrics, algorithm, selected_features, scale_data=True):
    """Display clustering results"""
    
    st.markdown("---")
    st.markdown("### Results")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Clusters", metrics.get('n_clusters', 'N/A'))
    
    with col2:
        if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}",
                     help="Range: -1 to 1. Higher is better. Measures how similar points are to their own cluster vs other clusters.")
    
    with col3:
        if 'davies_bouldin_score' in metrics and metrics['davies_bouldin_score'] is not None:
            st.metric("Davies-Bouldin Index", f"{metrics['davies_bouldin_score']:.3f}",
                     help="Lower is better. Measures average similarity between clusters.")
    
    # Additional metrics
    if 'calinski_harabasz_score' in metrics and metrics['calinski_harabasz_score'] is not None:
        st.metric("Calinski-Harabasz Score", f"{metrics['calinski_harabasz_score']:.2f}",
                 help="Higher is better. Ratio of between-cluster to within-cluster variance.")
    
    if 'inertia' in metrics:
        st.metric("Inertia", f"{metrics['inertia']:.2f}",
                 help="Sum of squared distances to nearest cluster center. Lower is better.")
    
    if 'bic' in metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BIC", f"{metrics['bic']:.2f}", help="Bayesian Information Criterion. Lower is better.")
        with col2:
            st.metric("AIC", f"{metrics['aic']:.2f}", help="Akaike Information Criterion. Lower is better.")
    
    if 'noise_percentage' in metrics:
        st.metric("Noise Points", f"{metrics['noise_percentage']:.2f}%",
                 help="Percentage of points classified as noise (DBSCAN)")
    
    # Cluster sizes
    st.markdown("#### Cluster Sizes")
    cluster_sizes = metrics.get('cluster_sizes', {})
    sizes_df = pd.DataFrame({
        'Cluster': list(cluster_sizes.keys()),
        'Size': list(cluster_sizes.values()),
        'Percentage': [v / sum(cluster_sizes.values()) * 100 for v in cluster_sizes.values()]
    })
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(sizes_df)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(sizes_df['Cluster'].astype(str), sizes_df['Size'])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Points')
        ax.set_title('Cluster Size Distribution')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Visualizations
    visualize_clusters(X, labels, selected_features, algorithm, scale_data)
    
    # Cluster profiling
    st.markdown("#### Cluster Profiles")
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
    📊 <b>What is a Cluster Profile?</b><br>
    A cluster profile shows the <b>average characteristics</b> of each cluster. Each row represents one cluster, 
    and each column shows the mean value of a feature for that cluster. This helps you understand what makes each 
    cluster unique and how they differ from each other.
    </div>
    """, unsafe_allow_html=True)
    
    # Get original data from session state
    X_original = st.session_state.data[selected_features]
    
    # Generate profiles for both scaled and original data
    if scale_data:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        profiles_result = cluster_profiling(X_scaled, labels, selected_features, X_original)
        
        if isinstance(profiles_result, tuple):
            profiles_scaled, profiles_original = profiles_result
            
            # Show option to view scaled or original values
            profile_view = st.radio(
                "Select view:",
                ["Scaled Values (Standardized)", "Original Values", "Comparison View"],
                horizontal=True,
                help="Scaled values show standardized data (mean=0, std=1). Original values show actual data ranges."
            )
            
            if profile_view == "Scaled Values (Standardized)":
                st.info("📏 **Scaled Values**: Data is standardized with mean=0 and standard deviation=1. Most values fall between -3 and +3.")
                st.dataframe(profiles_scaled.style.background_gradient(cmap='RdYlGn', axis=1))
                csv = profiles_scaled.to_csv(index=True)
                
            elif profile_view == "Original Values":
                st.info("📊 **Original Values**: Data shown in its original scale and units.")
                st.dataframe(profiles_original.style.background_gradient(cmap='RdYlGn', axis=1))
                csv = profiles_original.to_csv(index=True)
                
            else:  # Comparison View
                st.info("🔄 **Comparison**: Side-by-side view of scaled and original values.")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Scaled Values**")
                    st.dataframe(profiles_scaled.style.background_gradient(cmap='RdYlGn', axis=1))
                with col2:
                    st.markdown("**Original Values**")
                    st.dataframe(profiles_original.style.background_gradient(cmap='RdYlGn', axis=1))
                csv = profiles_scaled.to_csv(index=True) + "\n\n" + profiles_original.to_csv(index=True)
        else:
            profiles = profiles_result
            st.dataframe(profiles.style.background_gradient(cmap='RdYlGn', axis=1))
            csv = profiles.to_csv(index=True)
    else:
        profiles = cluster_profiling(X, labels, selected_features)
        st.info("📊 Data was not scaled during clustering.")
        st.dataframe(profiles.style.background_gradient(cmap='RdYlGn', axis=1))
        csv = profiles.to_csv(index=True)
    
    # Download results
    st.download_button(
        "📥 Download Cluster Profiles",
        csv,
        f"cluster_profiles_{algorithm}.csv",
        "text/csv"
    )


def visualize_clusters(X, labels, feature_names, algorithm, scale_data=True):
    """Visualize clustering results"""
    
    st.markdown("#### Cluster Visualization")
    
    n_features = X.shape[1]
    
    if n_features >= 2:
        # Get original data for comparison
        X_original = st.session_state.data[feature_names]
        
        # Add option to view scaled or original values
        if scale_data:
            viz_data_view = st.radio(
                "Data view for visualization:",
                ["Scaled Values (Used for Clustering)", "Original Values", "Side-by-Side Comparison"],
                horizontal=True,
                help="Scaled values show the standardized data used for clustering. Original values show actual data ranges."
            )
        else:
            viz_data_view = "Original Values"
            st.info("📊 Data was not scaled during clustering, showing original values.")
        
        # Prepare data based on selection
        if scale_data:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            
            if viz_data_view == "Scaled Values (Used for Clustering)":
                X_display = X_scaled
                data_label = " (Scaled)"
            elif viz_data_view == "Original Values":
                X_display = X_original
                data_label = " (Original)"
            else:  # Side-by-Side Comparison
                X_display = None  # Will show both
                data_label = ""
        else:
            X_display = X
            data_label = ""
        
        # 2D or 3D scatter plot
        viz_type = st.radio("Visualization type", ["2D Scatter", "3D Scatter (if 3+ features)", "Pair Plot"], horizontal=True)
        
        if viz_type == "2D Scatter":
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis feature", feature_names, index=0)
            with col2:
                y_feature = st.selectbox("Y-axis feature", feature_names, index=min(1, len(feature_names)-1))
            
            x_idx = feature_names.index(x_feature)
            y_idx = feature_names.index(y_feature)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            # Show side-by-side comparison if selected
            if scale_data and viz_data_view == "Side-by-Side Comparison":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Scaled plot
                scatter1 = ax1.scatter(X_scaled.iloc[:, x_idx], X_scaled.iloc[:, y_idx], 
                                      c=labels, cmap='viridis', alpha=0.6, s=50)
                ax1.set_xlabel(f"{x_feature} (Scaled)")
                ax1.set_ylabel(f"{y_feature} (Scaled)")
                ax1.set_title(f'{algorithm} - Scaled Values')
                plt.colorbar(scatter1, ax=ax1, label='Cluster')
                
                # Original plot
                scatter2 = ax2.scatter(X_original.iloc[:, x_idx], X_original.iloc[:, y_idx], 
                                      c=labels, cmap='viridis', alpha=0.6, s=50)
                ax2.set_xlabel(f"{x_feature} (Original)")
                ax2.set_ylabel(f"{y_feature} (Original)")
                ax2.set_title(f'{algorithm} - Original Values')
                plt.colorbar(scatter2, ax=ax2, label='Cluster')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                # Single plot
                fig, ax = plt.subplots(figsize=(10, 7))
                scatter = ax.scatter(X_display.iloc[:, x_idx], X_display.iloc[:, y_idx], 
                                   c=labels, cmap='viridis', alpha=0.6, s=50)
                ax.set_xlabel(x_feature + data_label)
                ax.set_ylabel(y_feature + data_label)
                ax.set_title(f'{algorithm} Clustering Results{data_label}')
                plt.colorbar(scatter, ax=ax, label='Cluster')
                plt.tight_layout()
                st.pyplot(fig)
        elif viz_type == "3D Scatter (if 3+ features)" and n_features >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_feature = st.selectbox("X-axis feature", feature_names, index=0)
            with col2:
                y_feature = st.selectbox("Y-axis feature", feature_names, index=1)
            with col3:
                z_feature = st.selectbox("Z-axis feature", feature_names, index=2)
            
            x_idx = feature_names.index(x_feature)
            y_idx = feature_names.index(y_feature)
            z_idx = feature_names.index(z_feature)
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X.iloc[:, x_idx], X.iloc[:, y_idx], X.iloc[:, z_idx], 
                               c=labels, cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            # Show side-by-side comparison if selected
            if scale_data and viz_data_view == "Side-by-Side Comparison":
                fig = plt.figure(figsize=(18, 7))
                
                # Scaled plot
                ax1 = fig.add_subplot(121, projection='3d')
                scatter1 = ax1.scatter(X_scaled.iloc[:, x_idx], X_scaled.iloc[:, y_idx], 
                                      X_scaled.iloc[:, z_idx],
                                      c=labels, cmap='viridis', alpha=0.6, s=50)
                ax1.set_xlabel(f"{x_feature} (Scaled)")
                ax1.set_ylabel(f"{y_feature} (Scaled)")
                ax1.set_zlabel(f"{z_feature} (Scaled)")
                ax1.set_title(f'{algorithm} - Scaled Values')
                
                # Original plot
                ax2 = fig.add_subplot(122, projection='3d')
                scatter2 = ax2.scatter(X_original.iloc[:, x_idx], X_original.iloc[:, y_idx], 
                                      X_original.iloc[:, z_idx],
                                      c=labels, cmap='viridis', alpha=0.6, s=50)
                ax2.set_xlabel(f"{x_feature} (Original)")
                ax2.set_ylabel(f"{y_feature} (Original)")
                ax2.set_zlabel(f"{z_feature} (Original)")
                ax2.set_title(f'{algorithm} - Original Values')
                
                # Add colorbars
                fig.colorbar(scatter1, ax=ax1, label='Cluster', pad=0.1, shrink=0.8)
                fig.colorbar(scatter2, ax=ax2, label='Cluster', pad=0.1, shrink=0.8)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                # Single plot
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(X_display.iloc[:, x_idx], X_display.iloc[:, y_idx], 
                                   X_display.iloc[:, z_idx], 
                                   c=labels, cmap='viridis', alpha=0.6, s=50)
                ax.set_xlabel(x_feature + data_label)
                ax.set_ylabel(y_feature + data_label)
                ax.set_zlabel(z_feature + data_label)
                ax.set_title(f'{algorithm} Clustering Results{data_label} (3D)')
                plt.colorbar(scatter, ax=ax, label='Cluster')
                st.pyplot(fig)
        
        elif viz_type == "Pair Plot":
            st.info("📊 Showing pairwise relationships between features colored by cluster.")
            # Limit number of features for pair plot to avoid performance issues
            max_features = min(5, n_features)
            if n_features > max_features:
                st.warning(f"⚠️ Showing only first {max_features} features for performance. Your data has {n_features} features.")
            
            # Show side-by-side comparison if selected
            if scale_data and viz_data_view == "Side-by-Side Comparison":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Scaled Values**")
                    df_scaled = pd.DataFrame(X_scaled.iloc[:, :max_features], 
                                           columns=feature_names[:max_features])
                    df_scaled['Cluster'] = labels
                    fig1 = sns.pairplot(df_scaled, hue='Cluster', palette='viridis', 
                                       diag_kind='kde', corner=True, height=2)
                    st.pyplot(fig1)
                
                with col2:
                    st.markdown("**Original Values**")
                    df_original_display = pd.DataFrame(X_original.iloc[:, :max_features], 
                                             columns=feature_names[:max_features])
                    df_original_display['Cluster'] = labels
                    fig2 = sns.pairplot(df_original_display, hue='Cluster', palette='viridis', 
                                       diag_kind='kde', corner=True, height=2)
                    st.pyplot(fig2)
            else:
                df_display = pd.DataFrame(X_display.iloc[:, :max_features], 
                                        columns=feature_names[:max_features])
                df_display['Cluster'] = labels
                fig = sns.pairplot(df_display, hue='Cluster', palette='viridis', 
                                 diag_kind='kde', corner=True)
                st.pyplot(fig)
    else:
        st.warning("⚠️ Need at least 2 features for visualization")


def show_optimal_k_finder(X, selected_features):
    """Find optimal number of clusters"""
    
    st.markdown("#### Find Optimal Number of Clusters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        method = st.selectbox(
            "Optimization method",
            ["Elbow Method", "Silhouette Analysis", "BIC (Gaussian Mixture)"],
            help="Method to determine optimal number of clusters"
        )
    
    with col2:
        k_min = st.number_input("Minimum K", 2, 10, 2)
    
    with col3:
        k_max = st.number_input("Maximum K", k_min + 1, 20, 10)
    
    scale_data = st.checkbox("Scale data", value=True, key="optimal_k_scale")
    
    if st.button("Find Optimal K", type="primary"):
        try:
            with st.spinner("Analyzing different values of K..."):
                method_map = {
                    "Elbow Method": "elbow",
                    "Silhouette Analysis": "silhouette",
                    "BIC (Gaussian Mixture)": "bic"
                }
                
                results = find_optimal_clusters(
                    X, method_map[method], (k_min, k_max), scale_data
                )
                
                st.success(f"✅ Optimal K: **{results['optimal_k']}**")
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results['k_values'], results['scores'], 'bo-', linewidth=2, markersize=8)
                ax.axvline(x=results['optimal_k'], color='r', linestyle='--', 
                          label=f"Optimal K = {results['optimal_k']}")
                ax.set_xlabel('Number of Clusters (K)')
                ax.set_ylabel('Score')
                ax.set_title(f'{method} Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show scores table
                scores_df = pd.DataFrame({
                    'K': results['k_values'],
                    'Score': results['scores']
                })
                st.dataframe(scores_df)
                
        except Exception as e:
            st.error(f"Error finding optimal K: {str(e)}")
            st.code(traceback.format_exc())


def show_clustering_comparison(X, selected_features):
    """Compare different clustering algorithms"""
    
    st.markdown("#### Compare Clustering Algorithms")
    st.markdown("Run multiple algorithms and compare their performance.")
    
    algorithms = st.multiselect(
        "Select algorithms to compare",
        ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"],
        default=["K-Means", "Hierarchical"]
    )
    
    if "K-Means" in algorithms or "Hierarchical" in algorithms or "Gaussian Mixture" in algorithms:
        n_clusters = st.slider("Number of clusters (for applicable algorithms)", 2, 10, 3)
    
    scale_data = st.checkbox("Scale data", value=True, key="clustering_compare_scale")
    
    if st.button("Compare Algorithms", type="primary", key="compare_clustering_btn") and algorithms:
        try:
            with st.spinner("Training and comparing algorithms..."):
                comparison_results = {}
                
                for algo in algorithms:
                    # Set default parameters
                    if algo == "K-Means":
                        params = {'n_clusters': n_clusters, 'random_state': 42}
                        model_type = 'kmeans'
                    elif algo == "DBSCAN":
                        params = {'eps': 0.5, 'min_samples': 5}
                        model_type = 'dbscan'
                    elif algo == "Hierarchical":
                        params = {'n_clusters': n_clusters, 'linkage': 'ward'}
                        model_type = 'hierarchical'
                    elif algo == "Gaussian Mixture":
                        params = {'n_components': n_clusters, 'random_state': 42}
                        model_type = 'gmm'
                    
                    model, labels, metrics = train_clustering_model(X, model_type, params, scale_data)
                    comparison_results[algo] = {'labels': labels, 'metrics': metrics}
                
                # Display comparison
                st.markdown("### Comparison Results")
                
                # Metrics comparison table
                metrics_data = []
                for algo, result in comparison_results.items():
                    metrics = result['metrics']
                    row = {'Algorithm': algo}
                    row['N Clusters'] = metrics.get('n_clusters', 'N/A')
                    row['Silhouette'] = f"{metrics.get('silhouette_score', 0):.3f}" if metrics.get('silhouette_score') else 'N/A'
                    row['Davies-Bouldin'] = f"{metrics.get('davies_bouldin_score', 0):.3f}" if metrics.get('davies_bouldin_score') else 'N/A'
                    row['Calinski-Harabasz'] = f"{metrics.get('calinski_harabasz_score', 0):.2f}" if metrics.get('calinski_harabasz_score') else 'N/A'
                    metrics_data.append(row)
                
                comparison_df = pd.DataFrame(metrics_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualize comparison
                if len(selected_features) >= 2:
                    n_algos = len(algorithms)
                    cols = min(2, n_algos)
                    rows = (n_algos + 1) // 2
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(14, 6*rows))
                    if n_algos == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()
                    
                    for idx, (algo, result) in enumerate(comparison_results.items()):
                        ax = axes[idx]
                        labels = result['labels']
                        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.6)
                        ax.set_xlabel(selected_features[0])
                        ax.set_ylabel(selected_features[1])
                        ax.set_title(f'{algo}')
                        plt.colorbar(scatter, ax=ax)
                    
                    # Hide unused subplots
                    for idx in range(n_algos, len(axes)):
                        axes[idx].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error comparing algorithms: {str(e)}")
            st.code(traceback.format_exc())


def show_dimensionality_reduction_tab(X, selected_features):
    """Dimensionality reduction tab"""
    
    st.markdown("### Dimensionality Reduction")
    st.markdown("Reduce the number of features while preserving important information.")
    
    # Create sub-tabs
    dr_tabs = st.tabs(["Apply Reduction", "Find Optimal Components", "Compare Methods"])
    
    with dr_tabs[0]:
        show_dimensionality_reduction_training(X, selected_features)
    
    with dr_tabs[1]:
        show_optimal_components_finder(X)
    
    with dr_tabs[2]:
        show_dr_comparison(X, selected_features)


def show_dimensionality_reduction_training(X, selected_features):
    """Train dimensionality reduction model"""
    
    st.markdown("#### Select Reduction Method")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        method = st.selectbox(
            "Method",
            ["PCA", "t-SNE", "ICA", "NMF", "SVD", "Isomap", "LLE", "MDS"],
            help="Choose the dimensionality reduction technique"
        )
        
        scale_data = st.checkbox("Scale data before reduction", value=True, key="dr_scale")
    
    with col2:
        st.markdown("**Method Description:**")
        
        descriptions = {
            "PCA": "📊 **Principal Component Analysis** - Linear technique that finds orthogonal axes of maximum variance. **Most popular dimensionality reduction**. **Pros:** Fast, interpretable components, handles multicollinearity. **Cons:** Only linear relationships, sensitive to scaling. **Best for:** Feature extraction, visualization, removing correlated features, compression. **Preserves:** Global structure, variance.",
            "t-SNE": "🎨 **t-Distributed Stochastic Neighbor Embedding** - Non-linear technique that maps high-dimensional data to 2D/3D. **Excellent for visualization**. **Pros:** Reveals clusters beautifully, preserves local structure. **Cons:** Slow on large datasets, different runs give different results, can't transform new data easily. **Best for:** Exploratory visualization, revealing hidden clusters. **Preserves:** Local neighborhood structure.",
            "ICA": "🔊 **Independent Component Analysis** - Separates mixed signals into independent sources. **Statistical independence focus**. **Pros:** Finds hidden factors, good for blind source separation. **Cons:** Sensitive to non-Gaussian data, components not ordered by importance. **Best for:** Signal processing, separating mixed audio sources, neuroimaging. **Preserves:** Statistical independence.",
            "NMF": "➕ **Non-negative Matrix Factorization** - Decomposes data into non-negative parts. **Interpretable components**. **Pros:** Parts-based representation, components are additive and interpretable. **Cons:** Only works with non-negative data, can converge to local minimum. **Best for:** Topic modeling, image analysis, recommendation systems. **Requirement:** All values must be ≥ 0.",
            "SVD": "🔢 **Singular Value Decomposition** - Matrix factorization technique similar to PCA. **Works with any matrix**. **Pros:** Handles sparse matrices efficiently, numerically stable. **Cons:** Computationally expensive for very large matrices. **Best for:** Recommender systems, text mining (LSA), sparse data. **Preserves:** Singular values (importance ranking).",
            "Isomap": "🗺️ **Isometric Mapping** - Non-linear technique using geodesic distances on a neighborhood graph. **Manifold learning**. **Pros:** Preserves global non-linear structure, good with Swiss roll-type data. **Cons:** Sensitive to noise, computational cost, needs connected graph. **Best for:** Non-linear manifolds, when data lies on a curved surface. **Preserves:** Geodesic distances.",
            "LLE": "🏘️ **Locally Linear Embedding** - Assumes data lies on a locally linear manifold. **Local structure preservation**. **Pros:** Preserves local relationships, handles non-linear manifolds. **Cons:** Sensitive to noise, no out-of-sample extension, one free parameter. **Best for:** Face recognition, handwriting, when local structure matters. **Preserves:** Local linear relationships.",
            "MDS": "📏 **Multidimensional Scaling** - Maps points to lower dimensions while preserving pairwise distances. **Distance preservation**. **Pros:** Preserves distances well, classical technique. **Cons:** Computationally expensive O(n²), doesn't scale well. **Best for:** Psychology studies, small-to-medium datasets, when distances are key. **Preserves:** Pairwise distances."
        }
        
        st.info(descriptions[method])
    
    # Method-specific parameters
    st.markdown("#### Configure Parameters")
    
    params = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        params['n_components'] = st.slider("Number of components", 2, min(10, X.shape[1]), 2,
                                          help="Target dimensionality")
    
    with col2:
        params['random_state'] = 42
    
    if method == "t-SNE":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['perplexity'] = st.slider("Perplexity", 5, 50, 30,
                                            help="Balance between local and global structure")
        with col2:
            params['learning_rate'] = st.slider("Learning rate", 10, 1000, 200)
        with col3:
            params['max_iter'] = st.slider("Iterations", 250, 2000, 1000)
    
    elif method == "ICA":
        params['max_iter'] = st.slider("Max iterations", 100, 500, 200)
    
    elif method == "NMF":
        params['init'] = st.selectbox("Initialization", ['nndsvd', 'nndsvda', 'nndsvdar', 'random'])
        params['max_iter'] = st.slider("Max iterations", 100, 500, 200)
    
    elif method in ["Isomap", "LLE"]:
        params['n_neighbors'] = st.slider("Number of neighbors", 2, 20, 5)
    
    # Train button
    if st.button("Apply Dimensionality Reduction", type="primary"):
        try:
            with st.spinner(f"Applying {method}..."):
                method_map = {
                    "PCA": "pca",
                    "t-SNE": "tsne",
                    "ICA": "ica",
                    "NMF": "nmf",
                    "SVD": "svd",
                    "Isomap": "isomap",
                    "LLE": "lle",
                    "MDS": "mds"
                }
                
                X_reduced, model, metrics = apply_dimensionality_reduction(
                    X, method_map[method], params, scale_data
                )
                
                # Store results
                if 'dr_results' not in st.session_state:
                    st.session_state.dr_results = {}
                
                st.session_state.dr_results[method] = {
                    'X_reduced': X_reduced,
                    'model': model,
                    'metrics': metrics,
                    'method': method,
                    'params': params
                }
                
                st.success(f"✅ Successfully applied {method}!")
                
        except Exception as e:
            st.error(f"Error applying dimensionality reduction: {str(e)}")
            st.code(traceback.format_exc())
    
    # Always display results if they exist for this method
    if 'dr_results' in st.session_state and method in st.session_state.dr_results:
        result = st.session_state.dr_results[method]
        st.markdown("---")
        st.markdown(f"### Current Results for {method}")
        display_dr_results(result['X_reduced'], result['metrics'], method, selected_features)



def display_dr_results(X_reduced, metrics, method, feature_names):
    """Display dimensionality reduction results"""
    
    st.markdown("---")
    st.markdown("### Results")
    
    # Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Original Dimensions", len(feature_names))
        st.metric("Reduced Dimensions", X_reduced.shape[1])
    
    with col2:
        if 'explained_variance_ratio' in metrics:
            total_var = sum(metrics['explained_variance_ratio'])
            st.metric("Variance Explained", f"{total_var*100:.2f}%")
        
        if 'kl_divergence' in metrics:
            st.metric("KL Divergence", f"{metrics['kl_divergence']:.4f}")
        
        if 'stress' in metrics:
            st.metric("Stress", f"{metrics['stress']:.4f}")
        
        if 'reconstruction_error' in metrics:
            st.metric("Reconstruction Error", f"{metrics['reconstruction_error']:.4f}")
    
    # Variance explained plot (for PCA/SVD)
    if 'explained_variance_ratio' in metrics:
        st.markdown("#### Variance Explained by Components")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual variance
            fig, ax = plt.subplots(figsize=(8, 5))
            components = range(1, len(metrics['explained_variance_ratio']) + 1)
            ax.bar(components, metrics['explained_variance_ratio'])
            ax.set_xlabel('Component')
            ax.set_ylabel('Variance Explained Ratio')
            ax.set_title('Variance Explained by Each Component')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Cumulative variance
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(components, metrics['cumulative_variance_ratio'], 'bo-')
            ax.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Variance Explained')
            ax.set_title('Cumulative Variance Explained')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Visualizations
    visualize_reduced_data(X_reduced, method)


def visualize_reduced_data(X_reduced, method):
    """Visualize reduced dimensional data"""
    
    st.markdown("#### Reduced Data Visualization")
    
    if X_reduced.shape[1] == 2:
        # 2D scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=50)
        ax.set_xlabel(f'{method} Component 1')
        ax.set_ylabel(f'{method} Component 2')
        ax.set_title(f'{method} - 2D Projection')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
    elif X_reduced.shape[1] == 3:
        # 3D scatter plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], alpha=0.6, s=50)
        ax.set_xlabel(f'{method} Component 1')
        ax.set_ylabel(f'{method} Component 2')
        ax.set_zlabel(f'{method} Component 3')
        ax.set_title(f'{method} - 3D Projection')
        st.pyplot(fig)
    
    elif X_reduced.shape[1] > 3:
        # Scatter matrix for first 3 components
        df_reduced = pd.DataFrame(X_reduced[:, :3], columns=[f'Component {i+1}' for i in range(3)])
        fig = sns.pairplot(df_reduced, diag_kind='kde')
        st.pyplot(fig)
    
    # Download reduced data
    df_download = pd.DataFrame(X_reduced, columns=[f'{method}_Component_{i+1}' for i in range(X_reduced.shape[1])])
    csv = df_download.to_csv(index=False)
    st.download_button(
        "📥 Download Reduced Data",
        csv,
        f"reduced_data_{method}.csv",
        "text/csv"
    )


def show_optimal_components_finder(X):
    """Find optimal number of components"""
    
    st.markdown("#### Find Optimal Number of Components")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox("Method", ["PCA", "SVD"], key="optimal_comp_method")
    
    with col2:
        variance_threshold = st.slider("Variance threshold", 0.80, 0.99, 0.95, 0.01,
                                       help="Minimum cumulative variance to explain")
    
    if st.button("Find Optimal Components", type="primary"):
        try:
            with st.spinner("Calculating optimal components..."):
                results = calculate_optimal_components(X, method.lower(), variance_threshold)
                
                st.success(f"✅ Optimal number of components: **{results['optimal_components']}**")
                st.info(f"This explains {results['variance_explained']*100:.2f}% of the total variance")
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                components = range(1, len(results['cumulative_variance']) + 1)
                ax.plot(components, results['cumulative_variance'], 'bo-', linewidth=2)
                ax.axhline(y=variance_threshold, color='r', linestyle='--', 
                          label=f'{variance_threshold*100}% threshold')
                ax.axvline(x=results['optimal_components'], color='g', linestyle='--',
                          label=f"Optimal K = {results['optimal_components']}")
                ax.set_xlabel('Number of Components')
                ax.set_ylabel('Cumulative Variance Explained')
                ax.set_title(f'{method} - Cumulative Variance Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error finding optimal components: {str(e)}")
            st.code(traceback.format_exc())


def show_dr_comparison(X, selected_features):
    """Compare dimensionality reduction methods"""
    
    st.markdown("#### Compare Dimensionality Reduction Methods")
    
    methods = st.multiselect(
        "Select methods to compare",
        ["PCA", "t-SNE", "ICA", "SVD"],
        default=["PCA", "t-SNE"]
    )
    
    n_components = st.slider("Number of components", 2, 3, 2, key="dr_compare_components")
    
    if st.button("Compare Methods", type="primary", key="dr_compare_methods_btn") and methods:
        try:
            with st.spinner("Applying and comparing methods..."):
                comparison_results = {}
                
                for method in methods:
                    method_map = {"PCA": "pca", "t-SNE": "tsne", "ICA": "ica", "SVD": "svd"}
                    params = {'n_components': n_components, 'random_state': 42}
                    
                    X_reduced, model, metrics = apply_dimensionality_reduction(
                        X, method_map[method], params, True
                    )
                    
                    comparison_results[method] = {
                        'X_reduced': X_reduced,
                        'metrics': metrics
                    }
                
                # Visualize comparison
                st.markdown("### Comparison Results")
                
                if n_components == 2:
                    n_methods = len(methods)
                    cols = min(2, n_methods)
                    rows = (n_methods + 1) // 2
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(14, 6*rows))
                    if n_methods == 1:
                        axes = [axes]
                    elif rows == 1:
                        axes = axes if n_methods > 1 else [axes]
                    else:
                        axes = axes.flatten()
                    
                    for idx, (method, result) in enumerate(comparison_results.items()):
                        ax = axes[idx]
                        X_reduced = result['X_reduced']
                        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=30)
                        ax.set_xlabel(f'Component 1')
                        ax.set_ylabel(f'Component 2')
                        ax.set_title(f'{method}')
                        ax.grid(True, alpha=0.3)
                    
                    # Hide unused subplots
                    for idx in range(n_methods, len(axes)):
                        axes[idx].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Metrics comparison
                metrics_data = []
                for method, result in comparison_results.items():
                    metrics = result['metrics']
                    row = {'Method': method}
                    if 'explained_variance_ratio' in metrics:
                        row['Variance Explained'] = f"{sum(metrics['explained_variance_ratio'])*100:.2f}%"
                    metrics_data.append(row)
                
                if metrics_data:
                    comparison_df = pd.DataFrame(metrics_data)
                    st.dataframe(comparison_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error comparing methods: {str(e)}")
            st.code(traceback.format_exc())


def show_anomaly_detection_tab(X, selected_features):
    """Anomaly detection tab"""
    
    st.markdown("### Anomaly Detection")
    st.markdown("Identify unusual patterns or outliers in your data.")
    
    # Create sub-tabs
    anomaly_tabs = st.tabs(["Single Method", "Compare Methods"])
    
    with anomaly_tabs[0]:
        show_anomaly_detection_single(X, selected_features)
    
    with anomaly_tabs[1]:
        show_anomaly_detection_comparison(X, selected_features)


def show_anomaly_detection_single(X, selected_features):
    """Single anomaly detection method"""
    
    st.markdown("#### Select Detection Method")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        method = st.selectbox(
            "Method",
            ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Elliptic Envelope"],
            help="Choose the anomaly detection method"
        )
        
        scale_data = st.checkbox("Scale data", value=True, key="anomaly_scale")
    
    with col2:
        st.markdown("**Method Description:**")
        
        descriptions = {
            "Isolation Forest": "🌲 **Tree-Based Isolation** - Isolates anomalies by building random trees. Anomalies are easier to isolate (require fewer splits). **Pros:** Fast, scalable, works well with high-dimensional data, no need for distance calculations. **Cons:** May struggle with local anomalies, parameter tuning needed. **Best for:** Large datasets, high dimensions, fraud detection, network intrusion. **Assumption:** Anomalies are few and different.",
            "Local Outlier Factor": "🎯 **Density-Based** - Measures how isolated a point is relative to its neighbors' densities. **Finds local outliers**. **Pros:** Detects local anomalies in varying density regions, no global threshold needed. **Cons:** Computationally expensive O(n²), sensitive to k parameter, doesn't scale well. **Best for:** Small-to-medium datasets with varying densities, local anomaly detection. **Assumption:** Anomalies have lower density than neighbors.",
            "One-Class SVM": "📐 **Boundary Learning** - Learns a decision boundary that encompasses normal data in high-dimensional space. **Kernel-based**. **Pros:** Works well with small datasets, effective in high dimensions, kernel trick for non-linear boundaries. **Cons:** Sensitive to parameter tuning (nu, gamma), doesn't scale well, needs feature scaling. **Best for:** Small datasets, novelty detection, when you only have normal examples. **Assumption:** Normal data lies in a dense region.",
            "Elliptic Envelope": "⭕ **Gaussian Assumption** - Fits an ellipse around normal data assuming it's Gaussian distributed. **Fast parametric method**. **Pros:** Very fast, works well when assumption holds, minimal parameters. **Cons:** Assumes Gaussian distribution, sensitive to outliers in training, struggles with complex shapes. **Best for:** Low-dimensional data that's approximately Gaussian, quick screening. **Assumption:** Normal data follows multivariate Gaussian distribution."
        }
        
        st.info(descriptions[method])
    
    # Method-specific parameters
    st.markdown("#### Configure Parameters")
    
    params = {}
    
    contamination = st.slider("Expected contamination", 0.01, 0.5, 0.1, 0.01,
                             help="Expected proportion of anomalies in the dataset")
    params['contamination'] = contamination
    
    if method == "Isolation Forest":
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider("Number of estimators", 50, 300, 100)
        with col2:
            params['max_samples'] = st.selectbox("Max samples", ['auto', 256, 512, 1024])
    
    elif method == "Local Outlier Factor":
        params['n_neighbors'] = st.slider("Number of neighbors", 5, 50, 20)
        params['novelty'] = st.checkbox("Novelty detection mode", value=False,
                                       help="Set to True if you want to use the model on new data")
    
    elif method == "One-Class SVM":
        col1, col2 = st.columns(2)
        with col1:
            params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
        with col2:
            params['nu'] = st.slider("Nu", 0.01, 0.5, 0.1, 0.01,
                                    help="Upper bound on fraction of training errors")
        params['gamma'] = st.selectbox("Gamma", ['scale', 'auto'])
    
    params['random_state'] = 42
    
    # Detect button
    if st.button("Detect Anomalies", type="primary"):
        try:
            with st.spinner(f"Detecting anomalies using {method}..."):
                method_map = {
                    "Isolation Forest": "isolation_forest",
                    "Local Outlier Factor": "lof",
                    "One-Class SVM": "one_class_svm",
                    "Elliptic Envelope": "elliptic_envelope"
                }
                
                predictions, scores, model, metrics = detect_anomalies(
                    X, method_map[method], params, scale_data
                )
                
                # Store results
                if 'anomaly_results' not in st.session_state:
                    st.session_state.anomaly_results = {}
                
                st.session_state.anomaly_results[method] = {
                    'predictions': predictions,
                    'scores': scores,
                    'model': model,
                    'metrics': metrics,
                    'method': method
                }
                
                st.success(f"✅ Anomaly detection complete!")
                
        except Exception as e:
            st.error(f"Error detecting anomalies: {str(e)}")
            st.code(traceback.format_exc())
    
    # Always display results if they exist for this method
    if 'anomaly_results' in st.session_state and method in st.session_state.anomaly_results:
        result = st.session_state.anomaly_results[method]
        st.markdown("---")
        st.markdown(f"### Current Results for {method}")
        display_anomaly_results(X, result['predictions'], result['scores'], result['metrics'], method, selected_features)



def display_anomaly_results(X, predictions, scores, metrics, method, feature_names):
    """Display anomaly detection results"""
    
    st.markdown("---")
    st.markdown("### Results")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Points", len(predictions))
    
    with col2:
        st.metric("Anomalies Detected", metrics['n_anomalies'])
    
    with col3:
        st.metric("Anomaly Percentage", f"{metrics['anomaly_percentage']:.2f}%")
    
    # Anomaly scores distribution
    st.markdown("#### Anomaly Scores Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Anomaly Scores - {method}')
    # Calculate the threshold based on the actual anomaly percentage
    threshold_percentile = 100 - metrics.get("anomaly_percentage", 10)
    ax.axvline(x=np.percentile(scores, threshold_percentile), color='r', linestyle='--',
               label=f'{metrics.get("anomaly_percentage", 0):.1f}% anomalies threshold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Visualizations
    if len(feature_names) >= 2:
        st.markdown("#### Anomaly Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis feature", feature_names, index=0, key="anomaly_x")
        with col2:
            y_feature = st.selectbox("Y-axis feature", feature_names, index=min(1, len(feature_names)-1), key="anomaly_y")
        
        x_idx = feature_names.index(x_feature)
        y_idx = feature_names.index(y_feature)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot inliers
        inliers = predictions == 1
        outliers = predictions == -1
        
        ax.scatter(X.iloc[inliers, x_idx], X.iloc[inliers, y_idx], 
                  c='blue', label='Normal', alpha=0.6, s=50)
        ax.scatter(X.iloc[outliers, x_idx], X.iloc[outliers, y_idx], 
                  c='red', label='Anomaly', alpha=0.8, s=100, marker='x')
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'Anomaly Detection Results - {method}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Anomalies data table
    if metrics['n_anomalies'] > 0:
        st.markdown("#### Detected Anomalies")
        
        anomaly_indices = np.where(predictions == -1)[0]
        anomalies_df = X.iloc[anomaly_indices].copy()
        anomalies_df['Anomaly_Score'] = scores[anomaly_indices]
        anomalies_df = anomalies_df.sort_values('Anomaly_Score')
        
        st.dataframe(anomalies_df.head(20))
        
        # Download anomalies
        csv = anomalies_df.to_csv(index=True)
        st.download_button(
            "📥 Download Anomalies",
            csv,
            f"anomalies_{method}.csv",
            "text/csv"
        )


def show_anomaly_detection_comparison(X, selected_features):
    """Compare anomaly detection methods"""
    
    st.markdown("#### Compare Anomaly Detection Methods")
    
    methods = st.multiselect(
        "Select methods to compare",
        ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Elliptic Envelope"],
        default=["Isolation Forest", "Local Outlier Factor"]
    )
    
    contamination = st.slider("Expected contamination", 0.01, 0.5, 0.1, 0.01, key="compare_contamination")
    
    if st.button("Compare Methods", type="primary", key="anomaly_compare_methods_btn") and methods:
        try:
            with st.spinner("Comparing anomaly detection methods..."):
                method_map = {
                    "Isolation Forest": "isolation_forest",
                    "Local Outlier Factor": "lof",
                    "One-Class SVM": "one_class_svm",
                    "Elliptic Envelope": "elliptic_envelope"
                }
                
                methods_to_compare = [method_map[m] for m in methods]
                
                results = compare_anomaly_methods(X, methods_to_compare, contamination, True)
                
                # Display comparison
                st.markdown("### Comparison Results")
                
                # Metrics table
                metrics_data = []
                for method_name, result in results.items():
                    if 'error' not in result:
                        metrics = result['metrics']
                        row = {
                            'Method': method_name.replace('_', ' ').title(),
                            'Anomalies': metrics['n_anomalies'],
                            'Percentage': f"{metrics['anomaly_percentage']:.2f}%"
                        }
                        metrics_data.append(row)
                
                comparison_df = pd.DataFrame(metrics_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualize comparison
                if len(selected_features) >= 2:
                    n_methods = len([r for r in results.values() if 'error' not in r])
                    cols = min(2, n_methods)
                    rows = (n_methods + 1) // 2
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(14, 6*rows))
                    if n_methods == 1:
                        axes = [axes]
                    elif rows == 1 and cols > 1:
                        pass  # axes is already correct
                    elif rows > 1:
                        axes = axes.flatten()
                    else:
                        axes = [axes]
                    
                    idx = 0
                    for method_name, result in results.items():
                        if 'error' not in result:
                            ax = axes[idx]
                            predictions = result['predictions']
                            
                            inliers = predictions == 1
                            outliers = predictions == -1
                            
                            ax.scatter(X.iloc[inliers, 0], X.iloc[inliers, 1], 
                                      c='blue', alpha=0.5, s=30, label='Normal')
                            ax.scatter(X.iloc[outliers, 0], X.iloc[outliers, 1], 
                                      c='red', alpha=0.8, s=80, marker='x', label='Anomaly')
                            
                            ax.set_xlabel(selected_features[0])
                            ax.set_ylabel(selected_features[1])
                            ax.set_title(method_name.replace('_', ' ').title())
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            idx += 1
                    
                    # Hide unused subplots
                    for i in range(idx, len(axes)):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error comparing methods: {str(e)}")
            st.code(traceback.format_exc())


def show_results_summary():
    """Show summary of all unsupervised learning results"""
    
    st.markdown("### Results Summary")
    st.markdown("View and export all unsupervised learning results.")
    
    # Check if any results exist
    has_clustering = 'clustering_results' in st.session_state and st.session_state.clustering_results
    has_dr = 'dr_results' in st.session_state and st.session_state.dr_results
    has_anomaly = 'anomaly_results' in st.session_state and st.session_state.anomaly_results
    
    if not (has_clustering or has_dr or has_anomaly):
        st.info("No results yet. Run some unsupervised learning methods to see results here.")
        return
    
    # Clustering results
    if has_clustering:
        st.markdown("#### Clustering Results")
        for algo, result in st.session_state.clustering_results.items():
            with st.expander(f"📊 {algo}"):
                metrics = result['metrics']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters", metrics.get('n_clusters', 'N/A'))
                with col2:
                    if 'silhouette_score' in metrics and metrics['silhouette_score']:
                        st.metric("Silhouette", f"{metrics['silhouette_score']:.3f}")
                with col3:
                    if 'davies_bouldin_score' in metrics and metrics['davies_bouldin_score']:
                        st.metric("Davies-Bouldin", f"{metrics['davies_bouldin_score']:.3f}")
    
    # Dimensionality reduction results
    if has_dr:
        st.markdown("#### Dimensionality Reduction Results")
        for method, result in st.session_state.dr_results.items():
            with st.expander(f"📉 {method}"):
                metrics = result['metrics']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Components", result['X_reduced'].shape[1])
                with col2:
                    if 'explained_variance_ratio' in metrics:
                        total_var = sum(metrics['explained_variance_ratio'])
                        st.metric("Variance Explained", f"{total_var*100:.2f}%")
    
    # Anomaly detection results
    if has_anomaly:
        st.markdown("#### Anomaly Detection Results")
        for method, result in st.session_state.anomaly_results.items():
            with st.expander(f"⚠️ {method}"):
                metrics = result['metrics']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Anomalies", metrics['n_anomalies'])
                with col2:
                    st.metric("Percentage", f"{metrics['anomaly_percentage']:.2f}%")


if __name__ == "__main__":
    show_unsupervised_learning()
