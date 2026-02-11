"""
Unsupervised Learning Module
Contains functions for clustering, dimensionality reduction, and anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import logging

logger = logging.getLogger(__name__)

# ==================== CLUSTERING METHODS ====================

def train_clustering_model(
    X: pd.DataFrame,
    model_type: str,
    params: Dict[str, Any],
    scale_data: bool = True,
    return_scaler: bool = False
) -> Tuple[Any, np.ndarray, Dict[str, float], Optional[Any]]:
    """
    Train a clustering model
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    model_type : str
        Type of clustering algorithm
    params : Dict
        Model parameters
    scale_data : bool
        Whether to scale data before clustering
    
    Returns:
    --------
    Tuple containing:
    - Trained model
    - Cluster labels
    - Clustering metrics 
    - Scaler object (if return_scaler=True and scale_data=True)
    """
    # Scale data if requested
    scaler = None
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    else:
        X_scaled = X.copy()
    
    # Initialize model based on type
    if model_type == 'kmeans':
        model = KMeans(
            n_clusters=params.get('n_clusters', 3),
            init=params.get('init', 'k-means++'),
            n_init=params.get('n_init', 10),
            max_iter=params.get('max_iter', 300),
            random_state=params.get('random_state', 42)
        )
        
    elif model_type == 'dbscan':
        model = DBSCAN(
            eps=params.get('eps', 0.5),
            min_samples=params.get('min_samples', 5),
            metric=params.get('metric', 'euclidean'),
            algorithm=params.get('algorithm', 'auto')
        )
        
    elif model_type == 'hierarchical':
        model = AgglomerativeClustering(
            n_clusters=params.get('n_clusters', 3),
            linkage=params.get('linkage', 'ward'),
            metric=params.get('metric', 'euclidean')
        )
        
    elif model_type == 'meanshift':
        model = MeanShift(
            bandwidth=params.get('bandwidth', None),
            bin_seeding=params.get('bin_seeding', False),
            cluster_all=params.get('cluster_all', True)
        )
        
    elif model_type == 'spectral':
        model = SpectralClustering(
            n_clusters=params.get('n_clusters', 3),
            affinity=params.get('affinity', 'rbf'),
            n_init=params.get('n_init', 10),
            random_state=params.get('random_state', 42)
        )
        
    elif model_type == 'gmm':
        model = GaussianMixture(
            n_components=params.get('n_components', 3),
            covariance_type=params.get('covariance_type', 'full'),
            max_iter=params.get('max_iter', 100),
            n_init=params.get('n_init', 1),
            random_state=params.get('random_state', 42)
        )
        
    else:
        raise ValueError(f"Unknown clustering model type: {model_type}")
    
    # Fit model and get labels
    if model_type == 'gmm':
        labels = model.fit_predict(X_scaled)
    else:
        model.fit(X_scaled)
        labels = model.labels_
    
    # Calculate clustering metrics
    metrics = {}
    
    # Only calculate metrics if we have more than 1 cluster and less than n_samples
    n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise points (-1)
    
    if n_clusters > 1 and n_clusters < len(X_scaled):
        try:
            # Silhouette Score (higher is better, range: -1 to 1)
            metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
        except:
            metrics['silhouette_score'] = None
        
        try:
            # Davies-Bouldin Index (lower is better)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
        except:
            metrics['davies_bouldin_score'] = None
        
        try:
            # Calinski-Harabasz Score (higher is better)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
        except:
            metrics['calinski_harabasz_score'] = None
    
    # Add model-specific metrics
    if model_type == 'kmeans':
        metrics['inertia'] = model.inertia_
    elif model_type == 'gmm':
        metrics['bic'] = model.bic(X_scaled)
        metrics['aic'] = model.aic(X_scaled)
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
    metrics['n_clusters'] = n_clusters
    
    # Percentage of noise points (for DBSCAN)
    if -1 in labels:
        metrics['noise_percentage'] = (labels == -1).sum() / len(labels) * 100
    
    logger.info(f"Trained {model_type} clustering model with {n_clusters} clusters")
    
    if return_scaler:
        return model, labels, metrics, scaler
    else:
        return model, labels, metrics


def find_optimal_clusters(
    X: pd.DataFrame,
    method: str = 'elbow',
    k_range: Tuple[int, int] = (2, 10),
    scale_data: bool = True
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using various methods
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    method : str
        Method to use ('elbow', 'silhouette', 'bic', 'gap')
    k_range : Tuple
        Range of k values to test
    scale_data : bool
        Whether to scale data
    
    Returns:
    --------
    Dictionary with results including optimal k and scores
    """
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    k_min, k_max = k_range
    k_values = range(k_min, k_max + 1)
    
    results = {
        'k_values': list(k_values),
        'scores': [],
        'method': method
    }
    
    for k in k_values:
        if method == 'elbow':
            # Use inertia (within-cluster sum of squares)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            results['scores'].append(kmeans.inertia_)
            
        elif method == 'silhouette':
            # Use silhouette score
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            results['scores'].append(score)
            
        elif method == 'bic':
            # Use BIC for Gaussian Mixture Models
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X_scaled)
            results['scores'].append(gmm.bic(X_scaled))
    
    # Determine optimal k based on method
    if method == 'elbow':
        # For elbow, we want the "elbow point" - calculate rate of change
        scores = np.array(results['scores'])
        # Simple method: find max second derivative
        if len(scores) > 2:
            second_deriv = np.diff(scores, 2)
            optimal_idx = np.argmax(np.abs(second_deriv)) + 1
            results['optimal_k'] = k_values[optimal_idx]
        else:
            results['optimal_k'] = k_values[0]
    elif method == 'silhouette':
        # For silhouette, higher is better
        results['optimal_k'] = k_values[np.argmax(results['scores'])]
    elif method == 'bic':
        # For BIC, lower is better
        results['optimal_k'] = k_values[np.argmin(results['scores'])]
    
    return results


# ==================== DIMENSIONALITY REDUCTION ====================

def apply_dimensionality_reduction(
    X: pd.DataFrame,
    method: str,
    params: Dict[str, Any],
    scale_data: bool = True
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Apply dimensionality reduction technique
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    method : str
        Reduction method ('pca', 'tsne', 'ica', etc.)
    params : Dict
        Method parameters
    scale_data : bool
        Whether to scale data
    
    Returns:
    --------
    Tuple containing:
    - Transformed data
    - Fitted model
    - Method-specific metrics
    """
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    metrics = {}
    
    # Initialize reduction model
    if method == 'pca':
        model = PCA(
            n_components=params.get('n_components', 2),
            random_state=params.get('random_state', 42)
        )
        X_reduced = model.fit_transform(X_scaled)
        
        # PCA-specific metrics
        metrics['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
        metrics['cumulative_variance_ratio'] = np.cumsum(model.explained_variance_ratio_).tolist()
        metrics['singular_values'] = model.singular_values_.tolist()
        
    elif method == 'tsne':
        model = TSNE(
            n_components=params.get('n_components', 2),
            perplexity=params.get('perplexity', 30),
            learning_rate=params.get('learning_rate', 200),
            max_iter=params.get('max_iter', 1000),
            random_state=params.get('random_state', 42)
        )
        X_reduced = model.fit_transform(X_scaled)
        
        # t-SNE specific metrics
        metrics['kl_divergence'] = model.kl_divergence_
        
    elif method == 'ica':
        model = FastICA(
            n_components=params.get('n_components', 2),
            max_iter=params.get('max_iter', 200),
            random_state=params.get('random_state', 42)
        )
        X_reduced = model.fit_transform(X_scaled)
        
    elif method == 'nmf':
        # NMF requires non-negative data
        X_non_neg = X_scaled - X_scaled.min()
        model = NMF(
            n_components=params.get('n_components', 2),
            init=params.get('init', 'nndsvd'),
            max_iter=params.get('max_iter', 200),
            random_state=params.get('random_state', 42)
        )
        X_reduced = model.fit_transform(X_non_neg)
        
        # NMF specific metrics
        metrics['reconstruction_error'] = model.reconstruction_err_
        
    elif method == 'svd':
        model = TruncatedSVD(
            n_components=params.get('n_components', 2),
            n_iter=params.get('n_iter', 5),
            random_state=params.get('random_state', 42)
        )
        X_reduced = model.fit_transform(X_scaled)
        
        # SVD specific metrics
        metrics['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
        metrics['cumulative_variance_ratio'] = np.cumsum(model.explained_variance_ratio_).tolist()
        
    elif method == 'isomap':
        model = Isomap(
            n_components=params.get('n_components', 2),
            n_neighbors=params.get('n_neighbors', 5)
        )
        X_reduced = model.fit_transform(X_scaled)
        
    elif method == 'lle':
        model = LocallyLinearEmbedding(
            n_components=params.get('n_components', 2),
            n_neighbors=params.get('n_neighbors', 5),
            random_state=params.get('random_state', 42)
        )
        X_reduced = model.fit_transform(X_scaled)
        
    elif method == 'mds':
        model = MDS(
            n_components=params.get('n_components', 2),
            random_state=params.get('random_state', 42)
        )
        X_reduced = model.fit_transform(X_scaled)
        
        # MDS specific metrics
        metrics['stress'] = model.stress_
        
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    logger.info(f"Applied {method} dimensionality reduction to {params.get('n_components', 2)} components")
    
    return X_reduced, model, metrics


def calculate_optimal_components(
    X: pd.DataFrame,
    method: str = 'pca',
    variance_threshold: float = 0.95,
    max_components: int = None
) -> Dict[str, Any]:
    """
    Calculate optimal number of components for dimensionality reduction
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    method : str
        Reduction method
    variance_threshold : float
        Cumulative variance threshold for PCA/SVD
    max_components : int
        Maximum components to consider
    
    Returns:
    --------
    Dictionary with optimal number of components and related metrics
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if max_components is None:
        max_components = min(X.shape[0], X.shape[1])
    
    results = {}
    
    if method == 'pca':
        pca = PCA(n_components=max_components)
        pca.fit(X_scaled)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        optimal_n = np.argmax(cumsum >= variance_threshold) + 1
        
        results['optimal_components'] = optimal_n
        results['variance_explained'] = cumsum[optimal_n - 1]
        results['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
        results['cumulative_variance'] = cumsum.tolist()
        
    elif method == 'svd':
        svd = TruncatedSVD(n_components=min(max_components, min(X.shape) - 1))
        svd.fit(X_scaled)
        
        cumsum = np.cumsum(svd.explained_variance_ratio_)
        optimal_n = np.argmax(cumsum >= variance_threshold) + 1
        
        results['optimal_components'] = optimal_n
        results['variance_explained'] = cumsum[optimal_n - 1]
        results['explained_variance_ratio'] = svd.explained_variance_ratio_.tolist()
        results['cumulative_variance'] = cumsum.tolist()
    
    return results


# ==================== ANOMALY DETECTION ====================

def detect_anomalies(
    X: pd.DataFrame,
    method: str,
    params: Dict[str, Any],
    scale_data: bool = True
) -> Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """
    Detect anomalies using various methods
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    method : str
        Detection method ('isolation_forest', 'lof', 'one_class_svm', 'elliptic_envelope')
    params : Dict
        Method parameters
    scale_data : bool
        Whether to scale data
    
    Returns:
    --------
    Tuple containing:
    - Predictions (1 for inliers, -1 for outliers)
    - Anomaly scores
    - Fitted model
    - Detection metrics
    """
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Initialize model
    if method == 'isolation_forest':
        model = IsolationForest(
            n_estimators=params.get('n_estimators', 100),
            contamination=params.get('contamination', 0.1),
            max_samples=params.get('max_samples', 'auto'),
            random_state=params.get('random_state', 42)
        )
        
    elif method == 'lof':
        model = LocalOutlierFactor(
            n_neighbors=params.get('n_neighbors', 20),
            contamination=params.get('contamination', 0.1),
            novelty=params.get('novelty', False)
        )
        
    elif method == 'one_class_svm':
        model = OneClassSVM(
            kernel=params.get('kernel', 'rbf'),
            nu=params.get('nu', 0.1),
            gamma=params.get('gamma', 'scale')
        )
        
    elif method == 'elliptic_envelope':
        model = EllipticEnvelope(
            contamination=params.get('contamination', 0.1),
            random_state=params.get('random_state', 42)
        )
        
    else:
        raise ValueError(f"Unknown anomaly detection method: {method}")
    
    # Fit and predict
    predictions = model.fit_predict(X_scaled)
    
    # Get anomaly scores
    if method == 'lof' and not params.get('novelty', False):
        scores = model.negative_outlier_factor_
    elif hasattr(model, 'score_samples'):
        scores = model.score_samples(X_scaled)
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_scaled)
    else:
        scores = np.zeros(len(X_scaled))
    
    # Calculate metrics
    metrics = {}
    n_anomalies = (predictions == -1).sum()
    metrics['n_anomalies'] = int(n_anomalies)
    metrics['anomaly_percentage'] = float(n_anomalies / len(X) * 100)
    metrics['n_inliers'] = int((predictions == 1).sum())
    
    # Method-specific metrics
    if method == 'isolation_forest':
        metrics['contamination'] = params.get('contamination', 0.1)
    elif method == 'lof':
        if not params.get('novelty', False):
            metrics['mean_lof_score'] = float(scores.mean())
            metrics['min_lof_score'] = float(scores.min())
    
    logger.info(f"Detected {n_anomalies} anomalies using {method} ({metrics['anomaly_percentage']:.2f}%)")
    
    return predictions, scores, model, metrics


def compare_anomaly_methods(
    X: pd.DataFrame,
    methods: List[str],
    contamination: float = 0.1,
    scale_data: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple anomaly detection methods
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    methods : List[str]
        List of methods to compare
    contamination : float
        Expected proportion of anomalies
    scale_data : bool
        Whether to scale data
    
    Returns:
    --------
    Dictionary with results for each method
    """
    results = {}
    
    for method in methods:
        try:
            params = {'contamination': contamination, 'random_state': 42}
            predictions, scores, model, metrics = detect_anomalies(
                X, method, params, scale_data
            )
            
            results[method] = {
                'predictions': predictions,
                'scores': scores,
                'model': model,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Error with {method}: {str(e)}")
            results[method] = {'error': str(e)}
    
    return results


# ==================== UTILITY FUNCTIONS ====================

def cluster_profiling(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_names: List[str] = None,
    X_original: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Create cluster profiles showing mean values for each feature
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    labels : ndarray
        Cluster labels
    feature_names : List[str]
        Names of features
    X_original : DataFrame, optional
        Original (unscaled) feature data for comparison
    
    Returns:
    --------
    DataFrame with cluster profiles
    """
    if feature_names is None:
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrame with labels
    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = labels
    
    # Calculate mean values per cluster
    profiles = df.groupby('cluster').mean()
    
    # Add cluster sizes
    profiles['cluster_size'] = df.groupby('cluster').size()
    
    # If original data provided, also return profiles for original scale
    if X_original is not None:
        df_orig = pd.DataFrame(X_original, columns=feature_names)
        df_orig['cluster'] = labels
        
        profiles_orig = df_orig.groupby('cluster').mean()
        profiles_orig['cluster_size'] = df_orig.groupby('cluster').size()
        
        # Rename columns to indicate original scale
        profiles_orig.columns = [f"{col}_original" if col != 'cluster_size' else col 
                                for col in profiles_orig.columns]
        
        return profiles, profiles_orig
    
    return profiles


def calculate_cluster_separation(
    X: pd.DataFrame,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate cluster separation metrics
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    labels : ndarray
        Cluster labels
    
    Returns:
    --------
    Dictionary with separation metrics
    """
    from scipy.spatial.distance import cdist
    
    metrics = {}
    
    # Get unique clusters (excluding noise if present)
    unique_labels = np.unique(labels[labels != -1])
    
    if len(unique_labels) < 2:
        return metrics
    
    # Calculate cluster centers
    centers = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        centers.append(cluster_points.mean(axis=0))
    
    centers = np.array(centers)
    
    # Inter-cluster distance (higher is better)
    inter_cluster_dist = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(inter_cluster_dist, np.inf)
    metrics['min_inter_cluster_distance'] = float(inter_cluster_dist.min())
    metrics['mean_inter_cluster_distance'] = float(inter_cluster_dist[inter_cluster_dist != np.inf].mean())
    
    # Intra-cluster distance (lower is better)
    intra_cluster_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            center = centers[list(unique_labels).index(label)]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            intra_cluster_distances.append(distances.mean())
    
    if intra_cluster_distances:
        metrics['mean_intra_cluster_distance'] = float(np.mean(intra_cluster_distances))
    
    # Separation ratio (higher is better)
    if 'mean_intra_cluster_distance' in metrics:
        metrics['separation_ratio'] = metrics['mean_inter_cluster_distance'] / metrics['mean_intra_cluster_distance']
    
    return metrics
