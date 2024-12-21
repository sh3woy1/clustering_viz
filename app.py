import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Dictionary for multilingual support
translations = {
    "en": {
        "app_title": "Enhanced Clustering Visualization",
        "app_description": """
        An advanced clustering analysis tool with multiple algorithms, evaluation metrics, 
        and interactive visualizations. Upload your dataset or use the example data to explore 
        patterns in your data.
        
        **Key Features:**
        - Multiple clustering algorithms (K-Means, DBSCAN, Hierarchical)
        - Advanced data preprocessing options
        - Interactive visualizations
        - Comprehensive cluster analysis
        - Data quality assessment
        """,
        "settings": "Settings",
        "language": "Language",
        "data_source": "Choose data source",
        "use_example": "Use example data",
        "upload_data": "Upload your own data",
        "example_info": "Using example dataset with 3 features and distinct clusters",
        "feature_selection": "Feature Selection",
        "choose_features": "Choose features for clustering",
        "preprocessing": "Preprocessing",
        "scaling_method": "Scaling method",
        "missing_values": "Missing values strategy",
        "fill_value": "Fill value for missing data",
        "pca_option": "Use PCA for visualization",
        "algorithm_selection": "Clustering Algorithm",
        "choose_algorithm": "Choose algorithm",
        "cluster_viz": "Cluster Visualizations",
        "scatter_plots": "Scatter Plots",
        "feature_analysis": "Feature Analysis",
        "cluster_stats": "Cluster Statistics",
        "quality_metrics": "Clustering Quality Metrics",
        "download_results": "Download Results",
        "download_data": "Download clustered data (CSV)",
        "download_stats": "Download cluster statistics (CSV)",
        "data_quality": "Data Quality Information",
        "missing_values_tab": "Missing Values",
        "feature_ranges": "Feature Ranges",
        "outliers": "Outliers",
        "algorithm_explanation": {
            "kmeans": """
            **K-Means Clustering**
            - Groups data into K clusters based on similarity
            - Each cluster has a center (centroid)
            - Best for spherical clusters of similar size
            - Number of clusters (K) must be specified
            """,
            "dbscan": """
            **DBSCAN Clustering**
            - Density-based clustering algorithm
            - Finds clusters of arbitrary shape
            - Automatically detects noise points
            - No need to specify number of clusters
            - Parameters:
              - Epsilon: Maximum distance between points in a cluster
              - Min Samples: Minimum points to form a dense region
            """,
            "hierarchical": """
            **Hierarchical Clustering**
            - Creates a tree of clusters (dendrogram)
            - Multiple linking methods available
            - Can reveal hierarchical relationships
            - Parameters:
              - Number of clusters
              - Linkage method (determines how distances are measured)
            """
        },
        "metric_explanations": {
            "silhouette": "Measures how similar an object is to its cluster compared to other clusters (Range: -1 to 1, higher is better)",
            "calinski": "Ratio of between-cluster variance to within-cluster variance (Higher is better)",
            "davies": "Average similarity measure of each cluster with its most similar cluster (Lower is better)"
        }
    },
    "ar": {
        "app_title": "تحليل وتصور العناقيد المتقدم",
        "app_description": """
        أداة متقدمة لتحليل العناقيد مع خوارزميات متعددة ومقاييس تقييم 
        وتصورات تفاعلية. قم بتحميل مجموعة البيانات الخاصة بك أو استخدم البيانات المثال 
        لاستكشاف الأنماط في بياناتك.
        
        **الميزات الرئيسية:**
        - خوارزميات تجميع متعددة (K-Means, DBSCAN, Hierarchical)
        - خيارات متقدمة لمعالجة البيانات
        - تصورات تفاعلية
        - تحليل شامل للعناقيد
        - تقييم جودة البيانات
        """,
        "settings": "الإعدادات",
        "language": "اللغة",
        "data_source": "اختر مصدر البيانات",
        "use_example": "استخدم البيانات المثال",
        "upload_data": "تحميل البيانات الخاصة بك",
        "example_info": "استخدام مجموعة بيانات مثال مع 3 خصائص وعناقيد متميزة",
        "feature_selection": "اختيار الخصائص",
        "choose_features": "اختر الخصائص للتجميع",
        "preprocessing": "المعالجة المسبقة",
        "scaling_method": "طريقة التحجيم",
        "missing_values": "استراتيجية القيم المفقودة",
        "fill_value": "قيمة ملء البيانات المفقودة",
        "pca_option": "استخدام PCA للتصور",
        "algorithm_selection": "خوارزمية التجميع",
        "choose_algorithm": "اختر الخوارزمية",
        "cluster_viz": "تصورات العناقيد",
        "scatter_plots": "مخططات الانتشار",
        "feature_analysis": "تحليل الخصائص",
        "cluster_stats": "إحصائيات العناقيد",
        "quality_metrics": "مقاييس جودة التجميع",
        "download_results": "تحميل النتائج",
        "download_data": "تحميل البيانات المجمعة (CSV)",
        "download_stats": "تحميل إحصائيات العناقيد (CSV)",
        "data_quality": "معلومات جودة البيانات",
        "missing_values_tab": "القيم المفقودة",
        "feature_ranges": "نطاقات الخصائص",
        "outliers": "القيم الشاذة",
        "algorithm_explanation": {
            "kmeans": """
            **خوارزمية K-Means للتجميع**
            - تجمع البيانات في K من العناقيد بناءً على التشابه
            - كل عنقود له مركز
            - الأفضل للعناقيد الكروية ذات الحجم المتشابه
            - يجب تحديد عدد العناقيد (K)
            """,
            "dbscan": """
            **خوارزمية DBSCAN للتجميع**
            - خوارزمية تجميع قائمة على الكثافة
            - تجد العناقيد بأي شكل
            - تكتشف نقاط الضوضاء تلقائياً
            - لا حاجة لتحديد عدد العناقيد
            - المعلمات:
              - Epsilon: أقصى مسافة بين النقاط في العنقود
              - الحد الأدنى للعينات: الحد الأدنى للنقاط لتشكيل منطقة كثيفة
            """,
            "hierarchical": """
            **التجميع الهرمي**
            - ينشئ شجرة من العناقيد
            - طرق ربط متعددة متاحة
            - يمكن أن يكشف العلاقات الهرمية
            - المعلمات:
              - عدد العناقيد
              - طريقة الربط (تحدد كيفية قياس المسافات)
            """
        },
        "metric_explanations": {
            "silhouette": "يقيس مدى تشابه العنصر مع عنقوده مقارنة بالعناقيد الأخرى (النطاق: -1 إلى 1، الأعلى أفضل)",
            "calinski": "نسبة التباين بين العناقيد إلى التباين داخل العناقيد (الأعلى أفضل)",
            "davies": "متوسط مقياس التشابه لكل عنقود مع أقرب عنقود له (الأقل أفضل)"
        }
    }
}

# Function to get translated text
def get_text(key, lang="en"):
    # Handle nested dictionary keys
    if "." in key:
        main_key, sub_key = key.split(".")
        return translations[lang][main_key][sub_key]
    return translations[lang].get(key, key)

# Set page config
st.set_page_config(
    page_title=get_text("app_title"),
    page_icon="📊",
    layout="wide"
)

# Custom CSS with RTL support
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .uploadedFile {
        margin: 2rem 0;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .explanation-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Language selection
lang = st.sidebar.selectbox(
    get_text("language"),
    ["English", "العربية"],
    key="language_selector"
)

# Set language code
lang_code = "en" if lang == "English" else "ar"

# Add CSS class for RTL if Arabic is selected
if lang_code == "ar":
    st.markdown("""
        <style>
        .rtl-content {
            direction: rtl;
            text-align: right;
        }
        </style>
    """, unsafe_allow_html=True)

# Title and description
st.title(get_text("app_title", lang_code))
st.markdown(get_text("app_description", lang_code))

# Load example data
@st.cache_data
def load_example_data():
    np.random.seed(42)
    n_samples = 300
    
    # Generate three clusters
    cluster1 = np.random.normal(0, 1, (n_samples//3, 3))
    cluster2 = np.random.normal(5, 1, (n_samples//3, 3))
    cluster3 = np.random.normal(-5, 1, (n_samples//3, 3))
    
    # Combine clusters
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Add some noise and missing values
    mask = np.random.random(data.shape) < 0.1
    data[mask] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    return df

# Sidebar
with st.sidebar:
    st.header(get_text("settings"))
    
    # Data source selection
    data_option = st.radio(
        get_text("data_source"),
        [get_text("use_example"), get_text("upload_data")]
    )
    
    if data_option == get_text("upload_data"):
        uploaded_file = st.file_uploader(get_text("upload_data"), type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                df = None
    else:
        df = load_example_data()
        st.info(get_text("example_info"))
    
    if df is not None:
        # Feature selection
        st.subheader(get_text("feature_selection"))
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        selected_features = st.multiselect(
            get_text("choose_features"),
            options=numeric_columns,
            default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
        )
        
        # Preprocessing options
        st.subheader(get_text("preprocessing"))
        scaling_method = st.selectbox(
            get_text("scaling_method"),
            ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"],
            help=get_text("scaling_method")
        )
        
        impute_strategy = st.selectbox(
            get_text("missing_values"),
            ["mean", "median", "most_frequent", "constant"],
            help=get_text("missing_values")
        )
        
        if impute_strategy == "constant":
            fill_value = st.number_input(get_text("fill_value"), value=0)
        
        # Dimensionality reduction
        if len(selected_features) > 2:
            use_pca = st.checkbox(get_text("pca_option"), value=True)
            if use_pca:
                n_components = st.slider(get_text("pca_option"), 2, 
                                      min(len(selected_features), 5), 2)
        
        # Algorithm selection
        st.subheader(get_text("algorithm_selection"))
        algorithm = st.selectbox(
            get_text("choose_algorithm"),
            ["K-Means", "DBSCAN", "Hierarchical"],
            help=get_text("algorithm_selection")
        )
        
        # Algorithm-specific parameters
        if algorithm == "K-Means":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
        elif algorithm == "DBSCAN":
            eps = st.slider("Epsilon (neighborhood size)", 0.1, 2.0, 0.5)
            min_samples = st.slider("Minimum samples per cluster", 2, 10, 5)
        else:  # Hierarchical
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            linkage = st.selectbox("Linkage method", 
                                 ["ward", "complete", "average", "single"])

# Main content
if df is not None and len(selected_features) >= 2:
    # Prepare data
    X = df[selected_features].copy()
    
    # Handle missing values
    if impute_strategy == "constant":
        imputer = SimpleImputer(strategy=impute_strategy, fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=impute_strategy)
    X_imputed = imputer.fit_transform(X)
    
    # Scale the data
    if scaling_method == "Standard Scaling":
        scaler = StandardScaler()
    elif scaling_method == "Min-Max Scaling":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
    
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Apply PCA if selected
    if len(selected_features) > 2 and 'use_pca' in locals() and use_pca:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        # Create feature names for PCA components
        pca_features = [f'PC{i+1}' for i in range(n_components)]
        X_viz = pd.DataFrame(X_pca, columns=pca_features)
        
        # Display explained variance
        explained_var = pca.explained_variance_ratio_
        st.subheader(get_text("pca_option"))
        fig_var = px.bar(
            x=[f'PC{i+1}' for i in range(len(explained_var))],
            y=explained_var,
            title=get_text("pca_option")
        )
        st.plotly_chart(fig_var, use_container_width=True)
    else:
        X_viz = pd.DataFrame(X_scaled, columns=selected_features)
    
    # Perform clustering
    if algorithm == "K-Means":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "DBSCAN":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage
        )
    
    clusters = clusterer.fit_predict(X_scaled)
    
    # Store clustered data
    clustered_df = df.copy()
    clustered_df['Cluster'] = clusters
    st.session_state.clustered_data = clustered_df
    
    # Calculate clustering metrics
    if algorithm != "DBSCAN":  # Some metrics require predicted clusters
        silhouette = silhouette_score(X_scaled, clusters)
        calinski = calinski_harabasz_score(X_scaled, clusters)
        davies = davies_bouldin_score(X_scaled, clusters)
        
        # Display metrics
        st.subheader(get_text("quality_metrics"))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", f"{silhouette:.3f}",
                     help=get_text("metric_explanations.silhouette"))
        with col2:
            st.metric("Calinski-Harabasz Score", f"{calinski:.1f}",
                     help=get_text("metric_explanations.calinski"))
        with col3:
            st.metric("Davies-Bouldin Score", f"{davies:.3f}",
                     help=get_text("metric_explanations.davies"))
    
    # Visualizations
    st.subheader(get_text("cluster_viz"))
    viz_tabs = st.tabs([get_text("scatter_plots"), get_text("feature_analysis"), get_text("cluster_stats")])
    
    with viz_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # 2D Scatter plot
            fig1 = px.scatter(
                X_viz,
                x=X_viz.columns[0],
                y=X_viz.columns[1],
                color=clusters,
                title=f'{get_text("cluster_viz")}: {X_viz.columns[0]} vs {X_viz.columns[1]}',
                template='plotly_white'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            if len(X_viz.columns) >= 3:
                # 3D Scatter plot
                fig2 = px.scatter_3d(
                    X_viz,
                    x=X_viz.columns[0],
                    y=X_viz.columns[1],
                    z=X_viz.columns[2],
                    color=clusters,
                    title=get_text("cluster_viz"),
                    template='plotly_white'
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with viz_tabs[1]:
        # Feature importance and correlation analysis
        st.subheader(get_text("feature_analysis"))
        
        # Correlation matrix
        corr_matrix = X.corr()
        fig_corr = px.imshow(
            corr_matrix,
            title=get_text("feature_analysis"),
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature distributions by cluster
        st.subheader(get_text("feature_analysis"))
        feature_to_analyze = st.selectbox(get_text("choose_features"), selected_features)
        fig_dist = px.box(
            clustered_df,
            x='Cluster',
            y=feature_to_analyze,
            title=f'{get_text("feature_analysis")} {feature_to_analyze} {get_text("cluster_viz")}'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with viz_tabs[2]:
        # Cluster statistics
        st.subheader(get_text("cluster_stats"))
        
        # Calculate statistics for each cluster
        cluster_stats = clustered_df.groupby('Cluster')[selected_features].agg([
            'mean', 'std', 'min', 'max'
        ]).round(2)
        
        # Display statistics
        st.write(get_text("cluster_stats"))
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Cluster sizes
        cluster_sizes = clustered_df['Cluster'].value_counts().sort_index()
        fig_sizes = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            title=get_text("cluster_stats"),
            labels={'x': get_text("cluster_viz"), 'y': 'Number of Samples'}
        )
        st.plotly_chart(fig_sizes, use_container_width=True)
    
    # Data Quality Information
    st.subheader(get_text("data_quality"))
    quality_tabs = st.tabs([get_text("missing_values_tab"), get_text("feature_ranges"), get_text("outliers")])
    
    with quality_tabs[0]:
        missing_values = X.isnull().sum()
        if missing_values.sum() > 0:
            st.write(get_text("missing_values_tab"))
            for feature, count in missing_values.items():
                if count > 0:
                    st.write(f"- {feature}: {count} values imputed using {impute_strategy}")
        else:
            st.write(get_text("missing_values_tab"))
    
    with quality_tabs[1]:
        feature_ranges = pd.DataFrame({
            'Min': X.min(),
            'Max': X.max(),
            'Range': X.max() - X.min(),
            'Std Dev': X.std()
        }).round(2)
        st.write(get_text("feature_ranges"))
        st.dataframe(feature_ranges, use_container_width=True)
    
    with quality_tabs[2]:
        st.write(get_text("outliers"))
        fig_box = px.box(X, title=get_text("outliers"))
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Download options
    st.subheader(get_text("download_results"))
    col1, col2 = st.columns(2)
    with col1:
        csv = clustered_df.to_csv(index=False)
        st.download_button(
            label=get_text("download_data"),
            data=csv,
            file_name="clustered_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export cluster statistics
        stats_csv = cluster_stats.to_csv()
        st.download_button(
            label=get_text("download_stats"),
            data=stats_csv,
            file_name="cluster_statistics.csv",
            mime="text/csv"
        )
            
else:
    if df is None:
        st.info(get_text("upload_data"))
    elif len(selected_features) < 2:
        st.warning(get_text("choose_features"))
