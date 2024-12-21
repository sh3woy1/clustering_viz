# 📊 Enhanced Clustering Visualization Tool | أداة تحليل وتصور العناقيد المتقدم

An advanced, bilingual (English/Arabic) clustering analysis tool built with Streamlit. This interactive application allows users to perform cluster analysis using multiple algorithms, evaluate results with various metrics, and visualize patterns in their data.

[English](#english) | [العربية](#arabic)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.24.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## English

### 🌟 Features

#### Multiple Clustering Algorithms
- **K-Means**: Traditional centroid-based clustering
- **DBSCAN**: Density-based clustering for non-spherical clusters
- **Hierarchical**: Tree-based clustering with different linkage options

#### Advanced Data Preprocessing
- Multiple scaling methods:
  - Standard Scaling
  - Min-Max Scaling
  - Robust Scaling
- Missing value handling strategies
- Dimensionality reduction using PCA

#### Comprehensive Analysis
- Interactive visualizations (2D and 3D)
- Multiple evaluation metrics:
  - Silhouette Score
  - Calinski-Harabasz Score
  - Davies-Bouldin Score
- Feature correlation analysis
- Cluster statistics and distributions

#### Data Quality Assessment
- Missing value analysis
- Feature range information
- Outlier detection
- Downloadable results

### 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/sh3woy1/clustering-viz.git
cd clustering-viz
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

### 📦 Requirements
- Python 3.8+
- Streamlit 1.24.0
- Pandas 1.5.3
- NumPy 1.23.5
- Scikit-learn 1.2.2
- Plotly 5.15.0

### 🎯 Usage

1. **Data Input**
   - Upload your CSV file or use the example dataset
   - Select features for clustering

2. **Preprocessing**
   - Choose scaling method
   - Handle missing values
   - Apply dimensionality reduction (optional)

3. **Algorithm Selection**
   - Choose clustering algorithm
   - Set algorithm-specific parameters
   - View clustering results and metrics

4. **Analysis**
   - Explore interactive visualizations
   - Analyze cluster statistics
   - Download results

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## العربية

### 🌟 المميزات

#### خوارزميات التجميع المتعددة
- **K-Means**: خوارزمية التجميع المركزية التقليدية
- **DBSCAN**: خوارزمية التجميع القائمة على الكثافة
- **التجميع الهرمي**: التجميع القائم على الشجرة مع خيارات ربط مختلفة

#### معالجة البيانات المتقدمة
- طرق تحجيم متعددة:
  - التحجيم المعياري
  - تحجيم Min-Max
  - التحجيم القوي
- استراتيجيات معالجة القيم المفقودة
- تقليل الأبعاد باستخدام PCA

#### تحليل شامل
- تصورات تفاعلية (ثنائية وثلاثية الأبعاد)
- مقاييس تقييم متعددة:
  - معامل Silhouette
  - معامل Calinski-Harabasz
  - معامل Davies-Bouldin
- تحليل الارتباط بين الخصائص
- إحصائيات وتوزيعات العناقيد

#### تقييم جودة البيانات
- تحليل القيم المفقودة
- معلومات نطاق الخصائص
- كشف القيم الشاذة
- نتائج قابلة للتحميل

### 🚀 البدء السريع

1. **استنساخ المستودع**
```bash
git clone https://github.com/yourusername/clustering-viz.git
cd clustering-viz
```

2. **تثبيت المتطلبات**
```bash
pip install -r requirements.txt
```

3. **تشغيل التطبيق**
```bash
streamlit run app.py
```

### 📦 المتطلبات
- Python 3.8+
- Streamlit 1.24.0
- Pandas 1.5.3
- NumPy 1.23.5
- Scikit-learn 1.2.2
- Plotly 5.15.0

### 🎯 الاستخدام

1. **إدخال البيانات**
   - تحميل ملف CSV أو استخدام مجموعة البيانات المثال
   - اختيار الخصائص للتجميع

2. **المعالجة المسبقة**
   - اختيار طريقة التحجيم
   - معالجة القيم المفقودة
   - تطبيق تقليل الأبعاد (اختياري)

3. **اختيار الخوارزمية**
   - اختيار خوارزمية التجميع
   - تعيين معلمات الخوارزمية
   - عرض نتائج ومقاييس التجميع

4. **التحليل**
   - استكشاف التصورات التفاعلية
   - تحليل إحصائيات العناقيد
   - تحميل النتائج

### 🤝 المساهمة

نرحب بالمساهمات! لا تتردد في تقديم طلب سحب.

1. انسخ المستودع
2. أنشئ فرع الميزة الخاص بك (`git checkout -b feature/AmazingFeature`)
3. قم بتأكيد تغييراتك (`git commit -m 'Add some AmazingFeature'`)
4. ادفع إلى الفرع (`git push origin feature/AmazingFeature`)
5. افتح طلب سحب

### 📄 License | الترخيص

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

هذا المشروع مرخص بموجب رخصة MIT - راجع ملف [LICENSE](LICENSE) للحصول على التفاصيل.
