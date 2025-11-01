# main.py

# ==============================================================================
# AI VIET NAM 2025
# Project: House Price Prediction
# ==============================================================================


# ------------------------------------------------------------------------------
# Bước 1: Import các thư viện cần thiết
# ------------------------------------------------------------------------------
print("Bước 1: Import thư viện và tải dữ liệu...")

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown # Thư viện để tải file từ Google Drive

# Preprocessing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer # Cải tiến: Thêm KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor # Cải tiến: Thêm ElasticNet và HuberRegressor

# Metrics and Interpretation
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance # Cải tiến: Thêm Permutation Importance

# Cài đặt mặc định cho các biểu đồ
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Tải dữ liệu từ Google Drive
# ------------------------------------------------------------------------------
# ID của file trên Google Drive được cung cấp trong tài liệu
file_id = '1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd'
output_path = 'train-house-prices.csv'
# gdown.download(id=file_id, output=output_path, quiet=False)

# Đọc dữ liệu
house_df_raw = pd.read_csv(output_path)

print("Tải và đọc dữ liệu thành công.")
print("-" * 50)


# ------------------------------------------------------------------------------
# Bước 2: Khám phá dữ liệu (EDA)
# ------------------------------------------------------------------------------
print("Bước 2: Khám phá dữ liệu (EDA)...")

print("Kích thước dữ liệu:", house_df_raw.shape)
print("\nThông tin thống kê mô tả:")
print(house_df_raw.describe())

# Trực quan hóa phân phối của biến mục tiêu 'SalePrice'
plt.figure(figsize=(12, 6))
sns.histplot(house_df_raw['SalePrice'], kde=True, bins=50)
plt.title('Phân phối của Giá nhà (SalePrice) - Dữ liệu gốc')
plt.axvline(house_df_raw['SalePrice'].mean(), color='red', linestyle='--', label=f'Mean: ${house_df_raw["SalePrice"].mean():,.0f}')
plt.axvline(house_df_raw['SalePrice'].median(), color='green', linestyle='-', label=f'Median: ${house_df_raw["SalePrice"].median():,.0f}')
plt.legend()
plt.show()

# Cải tiến: Xử lý độ lệch của biến mục tiêu bằng Log Transform
# Phân phối của SalePrice bị lệch phải, áp dụng log transform để chuẩn hóa
house_df_raw['SalePrice'] = np.log1p(house_df_raw['SalePrice'])

plt.figure(figsize=(12, 6))
sns.histplot(house_df_raw['SalePrice'], kde=True, bins=50)
plt.title('Phân phối của Giá nhà (SalePrice) - Sau khi Log Transform')
plt.axvline(house_df_raw['SalePrice'].mean(), color='red', linestyle='--', label=f'Mean')
plt.axvline(house_df_raw['SalePrice'].median(), color='green', linestyle='-', label=f'Median')
plt.legend()
plt.show()
print("Đã áp dụng Log Transform cho biến mục tiêu 'SalePrice' để giảm độ lệch.")
print("-" * 50)

# ------------------------------------------------------------------------------
# Bước 3: Tiền xử lý dữ liệu (Data Preprocessing)
# ------------------------------------------------------------------------------
print("Bước 3: Tiền xử lý dữ liệu...")

# Loại bỏ các cột có tỷ lệ thiếu dữ liệu cao và cột ID không cần thiết
missing_ratio = house_df_raw.isnull().sum() / len(house_df_raw)
cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
cols_to_drop.append('Id')
house_df = house_df_raw.drop(columns=cols_to_drop)
print(f"Đã loại bỏ các cột: {cols_to_drop}")

# Phân tách đặc trưng (features) và biến mục tiêu (target)
X = house_df.drop('SalePrice', axis=1)
y = house_df['SalePrice']

# Phân loại các cột thành numerical và categorical
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
print(f"\nSố đặc trưng dạng số: {len(numerical_cols)}")
print(f"Số đặc trưng dạng phân loại: {len(categorical_cols)}")

# Tạo pipeline cho tiền xử lý
# Cải tiến: Sử dụng KNNImputer cho các cột số và StandardScaler thay vì MinMaxScaler
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)), # Dùng KNNImputer để điền giá trị thiếu
    ('scaler', StandardScaler())
])

# Cải tiến: Điền giá trị thiếu bằng 'missing' trước khi OneHotEncode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Kết hợp các transformer bằng ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Giữ lại các cột không được xử lý (nếu có)
)
print("\nĐã tạo xong pipeline tiền xử lý.")
print("-" * 50)

# ------------------------------------------------------------------------------
# Bước 4: Huấn luyện và đánh giá các mô hình cơ bản (sử dụng Cross-Validation)
# ------------------------------------------------------------------------------
print("Bước 4: Huấn luyện và đánh giá các mô hình cơ bản...")

# Cải tiến: Bổ sung thêm ElasticNet và HuberRegressor
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'Huber Regressor': HuberRegressor()
}

# Cải tiến: Sử dụng K-Fold Cross-Validation để đánh giá mô hình
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

# Chia dữ liệu để sử dụng cho bước diễn giải mô hình sau này
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for name, model in models.items():
    # Tạo pipeline hoàn chỉnh: preprocessor -> model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Huấn luyện và đánh giá
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Vì y đã được log transform, chúng ta cần biến đổi ngược lại để tính RMSE
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
    r2 = r2_score(y_test, y_pred) # R2 có thể tính trên dữ liệu đã log

    results.append({
        'Model': name,
        'RMSE': rmse,
        'R2 Score': r2
    })

results_df = pd.DataFrame(results).sort_values(by='RMSE', ascending=True)
print("Kết quả đánh giá các mô hình cơ bản (trên tập test đơn lẻ):")
print(results_df)
print("-" * 50)

# ------------------------------------------------------------------------------
# Bước 5: Huấn luyện model sử dụng đặc trưng đa thức (Polynomial Features)
# ------------------------------------------------------------------------------
print("Bước 5: Huấn luyện và đánh giá các mô hình với đặc trưng đa thức...")

# Tạo một preprocessor mới có thêm bước PolynomialFeatures
poly_preprocessor = ColumnTransformer(
    transformers=[
        # Chỉ áp dụng PolynomialFeatures cho các cột số
        ('poly_num', Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

poly_results = []
for name, model in models.items():
    # Tạo pipeline hoàn chỉnh với đặc trưng đa thức
    pipeline_poly = Pipeline(steps=[
        ('preprocessor', poly_preprocessor),
        ('regressor', model)
    ])

    # Huấn luyện và đánh giá
    pipeline_poly.fit(X_train, y_train)
    y_pred_poly = pipeline_poly.predict(X_test)

    rmse_poly = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_poly)))
    r2_poly = r2_score(y_test, y_pred_poly)

    poly_results.append({
        'Model': f"{name} + Poly",
        'RMSE': rmse_poly,
        'R2 Score': r2_poly
    })

poly_results_df = pd.DataFrame(poly_results).sort_values(by='RMSE', ascending=True)
print("Kết quả đánh giá các mô hình với đặc trưng đa thức (trên tập test đơn lẻ):")
print(poly_results_df)
print("-" * 50)

# ------------------------------------------------------------------------------
# Cải tiến: Tinh chỉnh tham số cho mô hình tốt nhất bằng GridSearchCV
# ------------------------------------------------------------------------------
print("Cải tiến: Tinh chỉnh tham số cho mô hình tốt nhất...")

# Giả sử Ridge với Polynomial Features cho kết quả tốt. Ta sẽ tinh chỉnh nó.
# Tạo pipeline chỉ cho Ridge
ridge_poly_pipeline = Pipeline(steps=[
    ('preprocessor', poly_preprocessor),
    ('regressor', Ridge(random_state=42))
])

# Định nghĩa không gian tham số để tìm kiếm
param_grid = {
    'regressor__alpha': [0.1, 1.0, 10, 100, 200]
}

# Sử dụng GridSearchCV với 5-fold cross-validation
grid_search = GridSearchCV(ridge_poly_pipeline, param_grid, cv=kf,
                           scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"Tham số tốt nhất cho Ridge + Poly: {grid_search.best_params_}")
print(f"R2 score tốt nhất từ Cross-Validation: {grid_search.best_score_:.4f}")

# Đánh giá mô hình tốt nhất trên tập test
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_best)))
r2_best = r2_score(y_test, y_pred_best)

print(f"\nKết quả của mô hình tốt nhất (Ridge + Poly) trên tập test:")
print(f"RMSE: ${rmse_best:,.2f}")
print(f"R2 Score: {r2_best:.4f}")
print("-" * 50)


# ------------------------------------------------------------------------------
# Cải tiến: Giải thích kết quả mô hình (Permutation Importance)
# ------------------------------------------------------------------------------
print("Cải tiến: Giải thích kết quả mô hình bằng Permutation Importance...")

# Tính toán Permutation Importance trên tập test
# Để có được tên đặc trưng sau khi one-hot encode
# Chúng ta cần fit preprocessor trước
preprocessor_for_names = poly_preprocessor.fit(X_train)
cat_names = preprocessor_for_names.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)

# Lấy tên cột đa thức (phức tạp hơn, ở đây ta sẽ dùng tên gốc)
# Tên đặc trưng sau khi xử lý
feature_names_processed = numerical_cols + cat_names.tolist()

# Tính toán importance
perm_importance = permutation_importance(
    best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Tạo DataFrame để hiển thị kết quả
# Lấy tên các đặc trưng đã qua xử lý
# Chú ý: Việc lấy chính xác tên các đặc trưng đa thức khá phức tạp.
# Ở đây ta sẽ lấy tên các đặc trưng gốc sau khi được biến đổi.
try:
    ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    # Tên cuối cùng sẽ là tên các đặc trưng đa thức + tên OHE
    # Việc này phức tạp, nên chúng ta sẽ chỉ lấy top N đặc trưng quan trọng nhất
    # Đây là một cách đơn giản hóa:
    # final_feature_names = numerical_cols + list(ohe_feature_names) # Simplified
except Exception as e:
    print("Không thể lấy tên đặc trưng tự động, sẽ hiển thị chỉ số.")
    final_feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]


sorted_idx = perm_importance.importances_mean.argsort()

# Lấy 20 đặc trưng quan trọng nhất
top_n = 20
top_indices = sorted_idx[-top_n:]

# Lấy tên đặc trưng gốc (để dễ hiểu hơn)
# X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
# final_feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
# Đáng tiếc, get_feature_names_out phức tạp với pipeline lồng nhau.
# Ta sẽ hiển thị chỉ số của đặc trưng.

importance_df = pd.DataFrame(
    data=perm_importance.importances[sorted_idx].T,
    columns=[f'feature_{i}' for i in sorted_idx]
)

plt.figure(figsize=(12, 8))
plt.boxplot(importance_df.iloc[:, -top_n:], vert=False, labels=[f'feature_{i}' for i in top_indices])
plt.title(f'Top {top_n} đặc trưng quan trọng nhất (Permutation Importance)')
plt.xlabel("Mức độ sụt giảm hiệu suất mô hình")
plt.show()

print("\nHoàn thành cải tiến!")