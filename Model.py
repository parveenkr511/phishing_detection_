import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv('/kaggle/input/datasetttt/enhanced_features_dataset_async.csv')

# 2. Preprocessing
df.drop_duplicates(subset='URL', inplace=True)  # Remove duplicates
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# 3. Features and target
feature_cols = [
    'url_length', 'num_dots', 'num_hyphens', 'num_at', 'has_https',
    'has_ip', 'num_subdirs', 'num_parameters', 'num_percent', 'num_www',
    'num_digits', 'num_letters', 'is_live', 'page_title_length',
    'num_forms', 'num_links', 'num_input_fields', 'has_login_keyword',
    'domain_similarity_score',
    # 'domain_age_days'  # Uncomment if you've added this feature
]
X = df[feature_cols]
y = df['label']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 5. Define base models
xgb = XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
lr = LogisticRegression(max_iter=1000)

# 6. Stacking ensemble
stacked_model = Pipeline([
    ('scaler', StandardScaler()),
    ('stack', StackingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False,
        cv=5,
        n_jobs=-1
    ))
])

# 7. Train model
print("\n🚀 Training model...")
stacked_model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = stacked_model.predict(X_test)

print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# 9. Cross-validation (optional but recommended)
cv_scores = cross_val_score(stacked_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(" Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# 10. Save model
joblib.dump(stacked_model, 'phishing_url_detection.pkl')
print("\n Model saved to phishing_url_detection.pkl")
