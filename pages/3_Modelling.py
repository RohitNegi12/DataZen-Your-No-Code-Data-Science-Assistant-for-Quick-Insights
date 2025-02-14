import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import pickle

# Streamlit App Title
st.title("Modeling")

data = st.session_state.get('dataset')

if data is None:
    st.warning("Please upload a dataset on the Home page first.")
else:
    target_feature = st.selectbox("Select Target Feature:", data.columns)

    if target_feature:
        target_type = "categorical" if data[target_feature].dtype == "object" else "continuous"
        st.write(f"Target feature '{target_feature}' is **{target_type}**.")

        X = data.drop(columns=[target_feature])
        y = data[target_feature]

        try:
            # Feature Selection Option
            use_feature_selection = st.checkbox("Apply Feature Selection")
            if use_feature_selection:
                model_for_rfe = LogisticRegression(max_iter=1000) if target_type == "categorical" else LinearRegression()
                rfe = RFE(model_for_rfe, n_features_to_select=5)
                rfe.fit(X.select_dtypes(include=['number']), y)
                selected_features = X.select_dtypes(include=['number']).columns[rfe.support_]
                X = X[selected_features]
                st.write("### Selected Features:")
                st.write(list(selected_features))
        except Exception as e:
            st.error("Unable to apply Feature selection")

        st.subheader("Preprocessing")
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns

        preprocess = ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        if st.checkbox("Add Polynomial Features"):
            poly_degree = st.slider("Select Polynomial Degree:", 2, 3, 2)
            preprocess = Pipeline([
                ('preprocessing', preprocess),
                ('polynomial', PolynomialFeatures(degree=poly_degree))
            ])

        st.subheader("Model Evaluation")

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor()
        } if target_type == "continuous" else {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier()
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_model = None
        best_score = -np.inf

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "R2 Score": r2_score(y_test, y_pred),
                "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
                "Mean Squared Error": mean_squared_error(y_test, y_pred)
            } if target_type == "continuous" else {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test, y_pred, average='weighted')
            }

            fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()), title=f"{model_name} Performance", labels={'x': 'Metric', 'y': 'Score'})
            st.plotly_chart(fig)

            overall_score = np.mean(list(metrics.values()))
            if overall_score > best_score:
                best_score = overall_score
                best_model = model_name  # Store only model name

            with open(f"{model_name}.pkl", "wb") as f:
                pickle.dump(pipeline, f)

        

        
        if target_type == "categorical":
            cm = confusion_matrix(y_test, pipeline.predict(X_test))
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
            st.plotly_chart(fig_cm)
        else:
            residuals = y_test - pipeline.predict(X_test)
            fig_residuals = px.scatter(x=y_test, y=residuals, labels={"x": "Actual Values", "y": "Residuals"}, title="Residual Plot")
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals)

        st.write(f"### Best Model: {best_model} with Score: {best_score:.4f}")

        st.subheader("Download Models")
        for model_name in models.keys():
            with open(f"{model_name}.pkl", "rb") as f:
                st.download_button(label=f"Download {model_name} Model", data=f, file_name=f"{model_name}.pkl")

        st.success("Modeling results have been saved successfully.")
