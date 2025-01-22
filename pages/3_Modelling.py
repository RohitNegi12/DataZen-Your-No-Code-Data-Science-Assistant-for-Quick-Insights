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
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import pickle
import os
import json


# Streamlit App Title
st.title("Modeling")

data = st.session_state.get('dataset')


if data is None:
    st.warning("Please upload a dataset on the Home page first.")
else:
    # Step 1: User input for target feature
    target_feature = st.selectbox("Select Target Feature:", data.columns)

    if target_feature:


        # Determine if target is categorical or continuous
        target_type = "categorical" if data[target_feature].dtype == "object" else "continuous"
        st.write(f"Target feature '{target_feature}' is **{target_type}**.")
    
        if st.button("Train Model",key="model training"):

            # Step 2: Split dataset into features and target
            X = data.drop(columns=[target_feature])
            y = data[target_feature]

            # Step 3: Feature Selection with RFE
            st.subheader("Feature Selection")
            model_for_rfe = LogisticRegression(max_iter=1000) if target_type == "categorical" else LinearRegression()
            rfe = RFE(model_for_rfe, n_features_to_select=5)
            rfe.fit(X.select_dtypes(include=['number']), y)
            selected_features = X.select_dtypes(include=['number']).columns[rfe.support_]

            st.write("### Selected Features:")
            st.write(list(selected_features))

            # Keep only selected features
            X = X[selected_features]

            # Step 4: Preprocessing
            st.subheader("Preprocessing")
            categorical_cols = X.select_dtypes(include=['object']).columns
            numerical_cols = X.select_dtypes(include=['number']).columns

            preprocess = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ]
            )

            # Step 5: Polynomial Features (Optional)
            if st.checkbox("Add Polynomial Features"):
                poly_degree = st.slider("Select Polynomial Degree:", 2, 3, 2)
                preprocess = Pipeline([
                    ('preprocessing', preprocess),
                    ('polynomial', PolynomialFeatures(degree=poly_degree))
                ])

            # Step 6: Models and Training
            # st.subheader("Model Training")
            if target_type == "continuous":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Gradient Boosting Regressor": GradientBoostingRegressor()
                }
            else:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "Gradient Boosting Classifier": GradientBoostingClassifier()
                }

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            results = []
            best_model = None
            best_score = -np.inf

        
            for model_name, model in models.items():
                # Pipeline
                pipeline = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # Scoring
                if target_type == "continuous":
                    score = r2_score(y_test, y_pred)
                else:
                    score = accuracy_score(y_test, y_pred)

                results.append({"Model": model_name, "Score": score})

                # Save the best model
                if score > best_score:
                    best_score = score
                    best_model = pipeline

                # Save model as .pkl
                with open(f"models/{model_name}.pkl", "wb") as f:
                    pickle.dump(pipeline, f)


            modeling_results = {
            "selected_features": list(selected_features),
            "target_feature": target_feature,
            "target_type": target_type,
            "model_performance": [],
            "best_model": None,
            "confusion_matrix": None,
            "shap_values": None,
            "residuals": None
                    }
            
            
            
            # Add model performance to the results
            for result in results:
                modeling_results["model_performance"].append({
                    "model": result["Model"],
                    "score": result["Score"]
                })


            # Display Results with Plotly
            results_df = pd.DataFrame(results)
            st.write("### Model Performance:")
            fig = px.bar(results_df, x="Model", y="Score", color="Score", title="Model Performance")
            st.plotly_chart(fig)

            # Add best model details
            modeling_results["best_model"] = {
                "model": results_df.loc[results_df['Score'].idxmax(), 'Model'],
                "score": best_score
            }

            # Best Model Recommendation
            st.write(f"### Best Model: {results_df.loc[results_df['Score'].idxmax(), 'Model']} with Score: {best_score:.4f}")

            # Step 7: Advanced Model Evaluation
            st.subheader("Advanced Model Evaluation")
            if target_type == "categorical":
                cm = confusion_matrix(y_test, best_model.predict(X_test))
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                labels=dict(x="Predicted", y="Actual"),
                                title="Confusion Matrix")
                st.plotly_chart(fig_cm)
            else:
                residuals = y_test - best_model.predict(X_test)
                fig_residuals = px.scatter(x=y_test, y=residuals, labels={"x": "Actual Values", "y": "Residuals"},
                                        title="Residual Plot")
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals)


            # Add confusion matrix (if applicable)
            if target_type == "categorical":
                cm = confusion_matrix(y_test, best_model.predict(X_test))
                modeling_results["confusion_matrix"] = cm.tolist()  # Convert NumPy array to list for JSON serialization
            else:
                # Add residuals for regression
                residuals = (y_test - best_model.predict(X_test)).tolist()
                modeling_results["residuals"] = residuals

            # Step 8: Model Explainability with SHAP
            st.subheader("Model Explainability")
            if st.checkbox("Show SHAP Values"):
                explainer = shap.Explainer(best_model.named_steps['model'], best_model.named_steps['preprocess'].transform(X_train))
                shap_values = explainer(X_test)
                # Create the SHAP summary plot
                shap_summary = shap_values.values.tolist()  # Convert SHAP values to list for JSON
                modeling_results["shap_values"] = shap_summary
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(plt.gcf())  # Display the Matplotlib plot in Streamlit

            # Step 9: Provide Download Links
            st.subheader("Download Models")
            for model_name in models.keys():
                with open(f"models/{model_name}.pkl", "rb") as f:
                    st.download_button(
                        label=f"Download {model_name} Model",
                        data=f,
                        file_name=f"{model_name}.pkl"
                    )

            # Save the results as a JSON file
            output_path = os.path.join("models/modeling_results.json")
            with open(output_path, "w") as json_file:
                json.dump(modeling_results, json_file, indent=4)

            st.success(f"Modeling results have been saved to {output_path}")

       
       
