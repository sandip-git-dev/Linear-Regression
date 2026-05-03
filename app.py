from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


st.set_page_config(
    page_title="Linear Regression Learning Studio",
    page_icon="📈",
    layout="wide",
)


@dataclass
class PrepArtifacts:
    imputations: dict
    scalers: dict
    encoders: dict
    processed_feature_names: list


def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    X_bias = np.c_[np.ones(m), X]
    theta = np.zeros(n + 1)
    cost_history = []
    snapshots = []

    for iteration in range(iterations):
        predictions = X_bias @ theta
        errors = predictions - y
        gradients = (X_bias.T @ errors) / m
        theta = theta - learning_rate * gradients
        cost = np.sum(errors ** 2) / (2 * m)
        cost_history.append(cost)

        if iteration < 5 or iteration == iterations - 1:
            snapshots.append(
                {
                    "iteration": iteration + 1,
                    "cost": float(cost),
                    "gradients": gradients.copy(),
                    "theta": theta.copy(),
                }
            )

    return theta, cost_history, snapshots


def predict_with_theta(theta, X):
    X_bias = np.c_[np.ones(len(X)), X]
    return X_bias @ theta


def infer_types(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in df.columns if column not in numeric]
    return numeric, categorical


def preprocess_dataframe(df, target_column, selected_features, missing_strategy, encoding_strategy, scaling_strategy):
    working = df[selected_features + [target_column]].copy()
    imputations = {}
    scalers = {}
    encoders = {}

    if missing_strategy == "Drop rows":
        before_rows = len(working)
        working = working.dropna()
        dropped_rows = before_rows - len(working)
    else:
        dropped_rows = 0
        for column in working.columns:
            if pd.api.types.is_numeric_dtype(working[column]):
                fill_value = float(working[column].mean())
            else:
                mode_series = working[column].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            working[column] = working[column].fillna(fill_value)
            imputations[column] = fill_value

    X_frame = pd.DataFrame(index=working.index)

    for feature in selected_features:
        series = working[feature]
        if pd.api.types.is_numeric_dtype(series):
            transformed = series.astype(float).copy()
            if scaling_strategy == "Standardization":
                mean = float(transformed.mean())
                std = float(transformed.std(ddof=0)) or 1.0
                transformed = (transformed - mean) / std
                scalers[feature] = {"type": "standardize", "mean": mean, "std": std}
            elif scaling_strategy == "Normalization":
                min_value = float(transformed.min())
                max_value = float(transformed.max())
                range_value = (max_value - min_value) or 1.0
                transformed = (transformed - min_value) / range_value
                scalers[feature] = {"type": "normalize", "min": min_value, "max": max_value}
            X_frame[feature] = transformed
        else:
            if encoding_strategy == "Label Encoding":
                categories = sorted(series.astype(str).unique().tolist())
                mapping = {category: index for index, category in enumerate(categories)}
                X_frame[feature] = series.astype(str).map(mapping)
                encoders[feature] = {"type": "label", "mapping": mapping}
            elif encoding_strategy == "One-Hot Encoding":
                dummies = pd.get_dummies(series.astype(str), prefix=feature)
                for column in dummies.columns:
                    X_frame[column] = dummies[column]
                encoders[feature] = {"type": "onehot", "categories": sorted(series.astype(str).unique().tolist())}

    y = working[target_column].astype(float)
    artifacts = PrepArtifacts(
        imputations=imputations,
        scalers=scalers,
        encoders=encoders,
        processed_feature_names=X_frame.columns.tolist(),
    )
    return working, X_frame, y, artifacts, dropped_rows


def correlation_table(X_frame, y, target_column):
    corr_df = X_frame.copy()
    corr_df[target_column] = y
    return corr_df.corr(numeric_only=True)


def transform_new_sample(raw_sample, selected_features, artifacts):
    vector = []
    notes = []

    for feature in selected_features:
        value = raw_sample.get(feature)
        if feature in artifacts.scalers or feature not in artifacts.encoders:
            if value in ("", None):
                value = artifacts.imputations.get(feature, 0)
                notes.append(f"{feature} was missing, so the training imputation value was used.")
            numeric_value = float(value)
            scaler = artifacts.scalers.get(feature)
            if scaler:
                if scaler["type"] == "standardize":
                    numeric_value = (numeric_value - scaler["mean"]) / scaler["std"]
                    notes.append(f"{feature} was standardized using the training mean and standard deviation.")
                elif scaler["type"] == "normalize":
                    numeric_value = (numeric_value - scaler["min"]) / ((scaler["max"] - scaler["min"]) or 1.0)
                    notes.append(f"{feature} was normalized using the training minimum and maximum.")
            vector.append(numeric_value)
        else:
            encoder = artifacts.encoders[feature]
            if value in ("", None):
                value = artifacts.imputations.get(feature, "")
                notes.append(f"{feature} was missing, so the training mode value was used.")
            if encoder["type"] == "label":
                vector.append(float(encoder["mapping"].get(str(value), 0)))
                notes.append(f"{feature} was label encoded before prediction.")
            elif encoder["type"] == "onehot":
                for category in encoder["categories"]:
                    vector.append(1.0 if str(value) == category else 0.0)
                notes.append(f"{feature} was converted into one-hot columns before prediction.")

    return np.array(vector, dtype=float), notes


def plot_histogram(series, title):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(series, bins=8, color="#c96b33", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_scatter(x, y, x_label, y_label):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(x, y, color="#1f7a52")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{x_label} vs {y_label}")
    fig.tight_layout()
    return fig


def plot_cost(cost_history):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(cost_history) + 1), cost_history, color="#c96b33")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Cost Convergence")
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(actual, predicted):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(actual, predicted, color="#1f7a52")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    return fig


def plot_regression_line(x, y, theta, feature_name, target_column):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(x, y, color="#1f7a52", label="Samples")
    order = np.argsort(x)
    ax.plot(x[order], theta[0] + theta[1] * x[order], color="#c96b33", linewidth=2.5, label="Regression line")
    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_column)
    ax.set_title("Simple Linear Regression Fit")
    ax.legend()
    fig.tight_layout()
    return fig


st.title("Linear Regression Learning Studio")
st.caption("Python + Streamlit ML project for learning linear regression step by step.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Upload your own dataset in CSV format to continue.")

if uploaded_file is None:
    st.info("Upload a CSV dataset from the sidebar to start the project workflow.")
    st.stop()

df = pd.read_csv(uploaded_file)

numeric_columns, categorical_columns = infer_types(df)
default_target = df.columns[-1]

with st.expander("1. Dataset Input Module", expanded=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("Dataset preview")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("Dataset summary")
        st.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        st.write("Data types")
        st.dataframe(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)}), use_container_width=True)
        st.write(f"Missing cells: {int(df.isna().sum().sum())}")

with st.expander("2. Preprocessing Interface", expanded=True):
    target_column = st.selectbox("Target column", df.columns.tolist(), index=df.columns.get_loc(default_target))
    feature_candidates = [column for column in df.columns if column != target_column]
    selected_features = st.multiselect("Feature selection", feature_candidates, default=feature_candidates)
    missing_strategy = st.selectbox("Missing value handling", ["Mean/Mode Imputation", "Drop rows"])
    encoding_strategy = st.selectbox("Encoding", ["One-Hot Encoding", "Label Encoding"])
    scaling_strategy = st.selectbox("Feature scaling", ["Standardization", "Normalization", "No Scaling"])

    st.info(
        "Why this step? Preprocessing converts the raw dataset into a form linear regression can learn from while showing students exactly what changed."
    )

    if not selected_features:
        st.warning("Select at least one feature to continue.")
        st.stop()

    working_df, X_frame, y, artifacts, dropped_rows = preprocess_dataframe(
        df,
        target_column,
        selected_features,
        missing_strategy,
        encoding_strategy,
        scaling_strategy,
    )

    before_after_col1, before_after_col2 = st.columns(2)
    with before_after_col1:
        st.subheader("Before")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        st.dataframe(df[selected_features + [target_column]].head(), use_container_width=True)
    with before_after_col2:
        st.subheader("After")
        st.write(f"Rows: {working_df.shape[0]}")
        st.write(f"Columns: {X_frame.shape[1] + 1}")
        st.write(f"Dropped rows: {dropped_rows}")
        st.dataframe(pd.concat([X_frame, y.rename(target_column)], axis=1).head(), use_container_width=True)

    st.write("Processed feature names")
    st.write(artifacts.processed_feature_names)

with st.expander("3. Exploratory Data Analysis", expanded=True):
    st.info("Why this step? EDA helps students see whether a feature changes enough and whether it appears to move with the target.")
    feature_for_eda = st.selectbox("Choose a processed feature for EDA", artifacts.processed_feature_names)
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        st.pyplot(plot_histogram(X_frame[feature_for_eda], f"Distribution of {feature_for_eda}"))
    with eda_col2:
        st.pyplot(plot_scatter(X_frame[feature_for_eda], y, feature_for_eda, target_column))

    corr_matrix = correlation_table(X_frame, y, target_column)
    st.write("Correlation analysis")
    st.dataframe(corr_matrix.style.background_gradient(cmap="Oranges"), use_container_width=True)
    st.write(
        f"Correlation between {feature_for_eda} and {target_column}: "
        f"{corr_matrix.loc[feature_for_eda, target_column]:.4f}"
    )

with st.expander("4. Linear Regression Learning Module", expanded=True):
    st.markdown("**Hypothesis formulation**")
    st.code("Simple: y_hat = theta0 + theta1*x\nMultiple: y_hat = theta0 + theta1*x1 + theta2*x2 + ... + thetan*xn")
    st.markdown("**Cost function**")
    st.code("J(theta) = (1 / 2m) * sum((y_hat - y)^2)")
    st.markdown("**Gradient update rule**")
    st.code("theta_j = theta_j - alpha * (1 / m) * sum((y_hat - y) * x_j)")
    st.info(
        "Why this step? Students should connect the equation, the code, and the changing parameter values instead of only seeing the final accuracy score."
    )

with st.expander("5. Training Configuration", expanded=True):
    split_ratio = st.slider("Training ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
    k_folds = st.number_input("K-fold cross validation", min_value=2, max_value=10, value=5, step=1)
    learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f")
    iterations = st.number_input("Iterations", min_value=50, max_value=5000, value=400, step=50)

if len(X_frame) < 2:
    st.error("At least two processed rows are required for training.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X_frame.values,
    y.values,
    train_size=split_ratio,
    random_state=42,
)

theta, cost_history, snapshots = gradient_descent(X_train, y_train, learning_rate, int(iterations))
test_predictions = predict_with_theta(theta, X_test)

metrics = {
    "MSE": mean_squared_error(y_test, test_predictions),
    "MAE": mean_absolute_error(y_test, test_predictions),
    "R2": r2_score(y_test, test_predictions),
}

kf = KFold(n_splits=int(k_folds), shuffle=True, random_state=42)
cv_rows = []
for fold_index, (train_idx, val_idx) in enumerate(kf.split(X_frame.values), start=1):
    fold_theta, _, _ = gradient_descent(X_frame.values[train_idx], y.values[train_idx], learning_rate, int(iterations))
    val_predictions = predict_with_theta(fold_theta, X_frame.values[val_idx])
    cv_rows.append(
        {
            "Fold": fold_index,
            "MSE": mean_squared_error(y.values[val_idx], val_predictions),
            "MAE": mean_absolute_error(y.values[val_idx], val_predictions),
            "R2": r2_score(y.values[val_idx], val_predictions),
        }
    )
cv_df = pd.DataFrame(cv_rows)

with st.expander("6. Model Training & Visualization", expanded=True):
    parameter_df = pd.DataFrame(
        {
            "Parameter": ["Intercept"] + artifacts.processed_feature_names,
            "Value": theta,
        }
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Learned parameters")
        st.dataframe(parameter_df, use_container_width=True)
    with col2:
        st.subheader("Cost convergence")
        st.pyplot(plot_cost(cost_history))

    if len(artifacts.processed_feature_names) == 1:
        st.pyplot(
            plot_regression_line(
                X_frame.values[:, 0],
                y.values,
                theta,
                artifacts.processed_feature_names[0],
                target_column,
            )
        )
    else:
        st.info("Multiple linear regression is active, so the fitted model exists in higher-dimensional space instead of a single 2D line.")

    st.pyplot(plot_predicted_vs_actual(y_test, test_predictions))

with st.expander("7. Prediction & Inference Module", expanded=True):
    st.write("Enter a new sample")
    raw_sample = {}
    input_columns = st.columns(min(3, max(1, len(selected_features))))
    for index, feature in enumerate(selected_features):
        with input_columns[index % len(input_columns)]:
            if pd.api.types.is_numeric_dtype(df[feature]):
                raw_sample[feature] = st.number_input(f"{feature}", value=float(df[feature].dropna().iloc[0]), key=f"pred_{feature}")
            else:
                raw_sample[feature] = st.selectbox(f"{feature}", sorted(df[feature].dropna().astype(str).unique().tolist()), key=f"pred_{feature}")

    transformed_vector, notes = transform_new_sample(raw_sample, selected_features, artifacts)
    prediction_value = theta[0] + np.dot(theta[1:], transformed_vector)
    contributions = []
    for feature_name, weight, value in zip(artifacts.processed_feature_names, theta[1:], transformed_vector):
        contributions.append(
            {
                "Feature": feature_name,
                "Weight": weight,
                "Input value": value,
                "Contribution": weight * value,
            }
        )

    st.write("Step-by-step prediction computation")
    st.dataframe(pd.DataFrame(contributions), use_container_width=True)
    st.write(f"Intercept contribution: {theta[0]:.4f}")
    st.success(f"Final predicted value: {prediction_value:.4f}")
    if notes:
        st.write("Transformation notes")
        for note in notes:
            st.write(f"- {note}")

with st.expander("8. Evaluation Metrics", expanded=True):
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("MSE", f"{metrics['MSE']:.4f}")
    metric_col2.metric("MAE", f"{metrics['MAE']:.4f}")
    metric_col3.metric("R² Score", f"{metrics['R2']:.4f}")
    st.write("Cross-validation results")
    st.dataframe(cv_df, use_container_width=True)
    st.write("Average cross-validation scores")
    st.dataframe(pd.DataFrame([cv_df.mean(numeric_only=True)]), use_container_width=True)

with st.expander("Educational Layer", expanded=True):
    st.write("Worked sample from the training set")
    sample_x = X_train[0]
    sample_y = y_train[0]
    sample_prediction = theta[0] + np.dot(theta[1:], sample_x)
    sample_error = sample_prediction - sample_y
    worked_rows = []
    for feature_name, weight, value in zip(artifacts.processed_feature_names, theta[1:], sample_x):
        worked_rows.append(
            {
                "Feature": feature_name,
                "Weight": weight,
                "Processed value": value,
                "Weight x Value": weight * value,
            }
        )
    st.dataframe(pd.DataFrame(worked_rows), use_container_width=True)
    st.write(f"Intercept: {theta[0]:.4f}")
    st.write(f"Predicted value: {sample_prediction:.4f}")
    st.write(f"Actual value: {sample_y:.4f}")
    st.write(f"Error: {sample_error:.4f}")
    st.write("Early gradient descent snapshots")
    snapshot_rows = []
    for snapshot in snapshots:
        snapshot_rows.append(
            {
                "Iteration": snapshot["iteration"],
                "Cost": snapshot["cost"],
                "Intercept gradient": snapshot["gradients"][0],
                "Intercept": snapshot["theta"][0],
            }
        )
    st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True)
