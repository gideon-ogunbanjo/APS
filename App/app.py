# Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title= "APS",
    layout="centered",
)
# Main Function
def main():
    st.title("A.P.S - Algorithm Predictive Studio")
    st.write("A.P.S is an Algorithm Predictive Studio. Upload a dataset, choose target variables and features, train a basic predictive model like linear regression, decision tree, or random forest and display evaluation metrics and predictions on new data.")
    
    # File Uploader Widget
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(uploaded_file)  # Read Excel file
        elif uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)     # Read CSV file
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()

        st.write("Uploaded dataset:")
        st.write(data.head())
        
        # Target variable and Feature Selection
        st.sidebar.header("Select Target Variable and Features")
        target_variable = st.sidebar.selectbox("Select target variable", data.columns)
        selected_features = st.sidebar.multiselect("Select features", data.columns)
        
        # Splitting data into train and test sets
        X = data[selected_features]
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model Selection
        model_choice = st.sidebar.radio(
            "Select an Algorithm",
            ("Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "k-Nearest Neighbors")
        )

        # Code Snippets
        if model_choice == "Linear Regression":
            model = LinearRegression()
            model_code = """
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)
                score = mean_squared_error(y_test, model.predict(X_test))
            """
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor()
            model_code = """
                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)  # Adjust parameters to fit your data
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
            """
        elif model_choice == "Support Vector Machine":
            model = SVR()
            model_code = """
                from sklearn.svm import SVR
                model = SVR(kernel=kernel)
                model.fit(X_train, y_train)
                score = mean_squared_error(y_test, model.predict(X_test))
            """
        elif model_choice == "k-Nearest Neighbors":
            model = KNeighborsRegressor()
            model_code = """
                from sklearn.neighbors import KNeighborsRegressor
                model = KNeighborsRegressor((n_neighbors=2)) # Adjust parameters to fit your data
                model.fit(X_train, y_train)
                model.predict(X_test)
            """
        else:
            model = RandomForestRegressor()
            model_code = """
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators = 1000, random_state = 42)  # Adjust parameters to fit your data
                model.fit(X_train, y_train)
            """
                   
        # Model Training
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Evaluation")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")
        
        # Code Snippet
        st.write("### Training Algorithm Code Snippet")
        st.code(model_code)
        
        # Predictions
        st.write("### Make Predictions")
        new_data = {}
        for feature in selected_features:
            new_data[feature] = st.number_input(f"Enter {feature}", value=0.0)
        new_df = pd.DataFrame([new_data])
        prediction = model.predict(new_df)
        st.write(f"Predicted {target_variable}: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
    
link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
st.markdown(link, unsafe_allow_html=True)