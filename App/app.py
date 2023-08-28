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

# Main Function
def main():
    st.title("APS - Algorithm Predictive Modeling App")
    st.write("APS is an Algorithm Predictive Modeling App. Upload a dataset, choose target variables and features, train a basic predictive model like linear regression, decision tree, or random forest and display evaluation metrics and predictions on new data.")
    
    # File Uploader Widget
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset: ")
        st.write(data.head())

if __name__ == "__main__":
    main()