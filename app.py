import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
import joblib

def main():
    st.header("Let's explore house dataset!")
    df = pd.read_csv('https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/formart_house.csv')
    st.subheader("A quick look at our dataset")
    st.write(df.head())
    df = df.drop([506],axis=0)
    df = df.astype(float)
    st.subheader("Explonatory Data Analysis")
    option = st.selectbox(
    'which graphs you want to see?',
    ('scatterplot', 'heatmap',"distribution","boxplot"))
    if option == "scatterplot":
        x = st.text_input("Choose your x")
        y = st.text_input("Choose your y")
        first_btn = st.button("let's see")
        if first_btn:
            if x and y:
                if x is not None and y is not None:
                    fig = plt.figure(figsize=(10,8))
                    sns.scatterplot(data=df,x=x,y=y)
                    st.pyplot(fig)
    elif option == "distribution":
        x = st.text_input("Choose a variable you want to see its distribution")
        first_btn = st.button("let's see")
        if first_btn:
            if x:
                fig = plt.figure(figsize=(10,8))
                sns.histplot(data=df,x=x,kde=True)
                st.pyplot(fig)
    elif option == "boxplot":
        columns = list(df.columns)
        columns.remove("chas")
        columns = tuple(columns)
        y = st.radio("Choose a variable",columns)
        first_btn = st.button("let's see")
        if first_btn:
            if y:
                fig = plt.figure(figsize=(10,8))
                sns.boxplot(data=df,x="chas",y=y)
                st.pyplot(fig)
    else :
        first_btn = st.button("let's see")
        if first_btn:
            st.write(df.corr())
            fig = plt.figure(figsize=(10,8))
            sns.heatmap(data=df.corr(),annot=True, cmap="Blues")
            st.pyplot(fig)

    st.subheader("Let's build our model")
    st.write("The model used for our dataset is Linear Regression")
    X = df.drop(["medv"],axis=1)
    y = df["medv"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_test)
    X_test = scaler.transform(X_test)
    
    model = joblib.load("house_model.pkl")
    y_pred = model.predict(X_test)

    z = st.radio("Choose a error metric to see",("R2_score","Mean absolute error","Root mean squared error"))
    if z == "R2_score":
        error = round(r2_score(y_test,y_pred),2)
        st.write(error)
    elif z == "Mean absolute error":
        error = round(mean_absolute_error(y_test,y_pred),2)
        st.write(error)
    else:
        error = round(np.sqrt(mean_squared_error(y_test,y_pred)),2)
        st.write(error)
    
    error_btn = st.button("Click to see error graph")
    if error_btn:
        length = y_pred.shape[0] 
        x = np.linspace(0,length,length)

        fig = plt.figure(figsize=(10,5))
        plt.plot(x, y_test, label='real y')
        plt.plot(x, y_pred, label="predicted y'")
        plt.legend(loc=2);
        st.pyplot(fig)
    

    
    st.subheader("Let's use our model to predict!")
    features = list(df.columns[:-1])
    X_predict =[]
    for feature in features:
        feature = st.number_input(f"Insert the value for {feature} here",df[feature].min(),df[feature].max(),df[feature].mean())
        X_predict.append(feature)
    X_predict = np.array(X_predict)
    
    X_predict = X_predict.reshape(-1,13)
    X_predict = scaler.transform(X_predict)
    value = round(model.predict(X_predict)[0],2)
    
    predict_btn = st.button("let's predict")
    if predict_btn:
            st.write("Medv is:",value)
    
    

    




















if __name__ == "__main__":
    main()