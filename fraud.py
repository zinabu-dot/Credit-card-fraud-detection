# Fraud Detection Algorithm

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# Handling Unbalanced Data-Over Sampling
#from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from collections import Counter
from PIL import Image



st.title("Credit Card Fraud Detection")

tabs = ['Data', 'Home', 'About', 'Vis', 'Contact']

menu_tab = st.sidebar.selectbox('Menu', tabs)

def file_upload():
    if menu_tab == 'Data':
        data = st.file_uploader(label='upload file', type=['csv', 'xlsx'])
        if data is not None:
            try:
                df = pd.read_csv(data)
            except Exception as e:
                df = pd.read_xlsx(data)

        #st.dataframe(data)
            #st.write(df)
            if st.checkbox('Shape of data'):
                st.write(df.shape)


            # Separate Data into independent features and target feature
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # Feature Scaling
            sc = StandardScaler()
            X = sc.fit_transform(X)

            # Splitting the dataset in to training and testing set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

            # Implementing Oversampling for Handling Imbalanced
            #smk = SMOTETomek(random_state=42)

            count_class_0 = max(len(y_train[y_train==0]), len(y_train[y_train==1]))
            count_class_1 = count_class_0
            pipe = make_pipeline(NearMiss(sampling_strategy={0: count_class_0}),
                                SMOTE(sampling_strategy={1: count_class_1}
                                ))

            X_train, y_train = pipe.fit_resample(X_train, y_train)

            shapes_train = st.checkbox('X train data shape')
            if shapes_train: # == 'X train data shape':
                st.write(X_train.shape)
            shapes_target = st.checkbox('target train data shape')
            if shapes_target: # == '':
                st.write(y_train.shape)

            #st.write('X train data shape:', X_train.shape, 'and', 'target train data shape:', y_train.shape)

            if st.checkbox('Resampled dataset shape'):
                st.write(' {}'.format(Counter(y_train)))
            #st.write('Resampled dataset shape {}'.format(Counter(y_train)))


            ML_Models = st.sidebar.selectbox('Select Classifiers',
                                options=['LR', 'NB', 'DT', 'RT', 'SVM', 'KNN'])
            def init_params_values(ML_Models):
                params = dict()
                if ML_Models == 'KNN':
                    n_neighbors = st.sidebar.slider("n_neighbors", 1,15)
                    params["n_neighbors"] = n_neighbors
                elif ML_Models == 'SVM':
                    C = st.sidebar.slider("C", 0.01,10.0)
                    params['C'] = C
                elif ML_Models == 'RT':
                    max_depth = st.sidebar.slider("max_depth", 2,15)
                    n_estimators = st.sidebar.slider("n_estimators", 1, 100)
                    params['max_depth'] = max_depth
                    params['n_estimators'] = n_estimators
                elif ML_Models == 'DT':
                    max_depth = st.sidebar.slider("max_depth", 2,15)
                    params['max_depth'] = max_depth
                elif ML_Models == 'NB':
                    var_smooth = st.sidebar.slider("var_smoothing", 1e-11, 1e-6)
                    params['var_smoothing'] = var_smooth
                else: #ML_Models == 'LR':
                    tol = st.sidebar.slider("tol", 1e-6, 1e-2)
                    C = st.sidebar.slider("C", 0.01, 10.0)
                    params['tol'] = tol
                    params['C'] = C
                return params

            params = init_params_values(ML_Models)

            def get_classifier(ML_Models, params):
                if ML_Models == 'KNN':
                    cls = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
                elif ML_Models == 'SVM':
                    cls = SVC(C=params['C'])
                elif ML_Models == 'RT':
                    cls = RandomForestClassifier(max_depth=params["max_depth"],
                                                n_estimators=params['n_estimators'])
                elif ML_Models == 'DT':
                    cls = DecisionTreeClassifier(max_depth=params["max_depth"])
                elif ML_Models == 'NB':
                    cls = GaussianNB(var_smoothing = params['var_smoothing'] )
                else: #ML_Models == 'LR':
                    cls = LogisticRegression(tol=params['tol'],
                                             C = params['C'])
                return cls

            cls = get_classifier(ML_Models, params)

            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            st.write(f"Classifier = {ML_Models}")
            st.write(f"Accuracy = {accuracy}")

            plot = px.scatter(df, x="V1",
                                 y="V2", color="Class",
                                    title=' title of the plot here')
            fig = go.Figure(data=[go.Scatter3d(z=np.array(df['Time']), x=np.array(df['V12']), y=np.array(df['Amount']),
                                               marker_color=df['Class'])])
            fig.update_layout(title='3D surface for fraud factors', autosize=False,
                             # width=500, height=500,
                              #margin=dict(l=65, r=50, b=65, t=90)
                                                        )
            with st.beta_expander('Visual', expanded=False):
                st.plotly_chart(plot)
                st.plotly_chart(fig)

            def val():
                rand_vals = df.sample(n=10)
                if st.button('save dataframe'):
                    open('rand_vals.csv','w').write(rand_vals.to_csv())

                val_df = st.file_uploader('upload data to validate')
                if val_df is not None:
                    try:
                        df_val = pd.read_csv(val_df)
                        #df_val = df_val.drop('Unnamed: 0')
                    except Exception as e:
                        print(e)

                    X_val = df_val.drop(columns=['Unnamed: 0', 'Class'],axis=1)

                    pred = cls.predict(X_val)
                    prob = cls.predict_proba(X_val)
                    prob = pd.DataFrame(prob)
                    df_val['predicted'] = pred
                    df_val['pred_proba_class_0'] = np.array(prob[0])
                    df_val['pred_proba_class_1'] = np.array(prob[1])

                    st.write(df_val)
            val()

            def predict_using_input():
                html_title = """
                <div style='background-color:silver;padding:15px'>
                <h1 style="color:black; text-align:center;"> Stramlit Fraud Detection DM/ML App </h1>
                </div>
                """
                st.markdown(html_title, unsafe_allow_html=True)

                html_temp = """
                     <div style='background-color:tomato;padding:10px'>
                     <h2 style='color:white; text-align:center;'> Fraud prediction </h2>
                     </div>
                """
                st.markdown(html_temp, unsafe_allow_html=True)
                feat = []
                #features = ['Time', 'Amount']
                for var in df.iloc[:,:-1].columns:
                    var1 = st.text_input(var, 'type here')
                    feat.append(var1)
                    #feat.append(var)
                dat = np.array(feat)
                feat = dat.reshape(1,-1)

                result = ''
                if st.button('Predict'):
                    result = cls.predict(feat)
                st.success('The predicted result is {}.'.format(result))

            predict_using_input()








            # Create a Logistic regression classifier
            #logreg = LogisticRegression()

            # Train the model using the training sets
            #logreg.fit(X_train, y_train)

            # Prediction on test data
            #y_pred = logreg.predict(X_test)
            #acc_logreg = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
            #print('Accuracy of Logistic Regression model : ', acc_logreg)

    elif menu_tab == 'Home':
        st.write("home page")
    elif menu_tab == 'About':
        st.write('About the app finds here')
    else:
        st.write('For your inquiries contact us here')
        uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Sunrise by the mountains', use_column_width = True)
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type,
                            "FileSize": uploaded_file.size}
            st.write(file_details)

        # Audio file reading
        audio_file = open('audio.oga', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/oga')

        # Video file reading
        video_file = open('star.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)


file_upload()

        # 'png', 'jpg', 'jpeg'