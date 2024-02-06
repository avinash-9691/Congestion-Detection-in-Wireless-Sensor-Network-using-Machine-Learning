import tkinter as tk
import pandas as pd
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,r2_score, mean_squared_error
import xgboost as xgb
from sklearn import preprocessing

class DatasetGUI:
    def __init__(self, master):
        self.master = master
        master.title("Congestion Detection in Wireless Sensor Network")
        self.label = tk.Label(master, fg="Black", text="Congestion Detection in Wireless Sensor Network using Machine Learning \n\n", font=("Arial Bold", 14))
        self.label.pack()
        
        self.filename_label = tk.Label(master, text="No file selected")
        self.filename_label.pack()

        self.display_button = tk.Button(master, text="Select Dataset", command=self.load_dataset)
        self.display_button.pack(pady=10)

        self.data_label = tk.Label(master, text="")
        self.data_label.pack(pady=10)

        self.rf_button = tk.Button(master, text="Random Forest Classifier", command=self.rf_classifier)
        self.rf_button.pack(pady=10)

        self.xgb_button = tk.Button(master, text="XGBoost Regressor", command=self.xgb_regressor)
        self.xgb_button.pack(pady=10)

        self.accuracy_label = tk.Label(master, text="")
        self.accuracy_label.pack(pady=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            self.filename_label.config(text="Selected file: " + file_path)

            self.data = pd.read_csv(file_path)
            le = preprocessing.LabelEncoder()
            self.data['class'] = le.fit_transform(self.data['class'])
            self.data_label.config(text="First 5 records:\n\n" + str(self.data.head()))

            self.X = self.data.drop(columns=['class'])
            self.y = self.data['class']

    def rf_classifier(self):
        if hasattr(self, 'X') and hasattr(self, 'y'):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            self.accuracy_label.config(text="Accuracy Using Random Forest Classifier: {:.2f} ".format(accuracy))
            self.data_label.config(text="First 5 records:\n\n" + str(self.data.head()))
        else:
            self.data_label.config(text="Please select a dataset first.")
            self.accuracy_label.config(text="")

    def xgb_regressor(self):
        if hasattr(self, 'X') and hasattr(self, 'y'):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            xgb_reg = xgb.XGBRegressor()
            xgb_reg.fit(X_train, y_train)

            y_pred = xgb_reg.predict(X_test)
            rsquare = r2_score(y_pred, y_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

           
            self.accuracy_label.config(text="R squared Score : {:.2f} " .format(rsquare) + " and Root Mean Squared Error: {:.2f} ".format(rmse))
            self.data_label.config(text="First 5 records:\n\n" + str(self.data.head()))
        else:
            self.data_label.config(text="Please select a dataset first.")
            self.accuracy_label.config(text="")

root = tk.Tk()
width=720
height=480
root.geometry(f"{width}x{height}")
gui = DatasetGUI(root)
root.mainloop()






