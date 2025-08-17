from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn import preprocessing
import tensorflow as tf
import pandas as pd
from flask import jsonify

app = Flask(__name__)
model = tf.keras.models.load_model('root_cause.h5')


# @app.route("/")
# def home():
#     root_cause_analysis = load_iris()
#     # model = KNeighborsClassifier(n_neighbors=3)
#     # X_train, X_test, y_train, y_test = train_test_split(root_cause_analysis.data, root_cause_analysis.target)
#     # model.fit(X_train, y_train)
#     # with open("root_cause_analysis.pkl", "wb") as f:
#     #     pickle.dump(model, f)
#     return render_template("loginpage.html")
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Here, you might want to do some validation of login credentials.
        # For simplicity, I'm just redirecting to test.html on any POST request.
        return redirect(url_for('test_page'))
    else:
        return render_template("loginpage.html")


@app.route("/test")
def test_page():
    return render_template("test.html")


@app.route("/form")
def form_page():
    return render_template("form2.html")


@app.route("/process", methods=["POST"])
def process():
    input1 = request.form.get("cpuload")
    input2 = request.form.get("memoryLeak")
    input3 = request.form.get("delay")
    input4 = request.form.get("error1000")
    input5 = request.form.get("error1001")
    input6 = request.form.get("error1002")
    input7 = request.form.get("error1003")
    form_data = np.array([[input1, input2, input3, input4, input5, input6, input7]], dtype=float)
    # with open("root_cause_analysis.pkl", "rb") as f:
    #     model = pickle.load(f)

    # prediction = model.predict(form_data)[0]
    prediction = np.argmax(model.predict(form_data), axis=1)
    label_encoder = preprocessing.LabelEncoder()
    symptom_data = pd.read_csv("root_cause_analysis.csv")
    symptom_data['ROOT_CAUSE'] = label_encoder.fit_transform(
        symptom_data['ROOT_CAUSE'])
    result = label_encoder.inverse_transform(prediction)

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

#
# @app.route("/predict", methods=["POST"])
# def predict():
#     CPU_load = float(request.form['cpuload'])
#     Memory_leak = float(request.form['memoryLeak'])
#     Delay = float(request.form['delay'])
#     Error_1000 = float(request.form['error1000'])
#     Error_1001 = float(request.form['error1001'])
#     Error_1002 = float(request.form['error1002'])
#     Error_1003 = float(request.form['error1003'])
#
#     form_array = np.array([[CPU_load, Memory_leak, Delay, Error_1000, Error_1001, Error_1002, Error_1003]])
#
#     with open("root_cause_analysis.pkl", "rb") as f:
#         model = pickle.load(f)
#
#     prediction = model.predict(form_array)[0]
#     return f"The predicted outcome is: {prediction}"
#
# def predict():
#     # Get the data from the POST request
#     data = request.json
#     form_data = np.array([
#         data['cpuLoad'],
#         data['memoryLeak'],
#         data['delay'],
#         data['error1000'],
#         data['error1001'],
#         data['error1002'],
#         data['error1003']
#     ], dtype=float)
#     # Perform the root cause analysis (dummy logic, replace with your actual analysis)
#     with open("root_cause_analysis.pkl", "rb") as f:
#         model = pickle.load(f)
#
#     prediction = model.predict(form_data)[0]
#     root_cause = prediction
#
#     # Return the result as JSON
#     return jsonify({"rootCause": root_cause})
# if __name__ == "__main__":
#     app.run(debug=True)
