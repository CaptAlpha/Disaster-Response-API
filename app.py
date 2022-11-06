from flask import Flask, request, jsonify, render_template
import os
import json
import numpy as np
import pandas as pd
import joblib 
import re
app = Flask(__name__)

# Load the model
model = joblib.load('static/model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the array from the POST request.
        data = request.get_json(force=True)
        text = data['text']

        dict = {}
        for i in range(len(text)):
            prediction = model.predict([text[i]])
            dict[text[i]] = prediction[0]

        sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=False)}
        print(sorted_dict)
        # Return the messages only in the sorted dictionary
        return jsonify(list(sorted_dict.keys()))

    return "No Response"


if __name__ == "__main__":
    app.run(debug=True)
