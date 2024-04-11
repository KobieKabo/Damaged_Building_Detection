from flask import Flask, request
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model('proj_models/Mod_LeNet5.keras')


@app.route('/proj_models/Mod_LeNet5/v1', methods=['GET'])
def model_info():
   return {
      "version": "v1",
      "name": "Mod_LeNet5",
      "description": "Classify images containing satellite data of Texas after a hurricane",
      "number_of_parameters": 505546
   }

def preprocess_input(im):
   """
   Converts user-provided input into an array that can be used with the model.
   This function could raise an exception.
   """
   # convert to a numpy array
   d = np.array(im)
   d = d / 255.0
   d = np.expand_dims(d, axis = 0)
   # then add an extra dimension
   return d


    
@app.route('/proj_models/Mod_LeNet5/v1', methods=['POST'])
def classify_building_image():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   return { "result": model.predict(data).tolist()}
    
# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')
