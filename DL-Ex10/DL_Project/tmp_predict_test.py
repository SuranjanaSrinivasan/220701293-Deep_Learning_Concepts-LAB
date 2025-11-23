import sys, numpy as np, pickle
sys.path.append('e:/DL_Project')
from tensorflow.keras.models import load_model
m = load_model('e:/DL_Project/models/phishing_model.h5')
print('model input shape:', m.input_shape)
arr = np.zeros((1,9))
try:
    with open('e:/DL_Project/models/scaler.pkl','rb') as f:
        s = pickle.load(f)
        arr = s.transform(arr)
        print('applied scaler')
except Exception as e:
    print('no scaler:', e)
print('arr shape', arr.shape)
print('prediction:', m.predict(arr))
