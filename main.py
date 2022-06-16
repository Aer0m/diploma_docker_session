import pickle
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier

# Импортируем Flask для создания API
from flask import Flask, request

# Загружаем обученную модель из текущего каталога
with open('model.pkl', 'rb') as model_pkl:
   knn = pickle.load(model_pkl)

# Инициализируем приложение Flask
app = Flask(__name__)

# Создайте конечную точку API
@app.route('/predict')
def predict_iris():
   age = request.args.get('age')
   sex = request.args.get('sex')
   bmi = request.args.get('bmi')
   children = request.args.get('children')
   smoker = request.args.get('smoker')
   region = request.args.get('region')

# Используем метод модели predict для
# получения прогноза для неизвестных данных
   unseen = np.array([[age, sex, bmi, children, smoker, region]], dtype=float)
   result = knn.predict(unseen)
  # возвращаем результат
   return 'Predicted result for observation ' + str(unseen) + ' is: ' + str(result)
if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True, port=5000)
