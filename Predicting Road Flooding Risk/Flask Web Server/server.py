# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# 모델 아키텍처를 정의합니다.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(1)
])

# checkpoint 로더를 생성합니다.
checkpoint_loader = tf.train.Checkpoint(model=model)
checkpoint_loader.restore('./model/saved.cpkt')

# 모델의 첫 번째 레이어를 가져옵니다.
first_layer = model.layers[0]

# 레이어의 가중치 이름을 저장합니다.
weights_names = [weight.name for weight in first_layer.weights]

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # 파라미터를 전달 받습니다.
        avg_temp = float(request.form['avg_temp'])
        min_temp = float(request.form['min_temp'])
        max_temp = float(request.form['max_temp'])
        rain_fall = float(request.form['rain_fall'])

        # 입력된 파라미터를 배열 형태로 준비합니다.
        data = np.array([(avg_temp, min_temp, max_temp, rain_fall)], dtype=np.float32)

        # 입력 값을 토대로 예측 값을 찾아냅니다.
        predictions = model.predict(data)
        price = predictions[0][0]

        return render_template('index.html', price=price)

if __name__ == '__main__':
    app.run(debug=True)
