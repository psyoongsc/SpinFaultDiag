import sys

import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
import pandas as pd

# 항상 같은 난수를 발생시키기 위해서 seed 를 지정함.
np.random.seed(1)
tf.random.set_seed(1)

# 입력 변수를 4개 받아야 정상적인 프로그램이 돌아가도록 처리함.
if len(sys.argv) != 5:
    print("Insufficient arguments")
    sys.exit()

# 입력변수 순서 및 리스트
# 1. 진단 모델 파일 (xxxx.h5)
# 2. 학습에 사용된 데이터 (xxxx.csv)
# 3. confusion matrix 생성을 위한 실제 loss를 계산할 비교 데이터 (xxxx.csv)
# 4. 진단할 데이터 (xxxx.csv)
diag_model_path = sys.argv[1]
train_file_path = sys.argv[2]
compare_file_path = sys.argv[3]
test_file_path = sys.argv[4]

#비교 데이터
df1 = pd.read_csv(compare_file_path)
df1 = df1[['TIME', 'CH1']]
df1['TIME'] = pd.to_numeric(df1['TIME'], downcast='signed')

#정상 데이터
df2 = pd.read_csv(train_file_path)
df2 = df2[['TIME', 'CH1']]
df2['TIME'] = pd.to_numeric(df2['TIME'], downcast='signed')

#진단 데이터
df3 = pd.read_csv(test_file_path)
df3 = df3[['TIME', 'CH1']]
df3['TIME'] = pd.to_numeric(df3['TIME'], downcast='signed')

test, train, compare = df3, df2, df1

TIME_STEPS=30

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])

    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['CH1']], train['CH1'])
X_test, y_test = create_sequences(test[['CH1']], test['CH1'])
X_compare, y_compare = create_sequences(compare[['CH1']], compare['CH1'])

from keras.models import load_model
model = load_model(diag_model_path)

model.evaluate(X_test, y_test)

X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

origin_mae_loss = np.mean(np.abs(X_compare - X_train), axis=1)

# origin_threshold = np.percentile(origin_mae_loss, 99)
origin_threshold = np.max(origin_mae_loss)
threshold = np.percentile(train_mae_loss, 99)

X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

test_loss_threshold = np.percentile(test_mae_loss, 99)

test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['origin_loss'] = origin_mae_loss
test_score_df['threshold'] = threshold
test_score_df['origin_threshold'] = origin_threshold
test_score_df['1percent_threshold'] = test_loss_threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['origin_anomaly'] = test_score_df['loss'] > test_score_df['origin_threshold']
test_score_df['CH1'] = test[TIME_STEPS:]['CH1']

origin_anomalies = test_score_df.loc[(test_score_df['origin_anomaly'] == True) & (test_score_df['loss'] <= test_loss_threshold)]
anomalies = test_score_df.loc[(test_score_df['anomaly'] == True) & (test_score_df['loss'] <= test_loss_threshold)]
print(f'origin threshold: {origin_threshold}, predict threshold: {threshold}')
print('============[예측 통계]============')
print(f'Origin Anomaly count: {origin_anomalies.shape[0]} / {(test_score_df.loc[test_score_df["loss"] <= test_loss_threshold]).shape[0]}')
print(f'Anomaly count: {anomalies.shape[0]} / {(test_score_df.loc[test_score_df["loss"] <= test_loss_threshold]).shape[0]}')

# 정상지점: False, 비정상지점: True
# Sequence 를 만들면서 빼놓은 30개 지점과 전체지점에서 1% 지점의 loss를 노이즈로 판단하여 무시하였기 때문에 지점이 진단 대상 지점
TP = (test_score_df.loc[(test_score_df['origin_anomaly'] == False) & (test_score_df['anomaly'] == False)]).shape[0]
TN = (test_score_df.loc[(test_score_df['origin_anomaly'] == True) & (test_score_df['anomaly'] == True) & (test_score_df['origin_loss'] <= test_loss_threshold) & (test_score_df['loss'] <= test_loss_threshold)]).shape[0]
FP = (test_score_df.loc[(test_score_df['origin_anomaly'] == True)  & (test_score_df['anomaly'] == False) & (test_score_df['origin_loss'] <= test_loss_threshold) & (test_score_df['loss'] <= test_loss_threshold)]).shape[0]
FN = (test_score_df.loc[(test_score_df['origin_anomaly'] == False) & (test_score_df['anomaly'] == True) & (test_score_df['origin_loss'] <= test_loss_threshold) & (test_score_df['loss'] <= test_loss_threshold)]).shape[0]

print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

try:
    Accuracy = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
except ZeroDivisionError:
    Accuracy = 100

try:
    Recall = round(TP / (TP + FN) * 100, 2)
except ZeroDivisionError:
    Recall = 100

try:
    Precision = round(TP / (TP + FP) * 100, 2)
except ZeroDivisionError:
    Precision = 100

try:
    F1 = round(2 * (Precision * Recall) / (Precision + Recall), 2)
except ZeroDivisionError:
    F1 = 100

print(f'Accuracy: {Accuracy}%')
print(f'Recall: {Recall}%, Precision: {Precision}%')
print(f'F1-score: {F1}%')

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['TIME'], y=test_score_df['loss'], mode='markers', name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df['TIME'], y=test_score_df['origin_threshold'], name='Origin Threshold'))
fig.add_trace(go.Scatter(x=test_score_df['TIME'], y=test_score_df['threshold'], name='Predict Threshold'))
fig.add_trace(go.Scatter(x=test_score_df['TIME'], y=test_score_df['1percent_threshold'], name='Test loss 1%'))
fig.update_layout(showlegend=True,
                  xaxis_title="Time(s)", yaxis_title="loss",
                  title='Test loss vs. Threshold')
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['TIME'], y=test_score_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df['TIME'], y=test_score_df['origin_threshold'], name='Origin Threshold'))
fig.add_trace(go.Scatter(x=test_score_df['TIME'], y=test_score_df['1percent_threshold'], name='Test Threshold'))
fig.update_layout(showlegend=True,
                  xaxis_title="Time(s)", yaxis_title="loss",
                  title='Compare loss vs. Threshold')
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['TIME'], y=test_score_df['CH1'], name='Spin Accelation'))
fig.add_trace(go.Scatter(x=anomalies['TIME'], y=anomalies['CH1'], mode='markers', name='Anomaly'))
fig.add_trace(go.Scatter(x=origin_anomalies['TIME'], y=origin_anomalies['CH1'], mode='markers', name='Origin Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
# fig.show()
