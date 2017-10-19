#! python3
# coding = utf-8
import numpy as np

file_path = "/data/yuyang/weather/result/beijing/"

gru = "bi-gru_pre-2017-10-16.csv"
rf = "rf_pre-2017-10-16.csv"
xgb = "xgb_pre-2017-10-16.csv"
ensemble = "ensemble-2017-10-16.csv"

preds_file = file_path + gru
print(preds_file)
label = []
with open(preds_file) as fp:
    for line in fp:
        if line:
            label.append(float(line.strip()))

print('length: ', len(label))

for i in label:
    print('{:.2f}'.format(i))
cnt_0, cnt_0_10, cnt_10_20, cnt_20_50, cnt_50_100, cnt_100 = 0, 0, 0, 0, 0, 0
for i in label:
    if i == 0:
        cnt_0 += 1
    elif 0 < i <= 10:
        cnt_0_10 += 1
    elif 10 < i <= 20:
        cnt_10_20 += 1
    elif 20 < i <= 50:
        cnt_20_50 += 1
    elif 50 < i <= 100:
        cnt_50_100 += 1
    else:
        cnt_100 += 1

print(' range      count')
print('  0       {} '.format(cnt_0))
print(' 0-10     {} '.format(cnt_0_10))
print('10-20     {} '.format(cnt_10_20))
print('20-50     {} '.format(cnt_20_50))
print(' 100      {} '.format(cnt_100))


'''
def rmse(y, x):
    n = len(x)
    variances = list(map(lambda x_target, y_prediction: (x_target - y_prediction) ** 2, x, y))
    return np.sqrt(np.sum(variances) / float(n))


for i in range(0, 30, 3):
    value_assumed = i
    preds_assumed = [value_assumed] * 10000

    print("rmse == {:.2f} when preds_assumed == {}".format(rmse(label, preds_assumed), value_assumed))
'''
