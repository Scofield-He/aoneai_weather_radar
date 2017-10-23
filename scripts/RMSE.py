#!python3
# coding = utf-8
import numpy as np


def rmse(y, x):
    n = len(x)
    variances = list(map(lambda x_target, y_prediction: (x_target - y_prediction) ** 2, x, y))
    return np.sqrt(np.sum(variances) / float(n))


def mae(y, x):
    return np.mean(list(map(lambda x1, y1: abs(x1 - y1), x, y)))


label_file_path = "/data/yuyang/weather/result/beijing_morerain/"
#label_file_path = "/data/yuyang/weather/result/beijing/"

label_file = "label_test.txt"

file_time = '2017-10-19-22'

bigru_file = "bi-gru_pre-{}.csv".format(file_time)
xgb_file = "xgb_pre-{}.csv".format(file_time)
rf_file = "rf_pre-{}.csv".format(file_time)
ensemble_file = "ensemble_pre-{}.csv".format(file_time)

prediction_file = label_file_path + ensemble_file
target_file = label_file_path + label_file
print("\nmodel : ensemble model,    file_time: {}".format(file_time))
#print("\nmodel : xgboost,    file_time: {}".format(file_time))

targets = []
with open(target_file, 'r') as fi:
    for line in fi:
        if line:
            targets.append(float(line.strip()))
labels = targets[:2025]
length_labels = len(labels)
#print("count of labels: ", length_labels)

preds = []
with open(prediction_file, 'r') as fo:
    for line in fo:
        preds.append(float(line.strip()))
length_preds = len(preds)
print("count of preds: ", length_preds)
print()

if length_labels != length_preds:
    print("values error: count of preds != labels")

l0, l1, l2, l3, l4, l5 = [], [], [], [], [], []
p0, p1, p2, p3, p4, p5 = [], [], [], [], [], []
for i in range(length_preds):
    if labels[i] == 0:
        l0.append(labels[i])
        p0.append(preds[i])
    elif 0 < labels[i] <= 10:
        l1.append(labels[i])
        p1.append(preds[i])
    elif 10 < labels[i] <= 20:
        l2.append(labels[i])
        p2.append(preds[i])
    elif 20 < labels[i] <= 50:
        l3.append(labels[i])
        p3.append(preds[i])
    elif 50 < labels[i] <= 100:
        l4.append(labels[i])
        p4.append(preds[i])
    else:
        l5.append(labels[i])
        p5.append(preds[i])

cnt_pre_0, cnt_pre_0_2, cnt_pre_2_10, cnt_pre_10_20, cnt_pre_20 = 0, 0, 0, 0, 0
for i in range(len(l0)):
    if p0[i] == 0:
        cnt_pre_0 += 1
    elif 0 < p0[i] <= 2:
        cnt_pre_0_2 += 1
    elif 2 < p0[i] <= 10:
        cnt_pre_2_10 += 1
    elif 10 < p0[i] <= 20:
        cnt_pre_10_20 += 1
    else:
        cnt_pre_20 += 1

print('count of preds when label == 0: ')
print(' [ 0, 0 ] -> {0}\n ( 0, 2 ] -> {1}\n ( 2, 10] -> {2}\n (10, 20] -> {3}\n (20,   ) -> {4}\n'.format(cnt_pre_0, cnt_pre_0_2, cnt_pre_2_10, cnt_pre_10_20, cnt_pre_20))

print('rmse & mae in different ranges of label: ')
print('label range  ', 'count    rmse      mae')
print('[  0, 0  ] ->  {:4d}   {:6.2f}   {:6.2f}'.format(len(l0), rmse(l0, p0), mae(l0, p0)))
print('(  0, 10 ] ->  {:4d}   {:6.2f}   {:6.2f}'.format(len(l1), rmse(l1, p1), mae(l1, p1)))
print('( 10, 20 ] ->  {:4d}   {:6.2f}   {:6.2f}'.format(len(l2), rmse(l2, p2), mae(l2, p2)))
print('( 20, 50 ] ->  {:4d}   {:6.2f}   {:6.2f}'.format(len(l3), rmse(l3, p3), mae(l3, p3)))
print('( 50, 100] ->  {:4d}   {:6.2f}   {:6.2f}'.format(len(l4), rmse(l4, p4), mae(l4, p4)))
print('[100,    ) ->  {:4d}   {:6.2f}   {:6.2f}'.format(len(l5), rmse(l5, p5), mae(l5, p5)))

print(' all data  ->  {:4d}   {:6.2f}   {:6.2f}'.format(2025, rmse(labels, preds), mae(labels, preds)))
print('Done!')

