#!python3
# coding = utf-8
import numpy as np


def rmse(y, x, n):
    variances = list(map(lambda x_target, y_prediction: (x_target - y_prediction) ** 2, x, y))
    return np.sqrt(np.sum(variances) / float(n))


target_file = "/data/weather/data_beijing/testB.txt"
prediction_file = "/data/weather/CIKM_AnalytiCup_2017_yuyang/result/submit_Team4.csv"
target = []
with open(target_file, 'r') as fi:
    for line in fi:
        target.append(line.split(',')[1])
length_target = len(target)
print("len of target: ", length_target)

prediction = []
with open(prediction_file, 'r') as fo:
    for line in fo:
        prediction.append(float(line))
length_prediction = len(prediction)
print(length_prediction)
if length_target == length_prediction:
    print("target number == prediction number")

print(rmse(target, prediction, length_prediction))
