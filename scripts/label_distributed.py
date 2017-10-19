#! python3
# coding = utf-8
import numpy as np

shenzhen_raw = "/data/yuyang/weather/data/data_shenzhen/CIKM2017_train/train_label.txt"
beijing_morerain = "/data/yuyang/weather/data/data_aggregated_more/label_train.txt"
beijing_raw = "/data/yuyang/weather/data/data_aggregated_raw/label_train.txt"

data_train = beijing_raw

label = []
with open(data_train) as fd:
    for line in fd:
        if line:
            label.append(float(line.strip()))

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

print('  range   count')
print('   0      {:4d}'.format(cnt_0))
print('  0-10    {:4d}'.format(cnt_0_10))
print(' 10-20    {:4d}'.format(cnt_10_20))
print(' 20-50    {:4d}'.format(cnt_20_50))
print(' 50-100   {:4d}'.format(cnt_50_100))
print('   >100   {:4d}'.format(cnt_100))
print('  all     {:4d}'.format(len(label)))

def rmse(y, x):
    n = len(x)
    variances = list(map(lambda x_target, y_prediction: (x_target - y_prediction) ** 2, x, y))
    return np.sqrt(np.sum(variances) / float(n))


for i in range(0, 50, 3):
    value_assumed = i
    preds_assumed = [value_assumed] * len(label)

    print("rmse == {:.2f} when preds_assumed == {}".format(rmse(label, preds_assumed), value_assumed))
