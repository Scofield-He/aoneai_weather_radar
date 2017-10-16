import sys

sys.path.append('..')
import rfmodel as rf
import pandas as pd
import numpy as np
from dataprocess import FeatureSelect as fs
from dataprocess import data_process8 as dp
from dataprocess import generate_percentile as gp
import xgbmodel as xgbm
import bigrumodel as bigru
from multiprocessing import Pool
from functools import partial

def check_code(mode, gru_mode):
    if mode == 'simple':
        train_df = pd.read_csv('/data/yuyang/weather/data/data_processed/train_percentile.csv')
        test_df = pd.read_csv('/data/yuyang/weather/data/data_processed/testB_percentile.csv')
        train_add = pd.read_csv('/data/yuyang/weather/data/data_processed/train_old_wind_4240.csv')
        testA_add = pd.read_csv('/data/yuyang/weather/data/data_processed/testB_old_wind_4240.csv')
        train_1ave8extend = pd.read_csv('/data/yuyang/weather/data/data_processed/train_new_wind_1ave_8extend.csv')
        test_1ave = pd.read_csv('/data/yuyang/weather/data/data_processed/testB_new_wind_1ave_8extend.csv')
    else:
        trainfile = '/data/yuyang/weather/data/data_aggregated/train'
        testBfile = '/data/yuyang/weather/data/data_aggregated/test'

        args = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']
        p = Pool(15)
        '''
        # 生成训练集数据,老的风
        # train_add = dp.dataprocess(trainfile, data_type='train', windversion='old')
        print('start----dp-train-old')
        func_dp_train_old = partial(dp.dataprocess, trainfile, 'train', 'old')
        p.map(func_dp_train_old, args)

        # 生成测试集B数据,老的风
        # testA_add = dp.dataprocess(testBfile, data_type='testB', windversion='old')
        print('start----dp-test-old')
        func_dp_test_old = partial(dp.dataprocess, testBfile, 'testB', 'old')
        p.map(func_dp_test_old, args)

        # 生成训练集数据,1ave8extend
        # train_1ave8extend = dp.dataprocess(trainfile, data_type='train', windversion='new')
        func_dp_train_new = partial(dp.dataprocess, trainfile, 'train', 'new')
        p.map(func_dp_train_new, args)
        
        # 生成测试集B数据,1ave
        # test_1ave = dp.dataprocess(testBfile, data_type='testB', windversion='new')
        func_dp_test_new = partial(dp.dataprocess, testBfile, 'testB', 'new')
        p.map(func_dp_test_new, args)
        '''
        # 生成训练集数据
        # train_df = gp.data_process(trainfile, data_type='train')
        #func_gp_train = partial(gp.data_process, trainfile, 'train')
        #p.map(func_gp_train, args)

        # 生成测试集B数据
        # test_df = gp.data_process(testBfile, data_type='testB')
        func_gp_test = partial(gp.data_process, testBfile, 'testB')
        p.map(func_gp_test, args)
        
        print('data process has been done')
        return 0

    print('#data process has been done')

    result_xgb = xgbm.xgbmodeltrain(train_1ave8extend, test_1ave)
    np.savetxt("/data/yuyang/weather/result/" + "xgb_pre.csv", result_xgb)

    print('#xgb model has been done')

    index = fs.pre_train(train_df=train_df, test_df=test_df, train_add=train_add, test_add=testA_add)

    valid = rf.rf_model(train_df, test_df, 'train', train_add, testA_add, ne=100)

    ne = 500
    result_rf = rf.rf_model(train_df, test_df, 'trai', train_add, testA_add, ne, index=index)
    np.savetxt("/data/yuyang/weather/result/" + "rf_pre.csv", result_rf)
    print('#rf model has been done')

    result_bigru = bigru.BiGRU_train(train_df, test_df, valid, gru_mode).reshape(2025)
    np.savetxt("/data/yuyang/weather/result/" + "bi-gru_pre.csv", result_bigru)
    print('#bigru model has been done')

    ensemble = (result_xgb + result_rf + result_bigru) / 3.0

    np.savetxt("/data/yuyang/weather/result/" + "ensemble.csv", ensemble)


check_code('simple', 'online')

#check_code('all','no')
