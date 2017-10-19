#! python2
# coding = utf-8

import numpy as np
import netCDF4 as nc
from datetime import datetime
from datetime import timedelta
from os import listdir
import io
import gzip
import os
import time
import multiprocessing
from functools import partial


LAT_LON_KM_RATIO = 111

NC_FILE_FOLDER = '/data/yuyang/weather/data/radar'
STATION_INFO_FILE = '/data/yuyang/weather/resources/station_info_100'
OUT_FILE = '/data/yuyang/weather/data/data_aggregated_more/' + '2014-Station-Centered-Data'
LABEL_FILE = '/data/yuyang/weather/resources/label'
TIME_FILE = '/data/yuyang/weather/resources/time_info_120'
TEMP_NC_FILE = 'tmp.nc'


def station_center_data_gen(nc_data, station_lon, station_lat):
    lat_diff = int(nc_data.OriginX) - station_lon
    lon_diff = int(nc_data.OriginY) - station_lat
    # print int(nc_data.OriginX), int(nc_data.OriginY)
    # print station_lat, station_lon
    # print lat_diff, lon_diff
    center_x = int(lat_diff * LAT_LON_KM_RATIO) + 400
    center_y = int(lon_diff * LAT_LON_KM_RATIO) + 400
    # print center_x, center_y
    '''
    for height in range(0, 4):
        print height
        for x in range(center_x - 50, center_x + 51):
            for y in range(center_y - 50, center_y + 51):
                res = ''.join([ res, str(nc_data.variables['DBZ'][0, height, x, y].data), ' ' ])
                #print nc_data.variables['DBZ'][0, height, x, y]
    '''
    data = nc_data.variables['DBZ'][0, 0:4, center_x - 50:center_x + 51, center_y - 50:center_y + 51]
    # res = np.array_str(data.data.reshape(1, 4*101*101), max_line_width=999999999)
    data = data.data.reshape(1, 4*101*101)
    # print(len(data))
    str_arr = ["%d" % int(float(v)) for v in data[0]]
    res = ' '.join(str_arr)
    # print(res[:50])
    return res


def get_station_dict(station_file):
    f = open(station_file)
    d = {}
    for line in f.readlines():
        # print line
        items = line.strip().split(',')
        lon = items[3]
        lat = items[4]
        idx = items[0]
        d[idx] = [float(lon), float(lat)]
    f.close()
    return d


# dict key format: id#time
def get_label_dict(label_file):
    f = open(label_file)
    d = {}
    for line in f.readlines():
        items = line.strip().split(',')
        idx = items[0]
        t = items[1]
        label = items[2]
        d[idx + '#' + t] = label
    f.close()
    return d


def get_time_list(time_file):
    l = []
    with open(time_file, 'r') as fi:
        for line in fi.readlines():
            if line:
                l.append(line.strip())
    return l


def decompress_gzip_file(infile, outfile):
    f = open(infile)
    compressed_file = io.BytesIO(f.read())
    decompressed_file = gzip.GzipFile(fileobj=compressed_file)

    with open(outfile, 'wb') as outfile:
        outfile.write(decompressed_file.read())
    outfile.close()
    f.close()


time_list = get_time_list(TIME_FILE)
time_list = [element.strip() for element in time_list]
print time_list
print len(time_list)
file_list = listdir(NC_FILE_FOLDER)
print file_list[:10]
station_info = get_station_dict(STATION_INFO_FILE)

label_dict = get_label_dict(LABEL_FILE)
print("label_dict:")
print(label_dict.keys()[:5])

time_li = []
index = 0
for idx in range(15):
    time_count = 9 if idx <= 1 else 8
    if idx == 14:
        time_li.append([v for v in time_list[index:]])
    else:
        time_li.append([v for v in time_list[index: index + time_count]])
    index += time_count
for item in time_li:
    print item


def data_aggregation(time_index):
    count = 0
    time0 = time.time()
    output = open(OUT_FILE + '%02d' % time_index, 'w')
    for t in time_li[time_index]:
        current_time = datetime.strptime(t[:15], "%Y%m%d_%H%M%S")
        print current_time
        current_time_str = current_time.strftime("%Y%m%d_%H%M%S")
        print current_time_str[:-2]
        station_data = {}
        for i in range(0, 15):
            j = 0
            while True:
                tmp_history_time = current_time + timedelta(minutes=j)
                # print tmp_history_time
                tmp_history_time_str = tmp_history_time.strftime("%Y%m%d_%H%M%S")
                # print tmp_history_time_str
                history_file_names = filter(lambda x: tmp_history_time_str[:-2] in x, file_list)
                if history_file_names:
                    break
                elif j > -3:
                    j -= 1
                else:
                    break

            if history_file_names:
                history_file_name = history_file_names[0]
                if i == 14:
                    count += 1
                    good_time.append(current_time_str)
            else:
                print("current_time = %s, i = %d" % (current_time_str, i))
                break

            current_time = tmp_history_time + timedelta(minutes=-5)

            decompress_gzip_file(NC_FILE_FOLDER + '/' + history_file_name, TEMP_NC_FILE + str(time_index))
            nc_data = nc.Dataset(TEMP_NC_FILE + str(time_index))

            for station_id in station_info.keys():
                data = station_center_data_gen(nc_data, station_info[station_id][0], station_info[station_id][1])
                # output.write(station_id + '#' + current_time_str[:-2] + ',,' + data + '\n')
                key = station_id + '#' + current_time_str[:-2]
                # print key
                if key in station_data.keys():
                    station_data[key] = station_data[key] + ' ' + data
                else:
                    station_data[key] = data
                    # print 'station finished: ', station_id
                    # break
            # print '------------------------------all station finished-------------------------------'
            nc_data.close()
            os.remove(TEMP_NC_FILE + str(time_index))

        print current_time_str, ' finished, start writing data...'
        print t
        label_key = datetime.strptime(current_time_str, "%Y%m%d_%H%M%S")
        label_key_str = label_key.strftime("%Y%m%d%H%M%S")
        print("label_key = %s" % label_key_str)
        number = 0
        for key in station_data.keys():
            if (key[:6] + label_key_str) in label_dict:
                output.write(key + ',' + label_dict[key[:6] + label_key_str] + ',' + station_data[key] + '\n')
                number += 1
                # print(station_data[key][-50:])
            else:
                station_data.pop(key)
        print 'writing data finished!', time.time() - time0, count
    output.close()


good_time = []
args = [v for v in range(15)]
print(args)

p = multiprocessing.Pool(15)
print("start distributed aggregate.........................")
# func_data_aggregation = partial(data_aggregation)
# p.map(func_data_aggregation, args)
p.map(data_aggregation, args)

good_time.sort()
print("good time count = %d" % len(good_time))
print(good_time)

''' get non-overlapped integer o'clocks which has consequent 15 radar data in the last 1.5h
overlap_times = []
for item in good_time:
    idx = good_time.index(item) + 1
    if idx < len(good_time):
        next_time = good_time[idx]
    if item[:9] == next_time[:9]:
        if int(item[9:11]) + 1 == int(next_time[9:11]):
            overlap_times.append(good_time.pop(idx))

print("good_time left => %d" % len(good_time))
print(good_time)
print("overlap time del => %d" % len(overlap_times))
print(overlap_times)
time_info = "/data/yuyang/weather/dataflow/src/main/resources/time_info_121"
if not time_info:
    os.mkdir(time_info)
with open(time_info, "w") as fo:
    for item in good_time:
        fo.write(item + '\n')
'''

'''
filename='weather_test_data/radar/20141125_005937_bjanc_mergedDbz.nc'
data=nc.Dataset(filename, format="NETCDF3_CLASSIC")
for i in data.variables.keys():
    print(i)
for i in range(0, 19):
    print data.variables['DBZ'][0][i][00][100]
print data.variables['DBZ'][0, 5, 100, 100]
print data.variables['time'][:]
print data.variables['height'][:]
print data
print data.OriginX
'''
