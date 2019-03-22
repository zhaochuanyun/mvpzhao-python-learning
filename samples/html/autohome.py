import requests
import pandas as pd
import numpy as np

car_dict = {'大众朗逸': '614',
            '荣威RX5': '4080',
            '别克GL8': '166',
            '荣威ei6': '4263',
            '荣威eRX5': '4240',
            '奔驰C级': '588',
            '别克昂科威': '3554',
            '大众途观L': '4274',
            '大众途安': '333',
            '雪佛兰科鲁兹': '657',
            '奔驰GLC级': '3862',
            '斯柯达明锐': '519',
            '奥迪A4L': '692',
            '宝马3': '66',
            '大众帕萨特': '528',
            '宝马5': '65',
            '奔驰E级': '197',
            '雪佛兰迈锐宝': '2313',
            '别克英朗GT': '982',
            '比亚迪秦': '2761'}

url = 'https://www.autohome.com.cn/ashx/dealer/AjaxDealersBySeriesId.ashx?seriesId={}&cityId=310100&provinceId=310000&countyId=0&orderType=0&kindId=1'

columns = ['series', 'countyName', 'dealerName', 'companySimple', 'address', 'longitude', 'latitude']

df = pd.DataFrame(np.empty((0, len(columns))), columns=columns)

for car in car_dict.keys():
    r = requests.get(url.format(car_dict[car]))
    for info in r.json()['result']['list']:
        base_info = info['dealerInfoBaseOut']
        base_info['series'] = car
        d_info = {}
        for column in columns:
            d_info[column] = base_info[column]
        df.loc[df.shape[0] + 1] = d_info

df.to_csv('4s店信息.csv')
