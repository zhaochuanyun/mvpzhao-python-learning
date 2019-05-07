import requests
import pandas as pd
import numpy as np

car_dict = {'大众朗逸': '614',
            '奔驰C级': '588',
            '别克昂科威': '3554',
            '雪佛兰科沃兹': '4105',
            '斯柯达明锐': '519'}

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
