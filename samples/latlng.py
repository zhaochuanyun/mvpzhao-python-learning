import exifread


def imageread():
    f = open("/Users/mvpzhao/Downloads/WechatIMG140.jpeg", 'rb')

    imagetext = exifread.process_file(f)
    for key in imagetext:  # 打印键值对
        print(key, ":", imagetext[key])

    print('********************************************************')

    for q in imagetext:  # 打印该图片的经纬度 以及拍摄的时间
        if q == "GPS GPSLongitude":
            print("GPS经度 =", imagetext[q], imagetext['GPS GPSLatitudeRef'])
        elif q == "GPS GPSLatitude":
            print("GPS纬度 =", imagetext[q], imagetext['GPS GPSLongitudeRef'])
        elif q == 'Image DateTime':
            print("拍摄时间 =", imagetext[q])


imageread()
