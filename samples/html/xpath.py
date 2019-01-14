# -*- coding: utf-8 -*-
from selenium import webdriver


def heros():
    driver = webdriver.PhantomJS()
    driver.get("http://pvp.qq.com/web201605/herolist.shtml")
    ul = driver.find_element_by_xpath('/html/body/div[3]/div/div/div[2]/div[2]/ul')
    lis = ul.find_elements_by_xpath('li')
    for li in lis:
        a = li.find_elements_by_xpath('a')[-1]
        img = a.find_element_by_xpath('img')
        print(a.text)
        print(img.get_attribute('src') + '\n')
    driver.quit()


if __name__ == '__main__':
    heros()