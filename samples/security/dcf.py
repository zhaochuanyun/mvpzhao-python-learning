#!/usr/bin/python
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    fcf0 = 25e8  # 自由现金流
    shares = 10.41e8  # 股本
    i = 0.1  # i:10年稳定成长率
    g = 0.05  # g:一般估计为GDP增长率，处于衰退行业的公司相应降低
    R = 0.105  # R:折现率，根据企业现金流稳定性确定，晨星公司使用10.5%作为一般水平，低风险公司使用9%，高风险公司使用13-15%
    N = 10  # 成长年
    safe_margin = 0.3  # 正常范围：30%~40%  有优势：20%  高风险：60%

    fcfn = fcf0 * ((1 + i) ** N)
    pv = fcfn * (1 + g) / (R - g)
    dpv = pv / ((1 + R) ** N)

    dcf = 0
    for n in range(1, 10):
        dcf += fcf0 * ((1 + i) ** n) / ((1 + R) ** n)

    e = dpv + dcf

    vps = e / shares

    price = vps * (1 - safe_margin)

    print("vps: %.2f" % (vps))
    print("safe margin: %.1f%%, price: %.2f" % (safe_margin * 100, price))
