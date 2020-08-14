# encoding: UTF-8

from __future__ import division

import random, copy
import requests # pip3 install requests
import re

__FakedUserAgents=[
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.81 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.27 (KHTML, like Gecko) Chrome/12.0.712.0 Safari/534.27",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.24 Safari/535.1",
    "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/535.2 (KHTML, like Gecko) Chrome/15.0.874.120 Safari/535.2",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.36 Safari/535.7",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.81 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.27 (KHTML, like Gecko) Chrome/12.0.712.0 Safari/534.27",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.24 Safari/535.1",
    "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/535.2 (KHTML, like Gecko) Chrome/15.0.874.120 Safari/535.2",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.36 Safari/535.7",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0b4pre) Gecko/20100815 Minefield/4.0b4pre",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; zh-CN) AppleWebKit/533.19.4 (KHTML, like Gecko) Version/5.0.2 Safari/533.18.5",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-GB; rv:1.9.1.17) Gecko/20110123 (like Firefox/3.x) SeaMonkey/2.0.12",
    "Mozilla/5.0 (Windows NT 5.2; rv:10.0.1) Gecko/20100101 Firefox/10.0.1 SeaMonkey/2.7.1",
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_8; zh-CN) AppleWebKit/532.8 (KHTML, like Gecko) Chrome/4.0.302.2 Safari/532.8",
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_4; zh-CN) AppleWebKit/534.3 (KHTML, like Gecko) Chrome/6.0.464.0 Safari/534.3",
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_5; zh-CN) AppleWebKit/534.13 (KHTML, like Gecko) Chrome/9.0.597.15 Safari/534.13",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.186 Safari/535.1",
    "Mozilla/5.0 (Macintosh; U; PPC Mac OS X; en) AppleWebKit/125.2 (KHTML, like Gecko) Safari/125.8",
    "Mozilla/5.0 (Macintosh; U; PPC Mac OS X; fr-fr) AppleWebKit/312.5 (KHTML, like Gecko) Safari/312.3",
    "Mozilla/5.0 (Macintosh; U; PPC Mac OS X; en) AppleWebKit/418.8 (KHTML, like Gecko) Safari/419.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1 Camino/2.2.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0b6pre) Gecko/20100907 Firefox/4.0b6pre Camino/2.2a1pre",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.4 (KHTML like Gecko) Chrome/22.0.1229.79 Safari/537.4",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2; rv:10.0.1) Gecko/20100101 Firefox/10.0.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:16.0) Gecko/20120813 Firefox/16.0",
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X; zh-CN) AppleWebKit/528.16 (KHTML, like Gecko, Safari/528.16) OmniWeb/v622.8.0.112941",
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_6; zh-CN) AppleWebKit/528.16 (KHTML, like Gecko, Safari/528.16) OmniWeb/v622.8.0"
    ]

def nextUserAgent() :
    return random.choice(__FakedUserAgents)

__proxyList=[]
__everGoods=[]

def stampGoodProxy(priority = 10.0):
    global __proxyList, __everGoods
    if len(__proxyList) <=0: return
    stmt = '%05d>%s' %(int(priority * 1000) % 100000, __proxyList[0])
    if len(__everGoods) >0:
        oldstmt = __everGoods[-1] 
        tokens = oldstmt.split('>')
        if len(tokens) >1 and tokens[1] == __proxyList[0]:
            if stmt >= oldstmt: return
            else: del(__everGoods[-1])

    __everGoods.append(stmt)

def listFrom_skyriver():
    prxs = []
    # download the proxy list from https://skyrivermecatronic.com, but many of them may not work
    # wget -O- --no-check-certificate https://skyrivermecatronic.com/proxychains/ | grep -o '<br>http.*' |sed 's/<br>\([a-z0-9]*\)[&nbsp;]*\([0-9\.]*\)[&nbsp;]*\([0-9\.]*\)/\n\1:\/\/\2:\3/g'
    # potential version issue: https://blog.csdn.net/fangbinwei93/article/details/59526937?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
    response = requests.get('https://skyrivermecatronic.com/proxychains/')
    httperr = response.status_code
    if 200 != httperr:
        return prxs

    lines =  response.text.split('<br>')
    SYNTAX = re.compile('^([a-z0-9]*)[&nbsp;]*([0-9\.]*)[&nbsp;]*([0-9\.]*).*')
    for l in lines:
        m = SYNTAX.match(l)
        if not m : continue
        prot, ip, port = m.group(1).lower(), m.group(2).lower(), m.group(3)
        if len(prot) + len(ip) + len(port) <8 : continue # or 'http' in prot
        prxs.append('%s://%s:%s' % (prot, ip, port))
    
    return prxs

def listFrom_proxyranker():
    prxs = []
    response = requests.get('https://www.proxyranker.com/china/list/')
    httperr = response.status_code
    if 200 != httperr:
        return prxs

    # <td>115.47.45.82</td><td>China</td><td>Beijing</td><td><span title="Proxy port">8080</span></td><td>0.329s</td><td>anonymous (HTTPS/SSL)</td></tr>
    # <tr><td>117.79.73.166</td><td>China</td><td>Beijing</td><td><span title="Proxy port">8080</span></td><td>0.325s</td><td>anonymous</td></tr>
    lines =  response.text.split('</tr>')
    SYNTAX = re.compile('^.*<td.*>([0-9\.]+)</td>.*<span title="Proxy port">([0-9\.]*)</span>.*(anonymous[^<]*)</td>')
    for l in lines:
        m = SYNTAX.match(l)
        if not m : continue
        prot, ip, port = m.group(3).lower(), m.group(1), m.group(2)
        if len(ip) <=0 or len(prot) + len(port) <5 : continue # or 'http' in prot
        prot = 'https' if 'https' in prot else 'http'
        prxs.append('%s://%s:%s' % (prot, ip, port))
    
    return prxs

def listFrom_proxynova():
    prxs = []
    response = requests.get('https://www.proxynova.com/proxy-server-list/country-cn/')
    httperr = response.status_code
    if 200 != httperr:
        return prxs

    '''
    <td align="left"><abbr title="47.106.220.74"><script>document.write('47.106.220.74');</script></abbr></td>
    <td align="left"> 3380  </td>
    <td align="left"><time class="icon icon-check timeago" datetime="2020-08-13 14:23:48Z"></time></td>
    <td align="left"><div class="progress-bar" data-value="28.5991925" title="1343"></div><small>1343 ms</small>  </td>
    <td class="text-center text-sm"> <span class="uptime-high">96%</span> <span> (32)</span> </td>
    <td align="left"> <img src="/assets/images/blank.gif" class="flag flag-cn inline-block align-middle" alt="cn" />  <a href="/proxy-server-list/country-cn/" title="Proxies from China">China  <span class="proxy-city"> - Hangzhou </span></a> </td>
    <td align="left">  <span class="proxy_transparent font-weight-bold smallish">Transparent</span>  </td>
    </tr>
    '''
    txt = response.text.replace("\n", "").replace("\r", "")
    lines =  txt.split('</tr>')
    SYNTAX = re.compile('^.*document.write\(.([0-9\.]+).*<td.*>[ \t]*([0-9\.]*)[ \t]*</td>.*Proxies from China.*(Elite|Transparent|Anonymous).*</td>')
    for l in lines:
        m = SYNTAX.match(l)
        if not m : continue
        prot, ip, port = 'http', m.group(1), m.group(2) # m.group(3).lower(), m.group(1), m.group(2)
        if len(ip) <=0 or len(prot) + len(port) <5 : continue # or 'http' in prot
        prot = 'https' if 'https' in prot else 'http'
        prxs.append('%s://%s:%s' % (prot, ip, port))
    
    return prxs

def nextProxy():
    global __proxyList, __everGoods

    ret = None
    if len(__proxyList) <= 0:
        __everGoods.sort()
        __proxyList = [i.split('>')[1] for i in __everGoods]
        __everGoods = []

    try :
        if len(__proxyList) <=0:
            __proxyList = listFrom_proxynova() # listFrom_skyriver() 
    except:
        pass
            
    if len(__proxyList) >0:
        ret = __proxyList[0]
        del(__proxyList[0])
    
    return ret

if __name__ == '__main__':

    for i in range(9999):
        print('prx[%s]' % nextProxy())
