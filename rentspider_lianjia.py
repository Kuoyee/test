import requests
from bs4 import BeautifulSoup
from threading import Thread
import csv
import json

# num表示记录序号
#网络格式：https://xm.lianjia.com/zufang/siming/pg1/#contentList
Url_head = "https://xm.lianjia.com/zufang/"
Url_tail = "/#contentList"
Filename = "/home/hadoop/代码/rent2.csv"


# 把每一页的记录写入文件中
def write_csv(msg_list):
    out = open(Filename, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for msg in msg_list:
        csv_write.writerow(msg)
    out.close()

#访问每一个具体房源信息
def acc_everyhourse_msg(page_url):
    web_data = requests.get(page_url).content.decode('utf8')
    soup = BeautifulSoup(web_data, 'html.parser')
    for tag in soup.find_all(name='ul',attrs="content__aside__list"):
        if len(tag) ==3:
            break
        else:
            idli=0
            for li in tag.find_all(name='li'):
                if idli == 2: #朝向楼层
                    idspan=0
                    for span in li:
                        if idspan == 2:
                            break
                        elif idspan == 1:
                            chaoxiangtext=span.get_text()
                        idspan+=1
                    
                elif idli == 1: #房屋类型
                    idspan=0
                    for span in li:
                        if idspan ==0 :
                            idspan+=1
                        else:
                            fangwutext=span
                elif idli == 0: #租赁方式
                    idspan=0
                    for span in li:
                        if idspan ==0 :
                            idspan+=1
                        else:
                            zulintext=span
                idli+=1
                if idli == 3:
                    break
    liuse=[5,9,11,15]
    idli=1
    for tag in soup.find_all(name='li',attrs="fl oneline"):
        if idli in liuse:
            if idli == 5: #维护
                weihutext=tag.get_text()[3:]
            elif idli == 9: #电梯
                diantitext=tag.get_text()[3:]
            elif idli == 11: #车位
                cheweitext=tag.get_text()[3:]
            else:#燃气
                ranqitext=tag.get_text()[3:]
        idli+=1
    peizhi=[1 for i in range(10)] #配置列表，有为1，没有为0
    #分别代表：洗衣机、空调、衣柜、电视、冰箱、热水器、床、暖气、宽带、天然气
    for tag in soup.find_all(name='ul',attrs="content__article__info2"):
        #查找没有的配置
        for lino in tag.find_all(attrs="fl oneline facility_no"):
            without=lino.get_text().strip()  #缺失的配置
            if without == '洗衣机':
                peizhi[0]=0
            elif without == '空调':
                peizhi[1]=0
            elif without == '衣柜':
                peizhi[2]=0                
            elif without == '电视':
                peizhi[3]=0            
            elif without == '冰箱':
                peizhi[4]=0  
            elif without == '热水器':
                peizhi[5]=0
            elif without == '床':
                peizhi[6]=0                
            elif without == '暖气':
                peizhi[7]=0            
            elif without == '宽带':
                peizhi[8]=0
            elif without == '天然气':
                peizhi[9]=0
    #经纬度信息
    for tag in soup.find_all(name='div',attrs="wrapper"):
        idscript=0
        for script in tag.find_all(name='script'):
            idscript+=1
            if idscript == 2:
                getmap=script.get_text()[script.get_text().find('g_conf.coord'):script.get_text().find('g_conf.subway')-3]
                longitude=getmap[getmap.find('longitude')+12:getmap.find('longitude')+19]
                latitude=getmap[getmap.find('latitude')+11:getmap.find('latitude')+17]
    return zulintext,fangwutext,chaoxiangtext,weihutext,diantitext,cheweitext,ranqitext,peizhi,longitude,latitude
                    
    
# 访问每个区域房源的一页
def acc_page_msg(page_url):
    web_data = requests.get(page_url).content.decode('utf8')
    soup = BeautifulSoup(web_data, 'html.parser')
    rentmode=[]
    decoration=[] #装修
    #室-厅-卫
    room=[]
    hall=[]
    bathroom=[]
    area=[]#面积
    orientation=[] #朝向
    floor=[] #楼层
    maintain=[] #维护
    elevator=[] #电梯
    parking=[] #车位
    gas=[] #燃气
    
    #洗衣机、空调、衣柜、电视、冰箱、热水器、床、暖气、宽带、天然气
    washingmachine=[]
    airconditioner=[]
    wardrobe=[]
    TV=[]
    refrigerator=[]
    waterheater=[]
    bed=[]
    heating=[]
    broadband=[]
    naturalgas=[]
    #区域-地址-名称
    district=[]
    address=[]
    #价格
    price=[]
    msg_list = []
    #经纬度
    longitude=[]
    latitude=[]
    
    i=0
    for tag in soup.find_all(attrs="content__list--item"):
        i+=1
        print(i)
        em = tag.find(name='a',attrs="content__list--item--aside")  ##找到房源网址
        if em['href'][1:7] == 'zufang':   ##如果是zufang网站，否则为广告
            idp=1
            for p in tag.find(name='p',attrs="content__list--item--des"):
                if idp == 2 :
                    if p.string == '/' or len(p.string) == 1:
                        continue
                    else:
                        district.append(p.get_text())
                elif idp == 4 :
                    ad=p.get_text()
                elif idp == 6 :
                    ad=ad+'-'+p.get_text()
                    address.append(ad)
                idp+=1
            for span in tag.find(name='span',attrs="content__list--item-price"):
                price.append(span.get_text())
                break
            zulintext,fangwutext,chaoxiangtext,weihutext,diantitext,cheweitext,ranqitext,peizhi,longitudetext,latitudetext = acc_everyhourse_msg('https://xm.lianjia.com'+ em['href'])
            rentmode.append(zulintext)
            #预处理
            #将“房屋类型”拆分成 室-厅-卫-面积-（装修）
            fangwulist=fangwutext.split()
            if fangwulist[-1] == '精装修':
                decoration.append(1)
            else:
                decoration.append(0)
            room.append(fangwulist[0][:fangwulist[0].find('室')])
            hall.append(fangwulist[0][fangwulist[0].find('室')+1:fangwulist[0].find('厅')])
            bathroom.append(fangwulist[0][fangwulist[0].find('厅')+1:-1])
            area.append(fangwulist[1][:-1])
            #朝向楼层分为 朝向-楼层
            cxlclist=chaoxiangtext.split()
            orientation.append(cxlclist[0])
            floor.append(cxlclist[1].split('/')[1][:-1])
            
            maintain.append(weihutext)
            elevator.append(diantitext)
            parking.append(cheweitext)
            gas.append(ranqitext)
            
            washingmachine.append(peizhi[0])
            airconditioner.append(peizhi[1])
            wardrobe.append(peizhi[2])
            TV.append(peizhi[3])
            refrigerator.append(peizhi[4])
            waterheater.append(peizhi[5])
            bed.append(peizhi[6])
            heating.append(peizhi[7])
            broadband.append(peizhi[8])
            naturalgas.append(peizhi[9])
            
            longitude.append(longitudetext)
            latitude.append(latitudetext)

    # 组合成为一个新的tuple——list并加上序号
    for i in range(len(price)):
        txt = (district[i],address[i],price[i],rentmode[i],
            room[i],hall[i],bathroom[i],area[i],decoration[i],
            orientation[i],floor[i],maintain[i],elevator[i],parking[i],
            gas[i],washingmachine[i],airconditioner[i],
            wardrobe[i],TV[i],refrigerator[i],waterheater[i],bed[i],
            heating[i],broadband[i],naturalgas[i],longitude[i],latitude[i])
        msg_list.append(txt)
    # 写入csv
    write_csv(msg_list)
    print(msg_list)

# 爬所有的页面
def get_pages_urls():
    urls = []
    # 思明可访问页数100
    for i in range(1):
        urls.append(Url_head + "siming" + "/pg" + str(i + 1) + Url_tail )
    # 湖里可访问页数68
    for i in range(1):
        urls.append(Url_head + "huli" + "/pg" + str(i + 1) + Url_tail )
    # 集美可访问页数70
    for i in range(1):
        urls.append(Url_head + "jimei" + "/pg" + str(i + 1) + Url_tail )
    # 同安可访问页数15
    for i in range(1):
        urls.append(Url_head + "tongan" + "/pg" + str(i + 1) + Url_tail )
    # 翔安可访问页数16
    for i in range(1):
        urls.append(Url_head + "xiangan" + "/pg" + str(i + 1) + Url_tail )
    # 海沧可访问页数37
    for i in range(1):
        urls.append(Url_head + "haicang" + "/pg" + str(i + 1) + Url_tail )

    return urls

out = open(Filename, 'w', newline='')
csv_write = csv.writer(out, dialect='excel')
    #区域District,地址Address,每月租金Price,租赁方式Rentmode,室Room,厅Hall,卫Bathroom,面积Area,精装修Decoration,
    #朝向Orientation,楼层Floor,维护Maintain,电梯Elevator,车位Parking,燃气Gas,
    #洗衣机Washingmachine,空调Airconditioner,衣柜Wardrobe,电视TV,冰箱Refrigerator,热水器Waterheater,床Bed,
    #暖气Heating,宽带Broadband,天然气Naturalgas,经度Longitude,纬度Latitude
title = ("District","Address", "Price", "Rentmode","Room", "Hall", "Bathroom", "Area", "Decoration", 
             "Orientation","Floor","Maintain","Elevator","Parking","Gas",
             "Washingmachine","Airconditioner","Wardrobe","TV","Refrigerator","Waterheater","Bed",
             "Heating","Broadband","Naturalgas","Longitude","Latitude")
csv_write.writerow(title)
out.close()

def run(url_list):
    print("开始爬虫")
    
    print("listlen:",len(url_list))
    for url in url_list:
        try:
            acc_page_msg(url)
        except:
            print("格式出错", url)
    print("结束爬虫")
url_list = get_pages_urls()
#分4个线程
s=len(url_list)//4
url1=url_list[:s]
url2=url_list[s:s*2]
url3=url_list[s*2:s*3]
url4=url_list[s*3:]
url4proj=[url1,url2,url3,url4]
print(url4proj)
threads=[]
for i in range(4):
    t = Thread(target=run, args=(url4proj[i],))
    threads.append(t)
for t in threads:
    t.start()