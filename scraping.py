from re import sub
from decimal import Decimal
import requests
import csv
from lxml import html
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
page=0
for page in range(400):

    response = requests.get("https://www.zomato.com/bangalore/restaurants?ref_page=zone&page="+str(page+1),headers=headers)
    content = response.content
    sourceCode=content
    htmlElem = html.fromstring(sourceCode)
    soup = BeautifulSoup(content,"html.parser")

    i=0
    for i in range(15):
        row=[]

        name = soup.find_all("div", class_="col-s-12")
        try:
            l = name[i]
            z= (l.find_all('a' , class_="result-title hover_feedback zred bold ln24 fontsize0 "))
            row.append(z[0].text.strip())
        except:
            row.append(" ")

        # call = soup.find_all('div',class_="ui item menu search-result-action mt0")
        # l = call[0]
        # z = (l.find_all('a',class_="item res-snippet-ph-info"))
        # print(z[0].text)

        cost = soup.find_all('div',class_="res-cost clearfix")
        try:
            l=cost[i]

            z= (l.find_all('span',class_="col-s-11 col-m-12 pl0"))
            money = z[0].text.strip()
            value = Decimal(sub(r'[^\d.]', '', money))
            row.append(value)
        except:
            row.append(" ")
        cuisines = soup.find_all(class_="search-page-text clearfix row")
        try:
            z = cuisines[i]
            # print (z)
            l = (z.find_all(class_="col-s-11 col-m-12 nowrap pl0"))
            cuisine_list=[]
            a=l[0].text.split(",")
            row.append(len(a))
        except:
            row.append(" ")

        features = z.find_all("div",class_="col-s-11 col-m-12 pl0 search-grid-right-text ")
        try:
            l=features[1].find_all("a")
            list=[]
            for a in l:
                list.append(a.text)
            row.append(len(list))
        except Exception as e:
            row.append(" ")

        try:
            hours=features[0]
            row.append(hours.text.strip())
        except:
            row.append(" ")

        votes=soup.find_all(class_="ta-right floating search_result_rating col-s-4 clearfix")
        try:
            # print (votes[i].find_all("span")[0].text.strip().split(" ")[0])
            row.append(votes[i].find_all("span")[0].text.strip().split(" ")[0])
            rating=votes[i].find("div")
            row.append(float(rating.text.strip()))
        except:
            row.append(" ")
            row.append(" ")


        name = soup.find_all("a", class_="ln24 search-page-text mr10 zblack search_result_subzone left")
        try:
            l = name[i]
            row.append(l.text.strip())
        except:
            row.append(" ")

        rest_type = soup.find_all("div",class_="res-snippet-small-establishment mt5")
        try:
            l=rest_type[i].find_all("a")
            list=[]
            for a in l:
                list.append(a.text)
            row.append(len(list))
        except:
            row.append(" ")


        try:
            tdElems2 = htmlElem.xpath("/html/body/section/div/div[2]/div[3]/div[2]/div/div[6]/div/div[1]/section/div[1]/div[3]/div[" + str(i+1) + "]/div[2]/a[2]")
            if tdElems2[0].text_content().strip()=="View Menu":
                row.append("1")
            else:
                row.append("0")
        except:
            row.append("0")
        try:
            tdElems2 = htmlElem.xpath("/html/body/section/div/div[2]/div[3]/div[2]/div/div[6]/div/div[1]/section/div[1]/div[3]/div[" + str(i+1) + "]/div[2]/a[3]")
            if len(tdElems2[0].text_content().strip())!=0:
                row.append("1")
            else:
                row.append("0")
        except:
            row.append("0")
        try:
            outlets = htmlElem.xpath("/html/body/section/div/div[2]/div[3]/div[2]/div/div[6]/div/div[1]/section/div[1]/div[3]/div[" + str(i+1) + "]/div[3]")
            row.append(outlets[0].text_content().strip().split(" ")[0])
        except:
            row.append(" ")
        try:
            tdElems2 = htmlElem.xpath("/html/body/section/div/div[2]/div[3]/div[2]/div/div[6]/div/div[1]/section/div[1]/div[3]/div[" + str(i) + "]/div[2]/a[4]")
            if len(tdElems2[0].text_content().strip())!=0:
                row.append("1")
            else:
                row.append("0")
        except:
            row.append("0")
        try:
            tdElems2 = htmlElem.xpath("/html/body/section/div/div[2]/div[3]/div[2]/div/div[6]/div/div[1]/section/div[1]/div[3]/div[" + str(i) + "]/div[2]/a[1]")
            if len(tdElems2[0].text_content().strip())!=0:
                row.append("1")
            else:
                row.append("0")
        except:
            row.append("0")

        i=i+1
        with open('zomato_res_final.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
    print (page+1)
