import os
import csv
from xml.etree import ElementTree

import spacy

directory = 'D:\TCS\hievents'

csvfile = open("hievents.csv",'w',newline='',encoding='utf-8')
csvfile_writer = csv.writer(csvfile,delimiter=",")
csvfile_writer.writerow(["Article Name","Text"])

flag=1
for filename in os.listdir(directory):
    #print(filename)
    path=directory+ '\\'+ str(filename)
    xml = ElementTree.parse(path)

    
    ArticleText=xml.find("Text")
    print(ArticleText)
    csv_line = [filename, ArticleText.text]
    csvfile_writer.writerow(csv_line)
csvfile.close()     




