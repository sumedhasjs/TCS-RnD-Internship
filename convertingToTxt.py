import os
from xml.etree import ElementTree
directory='D:\TCS\hievents'

for doc in os.listdir(directory):
    path=directory+ '\\'+ str(doc)
    xml = ElementTree.parse(path)    
    ArticleText=xml.find("Text")
    print(ArticleText.text)
    
    output_directory='D:\TCS\hievents_txt'
    new_doc_name=doc.split('.xml')[0]
    op_path=output_directory+'\\'+str(new_doc_name)+(".txt")
    temp=ArticleText.text
    with open(op_path,'w') as output_file:
        output_file.write(temp.strip())
    output_file.close()
