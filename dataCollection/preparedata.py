from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pdfkit
import os
from datetime import datetime
from pdf2image import convert_from_path
import os
import layoutparser as lp
import cv2

baseURL="https://www.sec.gov/edgar/search/#/category=custom&forms=10-K"

def getUrls(outFile):
    ## List URLs in the file
    
    driver = webdriver.Firefoxdriver = webdriver.Firefox(executable_path=r'./geckodriver')

    driver.get(baseURL)
    time.sleep(3)


    links=driver.find_elements_by_css_selector("td.filetype a.preview-file")
    print(len(links))

    with open(outFile ,'a') as outputFile:
        for idx,link in enumerate(links):
            link.click()

            fileUrl = driver.find_element_by_css_selector("a#open-file")
            print(fileUrl.get_attribute("href"))
            outputFile.write(fileUrl.get_attribute("href")+"\n")

            closebtn = driver.find_element_by_css_selector("button.close")
            closebtn.click()
            time.sleep(1)



def htmlTopdf(urlFile,outFolder):
    # convert HTML to pdf
    with open(urlFile ,'r') as outputFile:
        lines = outputFile.readlines()
        for l in lines:
            filename=os.path.basename(l).split(".")[0]
            pdfkit.from_url(l, f"{outFolder}/{filename}.pdf")
            


def getTableImage(inputFolder,outputFolder):
    tmpimg='tmp.png'
    # on LayoutParser docker container
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7])

    for filename in os.listdir(inputFolder):   
        if filename.endswith(".pdf"):
            pages = convert_from_path(inputFolder+filename)
            outputfilename=outputFolder+"/"+filename.split(".")[0]+".png"
            
            for i,p in enumerate(pages):

                layout = model.detect(p)
                table_blocks = lp.Layout([b for b in layout if b.type=='Table'])
                
                # lp.draw_box(p,[b.set(id=f'{b.type}/{b.score:.2f}') for b in table_blocks],
                # show_element_id=True, id_font_size=20,
                # id_text_background_color='grey',
                # id_text_color='white').save(f"table/page{i}.jpg")

            # for idx,block in enumerate(table_blocks):
            #     segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(img)
            #     plt.imshow(segment_image, interpolation='nearest')
            #     plt.show()
            #     segment_image.save(f"output/pagecrop{i}_idx.jpg")          

                if len(table_blocks) >0 :
                    p.save(outputfilename, "PNG")

if __name__ == "__main__":
    ## GET URLS
    timestampstr=str(round(time.time()))
    outputFolder="pdfDataset_"+timestampstr
    os.mkdir(outputFolder)
    urlsList=f"{outputFolder}/_urlList.txt"
    getUrls(urlsList)

    ### Generate PDF
    htmlTopdf(urlsList,outputFolder)

    ### Extract Table (layoutparser docker container)
    pdfDataset="pdfDataset_1645287846"
    tableDataset="tableDataset_1645287846"
    getTableImage(pdfDataset,tableDataset)










