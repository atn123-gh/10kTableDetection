### 10kTableDetection

A model to retrieve table with header and footer information from 10k report.
Finetune with Detectron2LayoutModel (lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config ) on GPU.


Datacollection 
  - Web scrab the 10k data
  - convert html to pdf 
  - Use layout parser to retrieve pages with table
  - annotate table "with header and footer information" using lable studio.(coco format)
  - train model with dectron2
  - inference model
  

Can improve accuracy with more training data.
  

The model is trained on GPU.

This was intended to train on Habana Gaudi AI Processor(HPU) for AWS Deep Learning Challenge
 (https://amazon-ec2-dl1.devpost.com/?ref_feature=challenge&ref_medium=discover )

But found that HPU PyTorch migration doesn't work with dectron2.

### Reference

https://www.sec.gov/edgar/search/#/category=custom&forms=10-K (data)</br>
https://github.com/Layout-Parser/layout-parser (A unified toolkit for Deep Learning Based Document Image Analysis)</br>
https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html (base model)</br>
https://github.com/Layout-Parser/layout-model-training</br>
https://labelstud.io/ (Open Source Data Labeling Tool)</br>
https://github.com/facebookresearch/detectron2 ( model training library)</br>
https://docs.habana.ai/en/latest/PyTorch/PyTorch_User_Guide/index.html</br>
https://github.com/HabanaAI/Model-References ( habana migration referene)


