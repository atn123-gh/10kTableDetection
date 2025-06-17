### **10-K Financial Table Detection**

A deep learning project focused on detecting financial tables—including headers and footers—in 10-K SEC filings. Unlike general table detection models, this solution is specialized for financial statement layouts.

**Objective:**
Train a custom object detection model for 10-K financial statement tables using Detectron2 and Layout Parser.

---

#### **Workflow Summary**

1. **Data Collection**

   * Scraped 10-K reports from the SEC EDGAR database.
   * Converted HTML filings to PDF format.
   * Used Layout Parser to identify and extract pages containing tables.
2. **Annotation**

   * Annotated tables with header and footer information using Label Studio.
   * Saved labels in COCO format.
3. **Model Training**

   * Fine-tuned Detectron2 Layout Model:
     `lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config`
   * Trained using GPU on a custom dataset.
4. **Inference**

   * Evaluated model on unseen 10-K filings.
   * Demonstrated improved performance on structured financial table detection.

---

#### **Platform Note**

The project was originally planned for the **AWS Deep Learning Challenge** using **Habana Gaudi (HPU)** processors:
[https://amazon-ec2-dl1.devpost.com](https://amazon-ec2-dl1.devpost.com)
However, Detectron2 was found incompatible with Habana's PyTorch support at the time of testing.

---

#### **Key Tools and Libraries**

* Python, Detectron2, Layout Parser
* Label Studio (annotation)
* COCO format (dataset structure)
* SEC EDGAR (data source)

---

#### **Improvements & Future Work**

* Accuracy can be improved significantly with a larger annotated dataset.
* Potential future integration with text extraction for content-aware processing of table cells.

---

#### **References**

* [SEC EDGAR Search (10-K)](https://www.sec.gov/edgar/search/#/category=custom&forms=10-K)
* [Layout Parser](https://github.com/Layout-Parser/layout-parser)
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [Label Studio](https://labelstud.io/)
* [HabanaAI PyTorch Guide](https://docs.habana.ai/en/latest/PyTorch/PyTorch_User_Guide/index.html)
* [Habana Model References](https://github.com/HabanaAI/Model-References)

