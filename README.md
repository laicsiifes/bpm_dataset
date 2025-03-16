# **BPM Dataset: Extraction of Entities and Business Process Constraints from Text**  

## 📌 **Overview**  
This repository provides code and datasets related to the **extraction of entities and business process constraints from textual descriptions**. The work was developed as part of a **Master’s dissertation** at the **Instituto Federal do Espírito Santo (IFES), Brazil**, focusing on **Natural Language Processing (NLP) techniques** to improve business process modeling.  

## 🔍 **Research Motivation**  
Business process descriptions are often written in **natural language**, making them **ambiguous and difficult to formalize**. Traditional business process modeling is mostly **manual and time-consuming**, requiring experts to interpret and convert textual descriptions into formal representations like **BPMN** or **DECLARE**.  

This research proposes an **intermediate step** in the modeling process:  
1. **Extracting named entities and constraint relations** from text using NLP techniques.  
2. **Using these extracted constraints** to generate declarative business process models.  

## 🛠 **Methodology**  
The approach consists of two key tasks:  

1️⃣ **Named Entity Recognition (NER)**  
   - Identifies entities such as **actors**, **activities**, and **resources** in business process descriptions.  
   - Models tested:  
     - **Conditional Random Fields (CRF)**  
     - **BiLSTM-CRF**  
     - **Transformer-based models (BERT, DistilBERT, RoBERTa)**  

2️⃣ **Relation Classification (RC)**  
   - Identifies constraint relationships between extracted entities.  
   - Models tested:  
     - **BERT**, **DistilBERT**, and **RoBERTaLarge**  
   - The **RoBERTaLarge** model showed the best performance for identifying complex dependencies.  

## 📂 **Dataset**  
- **133 documents**  
- **1,361 sentences**  
- **5,395 annotated entities and relations**  
- Covers entities like **actors, activities** and relationships like **strict dependency and circumstantial dependency**  

## 🚀 **Results**  
- **NER Task:** The best-performing model was **BiLSTM-CRF** with **Glove and Flair embeddings**, achieving high **F1-score** for most entity types.  
- **Relation Classification Task:** **RoBERTaLarge** outperformed other models, especially in complex relationship types.  
- The approach **reduces ambiguity** and supports **automatic process model generation**.  

## 📌 **Key Contributions**  
✅ A **high-quality annotated dataset** for business process modeling from text.  
✅ A **novel NLP-based approach** for extracting business process constraints.  
✅ Insights on **model performance for entity recognition and relation classification**.  
✅ Potential for **semi-automatic process modeling**, reducing manual effort.  

## 📜 **How to Use the Code**  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/laicsiifes/bpm_dataset.git
   cd bpm_dataset
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run corpus construction**  
   ```bash
   python src/corpus/*.py --dataset data/bpm_texts.json
   ```
4. **Run entity extraction**  
   ```bash
   python src/ner/*.py --model --input data
   ```
5. **Run relation classification**  
   ```bash
   python src/relation/*.py --model roberta_large --input data/relations.json
   ```
6. **Evaluate models**  
   ```bash
   python src/evaluation/*.py --task ner --model bilstm_crf
   python src/evaluation/*.py --task relation_classification --model roberta_large
   ```

## 📖 **Reference**  
This research was conducted as part of the Master's dissertation:  
📄 **Diogo de Santana Candido**  
📌 **Title:** Extraction of Entities and Business Process Constraints from Text  
🏫 **Institution:** Instituto Federal do Espírito Santo (IFES), 2024  

### 📄 **Citation**  
If you use this dataset or code, please cite our work:  
```
@article{candido2024bpm,
  author    = {Diogo de Santana Candido},
  title     = {Extraction of Entities and Business Process Constraints from Text},
  journal   = {ENIAC 2024 - Simpósio Nacional de Inteligência Artificial e Computacional},
  year      = {2024},
  url       = {https://sol.sbc.org.br/index.php/eniac/article/view/33860/33651}
}
```

🔗 **Full Paper Available at:** [SBC OpenLib - ENIAC 2024](https://sol.sbc.org.br/index.php/eniac/article/view/33860/33651)
 

## 📫 **Contact**  
For any questions or collaborations, feel free to reach out!  
✉️ Email: diogo.candido@senado.leg.br  
🔗 LinkedIn: www.linkedin.com/in/diogo-candido-a6440725  







