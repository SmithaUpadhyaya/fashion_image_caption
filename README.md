</br>

# <u>**Automate Fashion Image Captioning using BLIP-2**</u> # 
</br>
The fashion industry is worth trillions of dollars. The goal of any company/seller is to help customer tofind the right product from a huge corpus of products that they are searching for. So, when customer find the right product they are mostly going to add the item to their cart and which help in company revenue.</br>
Accurate and enchanting descriptions of clothes on shopping websites can help customers without fashion knowledge to better understand the features (attributes, style, functionality, etc.) of the items and increase online sales by enticing more customers. Also, most of the time when any customer visits shopping websites, they are looking for a certain style or type of clothes that wish to purchase, they search for the item by providing a description of the item and the system finds the relevant items that match the search query by computing the similarity score between the query and the item caption. In such use cases having an accurate description of the clothes is useful.</br>
Manually writing the descriptions is a non-trivial and highly expensive task. Thus, the automatic generation of descriptions is an urgent need and will help the seller (while uploading the product to recommend captions).
</br>
</br>
</br>

# **Problem Statement** #
</br>
Given the clothes image provide a short caption that describes the item. In general, in image captioning datasets (e.g., COCO, Fliker), the descriptions of fashion items have three unique features, which makes the automatic generation of captions a challenging task. First, fashion captioning needs to describe the attributes of an item, while image captioning generally narrates the objects and their relations in the image.</br>
e.g. image where the model is wearing a shirt, the general caption model describes such images as "male wearing a white shirt". This is incorrect since we want the model to describe the item. In this application, it is much more important to have a performant to caption the image than an interpretable model.
</br>
</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src = "images\challange_image_text.png"/>
</br>
</br>

# <u>**Dataset**</u> #
**FAshion CAptioning Dataset (FACAD)**, the fashion captioning dataset consisting of over 993K images. 

Properties of FACAD dataset:
1. Diverse fashion images of all four seasons, ages (kids and adults), categories (clothing, shoes, bag, accessories,
etc.), angles of a human body (front, back, side, etc.). 
2. It tackles the captioning problem for fashion items.
    * FACAD contains fine-grained descriptions of attributes of fashion-related items, while MS COCO narrates the objects and their relations in general images. FACAD has longer captions (21 words per sentence on average) compared with the 10.4 words per sentence of the MS COCO caption dataset
    * Expression style of FACAD is enchanting, while that of MS COCO is plain without rich expressions. e.g. words like "pearly", "so-simple yet so-chic", and 
"retro flair" are more attractive than the plain MS COCO descriptions, like "person in a dress".

</br>
<b><a href = "https://github.com/xuewyang/Fashion_Captioning"> Source of the dataset</a></b>
</br>
<b><i>Citation:</i></b>
<b>@inproceedings{</b></br>
&emsp; &emsp; XuewenECCV20Fashion,</br>
&emsp; &emsp; Author = {Xuewen Yang and Heming Zhang and Di Jin and Yingru Liu and Chi-Hao Wu and Jianchao Tan and Dongliang Xie and Jue Wang and Xin Wang},</br>
&emsp; &emsp; Title = {Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards},</br>
&emsp; &emsp; booktitle = {ECCV},</br>
&emsp; &emsp; Year = {2020}</br>
&emsp; }</br>
</br>
</br>
For this project, have only considered 20k images for the pre-trained model due to resource limitations. </br> 
When trained on the entire dataset would allow the model to see different patterns, design of the fashion item which help in the create a large vocabulary that will help in describing new item caption.</br>
For caption an item , have consider item description, color, and brand. Because when a user search for an item they usually either mention specific color or particular brand along with the style they are interested to buy. </br>
For cleaning caption did not apply to the stem as we want the caption with proper grammar words. If this was a classification problem we apply stemming, since in that case, predict output is either 1 or 0 whereas in our case, we want a proper word/sentence.
</br>
</br>
</br>

# **Solution: Using Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (BLIP-2)** #
</br>
BLIP2 is a recent powerful model by Salesforce that is capable of performing visual question answering as well as image captioning.
</br>
</br>

## <u>*Overview on BLIP-2*</u> ##

</br>
<b><a href = "https://arxiv.org/abs/2301.12597">BLIP2</a></b> introduces a lightweight module called Querying Transformer (Q-Former) that effectively enhances the vision-language model. Q-Former is a lightweight transformer that uses learnable query vectors to extract visual features from the frozen image encoder. </br>
It acts as an information bottleneck between the frozen image encoder and the frozen Large Language Model (LLM), where it feeds the most useful visual feature for the LLM to output the desired text. 
</br>
BLIP2 has mainly two different versions based on the pre-trained LLM model used:</br>

1. Open Pre-trained Transformer Language Models(opt-2.7b) by Meta. Pre-trained model weights in HuggingFace: <b><a href = "https://huggingface.co/Salesforce/blip2-opt-2.7b">Salesforce/blip2-opt-6.7b</a></b>
2. FlanT5 model by Google. Pre-trained model weights in HuggingFace: <b><a href = "https://huggingface.co/Salesforce/blip2-flan-t5-xl"> Salesforce/blip2-flan-t5-xl</a></b> or <b><a href = "https://huggingface.co/Salesforce/blip2-flan-t5-xxl">Salesforce/blip2-flan-t5-xxl</a></b>

</br>

In both these versions <i>Vision Encoder</i> for image extraction used was <b><a href = "https://huggingface.co/google/vit-large-patch16-224"> Vision Transformer (large-sized model)</a></b> by Google.
</br>
</br>

## <u>*Architecture of BLIP-2 model*</u> ##
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="images\BLIP-2.png"/>
</br>
</br>

## <u>*Solution*</u> ##

Fine-tune pre-trained model **BLIP2** (trained on Fliker dataset) with Fashion dataset using **Low Rank Adaptation (LoRA)** a **Parameter-efficient fine-tuning technique (PEFT)**
</br>

The original model <b><u><i><a href = "https://huggingface.co/Salesforce/blip2-opt-2.7b">Salesforce/blip2-opt-2.7b</a></i></u></b> size was too large. It was quite challenging to fit and fine-tune the model on the 16GB GPU.
</br>

So, for this project have downloaded the pre-trained model <b><u><i><a href = "https://huggingface.co/ybelkada/blip2-opt-2.7b-fp16-sharded">ybelkada/blip2-opt-2.7b-fp16-sharded</a></i></u></b> from HuggingFace. This model uses OPT-2.7b LLM model with reduced precision to float16.
</br>
</br>

# **Requirements** #
Refer requirements.txt 
</br>

# **Tech Stack Used** #

1) Python 3.8 </br>
2) HuggingFace `transformers` 🤗</br>
3) peft</br>
4) bitsandbytes</br>
5) Streamlit
6) HuggingFace Spaces

</br>

# **Metric** #

Most commonly used metric for image caption task, that is used to measuring the quality of an predict text based on reference texts are:

1. <b>B</b>i<b>l</b>ingual <b>E</b>valuation <b>U</b>nderstudy (<i>Bleu</i>) Score: a concept build on precision.

        Bleu = Number of correct predicted words / Number of total predicted words

2. <b>R</b>ecall-<b>O</b>riented <b>U</b>nderstudy for <b>G</b>isting <b>E</b>valuation (<i>ROUGE</i>) Score: a set of metrics, rather than just one. ROUGE metric return recall, precision and f1-score. In our project have use F1-Rouge score. It's concept build on recall.

        Recall-N-gram = Number of correct predicted n-grams / Number of total target N-grams

        Precision-N-gram = Number of correct predicted n-grams / Number of total predict N-grams

        F1-Score = 2* ((Recall-N-gram * Precision-N-gram) / (Recall-N-gram + Precision-N-gram))


Both these score are build on concept of N-gram. In n-gram the value of n, is group n words and these words will always be in order. For this project have consider the value on n = 2. </br>
Because caption of the fashion item are the attributes, it does not matter in which order the model predits those attributes words.

</br>

# **Results** #
</br>

<table style = "width:100%">
<tr style = "border-bottom:1px solid black">
<th>
Dataset
</th>
<th>
F1-Rouge@1
</th>
<th>
F1-Rouge@2
</th>
<th>
F1-RougeL@2
</th>
<th>
BlEU@1
</th>
<th>
BlEU@2
</th>
</tr>

<tr>
<th>Train</th>
<th>0.45</th>
<th>0.16</th>
<th>0.44</th>
<th>0.42</th>
<th>0.26</th>
</tr>

<tr>
<th>Valid</th>
<th>0.42</th>
<th>0.13</th>
<th>0.41</th>
<th>0.39</th>
<th>0.22</th>
</tr>

<tr>
<th>Test</th>
<th>0.45</th>
<th>0.13</th>
<th>0.45</th>
<th>0.41</th>
<th>0.23</th>
</tr>

</table>
</br>
</br>

# **Try it Out** #

Deployed the model on HuggingFace Space. You can check it out <b><u><i><a href = "https://huggingface.co/spaces/Upyaya/Fashion-Image-Captioning-using-BLIP-2">here</a></i></u></b> 

## **Demo** ##
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src = "images\demo.gif" />
