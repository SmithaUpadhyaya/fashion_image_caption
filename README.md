# **Automate Fashion Image Captioning using BLIP-2** # 
</br>
Fashion industry is worth trillions of dollars. Goal of any company/seller is to help customer to find the right product from huge corpus of products that they are searching for. So when customer find the right product that are mostly like going to add the item to cart and which help in company revenue.</br>
Accurate and enchanting descriptions of clothes on shopping websites can help customers without fashion knowledge to better understand the features
(attributes, style, functionality etc.) of the items and increase online sales by enticing more customers. Also most of the time when any customer vist on shopping websites, they are looking for certain style or type of cloths that wish to purchase, they search the item by providing description of item and in backend it find the relevent items that matchs the search query by computing the similarity score between the query and the item caption. In such use cases having accurate description of the clothes is usefull. </br>
Manually writing the descriptions is a non-trivial and highly expensive task. Thus, the automatic generation of descriptions is in urgent need and will help the seller (while uploading the product to recommend captions). 
</br>

# **Problem Statement** #
</br>
Given clothes image provide short caption that describes the item. 
General image captioning datasets (e.g. COCO, Fliker), the descriptions of fashion items have three unique features, which makes the automatic generation of captions a challenging task. First, fashion captioning needs to describe the attributes of a item, while image captioning generally narrates the objects and their relations in the image.</br>
e.g image where model is wearing a shirt, general caption model describe such images as "male wearning a white shirt". Which is incorrect since we want the model to describe the item.
In this application, it is much more important to have a performant to caption the image than an interpretable model.
</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="images\challange_image_text.png"/>
</br>

# **Solution: Using Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (BLIP-2)** #
</br>

## *Overview on BLIP-2* ##
</br>
BLIP introduce a lightweight module called Querying Transformer (Q-Former) that effectively enhances the vision-language model. Q-Former is a lightweight transformer that uses learnable query vectors to extract visual features from the frozen image encoder. It acts as an information bottleneck between the frozen image encoder and the frozen LLM, where it feeds the most useful visual feature for the LLM to output the desired text. 
</br>
    <b>Architecture of BLIP-2 model</b>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="images\BLIP-2.png"/>
</br>
</br>

## *Solution* #
</br>
Fine-tune pre-trained model BLIP-2(trained on Fliker dataset) with Fashion dataset.
</br>

# **Dataset** #
</br>
FAshion CAptioning Dataset (FACAD), the fashion captioning dataset consisting of over 993K images. 
When trained on entire dataset will train the model to see different pattern of the fashion item and the description of the item(which help in the create large vocabulary which will help in describing new item caption). But due to memory retriction consider only 20k images in this project. 
</br>
For caption of an item have included color and brand. Since when user search for an item they could mention either specific color or particular brand they are interested. Since each brand have they own style want to see if model is able to learn brand based on the style of the items.
</br>
Source of the dataset: https://github.com/xuewyang/Fashion_Captioning
</br>
<b><i>Citation:</i></b>
@inproceedings{
    XuewenECCV20Fashion,
    Author = {Xuewen Yang and Heming Zhang and Di Jin and Yingru Liu and Chi-Hao Wu and Jianchao Tan and Dongliang Xie and Jue Wang and Xin Wang},
    Title = {Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards},
    booktitle = {ECCV},
    Year = {2020}
    }
</br>
</br>
<i>Data Cleaning:</i>For cleaning image caption did not apply stemming  as we want the caption with proper grammar words. If this was a classification problem we apply stemming, since in that case predict output is either 1 or 0 where as in this case we want a proper word/sentence.

</br>

# **Requirements** #
</br>
    Refer requirements.txt 
</br>

# **Tech Stack Used** #
</br>
1) Python 3.8 </br>
2) Hugging Face Transfromer Library </br>
</br>

# **Metric** #
</br>

</br>

# **Results** #
</br>

</br>


