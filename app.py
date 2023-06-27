from transformers import Blip2ForConditionalGeneration
from transformers import Blip2Processor
from peft import PeftModel
import streamlit as st
from PIL import Image
#import torch
import os

preprocess_ckp = "Salesforce/blip2-opt-2.7b" #Checkpoint path used for perprocess image
base_model_ckp = "./model/blip2-opt-2.7b-fp16-sharded" #Base model checkpoint path
peft_model_ckp = "./model/blip2_peft" #PEFT model checkpoint path
sample_img_path = "./sample_images"

map_sampleid_name = {
                    'dress' : '00fe223d-9d1f-4bd3-a556-7ece9d28e6fb.jpeg',
                    'earrings': '0b3862ae-f89e-419c-bc1e-57418abd4180.jpeg',
                    'sweater': '0c21ba7b-ceb6-4136-94a4-1d4394499986.jpeg',
                    'sunglasses': '0e44ec10-e53b-473a-a77f-ac8828bb5e01.jpeg',
                    'shoe': '4cd37d6d-e7ea-4c6e-aab2-af700e480bc1.jpeg',
                    'hat': '69aeb517-c66c-47b8-af7d-bdf1fde57ed0.jpeg',
                    'heels':'447abc42-6ac7-4458-a514-bdcd570b1cd1.jpeg',
                    'socks': 'd188836c-b734-4031-98e5-423d5ff1239d.jpeg',
                    'tee': 'e2d8637a-5478-429d-a2a8-3d5859dbc64d.jpeg',
                    'bracelet': 'e78518ac-0f54-4483-a233-fad6511f0b86.jpeg'
                    }

def init_model(init_model_required):

    if init_model_required:

        #Preprocess input 
        processor = Blip2Processor.from_pretrained(preprocess_ckp)

        #Model   
        #Inferance on GPU device. Will give error in CPU system, as "load_in_8bit" is an setting of bitsandbytes library and only works for GPU
        #model = Blip2ForConditionalGeneration.from_pretrained(base_model_ckp, load_in_8bit = True, device_map = "auto") 

        #Inferance on CPU device
        model = Blip2ForConditionalGeneration.from_pretrained(base_model_ckp) 

        model = PeftModel.from_pretrained(model, peft_model_ckp)

        init_model_required = False

    return processor, model, init_model_required

#def main():

st.header("Automate Fashion Image Captioning using BLIP-2")    
st.caption("The fashion industry is worth trillions of dollars. The goal of any company/seller is to help customer tofind the right product from a huge corpus of products that they are searching for.")
st.caption("So, when customer find the right product they are mostly going to add the item to their cart and which help in company revenue.")
st.caption("Accurate and enchanting descriptions of clothes on shopping websites can help customers without fashion knowledge to better understand the features (attributes, style, functionality, etc.) of the items and increase online sales by enticing more customers.")
st.caption("Also, most of the time when any customer visits shopping websites, they are looking for a certain style or type of clothes that wish to purchase, they search for the item by providing a description of the item and the system finds the relevant items that match the search query by computing the similarity score between the query and the item caption.")
st.caption("Given the clothes image provide a short caption that describes the item. In general, in image captioning datasets (e.g., COCO, Fliker), the descriptions of fashion items have three unique features, which makes the automatic generation of captions a challenging task. First, fashion captioning needs to describe the attributes of an item, while image captioning generally narrates the objects and their relations in the image.")
st.caption("Solution: Used Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models [(BLIP-2)](https://huggingface.co/Salesforce/blip2-opt-2.7b) by Salesforce. The original model size was too large. It was quite challenging to fit and fine-tune the model on the 16GB GPU.")
st.caption("So, for this project have downloaded the pre-trained model [ybelkada/blip2-opt-2.7b-fp16-sharded](https://huggingface.co/ybelkada/blip2-opt-2.7b-fp16-sharded). This model uses OPT-2.7b LLM model with reduced precision to float16.")

st.caption("For more detail: [Github link](https://github.com/SmithaUpadhyaya/fashion_image_caption)")    #write

#Select few sample images for the catagory of cloths
with st.form("app", clear_on_submit = True):

    st.caption("Select image:")
    
    option = 'None'
    option = st.selectbox('From sample', ('None', 'dress', 'earrings', 'sweater', 'sunglasses', 'shoe', 'hat', 'heels', 'socks', 'tee', 'bracelet'), index = 0)
    
    st.text("Or")
    
    file_name = None
    file_name = st.file_uploader(label = "Upload an image", accept_multiple_files = False)


    btn_click = st.form_submit_button('Generate')
    st.caption("Application deployed on CPU basic with 16GB RAM")

    if btn_click:

        image = None
        if file_name is not None:     

            image = Image.open(file_name)

        elif option is not 'None': 

            file_name = os.path.join(sample_img_path, map_sampleid_name[option])
            image = Image.open(file_name)

        if image is not None:

            image_col, caption_text = st.columns(2)
            image_col.header("Image")
            caption_text.header("Generated Caption")
            image_col.image(image.resize((252,252)), use_column_width = True)
            caption_text.text("")

            if 'init_model_required' not in st.session_state:
                with st.spinner('Initializing model...'):

                    init_model_required = True
                    processor, model, init_model_required = init_model(init_model_required)

                    #Save session init model in session state
                    if 'init_model_required' not in st.session_state:
                        st.session_state.init_model_required = init_model_required
                        st.session_state.processor = processor
                        st.session_state.model = model
            else:
                processor = st.session_state.processor
                model = st.session_state.model

            with st.spinner('Generating Caption...'):            

                #Preprocess the image
                #Inferance on GPU. When used this on GPU will get errors like: "slow_conv2d_cpu" not implemented for 'Half'" , " Input type (float) and bias type (struct c10::Half)"
                #inputs = processor(images = image, return_tensors = "pt").to('cuda', torch.float16)

                #Inferance on CPU 
                inputs = processor(images = image, return_tensors = "pt")

                pixel_values = inputs.pixel_values

                #Predict the caption for the imahe
                generated_ids = model.generate(pixel_values = pixel_values, max_length = 10)
                generated_caption = processor.batch_decode(generated_ids, skip_special_tokens = True)[0]  

                #Output the predict text            
                caption_text.text(generated_caption) 
        

#if __name__ == "__main__":
#   main()