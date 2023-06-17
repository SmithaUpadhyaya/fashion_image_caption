from tqdm import tqdm
from PIL import Image
import pandas as pd
import requests
import json
import os

#### This is helper class for main script preprocess_images
# Script file is used to read the raw json data to extract the information of the image metadata and download image

data_columns = ['id', 'title', 'desc', 'detail', 'category', 'color', 'image_url', 'image_name', 'image_extension', 'image_size']
processed_data = os.path.join('data', 'processed_data.parquet')

def read_data(limit_cnt = 20000):

    """
    Read raw json file to extract images and there metadata
    limit_cnt: Number of records to read. Due to resource limitiation we shall read approx top 20k 
    """
    numbers_records = 0
    base_path = os.getcwd()
    raw_data_path = os.path.join(base_path, 'data', 'raw_data', 'meta_all_129927.json')
    #print(raw_data_path)

    if os.path.exists(raw_data_path):

        with open(raw_data_path, encoding = 'utf-8', mode = 'r') as json_file:

            data = json.load(json_file)
            read_records = len(data) 

            #We have aprox 123K datapoint and each datapoint have one or more images. We will consider only 1st image for each time.
            print(f'Total number of data points: {read_records}')  
            lst_data = []   

            # Id's fail to download: 108539 , 100997, 129048, 113989, 102099, 108269, 131069, 130847, 133452, 107294, 113970, 118093, 116030, 117730       
            #for i in tqdm (range (read_records), desc = "Loading..."): 
            for i in tqdm (range (0, limit_cnt), desc = "Loading..."):                
                
                row = data[i]
                #print(row.keys())
                #['id', 'images', 'title', 'description', 'detail_info', 'categoryid', 'category', 'attr', 'attrid']

                #Consider only those records which has images. Some records has more the one images. Have consider only first image
                
                if '0' in row['images'][0]:
                                    
                    id = row['id']

                    title = row['title']
                    desc = row['description']
                    detail = row['detail_info']
                    category = row['category']     
                    
                    #Sample image path https://n.nordstrommedia.com/id/sr3/21e7a67c-0a54-4d09-a4a4-6a0e0840540b.jpeg?crop=pad&pad_color=FFF&format=jpeg&trim=color&trimcolor=FFF&w=60&h=90
                    img = row['images'][0]
                    color = img['color']
                    image_url = img['0']
                    image_url = image_url.split('?')[0]
                    image_name = image_url.split('/')[-1]
                    image_extension = image_name.split('.')[1] #Case when we can have images with different extension. Need to handel them

                    image_found = False
                    #Method 1:  Use PIL.Image
                    """
                    img = Image.open(requests.get(image_url, stream = True).raw)                
                    img.save(os.path.join(download_image_path, image_name))
                    """ 

                    #Method 2: 
                    #response = requests.get(image_url) #Used when using method 1 to save the file
                    try:
                        
                        response = requests.get(image_url, stream = True)

                        #If successful, a Status Code of 200 is returned.
                        if response.status_code == 200:
                            
                            if not os.path.exists(os.path.join(base_path, 'data', 'images')):
                                os.makedirs(os.path.join(base_path, 'data', 'images'))

                            save_path = os.path.join(base_path, 'data', 'images', image_name)
                            
                            #If file exists the delete and create a new file.
                            if os.path.exists(save_path):
                                os.remove(save_path)

                            #Save the file
                            #Method 1: Save using open
                            #fp = open(save_path, 'wb')
                            #fp.write(response.content)
                            #fp.close()
                            #file_size = os.path.getsize(save_path)

                            #Method 2: Using Image. used this as need the size check, so this help's not to open the file again to get the byte of the file
                            img = Image.open(response.raw)
                        
                            image_size = str(img.width) + 'x' + str(img.height)  #Check if all the images are of same size WxH
                            file_size = len(img.fp.read())

                            #Check the file size. If it's zero byte delete file as we do not need it                        
                            if file_size == 0:

                                if os.path.exists(save_path):
                                    os.remove(save_path)
                                
                                print(f'id: {id}, with zero byte size.')

                            else:
                                
                                img.save(save_path)
                                image_found = True

                        if image_found == True:
                            lst_data.append([id, title, desc, detail, category, color, image_url, image_name, image_extension, image_size])

                    except:
                        print(f'Error occured at: {id}')

            #When we have data records, save the file. Avoid empty file getting saved            
            if len(lst_data) > 0: 
                
                #This dataset also act as maping for image and it's caption, and other image data
                df = pd.DataFrame(data = lst_data, columns = data_columns)  
                numbers_records = df.shape[0]
                df.to_parquet(os.path.join(base_path, processed_data))

    else:
        print(f'File does not exits: {raw_data_path}')

    return numbers_records









   