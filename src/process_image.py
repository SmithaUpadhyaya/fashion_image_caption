from multiprocessing import shared_memory
from tqdm import tqdm
from PIL import Image
import numpy as np
import h5py
import glob #Search a file with specific pattern or name
import cv2
import os


#### This is helper class for main script preprocess_images.py


IMAGE_BASE_FOLDER = "data\\images\\"
NUM_WORKERS = 1
SHARED_MEMORY_NAME_IMAGE_DATA = 'np_image_data_array'
SHARED_MEMORY_NAME_IMAGE_ID_DATA = 'np_image_id_array'
#IMAGE_ARRAY_SHAPE = 0
#IMAGE_ID_ARRAY_SHAPE = 0

IMAGE_SIZE = 1024
CHANNEL = 3

def create_shared_memory_nparray(number_of_records, shared_memory_created):

    ###### Shared memory for Image Data
    
    #Create shared memory by define the shape and size required
    image_array_shape = (number_of_records, IMAGE_SIZE, IMAGE_SIZE, CHANNEL)
    if not shared_memory_created: 

        print('[INFO]: Create shared memory resource for numpy image data.')       

        img_dst = np.ndarray(shape = image_array_shape, dtype = np.float16)
        d_size = img_dst.nbytes
        #d_size = int(np.prod(image_array_shape)) * int(np.dtype(np.float16).itemsize)  #np.dtype(np.float16).itemsize: 2 
        
        img_shm = shared_memory.SharedMemory(create = True, size = d_size, name = SHARED_MEMORY_NAME_IMAGE_DATA)

        print('[INFO]: Shared memory resource created sucessfully.')

    else:

        print(f'[INFO]: Shared memory {SHARED_MEMORY_NAME_IMAGE_DATA} already exsts. Access to created resource')
        img_shm = shared_memory.SharedMemory(name = SHARED_MEMORY_NAME_IMAGE_DATA)

    ## Create numpy array on shared memory buffer.
    img_dst = np.ndarray(shape = image_array_shape, dtype = np.float16, buffer = img_shm.buf)
    img_dst.fill(0) #init/reset with 0 value

    ###### Shared memory for Image Id Data

    #Create shared memory by define the shape and size required
    image_id_array_shape = (number_of_records, )
    if not shared_memory_created: 

        print('[INFO]: Create shared memory resource for int image id data.')       

        img_id_dst = np.ndarray(shape = image_id_array_shape, dtype = np.uint64)
        d_size = img_id_dst.nbytes
        #d_size = int(np.prod(image_id_array_shape)) * int(np.dtype(np.uint64).itemsize)   #np.dtype(np.uint64).itemsize:8   

        img_id_shm = shared_memory.SharedMemory(create = True, size = d_size, name = SHARED_MEMORY_NAME_IMAGE_ID_DATA)
        
        print('[INFO]: Shared memory resource created sucessfully.')

    else:

        print(f'[INFO]: Shared memory {SHARED_MEMORY_NAME_IMAGE_ID_DATA} already exsts. Access to created resource')
        img_id_shm = shared_memory.SharedMemory(name = SHARED_MEMORY_NAME_IMAGE_ID_DATA)


    # create numpy array on shared memory buffer. 
    img_id_dst = np.ndarray(shape = image_id_array_shape, dtype = np.uint64, buffer = img_id_shm.buf)
    img_id_dst.fill(0) #init with 0 value

    # Reason to return the shm(Shared Memory) object due to bug of Shared_Memory in Window OS. 
    # Windows OS that the file (memory mapped file which backs the shared memory) is immediately removed if there are no currently open handles. 
    # If you create the shm in a function scope and don't keep a reference, it will be garbage collected when the function returns and the variable goes out of scope. 
    # So calling function should store these varaible. Else will give error "The system cannot find the file specified:"" shared memory python
    # Refer link: https://stackoverflow.com/questions/74193377/filenotfounderror-when-passing-a-shared-memory-to-a-new-process
    return img_shm, img_id_shm

def release_shared_memory():

    print('[INFO]: Release shared memory resource for image data.')
    release_shared(SHARED_MEMORY_NAME_IMAGE_DATA)

    print('[INFO]: Release shared memory resource for image data.')
    release_shared(SHARED_MEMORY_NAME_IMAGE_ID_DATA)

def release_shared(name):

    try:

        shm = shared_memory.SharedMemory(name = name)
    
        shm.close()
        shm.unlink()  # Free and release the shared memory block

    except Exception as e:
        print(f'[ERROR]: Error when try to release the lock - {e}')

#def process_images(index, image_id, image_name):
def process_images(arg): #multi arg is passed as array when using pool.map

    index = arg[0] 
    image_id = arg[1]
    image_name = arg[2]
    number_of_records = arg[3] #Since share memory size is defined in main process. So need to pass that to all the parallel process
    image_array_shape = (number_of_records, IMAGE_SIZE, IMAGE_SIZE, CHANNEL)
    image_id_array_shape = (number_of_records, )

    #Process to convert the image to numpy array in parallel
    #index: index location of the shared memory where the data need to be updated
    #image_id: Image id for each image name
    #image_name: Name of the image file name, whoch need to be converted into narray

    image_path = os.path.join(IMAGE_BASE_FOLDER, image_name)
    img = open_image(image_path)

    #Open shared memory for the image data and assign value
    img_shm = shared_memory.SharedMemory(name = SHARED_MEMORY_NAME_IMAGE_DATA)
    img_dst = np.ndarray(shape = image_array_shape, dtype = np.float16, buffer = img_shm.buf)
    img_dst[index] = img


    #Open shared memory for the image id data and assign value
    img_id_shm = shared_memory.SharedMemory(name = SHARED_MEMORY_NAME_IMAGE_ID_DATA)
    img_id_dst = np.ndarray(shape = image_id_array_shape, dtype = np.uint64, buffer = img_id_shm.buf)
    img_id_dst[index] = image_id

def open_image(image_path):

    # Read Images    
    #image_path = os.path.join(image_base_folder, image_name)

    #Method 1: User OpenCV library. Finally used OpenCV since it was faster campare to PIL.
    img = cv2.imread(image_path)  

    #Method 2: Use PIL Library
    #img = Image.open(image_path) 

    #Resize the image
    #img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_CUBIC )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Incase of image open using PIL to get np array of the image. 
    #This is where PIL library got slow, convert image to numpy where as cv2 open image by default in np.array
    #img = np.array(img)

    #Normalize the image pixel
    img = img/255

    return img

def save_to_hdf(hdf5_filename, number_of_records):

    image_array_shape = (number_of_records, IMAGE_SIZE, IMAGE_SIZE, CHANNEL)
    image_id_array_shape = (number_of_records, )

    print('[INFO]: Saved shared memory array to hdf5 started.')

    with h5py.File(hdf5_filename, mode = 'w') as hdf5_file:

        #Dataset for narray of image data
        img_shm = shared_memory.SharedMemory(name = SHARED_MEMORY_NAME_IMAGE_DATA)
        img_dst = np.ndarray(shape = image_array_shape, dtype = np.float16, buffer = img_shm.buf)
        hdf5_file.create_dataset('np_image', image_array_shape, np.float16, compression = "gzip", data = img_dst)

        #Dataset for image_id data
        img_id_shm = shared_memory.SharedMemory(name = SHARED_MEMORY_NAME_IMAGE_ID_DATA)
        img_id_dst = np.ndarray(shape = image_id_array_shape, dtype = np.uint64, buffer = img_id_shm.buf)
        hdf5_file.create_dataset('id_image', image_id_array_shape, np.uint64, data = img_id_dst)

        del [img_dst, img_id_dst]

    print('[INFO]: Saved shared memory array to hdf5 completed.')

    return

def combine_multiple_hdf(file_pattern, search_in_folder):
    """
    When working with large valid(1902 records)/train(18k records) fail to allocate resource for shared memory in preprocess_image.py script. Gave "[WinError 1450] Insufficient system resources exist to complete the requested service"
    To over come the issue decided to split the records range, create seperate hdf5 files and then merge them into one file

    Help refe: https://nicolasshu.com/appending_to_dataset_h5py.html
    """

    print(f'[INFO]: Combine multiple hdf5 files. Paramaters: file_pattern - {file_pattern}, search_in_folder - {search_in_folder}')
    #name = 'validate_data'  name -> file_pattern
    #path = 'data\processed' path -> search_in_folder

    #merge_file_name = os.path.join(PROJECT_ROOT, path , 'validate_data.h5')
    merge_file_name = os.path.join(search_in_folder , file_pattern + '.h5')

    with h5py.File(merge_file_name, "a") as hdf_merge: #'a': R/W if exists, create otherwise

        # Create an empty dataset with numpy array image of size (1024, 1024, 3)
        dset_img = hdf_merge.create_dataset( name = "np_image", 
                                    shape = (0, 1024, 1024, 3), # 0 means that it is an empty dataset. 
                                    maxshape = (None, 1024, 1024, 3), # None means that this axis 0 dimension can be extended to whatever we wish.
                                    dtype = np.float16,
                                    compression = "gzip"
                                    )

        dset_imgid = hdf_merge.create_dataset(name = 'id_image', 
                                            shape = (0,),
                                            maxshape = (None,),
                                            dtype = np.uint64, 
                                            compression = "gzip"
                                            )

        

        files = glob.glob(os.path.join(search_in_folder , file_pattern + '_*.h5'))

        for i in tqdm(range(len(files))):

            curr_file = files[i]

            with h5py.File(curr_file, 'r') as hdf_read: #'r': Readonly

                N = hdf_read['np_image'].shape[0]
                dset_img.resize(dset_img.shape[0] + N, axis = 0)
                dset_img[-N:] = hdf_read['np_image']
                #print(dset_img.shape)

                N = hdf_read['id_image'].shape[0]
                dset_imgid.resize(dset_imgid.shape[0] + N, axis = 0)
                dset_imgid[-N:] = hdf_read['id_image']
                #print(dset_imgid.shape)

        #Remove multiple files 
        for i in tqdm(range(len(files))):
            os.remove(files[i])

    print(f'[INFO]: Combine multiple hdf5 files completed. Saved at: {merge_file_name}')

    with h5py.File(merge_file_name, "r") as hdf:

        print('[INFO]: Details of the files: ')
        print(f'[INFO]: Shape of id_image: {hdf["id_image"]}')
        print(f'[INFO]: Type of id_image: {type(hdf["id_image"])}')
        print('='*50)

        print(f'[INFO]: Shape of np_image: {hdf["np_image"]}')
        print(f'[INFO]: Type of np_image: {type(hdf["np_image"])}')
        print('='*50)

def convert_to_hdf(dbset, hdf5_filename, image_base_folder, name = 'train'):

    #This method took verg long time to process the images and stored.

    # Specify shape of the hdf5 (Total Number of records, (Image Size (WxH), image_channel))
    shape = (dbset.shape[0], IMAGE_SIZE, IMAGE_SIZE, CHANNEL)

    # Create hdf5 
    with h5py.File(hdf5_filename, mode = 'w') as hdf5_file:

        x_name = 'x_'+ name
        y_name = 'y_'+ name

        # For image data, dtype: used uint8 as image data range from 0 to 255. Since we normalize by divide by 255 dtype is np.float16
        # FYI: We can compute the max and min range np.iinfo(np.uint8).min or np.iinfo(np.uint8).max. For float np.finfo(np.float64).max or np.finfo(np.float64).min
        hdf5_file.create_dataset(x_name, shape, np.float16, compression = "gzip")  #np.uint8
        hdf5_file.create_dataset(y_name, (dbset.shape[0],), np.uint64) #For image id. Image_id are of dtype +ve int
        #hdf5_file.create_dataset(y_name, (dbset.shape[0],), str ) #For text data.  #Did not find code to stored text in hdf5. so decided to store id instead

        for i in tqdm(range(0, dbset.shape[0]), desc = "Loading..."): 

            #if i%200 == 0 and i > 1 :
            #    print(f"Processed: Done {i} of {dbset.shape[0]}")
        
            image_name = dbset.iloc[i]['image_name']
            ids = dbset.iloc[i]['id']
            #title = dbset.iloc[i]['title'] #Since text data did not work used id's instead

            # Read Images
            image_path = os.path.join(image_base_folder, image_name)
            img = open_image(image_path)

            hdf5_file[x_name][i, ...] = img
            hdf5_file[y_name][i] = int(ids) #Store the image_ids as interger
            #hdf5_file[y_name][i] = title 

    ####Sample Code that tried###
    ####Too slow for validation dataset###

    ##Convert train to hd5 file
    #TRAIN_DATA_HDF5_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_data.h5')
    #convert_to_hdf(X_train, TRAIN_DATA_HDF5_FILE, IMAGE_FOLDER, 'train')

    ##Convert valid to hd5 file
    #VALID_DATA_HDF5_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'validate_data.h5')
    #convert_to_hdf(X_valid, VALID_DATA_HDF5_FILE, IMAGE_FOLDER, 'valid')

    ##Convert test to hd5 file
    #TEST_DATA_HDF5_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_data.h5')
    #convert_to_hdf(X_test, TEST_DATA_HDF5_FILE, IMAGE_FOLDER, 'test')
    
    return





