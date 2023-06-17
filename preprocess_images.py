from multiprocessing import cpu_count
from multiprocessing import Pool
import src.process_image as pi 
import pyarrow.parquet as pq
from tqdm import tqdm
import pandas as pd
import argparse
import math
import gc
import os

"""
########################## Script to Pre-process image files by converting them to numpy array and save numpy arrays to hdf5 format #####################
Working on limtted RAM resource, so convert 18k images into numpy array and them to hdf5 always caused "OutofMemory" error. Also convert 18k too long time.
Faced memory issue to load entire the .parquet file as well.

Solution took:
-> Load .parquet file: 
    Read data from parquet file in batch of N. N batch of records are sent to process.
-> Process image to numpy: 
    Create Shared Memory based on the batch size and the space need for an image. So shared memory size will be = N * image_size 
    Used Shared Memory that can now store N image at a time. Any more then that Shared Memory was not created and will give OutOfMemory error. 
    Image size was too large, so convert image to numpy will also be large. 
    While experiment found 300 batch size was ideal
We created shared memory only once and after each batch of N we clean up the shared memory to store next batch process data
-> Create pool of process, each process write to shared memory.
-> End of each batch N data we created a .hdf5. So by end of the script we will have (Num_Records/N) .hdf5 files 
-> Now we combine each .hdf5 into single .hdf5
"""
###################################################################################################

#Since we do not know the number of batch it will generted for any give dataset file. We shall created a generator to loop thrught and show a progress
def generator():
  while True:
    yield

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()	
    #ap.add_argument("-i", "--images", required = True, type = str, help = "base path to directory containing of images")
    ap.add_argument("-d", "--proc_data", required = False, type = str, default = '\data\\processed\\test_data.parquet' ,help = "filename with path, of dataset who's images need to be converted to numpy array")
    ap.add_argument("-p", "--procs", type = int, default = -1, help = "# of processes to spin up")
    #ap.add_argument("-r", "--datarng", type = str, default = '0:100', help = "range of records to process")
    ap.add_argument("-b", "--batch_size", type = int, default = 250, help = "chunk/batch size to read data from file")
    ap.add_argument("-a", "--action_type", type = int, default = 1, help = "action script will perform. 1-> convert image to numpy and combine files. 2-> combine the files. 3-> convert image to numpy")
    args = vars(ap.parse_args())

    # determine the number of concurrent processes to launch when
    # distributing the load across the system, then create the list
    # of process IDs
    NUM_WORKERS = args["procs"] if args["procs"] > 0 else cpu_count()

    #IMAGE_BASE_FOLDER = args["images"] #data\\images. Since we run in parallel process, We can not init this variable for each process. So defined it global
    DATA_TO_PROCESS = args["proc_data"] #data\\processed\\test_data.parquet       

    BATCH_SIZE = args["batch_size"] #Batch size

    ACTION_TYPE = args["action_type"]
    shared_memory_created = False

    try:
        
        if (ACTION_TYPE == 1) or (ACTION_TYPE == 3):

            print(f'[INFO]: Process to read the {DATA_TO_PROCESS}, convert images to numpy array and store. Started...')

            # When working with large valid(1902 records)/train(18k records) fail to allocate resource for shared memory. 
            # Gave "[WinError 1450] Insufficient system resources exist to complete the requested service"
            # To over come the issue decided to split the records range and then merge those records i.e pass/read the data in batch
            
            """ 
            # Code that read range from input argument, then read the entire dataset and then slide the range. 
            # But in this approach first entire file records is loaded, which take memory space and the script fail to allocate shared memory space.
            # Method 1: 
            RANGE = args["datarng"]
            start = int(RANGE.split(':')[0])
            end = int(RANGE.split(':')[1])

            data = pd.read_parquet(DATA_TO_PROCESS)

            if (start > 0) and (end > 0) and (start < end): 
                data = data[start:end]
            """
            #Method 2: Read the data in batch using parquet
            parquet_file = pq.ParquetFile(DATA_TO_PROCESS)
            number_of_records = parquet_file.metadata.num_rows
            start = 0
            end = 0             
            number_of_batch = math.ceil(number_of_records/BATCH_SIZE)

            with tqdm(total = number_of_batch) as pbar:

                for i in parquet_file.iter_batches(batch_size = BATCH_SIZE):

                    end +=BATCH_SIZE       
                    RANGE = str(start) + ':' + str(end)
                    data = i.to_pandas()
                    print(f'[INFO]: Process data range {RANGE} started.')

                    img_shm, img_id_shm = pi.create_shared_memory_nparray(data.shape[0], shared_memory_created)
                    shared_memory_created = True
                    print('[INFO]: Sucessfully created shared memory resource.')    

                    process_args = list(zip(range(0, data.shape[0]), data['id'], data['image_name'], [data.shape[0]] * data.shape[0]))

                    print('[INFO]: Starting Pool process...')
                    with Pool(NUM_WORKERS) as pror_pool:

                        #tqdm with pool not helpfull
                        #for _ in tqdm(pror_pool.map(pi.process_images, process_args), total = data.shape[0]):
                        #    pass
                        pror_pool.map(pi.process_images, process_args)
                        
                    print('[INFO]: Started saving data to hdf5 format...')
                    hdf5_filename, filename = os.path.split(DATA_TO_PROCESS)
                    
                    hdf5_filename = os.path.join(hdf5_filename, filename.split('.')[0] + '_' + RANGE.replace(':','_') + '.h5')
                    pi.save_to_hdf(hdf5_filename, data.shape[0])

                    print(f'[INFO]: Process data range {RANGE} completed.')
                    start = end
                    del [data]
                    pbar.update(1)
                        
            
            print('[INFO]: Process to convert images to numpy array and store in seperate files. Completed.')

        if (ACTION_TYPE == 1) or (ACTION_TYPE == 2):

            print('[INFO]: Combine multiple hdf5 files into one started...')

            path, name = os.path.split(DATA_TO_PROCESS)
            name = name.split('.')[0]
            pi.combine_multiple_hdf(name, path)
    

    except Exception as e:

        print(f'Error Occured: {e}')

    finally:       

        if shared_memory_created:
            pi.release_shared_memory()

        gc.collect()

    print('[INFO]: Script execution completed.')

############################################################333
#Sample code
#python preprocess_images.py -d data\processed\test_data.parquet
#python preprocess_images.py -d data\processed\validate_data.parquet -r 0:100
#python preprocess_images.py -d data\processed\validate_data.parquet
#python preprocess_images.py -d data\processed\validate_data.parquet -b 300
#python preprocess_images.py -d data\processed\train_data.parquet -b 300 -a 2

