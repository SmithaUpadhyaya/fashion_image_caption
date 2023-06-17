import src.process_json_data as pj 
import argparse

"""
Script to download the image from imageurl and convert json file to .parquet file
"""

if __name__ == '__main__':

    ap = argparse.ArgumentParser()	
    ap.add_argument("-n", "--noofrecords", required = True, type=int, help = "Number of records to read from json file")
    args = vars(ap.parse_args())

    read_records = args["noofrecords"]   
    
    numbers_records = pj.read_data(read_records)

    if numbers_records == 0:

        print('No records to process found')

    else:
        
        print(f'Sucessfully processed: {numbers_records} of raw data and saved at "{pj.processed_data}" file')
