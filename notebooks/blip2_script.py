#HuggingFace: Transformers Library
from transformers import get_cosine_schedule_with_warmup
from transformers import Blip2ForConditionalGeneration
from transformers import Blip2Processor
from transformers import Blip2Config
from transformers import set_seed

#HuggingFace: Dataset Library
from datasets import Dataset
from datasets import Image

#HuggingFace: Accelerator Library
from accelerate import find_executable_batch_size
from accelerate import DistributedType
from accelerate import Accelerator

#PIL Library
#from PIL import Image

#Pytorch Library
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

#Pandas and numpy
from pandas import read_parquet
from numpy import save,array

#Eval metric Library
from rouge_score import rouge_scorer, scoring

#General Library
from tqdm.auto import tqdm
import argparse
import os
import gc

MAX_GPU_BATCH_SIZE = 512

os.environ['WANDB_DISABLED'] = 'True'
    
#Script build with help of: https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
def get_dataloaders(accelerator, filetype, args, processor, batch_size):
    
    """
    Creates a set of `DataLoader`s for dataset, using "Salesforce/blip2-opt-2.7b" as the tokenizer.
    """

    #Read data
    if filetype == 'train':

        filepath = args["train_filepath"]
        data_sample = args["train_sample"]

    else:

        filepath = args["valid_filepath"]
        data_sample = args["valid_sample"]

    seed = args["seed"]
    #batch_size = args["batch_size"]
    image_base_path = args["image_base_path"]
    max_length = args["max_length"]
    padding = args["padding"]

    accelerator.print(f'Filepath: {filepath}')
    db_set = read_parquet(filepath)
    
    if data_sample != -1:
        db_set = db_set.sample(n = data_sample, random_state = seed)

    image_paths = list(image_base_path + '/' + db_set["image_name"])
    captions = list(db_set["caption"])
    
    del [db_set]
    
    db_set = Dataset.from_dict(
                                {
                                    "image": image_paths,
                                    "text": captions,
                                }
                            ).cast_column("image", Image())

    del [image_paths, captions]
    gc.collect()  
    
    #Dataloader
    def preprocess_fun(items):
                  
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:        
            pad_to_multiple_of = None
        
        
        #images = []
        #for image_path in item['image']:
        #    images.append(Image.open(image_path).resize(config.image_resize))
        #texts = item['text']
        
        images = []
        texts = []
        for item in items:
            images.append(item['image'])
            texts.append(item['text'])
            
        #encoding is a dict with keys 'pixel_values', 'input_ids', 'attention_mask'
        encoding = processor(
                            images = images, 
                            text = texts, 
                            max_length = max_length,
                            padding = padding, 
                            return_tensors = "pt",
                            pad_to_multiple_of = pad_to_multiple_of, 
                            #truncation = True,
                            )
        
        del [texts, images]
        
        #The output of the function passed to a batched map should be a dict with the structure {column_name → [list of values per element]}
        #This method take up too much of memory. As it calculate and store in memory. 
        #encoding = {k:list(v) for k,v in encoding.items()}
        #return encoding
        
        return encoding['pixel_values'], encoding['input_ids']

    dbloader = DataLoader(db_set, 
                          collate_fn = preprocess_fun, 
                          batch_size = batch_size,
                          shuffle = True,
                         )
    
    del [db_set]
    gc.collect() 
    
    return dbloader


def training_function(args):

    # Initialize accelerator
    accelerator = Accelerator() #cpu = args["cpu"], mixed_precision = args["mixed_precision"]

    # hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = args["lr"]
    num_epochs = args["num_epochs"]
    seed = args["seed"]
    gradient_accumulation_steps = args["gradient_accumulation_steps"]
    checkpoint = args["checkpoint"]
    pre_process_checkpoint = args["pre_process_checkpoint"]
    batch_size = args["batch_size"]

    set_seed(seed)

    # If the batch size is too big we use gradient accumulation
    #if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
    #    gradient_accumulation_steps = batch_size # MAX_GPU_BATCH_SIZE
    #    batch_size = MAX_GPU_BATCH_SIZE

     
    accelerator.print(f"Load input pre-processor with default: {pre_process_checkpoint}.")
    processor = Blip2Processor.from_pretrained(pre_process_checkpoint, torch_dtype = torch.float16)
    
    # Instantiate the model
    #devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #if we have pretrained weights for the model. Then load pretrained configuration but load your own weights. 
    if checkpoint.endswith('.pkl'):

        accelerator.print("Model initializing from pre-trained pkl model weights.") 
        
        accelerator.print(f"Load model weights from: {checkpoint}, pickle file.") 
        state_dict = torch.load(checkpoint) #, map_location='cpu')

        # Initializing a Blip2Config with Salesforce/blip2-opt-2.7b style configuration
        accelerator.print(f"Load model architecture with default: {pre_process_checkpoint}, configuration.") 
        configuration = Blip2Config().from_pretrained(pre_process_checkpoint)

        # Initializing a Blip2ForConditionalGeneration (with weights from 'pkl' file) from the Salesforce/blip2-opt-2.7b style configuration
        model = Blip2ForConditionalGeneration.from_pretrained(
                                                                pretrained_model_name_or_path = None, 
                                                                config = configuration, 
                                                                state_dict = state_dict,
                                                            )

        del [state_dict]
        
    else: #Load model and weight from pretrained model

        accelerator.print("Model initializing from hugging face checkpoint.")
        #model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, return_dict = True) 

        model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, 
                                                              device_map = "auto", 
                                                              torch_dtype = "auto",
                                                              offload_folder = "offload_model", 
                                                              offload_state_dict = True,
                                                            )
    
    accelerator.print("Model initialized.") 
    #model.to(accelerator.device) #model.to(devices)  

    @find_executable_batch_size(starting_batch_size = batch_size)
    def inner_training_loop(batch_size):

        nonlocal accelerator # Ensure they can be used in our context
        accelerator.free_memory() # Free all lingering references

        accelerator.print("Init data loaders.")  
        train_dbloader = get_dataloaders(accelerator, "train", args, processor, batch_size)
        valid_dbloader = get_dataloaders(accelerator, "valid", args, processor, batch_size)

        # Instantiate optimizer
        accelerator.print("Instantiate optimizer.") 
        optimizer = AdamW(model.parameters(), lr = lr)

        # Instantiate the learning rate scheduler    
        accelerator.print("Instantiate the learning rate scheduler.")
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, 
                                                    num_warmup_steps = 100, 
                                                    num_training_steps = (len(train_dbloader) * num_epochs) // gradient_accumulation_steps,
                                                    )


        # Prepare everything
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
        # prepare method.
        accelerator.print("Prepare everything accelerator.")
        model, optimizer, train_dbloader, valid_dbloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dbloader, valid_dbloader, lr_scheduler )

        # Eval metric method
        rouge_types = ["rouge1", "rouge2", "rougeL"]
        use_stemmer = False
        rouge_score_obj = rouge_scorer.RougeScorer(rouge_types = rouge_types, use_stemmer = use_stemmer)

        def compute_metrics(pred_text, labels_text):

            for ref, pred in zip(labels_text, pred_text):
                
                output_score = rouge_score_obj.score(prediction = pred.strip(), target = ref.strip())
                aggregator.add_scores(output_score)

            result = aggregator.aggregate()
            
            return {
                "rouge1_fmeasure": round(result['rouge1'].mid.fmeasure, 2),
                "rouge2_fmeasure": round(result['rouge2'].mid.fmeasure, 2),
                "rougeL_fmeasure": round(result['rougeL'].mid.fmeasure, 2),
            }

        # Now we train the model
        
        for epoch in range(num_epochs):

            accelerator.print('-'*55)
            accelerator.print('EPOCH {}/{}'.format(epoch + 1, num_epochs))
            accelerator.print('-'*55)  

            #Training model on train data
            model.train()
            step = 1
            for pixel_values, input_ids in tqdm(train_dbloader):
                
                accelerator.print('-- batch {} data received.'.format(step))
                
                #input_ids = input_ids.to(accelerator.device)  #devices
                #pixel_values = pixel_values.to(accelerator.device) #devices
                
                # Forward pass
                outputs = model(input_ids = input_ids, 
                                pixel_values = pixel_values, 
                                labels = input_ids) 

                del [input_ids,pixel_values]
                
                loss = outputs.loss # Compute loss function
                loss = loss / gradient_accumulation_steps # Normalize our loss            
                accelerator.backward(loss) #loss.backward() # Backward pass

                accelerator.print('-- batch {} | cur_loss = {:.6f}'.format(step, loss))

                # Compute gradient accumulation steps
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()  # Now we can do an optimizer step   
                    lr_scheduler.step() # scheduler step  
                    optimizer.zero_grad() # Reset gradients tensors
                    
                step += 1

            #Evaluate model train on eval data
            accelerator.print("Evaluate model train on eval data.")

            model.eval()
            aggregator = scoring.BootstrapAggregator()

            for pixel_values, input_ids in tqdm(valid_dbloader):
                
                #pixel_values = pixel_values.to(accelerator.device) #devices
                """
                with torch.no_grad():
                    outputs = model(input_ids = input_ids, 
                                    pixel_values = pixel_values, 
                                )
                predictions = outputs.logits.argmax(dim=-1)
                """
                predictions = model.generate(pixel_values)
                predictions = processor.batch_decode(predictions, skip_special_tokens = True)

                labels = processor.batch_decode(input_ids, skip_special_tokens = True)

                eval_metric = compute_metrics(predictions, labels) 

            # Use accelerator.print to print only on the main process.
            accelerator.print(f"epoch {epoch}:", eval_metric)

    inner_training_loop()

    accelerator.print("Saving model weights...")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
                                    args["output_folder"],
                                    is_main_process = accelerator.is_main_process,
                                    save_function = accelerator.save,
                                    state_dict = accelerator.get_state_dict(model),
                                )
    #model = accelerator.unwrap_model(model)
    #state_dict = model.state_dict()

    #checkpoint_save_dir = os.path.join(args["output_folder"], "blip2_model_state.pkl")
    #accelerator.save(state_dict, checkpoint_save_dir)
    accelerator.print("Saved model weight.")


def main():

    parser = argparse.ArgumentParser()

    #Modeling arguments
    parser.add_argument("--base_path", help = "Basepath for the image and data.")
    parser.add_argument("--output_dir", help = "Output where the saved model will be stored.")

    parser.add_argument("--train_samples", help = "Number of training sample.")
    parser.add_argument("--valid_samples", help = "Number of valid sample.")

    #Load for checkpoint
    parser.add_argument("--checkpoint", default = "", help = "Checkpoint for model.")

    #Hyperparamaters
    parser.add_argument("--lr", default = 0.2, help = "Learning Rate.")
    parser.add_argument("--num_epochs", default = 2, help = "Number of Epochs to train.")
    parser.add_argument("--batch_size", default = 128, help = "Batch size.") 
    parser.add_argument("--gradient_accumulation_steps", default = 3, help = "Gradient Accumulation steps.")        
         
    """
    parser.add_argument("--cpu", help="If passed, will train on the CPU.")
    parser.add_argument(
                        "--mixed_precision",
                        type = str,
                        default = "fp16",
                        choices = ["no", "fp16", "bf16", "fp8"],
                        help = "Whether to use mixed precision. Choose"
                        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                        "and an Nvidia Ampere GPU.",
        )
    """
    args = parser.parse_args()

    base_path = args.base_path
    output_folder = args.output_dir

    if len(args.checkpoint) == 0:
        checkpoint_path = "Salesforce/blip2-opt-2.7b" #Pre-trained checkpoint for BLIP-2 #https://huggingface.co/Salesforce/blip2-opt-2.7b
    else:
        checkpoint_path = str(args.checkpoint)


    #Define Hyperparam
    config = {
            "seed" : 44,
            
            "output_folder": output_folder,

            "train_filepath" : os.path.join(base_path, 'train_data_processed.parquet'),
            "train_sample" : int(args.train_samples), #10000,
            
            "valid_filepath" : os.path.join(base_path, 'validate_data_processed.parquet'),
            "valid_sample" : int(args.valid_samples),# 1000,
            
            "image_base_path" : os.path.join(base_path, 'images'),
            
            "checkpoint" : checkpoint_path,
            "pre_process_checkpoint": "Salesforce/blip2-opt-2.7b",
            
            #preprocessing setting
            "padding" : "longest", #"max_length",
            "max_length" : 16,
            #"mixed_precision" : args.mixed_precision,#"fp16",
            #"cpu": args.cpu, 
            "image_resize" : (256,256),
            
            #HyperParamater
            "lr" : float(args.lr), #2e-2,
            "num_epochs" : int(args.num_epochs), #2,
            "batch_size" : int(args.batch_size), #128,
            "gradient_accumulation_steps" : int(args.gradient_accumulation_steps), #3,
    }    
    
    training_function(config)

if __name__ == "__main__":
    
    main()


#Not used. Was for trial