import torch
import pandas as pd
from tqdm import tqdm
from transformers import BitsAndBytesConfig,AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import csv

df=pd.read_csv("test.csv") ###PROVIDE THE PATH TO THE TEST DATA(DOWNLOADED FROM README OF PARENT DIRECTORY) OVER WHICH INFERENCE SHOULD BE DONE###

#functions to preprocess the input and output 
def preprocess_input(text):
  if text == 'NaN': # Check if the text is empty
    return text
  max_tokens = 5000 #adjust according to max tokens you need from Input Case description
  if text==None:
     return text
  tokens = text.split(' ')
  num_tokens_to_extract = min(max_tokens, len(tokens))
  text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
  return text1

def preprocess_output(text):
  max_tokens = 500 #adjust according to max tokens you need from Official Reasoning
  tokens = text.split(' ')
  num_tokens_to_extract = min(max_tokens, len(tokens))
  text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
  return text1

# Preprocess the input cases
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    inp = row['Input']
    inpu = preprocess_input(inp)
    print(i)
    df.at[i, 'Input'] = inpu

# Set up model and tokenizer configurations
peft_model_dir = "Srinivastl/Nyaya" ###GIVE THE PATH TO YOUR LOCAL MODEL(CPT MODEL OBTAINED BY EXECUTING CODES IN CONTINUED PRE-TRAINED FOLDER)/HUGGINGFACE PATH FOR THAT MODEL,OVER WHICH INFERENCE NEEDS TO BE EVALUATED###
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
use_nested_quant = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load the model and tokenizer
#trained_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_dir, quantization_config=bnb_config)
#tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)
model_name = "microsoft/Phi-3-mini-128k-instruct" ###GIVE THE PATH TO YOUR LOCAL MODEL(CPT MODEL OBTAINED BY EXECUTING CODES IN CONTINUED PRE-TRAINED FOLDER)/HUGGINGFACE PATH FOR THAT MODEL,OVER WHICH INFERENCE NEEDS TO BE EVALUATED###
tokenizer = AutoTokenizer.from_pretrained(model_name)
trained_model=AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, quantization_config=bnb_config)
trained_model.eval()
# Open the CSV file in append mode
with open("Pedex_result_phi_q_1.csv", 'a', newline='', encoding='utf-8') as f: ###SPECIFY PATH TO RESULTING INFERED CSV FILE,WHERE AN EXTRA COLUMN NAMED "LLAMA2_PRED" WILL BE CREATED AS A REPRESENTATIVE OF MODEL PREDICTIONS ###
    writer = csv.writer(f)
    writer.writerow(list(df.columns) + ["phi3_pred_exp"])  # Write the header

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        case_pro = row["Input"]
        prompt = f""" ### Instructions:
        you have to act indian legal expert and dont give unwanted information.just support your decision with the help of the case proceeding.
        First, predict whether the appeal in case proceeding will be accepted (1) or not (0)\

        ### Input:
        case_proceeding: <{case_pro}>

        ### Response:
        """
        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True).input_ids.cuda()
        outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=1)
        output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
        writer.writerow(list(row) + [output])
        #print(output)
        