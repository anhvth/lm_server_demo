import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from flask import Flask, request, jsonify


USERNAME = 'anhvth226'
peft_model_id = f'{USERNAME}/zalo_test'

config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
model = PeftModel.from_pretrained(model, peft_model_id)


app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    instruction = data['instruction']
    print('Rev instruction:', instruction)

    # Perform LM inference
    batch = tokenizer(instruction, return_tensors='pt')
    batch = {k: v.to('cpu') for k, v in batch.items()}

    with torch.no_grad():
        output_tokens = model.generate(**batch, max_new_tokens=100)
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
