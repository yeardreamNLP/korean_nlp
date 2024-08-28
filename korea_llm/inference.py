import os

from sympy.physics.units import temperature
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig

import argparse
from transformers import set_seed
from datasets import Dataset, DatasetDict
from triton.language import bfloat16

import re
import random
import gc


# Define your evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {
        'accuracy': (predictions == labels).float().mean().item()
    }


def generate_dict(df):
    prompt_q = "질문은 다음과 같습니다."
    prompt_ca = "당신은 다음 주어진 보기에서 답을 선택하셔야 합니다."
    prompt_ex = ("예를 들어 보기가, "
                 "0번: a,"
                 "1번: b,"
                 "2번: c,"
                 "3번: d"
                 "이렇게 주어졌는데, 정답이 0번이라면, 당신은, 정답은 0입니다 라고 말해야 합니다. "
                 "만약에 정답이 1번이라면, 당신은 정답이 1입니다 라고 말해야 합니다."
                 "만약에 정답이 2번이라면, 당신은 정답이 2입니다 라고 말해야 합니다."
                 "만약에 정답이 3번이라면, 당신은 정답이 3입니다 라고 말해야 합니다.")

    instruction_list = [[prompt_q + ' ' + prompt_ca + ' ' + prompt_ex] for _ in range(len(df))]
    question_list = df['문제']
    ans_0_list = df['answer_0']
    ans_1_list = df['answer_1']
    ans_2_list = df['answer_2']
    ans_3_list = df['answer_3']
    dataset_dict = {'instruction': instruction_list, 'question': question_list, 'answer 0': ans_0_list,
                    'answer 1': ans_1_list, 'answer 2': ans_2_list, 'answer 3': ans_3_list}
    dataset = Dataset.from_dict(dataset_dict)

    return dataset


def create_datasets(df, tokenizer, apply_chat_template=False):
    """
    Customized function for converting dataframes to huggingface datasets
    """

    def preprocess(samples):
        batch = []
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task.\n"
                "아래는 작업을 설명하는 명령어입니다.\n\n"
                "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
            ),
        }
        for instruction, question, an0, an1, an2, an3 in zip(samples["instruction"], samples["question"],
                                                             samples["answer 0"], samples["answer 1"],
                                                             samples["answer 2"], samples["answer 3"]):
            user_input = (question + '<|sep|>' + '\n' + '0번: ' + an0 + '\n' +
                          '1번: ' + an1 + '\n' + '2번: ' + an2 + '\n'
                          + '3번: ' + an3)
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}',
                                                                                                        user_input)
            batch.append(conversation)

        return {"content": batch}

    dataset = generate_dict(df)

    raw_datasets = DatasetDict()
    raw_datasets["test"] = dataset

    raw_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    test_data = raw_datasets["test"]
    print(
        f"Size of the test set: {len(test_data)}"
    )
    print(f"A sample of test dataset: {test_data[1]}")

    return test_data

def df_preprocess(df):

    df['선택지'] = df['선택지'].str.strip('[]').apply(lambda x: re.split(r",\s(?=(?:[^']*'[^']*')*[^']*$)", x))

    def split(df):
        num_answers = len(df['선택지'])
        result = {}
        result['answer_0'] = df['선택지'][0]
        result['answer_1'] = df['선택지'][1]
        result['answer_2'] = df['선택지'][2]
        result['answer_3'] = df['선택지'][3]
        return result

    results = df.apply(split, axis=1).tolist()

    # 결과 리스트를 DataFrame으로 변환합니다.
    result_df = pd.DataFrame(results)

    # 원래 DataFrame에 새로운 컬럼을 추가합니다.
    df = pd.concat([df, result_df], axis=1)
    df = df.drop('선택지', axis=1)

    return df


if __name__ == "__main__":

    # set base directory
    BASE_DIR = os.path.dirname(__file__)

    # Confirm which GPUs are visible
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--response_split', required=False, default='\nThe answer is', help='add response splitter')
    parser.add_argument('--model_path', required=False, default='', help='add pretrained model path')
    parser.add_argument('--model_name', required=False, default='', help='add model name if you run inference code only')

    args = parser.parse_args()

    # set seed for reproducibility
    set_seed(args.seed)

    # 언어 모델이 큰경우 4bit 사용하세요
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_quant_type='fp4',
        bnb_4bit_use_double_quant=True,
    )

    if args.model_path:
        model_path = os.path.join(BASE_DIR, args.model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, truncation=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.float16,
                                                     quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, truncation=True)

    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')
    df = df_preprocess(df)

    test_dataset = create_datasets(
        df,
        tokenizer,
        apply_chat_template=False
    )

    device = "cuda" if torch.cuda.is_available else "cpu"

    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    gc.collect()
    # inference
    with open(os.path.join(BASE_DIR, 'submission.csv'), 'w') as f:
        f.write('id,answer\n')

    for i, test_data in enumerate(tqdm(test_dataset)):
        text = test_data['content']
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Remove 'token_type_ids' if present
        model_inputs.pop('token_type_ids', None)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

        # create submission.csv
        answer = response.split('정답은')[-1]
        match = re.search(r'\d', answer)
        if match:
            answer = match.group()
            if answer not in ['0', '1', '2', '3']:
                answer = random.randint(0,3)
            else:
                answer = int(answer)
        else:
            print('에러 입니다! 랜덤한 값으로 설정 됩니다!')
            answer = random.randint(0,3)

        print(answer)
        with open(os.path.join(BASE_DIR, 'submission.csv'), 'a') as f:
            f.write(f'{i},{answer}\n')
    gc.collect()