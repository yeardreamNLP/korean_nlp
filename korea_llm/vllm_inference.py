
import os
import torch
import pandas as pd
import re
import argparse
import random
from transformers import set_seed, AutoTokenizer
from datasets import Dataset, DatasetDict
from vllm import LLM, SamplingParams
from tqdm import tqdm
import gc
import torch.multiprocessing as mp

from awq import AutoAWQForCausalLM
from accelerate import Accelerator, cpu_offload
import torch.nn as nn



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
                 "answer 0 : a,"
                 "answer 1 : b,"
                 "answer 2: c,"
                 "answer 3: d"
                 "이렇게 주어졌는데, 정답이 0이라면, 당신은, 정답은 0입니다 라고 말해야 합니다. "
                 "만약에 정답이 1이라면, 당신은 정답이 1입니다 라고 말해야 합니다."
                 "만약에 정답이 2이라면, 당신은 정답이 2입니다 라고 말해야 합니다."
                 "만약에 정답이 3이라면, 당신은 정답이 3입니다 라고 말해야 합니다.")

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
                                                  samples["answer 0"], samples["answer 1"], samples["answer 2"], samples["answer 3"]):
            user_input = (question + '<|sep|>' + '\n' + 'answer 0: ' + an0 + '\n' +
                          'answer 1: ' + an1 + '\n' + 'answer 2: ' + an2 + '\n'
                          + 'answer 3: ' + an3)
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
    mp.set_start_method('spawn')
    # set base directory
    BASE_DIR = os.path.dirname(__file__)

    # Confirm which GPUs are visible
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--model_name', default="Saxo/Linkbricks-Horizon-AI-Korean-Mixtral-sft-dpo-8x7b")
    # parser.add_argument('--enable_chunked_prefill', required=False, default='False', help='enable chunked prefill')

    args = parser.parse_args()

    # set seed for reproducibility
    set_seed(args.seed)

    gc.collect()

    # accelerator = Accelerator()
    #
    # # bitsandbytes 대신에 autoawq로 처리
    # # balanced로 균등하게
    # model = AutoAWQForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, attn_implementation='sdpa',
    #                                            **{'low_cpu_mem_usage': True})
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, truncation=True)

    # quant_config = {
    #     'zero_point': True,
    #     'q_group_size': 128,
    #     'w_bit': 4,
    #     'version': 'GEMM'
    # }

    # model.to('cuda')
    # model.quantize(tokenizer, quant_config=quant_config)
    # model = accelerator.prepare_model(model, evaluation_mode=True)

    df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')
    df =df_preprocess(df)

    test_dataset = create_datasets(
        df,
        None,
        apply_chat_template=False
    )
    gc.collect()

    # # inference
    # # parallel_size 지정 시 bitsandbytes 양자화 아직 지원안함(vllm)
    # # 4로 지정하면  The number of required GPUs exceeds the total number of available GPUs in the placement group.
    # # bf16 -> 8.0 compute capability 요구(최소 조건), 그런데 T4는 compute capability 7.5
    # # vllm 조건 -> 큰 모델로 돌리는 경우 Out of Memory 발생 -> Lora/양자화 적용한건 아니기 때문
    #

    # # inference
    # with open(os.path.join(BASE_DIR, 'submission.csv'), 'w') as f:
    #     f.write('id,answer\n')
    #
    # for i, test_data in enumerate(tqdm(test_dataset)):
    #     text = test_data['content']
    #     model_inputs = tokenizer([text], return_tensors="pt")
    #
    #     # Remove 'token_type_ids' if present
    #     model_inputs.pop('token_typeç_ids', None)
    #
    #     with torch.no_grad():
    #         generated_ids = model.generate(
    #             **model_inputs,
    #             max_new_tokens=9,
    #             eos_token_id=tokenizer.eos_token_id,
    #             pad_token_id=tokenizer.pad_token_id
    #         )
    #
    #     generated_ids = [
    #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]
    #
    #     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #     print(response)
    #
    #     # create submission.csv
    #     answer = response.split('정답은')[-1]
    #     match = re.search(r'\d', answer)
    #     if match:
    #         answer = match.group()
    #         if answer not in ['0', '1', '2', '3']:
    #             answer = random.randint(0, 3)
    #         else:
    #             answer = int(answer)
    #     else:
    #         print('에러 입니다! 랜덤한 값으로 설정 됩니다!')
    #         answer = random.randint(0, 3)
    #
    #     print(answer)
    #     with open(os.path.join(BASE_DIR, 'submission.csv'), 'a') as f:
    #         f.write(f'{i},{answer}\n')
    #
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')
    # df = df_preprocess(df)
    #
    # test_dataset = create_datasets(
    #     df,
    #     None,
    #     apply_chat_template=False
    # )
    gc.collect()
    with open(os.path.join(BASE_DIR, 'submission.csv'), 'w') as f:
        f.write('id,answer\n')

    prompts = [sample['content'] for sample in test_dataset]
    sampling_params = SamplingParams(temperature=0, top_p=1)

    model = LLM(args.model_name, tensor_parallel_size=4)
    outputs = model.generate(prompts, sampling_params)

    id = 0
    for output in outputs:
        prompt = output.prompt
        answer = output.outputs[0].text

        match = re.search(r'\d', answer)
        if match:
            answer = match.group()
            if answer not in ['0', '1', '2', '3']:
                answer = random.randint(0, 3)
            else:
                answer = int(answer)
        else:
            print('에러 입니다! 랜덤한 값으로 설정 됩니다!')
            answer = random.randint(0, 3)

        print(answer)
        with open(os.path.join(BASE_DIR, 'submission.csv'), 'a') as f:
            f.write(f'{i},{answer}\n')



    gc.collect()



