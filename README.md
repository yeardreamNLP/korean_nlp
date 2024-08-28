# korean_nlp

## Directory Structure 
|-- README.md
`-- korea_llm
    |-- __pycache__
    |-- data
    |   |-- test_data.csv
    |   `-- train_data.csv
    |-- inference.py
    |-- inference.sh
    `-- vllm_inference.py

## Description
- `data/`: 학습데이터/평가데이터 csv 파일 
- `inference.py`: LLM 모델 추론하는 python script
- `vllm_inference.py`: LLM 모델 추론을 vllm 통해 실험
- `inference.sh`: inference.sh 추론 쉘 실행 스크립트

## 회고
Team07의 한국어 상식 문답기를 위한 좌충우돌 최적화 도전기입니다.
task를 진행하며, 최적화에 대해서 많이 고민하게 되었고, 에러와의 싸움이었습니다.
팀 구성원 : 이세민, 곽라흔, 최영우, 유창준