# KB-ALBERT-Infofinder

1. KB AI Challenge
- KB국민은행, 제2회 Future Finance AI Challenge
- KB-ALBERT를 활용한 금융 자연어 서비스 분야 참가

2. Problem
- 글과 숫자가 혼용된 문서 내에서 행원들이 원하는 정보를 찾아주는 것을 목표
- 일반적인 문서와 다르게 숫자와 단위 정보를 정확히 추출할 수 있도록 목표

3. Model Architecture
- 문서 내 문장의 의미를 추출할 수 있는 분류 모델과 정보를 추출할 수 있는 추출 모델로 구분
- 금융 문서 내 숫자와 단위 정보를 추출할 수 있는 char-tokenizer 사용

![image](https://user-images.githubusercontent.com/37866322/102351712-3e985580-3fea-11eb-9f1b-2934ffb494cf.png)

4. Result
![image](https://user-images.githubusercontent.com/37866322/102352559-3e4c8a00-3feb-11eb-98c1-ba6738c1fb69.png)
