# country-of-origin-project
# AI기술을 활용한 GIS 기반의 농산물 원산지 표시 단속 추천 서비스
- **1등 (대상·경상남도지사상 수상)** : 한국데이터산업진흥원(K-data)과 과학기술정보통신부가 주최 및 주관하는 `2020 데이터 청년 캠퍼스`
- **한국데이터산업진흥원장상 수상** : **대학별 우승팀 간 결선**인 `2020 데이터 청년 캠퍼스 우수 프로젝트 발표`
- 공공기관인 **국립농산물품질관리원**과의 협력으로 진행했던 프로젝트
- 최종 결과물 : https://bigleader.herokuapp.com/
(사이트가 커서 접속 시간이 조금 걸립니다..)

# 머신러닝 코드
## Monthly_Machine_Learning.ipynb
<목차>

0. 필요한 패키지 임포트
1. 데이터 불러오기
2. 데이터 전처리
3. 모델에 넣을 데이터 정하기
4. 여러가지 Regressor 중 RMSE(Root Mean Square Error;평균 제곱근 오차)가 가장 낮은 모델 찾기
5. 모델링하여 결과값 만들기
6. 변수 중요도(Feature Importance) 그려서 저장하기
7. 월별 머신러닝 모델들의 성능(RMSE) 비교

- 코드 설명: 이전의 데이터 전처리를 통해 만든, 최종으로 정제된 데이터를 사용하여 머신러닝 모델에 적용할 input data를 만들고, 여러 가지 머신러닝 모델을 비교하여 월별로 가장 최적의 결과값을 보여주는 모델을 선택하여 실행하는 코드입니다. 추가적으로 해당 모델링에는 34개의 변수(적발건수, 행정동, 업체개수, 품목별 가격과 수량, 거리변수, 소득수준 등)가 사용되었고, 모델링, 변수중요도, 머신러닝 성능(RMSE) 비교 등의 모든 머신러닝 관련 코드가 포함되어 있습니다.

- 작동 전, 준비 사항:
  - numpy 1.19.2
  - pandas 1.1.1
    
    추가적으로, pandas를 통해 excel 파일을 열 때, "Missing optional dependency 'xlrd'. Install xlrd >= 1.0.0 for Excel support Use pip or conda to install xlrd."라는 오류가 발생할 수 있으므로, pip install xlrd을 해줘야 함
  - pickle : 4.0
  아나콘다가 깔려있다면, conda install pickle을 권장
  - sklearn : 0.23.2 (StackingRegressor 등의 ensemble 모델은 최신버전에서 가능)
  - lightgbm : 2.3.0
  - xgboost : 1.1.0
  
    https://www.lfd.uci.edu/~gohlke/pythonlibs/ 사이트에서 로컬 파이썬 버전과 본인 윈도우 환경에 맞게 xgboost whl 파일 다운로드 
    (ex. 파이썬 환경이 3.6이면 cp36, 3.7이면 cp37, 본인 윈도우 환경이 64bit이면 amd64 다운로드)
    다운로드 받은 파일이 위치한 곳에 cmd 경로를 설정한 뒤 pip install xgboost ~ .wml 입력하여 설치
    (https://lsjsj92.tistory.com/546 참고)


# 딥러닝 코드
## Deep_Learning_input_data.ipynb
<목차>

0. 필요한 패키지 임포트
1. 데이터 불러오기
2. 데이터 전처리
3. X값 만들기
4. Y값 만들기

- 코드 설명: 이전의 데이터 전처리를 통해 만든, 최종으로 정제된 데이터를 사용하여 딥러닝에 적용할 3차원(16235, 108, 36) input data를 만드는 코드입니다. 여기서 3차원 각각의 의미는 고유한 grid_id값 16235개, 2010년부터 2018년까지 총 개월 수인 108개월(2019년은 test data로 사용함), 분석에 사용한 변수의 개수 36개입니다. 결과적으로, 10년 간의 데이터를 그리드 지역 단위로 36개의 변수(적발건수, 행정동, 업체개수, 품목별 가격과 수량, 거리변수, 소득수준 등)로 나타내었는데, 이 변수들을 사용하여 만든 input data는 바로 딥러닝 모델에 적용됩니다.

- 작동 전, 준비 사항:
  - Python 3.7.4
  - numpy 1.19.2
  - pandas 1.1.1
  - pickle 4.0
  - matplotlib 3.3.1
  - seaborn 0.11.0
  - sklearn 0.23.2

## CNN
### CNN1D_original.py - 분류, 회귀
<목차>

1. 라이브러리 임포트
2. 데이터 로드
3. 전체데이터를 트레이닝셋, 테스트셋으로 분리
4. 모델 빌딩
5. 모델 설정
6. 모델 피팅
7. 모델 평가

- 여기서 분류는 해당 지역(그리드)별로 적발유무(0, 1)를 예측하는 것이고, 회귀는 해당 지역(그리드)별로 적발건수(continuous variable)를 예측하는 것입니다.

<사용된 데이터 설명>
- Deep_input_X_scaled_36.npy : Deep_Learning_input_data.ipynb에서 만든, 딥러닝 모델에 사용된 스케일링된 3차원 36개 변수 데이터 (shape : (16235, 108, 36))
- Deep_input_Y_MSE_scaled.npy : Deep_Learning_input_data.ipynb에서 만든, 딥러닝 모델에 사용된 스케일링된 라벨 데이터 (shape : (16235, 1))

### CNN1D_cont_wandb.py - 회귀(continuous variable), wandb용 
<목차>

1. 라이브러리 임포트
2. wandb 설정
3. 데이터 로드
4. 전체데이터를 트레이닝셋, 테스트셋으로 분리
5. 모델 빌딩
6. 모델 설정
7. 모델 피팅

#### Masking Layer란?
- CNN1D(CNN 1-Dimensional) 모델을 빌딩할 때 사용한 Masking layer란, 특정 값(여기서는 -1값)은 실제로 데이터가 없는 결측치이므로 모델에 "이 부분은 없는 값이므로 모델을 학습시킬 때 빈 값으로 인식해라"라고 지정해줄 수 있는 layer입니다.
- 이 Masking layer를 사용하기 위해서, Deep_Learning_input_data.ipynb에서 코드를 짤 때, 20100101부터 20191231까지 총 10년의 기간동안 실제 데이터가 있는 부분을 제외하고 모든 값을 -1로 설정하였습니다.

#### wandb란?
- autoML 중 하나인 Weights & Biases 라는 Tool(웹 플랫폼)
- wandb를 썼을 때의 장점 :
  * 학습에 사용되는 하이퍼 파라미터들을 저장
  * 학습 과정을 검색하고, 비교하고, 시각화해주는 기능
  * 학습 과정 실행과 함께 시스템 사용량 metrics를 분석해주는 기능
  * 다른 사람들과 공유하여 협력할 수 있는 기능
  * 실행 과정과 결과를 반복할 수 있는 기능(가중치와 사용된 하이퍼 파라미터값들을 자동으로 저장하기 때문)
  * parameter sweep을 사용할 수 있음 : 여기서 parameter sweep이란, 각 parameter마다 원하는 범위를 지정하면 그 안에서 자동으로 값을 추출하여 하이퍼 파라미터값을 사용함
  * 실험, 실행의 과정과 결과들을 평생 저장하여 사용할 수 있음

- wandb를 적용하여 실행하는 방법(영어로 된 설명은 '_wandb.py' 파일에 적어놓았습니다.) : 
  1. wandb에서 회원가입을 한다.
  2. 터미널에서 'wandb login'을 치고 로그인한다.
  3. 링크를 통해 받은 API key를 터미널에 붙여넣고 실행한다.
  4. wandb.init 코드 밑 부분을 실행한다.
  5. 링크를 클릭하여 project에 접속한다.
  6. wandb project page에서 sweep을 누른다.
  7. 하이퍼 파라미터의 구간을 원하는대로 설정하고, 프로그램의 이름을 파이썬 파일의 이름과 동일하게 설정해준다.
  8. sweep을 시작하고 command를 터미널에 복사한다.
  9. sweep을 일찍 끝내고 싶다면, 복사한 command 뒤에 '--count 100' 등으로 실행하고 싶은 횟수를 적고 실행한다.
 
- (주의) wandb에 로그인하고 이 python 파일을 실행시킨다고 해서 필자가 과거에 모델을 돌렸던 DB로 접속하는 게 아닙니다.
  
  
### CNN1D_binary_wandb.py - 분류, wandb용
- 목차는 CNN1D_cont_wandb.py 와 동일.

## LSTM
### LSTM_original.py - 분류, 회귀
- LSTM은 모델 안에 Forget gate, Input Gate, Update cell state, Output gate 등 메모리를 좀 더 오래 기억하기 위한 장치인 Cell state와 Gates 등이 있기 때문에 hidden node의 개수를 조금만 늘려도 파라미터 수가 엄청나게 늘어납니다. 더 나아가, 데이터의 크기에 비해 파라미터 수가 월등히 많을 경우, Overfitting의 문제가 발생할 수 있습니다. 또한, 사용한 변수 또한 36개나 되기 때문에, 노드를 하나 늘릴 때마다 늘어나는 파라미터 수 등을 전반적으로 고려하여 LSTM layer의 개수와 노드 개수를 적절히 조정하며 LSTM 모델을 빌딩하였습니다. 

<목차>

1. 라이브러리 임포트
2. 데이터 로드
3. 전체데이터를 트레이닝셋, 테스트셋으로 분리
4. 모델 빌딩
5. 모델 설정
6. 모델 피팅
7. 모델 평가

- 여기서 분류는 해당 지역(그리드)별로 적발유무(0, 1)를 예측하는 것이고, 회귀는 해당 지역(그리드)별로 적발건수(continuous variable)를 예측하는 것입니다.

<사용된 데이터 설명>
- CNN1D_original.py 에서 사용된 데이터 파일과 동일.

### LSTM_cont_wandb.py - 회귀(continuous variable), wandb용 
- 목차는 CNN1D_cont_wandb.py 와 동일.

### LSTM_binary_wandb.py - 분류, wandb용
- 목차는 CNN1D_cont_wandb.py 와 동일.

## CNN-LSTM ★
### CNN-LSTM_original.py - 분류, 회귀
- 프로젝트를 진행하며 결과값을 도출하는 데에 최종적으로 선택하여 사용한 CNN + LSTM 모델입니다. Input layer인 CNN Masking layer를 시작으로 CNN layer와 LSTM layer를 거친 후, Output layer를 설정하였습니다.

 <목차>
 
1. 라이브러리 임포트
2. 데이터 로드
3. 전체데이터를 트레이닝셋, 테스트셋으로 분리
4. 모델 빌딩
5. 모델 설정
6. 모델 피팅
7. 모델 평가

- 여기서 분류는 해당 지역(그리드)별로 적발유무(0, 1)를 예측하는 것이고, 회귀는 해당 지역(그리드)별로 적발건수(continuous variable)를 예측하는 것입니다.

<사용된 데이터 설명>
- CNN1D_original.py 에서 사용된 데이터 파일과 동일.

### CNN-LSTM_cont_wandb.py - 회귀(continuous variable), wandb용
- 목차는 CNN1D_cont_wandb.py 와 동일.

### CNN-LSTM_binary_wandb.py - 분류, wandb용
- 목차는 CNN1D_cont_wandb.py 와 동일.
