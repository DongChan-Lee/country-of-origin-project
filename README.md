# country-of-origin-project
# AI기술을 활용한 GIS 기반의 농산물 원산지 표시 단속 추천 서비스

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

- 코드 설명: 이전의 데이터 전처리를 통해 만든, 최종으로 정제된 데이터를 사용하여 머신러닝 모델에 적용할 인풋 데이터를 만들고, 여러 가지 머신러닝 모델을 비교하여 월별로 가장 최적의 결과값을 보여주는 모델을 선택하여 실행하는 코드입니다. 추가적으로 해당 모델링에는 34개의 변수(적발건수, 행정동, 업체개수, 품목별 가격과 수량, 거리변수, 소득수준 등)가 사용되었고, 모델링, 변수중요도, 머신러닝 성능(RMSE) 비교 등의 모든 머신러닝 관련 코드가 포함되어 있습니다.
PPT와 실제 모델링에 사용된 예측값, 그 값을 산출한 머신러닝 모델은 데이터 폴더에 포함되어 있습니다. 

- 필요한 데이터: '그리드별 월단위 통합 탐색 (20200825_아파트 매매가 추가).pickle'

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

- 코드 설명: 이전의 데이터 전처리를 통해 만든, 최종으로 정제된 데이터를 사용하여 딥러닝에 적용할 3차원 인풋 데이터를 만드는 코드입니다. 10년 간의 데이터를 그리드 지역 단위로 36개의 변수(적발건수, 행정동, 업체개수, 품목별 가격과 수량, 거리변수, 소득수준 등)로 나타내었는데, 이 변수들을 사용하여 만든 인풋 데이터는 바로 딥러닝 모델에 적용됩니다.

- 필요한 데이터: '최종_데이터프레임.pickle'

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

<사용된 데이터 설명>
- Deep_input_X_scaled_36.npy : Deep_Learning_input_data.ipynb에서 만든, 딥러닝 모델에 사용된 3차원 36개 변수 데이터 (shape : (16235, 108, 36))
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


  # wandb 설명
  - (주의) wandb에 로그인하고 이 python 파일을 실행시킨다고 해서 필자가 과거에 모델을 돌렸던 DB로 접속하는 게 아님.
  - wandb란? Weights & Biases 라는 웹 플랫폼
