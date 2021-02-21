### Main Files

- `Data.py` - Top-k 추천 모델을 구현한 적은 처음이라 데이터를 가공하는데에 더 신경을 썼다. Explicit한 rating 정보를 Implicit feedback을 표현하는 바이너리 형태로 바꾸고 Dataframe을 sparse matrix로 바꾸는 작업이 주를 이룬다.

- `models/CDAE.py` - CDAE 모델 클래스가 있는 파일이다. 추후에 다른 모델을 만들 때도 BaseModel 구조를 만들고 함께 관리하여 패키지화 시키려고 한다.

- `utils/Evaluator.py` - 평가를 위한 클래스가 있는 파일이다. Top-k 추천 모델의 성능을 평가하는 여러 지표가 있는데 그 중 precision과 recall을 사용하였다.

   

### TODO

평가 결과가 예상보다 좋지 않았다. 개인적인 생각으로는 데이터셋을 가공하는 쪽에서 문제가 생긴 것 같아 이를 수정해야 한다.