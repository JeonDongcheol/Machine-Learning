import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

my_tensor = tf.zeros([1, 20])

row_dim = 2
col_dim = 3

##1. 고정 텐서
zero_tsr = tf.zeros([row_dim, col_dim]) # 0 값으롤 채워진 텐서
ones_tsr = tf.ones([row_dim, col_dim]) #1 값으로 채워진 텐서
filled_tsr = tf.fill([row_dim, col_dim], 42) #동일한 상수 값으로 채워진 텐서
constant_tsr = tf.constant([1,2,3]) #기존 상수를 이용해 텐서를 생성하는 경우


##2. 형태가 비슷한 텐서
zeros_similar = tf.zeros_like(constant_tsr)
ones_similar = tf.ones_like(constant_tsr)

##3. 순열 텐서
linear_tsr = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # 결과 값 : [0.0, 0.5, 1.0]
integer_seq_tsr = tf.range(start=6, limit=15, delta=3) # 결과 값 : [6, 9, 12]

##4. 랜덤 텐서
randunif_tsr = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1)
random_tsr = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)

##변수 선언
my_var = tf.Variable(tf.zeros([2,3])) # Variable 메소드 : 변수 선언만 진행
sess = tf.Session() # 값을 계산하기 위한 세션 생성
initialize_op = tf.global_variables_initializer() # 변수의 초기화 진행
sess.run(initialize_op) # 계산 그래프의 실행

##플레이스 홀더
sess = tf.Session()
# 계산 그래프의 초기화
x = tf.placeholder(tf.float32, shape=[2,2])
# 플레이스홀더 x 정의
y = tf.identity(x)
# 플레이스홀더 x를 그대로 반환하는 x에 대한 항등 연산 y
x_vals = np.random.rand(2,2)
# 플레이스 홀더 x에 투입할 데이터 생성
sess.run(x, feed_dict={x : x_vals})
# 항등 연산 실행

sess = tf.Session()

identity_matrix = tf.diag([1.0, 1.0, 1.0]) # 대각 행렬
A = tf.truncated_normal([2,3]) # 절단정규분포로부터의 난수값을 반환.
# 생성된 값들은 평균으로부터 떨어진 버려지고 재선택된 두 개의 표준편차보다 큰 값을 제외한 지정된 평균과 표준 편차를 가진 정규 분포를 따른다.
B = tf.fill([2,3], 5.0) # 5.0 으로 채워진 행렬
print(sess.run(B))
C = tf.random_uniform([3,2]) # 3X2 Random으로 만들어진 행렬
D = tf.convert_to_tensor(np.array([[1.,2.,3.,],[-3.,-7.,-1.],[0.,5.,-2.]])) # 3X3 으로 각각 숫자가 대입된 행렬
A+B # 행렬 덧셈(뺄셈은 A-B를 대입하면 된다.)
tf.matmul(B, identity_matrix) # 행렬 곱셈 : 행렬을 전치할 것인지, 행렬이 희소 행렬인지를 인자를 통해 지정 가능
tf.transpose(C) #행렬 인자의 전치
tf.matrix_determinant(D) # 행렬식의 계산
tf.matrix_inverse(D) # 역행렬
tf.cholesky(identity_matrix) # 숄레스키 분해
tf.self_adjoint_eig(D) # 행렬 고유값 및 고유 벡터 : 첫 행 - 고유값 / 다음 행 - 고유 벡터


### 단일 연산이 들어간 계산 그래프
# 1. 계산 그래프의 데이터 생성
x_vals = np.array([1.,3.,5.,7.,9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)
# 2. 계산 그래프의 연산 생성
my_product = tf.multiply(x_data, m_const) # x_data에 데이터를 투입하고 상수 m_const와 multiply 연산을 수행
# 3. 계산 그래프에 데이터를 투입한 후 결과 출력
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data:x_val})) # 결과 : 3.0 9.0 15.0 21.0 27.0
### 여러 개의 연산이 들어간 계산 그래프
my_array = np.array([[1.,3.,5.,7.,9.],
                     [-2.,0.,2.,4.,6.],
                     [-6.,-3.,0.,3.,6]]) # 3X5 행렬 데이터 생성
x_vals = np.array([my_array, my_array+1]) # 행렬 my_array와 my_array 행렬에 각각 1을 더한 행렬 생성
x_data = tf.placeholder(tf.float32, shape=(3,5)) # 플레이스홀더 생성
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]]) # 행렬 곱셈 상수 생성1
m2 = tf.constant([[2.]]) # 행렬 곱셈 상수 생성2
a1 = tf.constant([[10.]]) # 행렬 덧셈 상수 생성
prod1 = tf.matmul(x_data,m1) # 행렬 곱셈1
prod2 = tf.matmul(prod1, m2) # 행렬 곱셈2
add1 = tf.add(prod2,a1) # 행렬 덧셈
# 계산 결과 출력[[102.] [66.] [58.]]-my_array  [[114.] [78.] [70.]]-my_array + 1 계산 결과
for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data:x_val}))

## 회귀에 대한 비용 함수
x_vals = tf.linspace(-1.,1.,500) # -1부터 1사이의 값 500개 생성 - 예측 값으로 가정
target = tf.constant(0.) # 대상 값
#### L2 Norm
l2_y_vals = tf.square(target - x_vals) # square() 함수는 제곱 값을 구하는 함수이다.
l2_y_out = sess.run(l2_y_vals)
#### L1 Norm
l1_y_vals = tf.abs(target - x_vals) # abs() 함수는 절댓값을 구하는 함수이다.
l1_y_out = sess.run(l1_y_vals)
#### Pseudo-Huber
delta1 = tf.constant(0.25) # delta 값을 0.25로 지정하였을 때
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.) # Pseudo-Huber 비용 함수
phuber1_y_out = sess.run(phuber1_y_vals)
delta2 = tf.constant(5.) # delta 값을 5.0으로 지정하였을 때
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.) # Pseudo-Huber 비용 함수
phuber2_y_out = sess.run(phuber2_y_vals)

### 회귀 비용 함수 그래프
x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label='L2 Norm') # L2 Norm
plt.plot(x_array, l1_y_out, 'r-', label='L1 Norm') # L1 Norm
plt.plot(x_array, phuber1_y_out, 'k-.', label='Pseudo-Huber 0.25') # Delta 값이 0.25인 Pseudo-Huber
plt.plot(x_array, phuber2_y_out, 'g-', label='Pseudo-Huber 5.0') # Delta 값이 5.0인 Pseudo-Huber
plt.ylim(-0.2, 0.6) # y 값 범위
plt.legend(loc='lower right', prop={'size': 8}) # 그래프 설명 위치 및 사이즈 지정
# * 그래프 위치 지정 : best, upper/center/lower, right/center/left
plt.show()

### 분류 예측 결과에 대한 비용 평가 함수
x_vals = tf.linspace(-3., 5., 500) # -3부터 5까지의 값 500개 생성 - 예측 값 재정의
target = tf.constant(1.) # 대상 값
targets = tf.fill([500,], 1.) # 대상 값 - 동일한(1.0) 으로 채워진 값

#### Hinge
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals)) # Hinge 비용 함수
hinge_y_out = sess.run(hinge_y_vals)

#### Cross Entropy
xentropy_y_vals = - (tf.multiply(target, tf.log(x_vals)) + tf.multiply((1. - target), tf.log(1. - x_vals))) # Cross Entropy 비용 함수(target : y, x_vals : z 값)
xentropy_y_out = sess.run(xentropy_y_vals)

#### 분류 비용 함수 그래프
x_array = sess.run(x_vals)
plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.ylim(-1.5, 3)
plt.legend(loc='lower left', prop={'size': 11})
plt.show()

#### Softmax Cross Entropy
unscaled_logits = tf.constant([[1., -3., 10.]]) # 정규화가 이루어지지 않은 출력값
target_dist = tf.constant([[0.1, 0.02, 0.88]]) # 실제 확률 분포
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=target_dist) # Softmax Cross Entropy
print(sess.run(softmax_xentropy)) # 결과값 : [1.1601256]


