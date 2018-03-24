import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

sess = tf.Session()

# 붓꽃 데이터 로드
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target]) # setosa 종 대상 값 1로 변경
iris_2d = np.array([[x[2], x[3]] for x in iris.data]) # 붓꽃 데이터 중에서 꽃잎의 길이와 폭만 사용

# 플레이스홀더 및 변수 선언
batch_size = 20
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32) # 플레이스홀더 선언
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1])) # 변수 선언
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 선형 모델 정의 및 Sigmoid Cross Entropy 추가
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add) # x1 = x2*A+b

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target) # Sigmoid Cross Entropy 추가 z: my_output, y : y_target

my_opt = tf.train.GradientDescentOptimizer(0.05) # 경사하강법 최적화 방식 정의
train_step = my_opt.minimize(xentropy)

init = tf.global_variables_initializer() # 변수 초기화
sess.run(init)

for i in range(1000) :
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x]) # 꽃잎 길이 데이터
    rand_x2 = np.array([[x[1]] for x in rand_x]) # 꽃잎 폭 데이터
    rand_y = np.array([[y] for y in binary_target[rand_index]]) # 대상 값
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data : rand_x2, y_target:rand_y}) # x1, x2, y_target 데이터 투입
    if(i+1)%100 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A))+ ', b = ' + str(sess.run(b)))

[[slope]] = sess.run(A) # 모델 변수 추출 작업
[[intercept]] = sess.run(b)
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x: # 선형 모델 정의
    ablineValues.append(slope*i+intercept)
    
setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 1] # setosa 종일 때 x 좌표
setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 1] # setosa 종일 때 y 좌표
non_setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 0] # setosa 종이 아닐 때 x 좌표
non_setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 0] # setosa 종이 아닐 때 y 좌표

plt.plot(setosa_x, setosa_y, 'rx', ms = 10, mew = 2, label = 'setosa') # setosa 종일 때 좌표 위치
plt.plot(non_setosa_x, non_setosa_y, 'ro', label = 'Non-setosa') # setosa 종이 아닐 때 좌표 위치
plt.plot(x, ablineValues, 'b-') # 선형 모델 표시
plt.xlim([0.0, 2.7]) # x축 범위
plt.ylim([0.0, 7.1]) # y축 범위
plt.suptitle('Linear Separator For I.setosa', fontsize = 20) # 부제목 설정
plt.xlabel('Petal Length') # x 좌표 라벨
plt.ylabel('Petal Width') # y 좌표 라벨
plt.legend(loc='lower right') # 지표 위치
plt.show()
