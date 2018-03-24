import tensorflow as tf
import numpy as np

sess = tf.Session()

### 역전파 예제
x_vals = np.random.normal(1, 0.1, 100) # 평균이 1, 표준편차 0.1인 데이터 100개 생성
y_vals = np.repeat(10., 100) # 대상 값 : 10.0인 데이터 100개
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32) # Placeholder 생성
A = tf.Variable(tf.random_normal(shape=[1])) # 변수 A의 생성

my_output = tf.multiply(x_data, A) # 난수 표본과 변수 A의 곱하기 연산
loss = tf.square(my_output - y_target) # L2 Norm 비용 함수

init = tf.global_variables_initializer() # 변수의 초기화
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.0001)
train_step = my_opt.minimize(loss) # 경사 하강법을 적용한 최적화 함수. 학습률 : 0.02


for i in range(100) : # 임의의 표본 데이터 x, y를 대상으로 100회 반복을 하고, 변수 값을 변경한다.
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data:rand_x, y_target: rand_y})
    if (i+1)%25 == 0: # 학습 횟수가 25씩 늘어날 때마다 변수 A 값의 변화와 비용 값을 출력한다.
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target:rand_y})))