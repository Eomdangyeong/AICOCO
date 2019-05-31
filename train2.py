import tensorflow as tf
import random
import os
import numpy as np
from PIL import Image
import random
import time
   
def load_data(folder_name, label): # 데이터 불러오는 함수
   pic_names = os.listdir("./data/" + folder_name)
   temp = []

   for p in pic_names:
      img = Image.open("./data/" + folder_name + "/" + p) #괄호 안의 경로에 있는 이미지를 오픈.
      # + p 를 적은 이유는 data 폴더 안의 변수p 라는 이름을 가진 파일이라는 뜻을 나타내기 위함

      ####### 정규화 #######

      numerator = img - np.min(img)
      denominator = np.max(img)
      img = numerator / denominator + 1e-7 # 0값이 있으면 log 씌웠을 때 발산하니까 10^-7정도 작은 수를 넣어줌.

      ####### 정규화 #######
      
      img_array = np.expand_dims(np.array(img), axis=3) # 차원 맞춰줌.

      temp.append({label : img_array}) # 라벨 추가

   return temp

max_flower = 10 # 클래스 갯 수 

######### train 데이터  ######### 딕셔너리 형태
forsythia = load_data("forsythia",0)
buckwheat = load_data("buckwheat",1)
sunflower = load_data("sunflower",2)
cosmos = load_data("cosmos",3)
lily = load_data("lily",4)
roseofsharon = load_data("roseofsharon",5)
tulip = load_data("tulip",6)
cherryblossom = load_data("cherryblossom",7)
rose = load_data("rose",8)
korearose = load_data("korearose",9)

print("labeling finish")


x_data_list = forsythia + buckwheat + sunflower + cosmos + lily + roseofsharon + tulip + cherryblossom + rose + korearose

######### train 데이터  #########




######### test 데이터 ######### 딕셔너리 형태


######### test 데이터 #########





######### 셔플 #########

random.shuffle(x_data_list)


######### 셔플 #########


####################### 여기서부터 133줄 까지는 김성훈 교수 코드랑 완전 동일 ######################

x = tf.placeholder(tf.float32, [None, 28, 28, 1])

y_raw = tf.placeholder(tf.float32, [None, max_flower])

learning_rate = 0.001
epoch = 10
batch_size = 100

keep_prob = tf.placeholder(tf.float32) # 드랍아웃

#########################################################################################


#################### Layer 1 시작!!
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) #3 3 1 32 중 1은 색을 의미하니까 맞게 쓰도록// 3 3은 필터 사이즈 3x3
#    Conv     -> (?, 28, 28, 32)                                32는 필터 개수
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')     #conv2d 함수 패딩을 same으로 하면
                                                                   #넣는 이미지와 출력 이미지의 크기에 맞게 알아서 패딩해줌.
                                                                   #스트라이드 2x2로 하고 싶으면 1, 2, 2, 1로 하면 됨.
L1 = tf.nn.relu(L1) #relu 통과시킴
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME') #스트라이드가 2라서 100x100이 아웃풋이 되는 것.
L1 = tf.nn.dropout(L1, keep_prob=keep_prob) # 오버피팅을 막기위해 드랍아웃 사용

#################### --- 여기까지가 layer1 끝 ---

#################### layer2 시작!!
#ImgIn shape=(?, 100, 100, 32) 첫 번째 layer와 크기가 같아야 하기에 100x100
# 내가 인풋이 200x200 이었지만 스트라이드가 2라서 아웃풋이 100x100이 될 것이라 생각.

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # 32는 위에서 정한 값 그대로
                                                                # 필터 개수 64는 임의로 선정한 것
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],                      # 강의에서 대부분 max pooling 사용한다고 하였으니 나도 그대로 간다.
                    strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 50 * 50 * 64])                     # 이제 최종 layer에 넣어야 하는데
                                                                 # 지금 50x50 짜리가 64개 있는 거니까(100x100짜리를 스트라이드2 짤리로 돌렸으니까 50x50이 될 것.)
                                                                 # 50*50*64가 되는 것. -1은 n개를 뜻함

L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


W3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # 32는 위에서 정한 값 그대로
                                                                # 필터 개수 64는 임의로 선정한 것
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L3 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],                      # 강의에서 대부분 max pooling 사용한다고 하였으니 나도 그대로 간다.
                    strides=[1, 2, 2, 1], padding='SAME')
L3_flat = tf.reshape(L3, [-1, 50 * 50 * 64])                     # 이제 최종 layer에 넣어야 하는데
                                                                 # 지금 50x50 짜리가 64개 있는 거니까(100x100짜리를 스트라이드2 짤리로 돌렸으니까 50x50이 될 것.)
                                                                 # 50*50*64가 되는 것. -1은 n개를 뜻함

L3 = tf.nn.dropout(L3, keep_prob=keep_prob)


# Final !!

W4 = tf.get_variable("W4", shape=[50 * 50 * 64, max_flower],   # 50*50*64,1에서 1은 mnist에서는 10이었는데 그 이유가 숫자가 0~9까지 10개니까 10이라고 김성훈 교수가 말했음
                                                               # 지금 나의 경우에는 사진이 조원 수인 max_people수에 해당.
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([max_flower]))
logits = tf.matmul(L3_flat, W4) + b                            # 이게 hypothesis가 되는 것.

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=y_raw))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 아담이 제일 좋다고 교수가 말했으니 나도 똑같이 AdamOptimizer로 간다.

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

############## 학습 가즈아!!!! ##############

print('마!! 학습 시작했다. 시간 좀 걸릴거다. 기다려라.')
      
start_time = time.time()     # 걸리는 시간 재는 용도

for e in range(epoch):
   avg_cost = 0

   ## TODO
   #print("training", e)

   for i in range(len(x_data_list)): # 이제 트레인 데이터를 넣을거임.

      temp_label = list(x_data_list[i].keys())[0]   # 딕셔너리의 key를 반환함.
      # gitaek이 딕셔너리의 list 형태라 하나하나씩 빼와야함. 그래서 for문을 썼음.
      # i 번째 데이터의 key값 즉 label을 list 형태로 가져와서, 0번째 원소를 temp_label에 저장
      # e.g. list(gitaek[i].keys()) 는 [0] , list(gitaek[i].keys())[0]는 0
      # e.g. list(gitaek[i].keys()) 는 [1] , list(gitaek[i].keys())[0]는 1
      
      y_tunned = np.zeros(max_flower)
      # max_people 길이의 0으로 채워진 list를 만듦

      y_tunned[temp_label] = 1


      y_tunned = np.expand_dims(y_tunned, axis=0)
      #print(y_tunned, temp_label)
      
      x_feed = list(x_data_list[i].values())[0]      

      x_feed = np.expand_dims(x_feed, axis=0)
      
      feed_dict = {x : x_feed, y_raw : y_tunned, keep_prob: 0.7} # 드랍아웃 0.7
      
      c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
      
      avg_cost += c / len(x_data_list)

      

   print('Epoch:', '%04d' % (e + 1), 'cost =', '{:.5f}'.format(avg_cost), c)

   ########## 저장 ##########
   
   saver = tf.train.Saver()
   saver.save(sess,"./model/"+'my_data' + str(e))     # model 이라는 폴더를 만들어서 거기 안으로 디렉토리 설정한 후 저장.
                                                      # 이렇게 하면 파일 이름이 my_data 에다가 에폭 수만큼 붙어서 my_data epoch이렇게 저장될 것.
   
   ########## 저장 ##########

print('학습끝!')

print("spend_time : ", time.time() - start_time)
# 1800장 기준으로 41.45분 걸림.
# 2730장 기준으로 46.51분 걸림

########## TEST~~~ ##########

