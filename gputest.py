import tensorflow as tf
from tensorflow.python.client import device_lib

# 현재 사용 가능한 장치 목록 출력
print("Devices available: ", device_lib.list_local_devices())

# GPU가 사용 가능한지 확인하는 간단한 연산 실행
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
    c = a + b
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(c))
