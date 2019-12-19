import tensorflow as tf
import numpy as np
import PIL.Image as Image
from captcha.image import ImageCaptcha    # 先安装captcha，pip install captcha
import random
import string
# 屏蔽tensorflow log
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('istrain', 1, '是否训练,1、训练，2、测试,3、检测图片')
tf.compat.v1.flags.DEFINE_integer('iter_num', 200000, '迭代次数,默认10000')
tf.compat.v1.flags.DEFINE_integer('batch_size', 64, '批次大小,默认64')
tf.compat.v1.flags.DEFINE_float('lr_rate', 0.0001, '学习率,默认0.0001')
tf.compat.v1.flags.DEFINE_string('img_path', '1.jpg', '检测图片路径,默认1.jpg')

# 验证码字符集 数字+大写字母
characters = string.digits + string.ascii_uppercase
# characters = '0123456789'
# 验证码宽高
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 130
# 验证码字符数量
CAPTCHA_NUM = 4
CAPTCHA_SIZE = len(characters)
width, height, n_len, n_class = IMAGE_WIDTH, IMAGE_HEIGHT, CAPTCHA_NUM, CAPTCHA_SIZE
generator = ImageCaptcha(width=width, height=height)

tf.compat.v1.disable_eager_execution()
tf.compat.v1.flags.DEFINE_string('f', '', 'kernel')     # 注释此行如果你不在jupyter里运行


# 将彩色图像转化为灰色图像，识别验证码内文字与颜色无关，减为单通道
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return np.reshape(gray, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    else:
        return img

# 生成一批次验证码
def gen_image(batch_size=FLAGS.batch_size):
    random_str = [''.join([random.choice(characters) for j in range(CAPTCHA_NUM)]) for i in range(batch_size)]
    imgs = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=np.int)
    for i, str in enumerate(random_str):
        img = generator.generate_image(str)
        img = np.array(img)
        img = convert2gray(img)
        imgs[i] = img
    return imgs, random_str

# 保存一张验证码图片
def saveimg():
    s = ''.join([random.choice(characters) for j in range(CAPTCHA_NUM)])
    print(s)
    img = generator.generate_image(s)
    img.save(s + '.jpg')

# w_alpha参考别人的代码，不加这个效果不太好控制
def init_weight(shape, w_alpha=0.01):
    w = tf.Variable(w_alpha * tf.random.normal(shape=shape))
    return w

# b_alpha参考别人的代码，不加这个效果不太好控制
def init_bias(shape, b_alpha=0.1):
    b = tf.Variable(b_alpha * tf.random.normal(shape=shape))
    return b

# 转换验证码字符为onehot编码
def y_to_onehot(y):
    y_res = np.zeros([FLAGS.batch_size, CAPTCHA_NUM, CAPTCHA_SIZE])
    for m, i in enumerate(y):
        for n, j in enumerate(i):
            for k, l in enumerate(characters):
                if j == l:
                    y_res[m][n][k] = 1
    return y_res

# 解码预测值为验证码字符
def decode_y_pre(y_pre):
    y = []
    for i in y_pre:
        y.append(onehot_to_y(i))
    return y

# 解码预测值为验证码字符
def onehot_to_y(y_onehot):
    y = []
    for i in y_onehot:
        y.append(characters[i])
    return ''.join(y)

# 定义卷积神经网络，深层会更慢
def model_2d():
    # 卷积层1
    with tf.compat.v1.variable_scope("conv1"):
        w_conv1 = init_weight([3, 3, 1, 32])
        b_conv1 = init_bias([32])
        x_reshape = tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        x_relu1 = tf.nn.relu(tf.nn.conv2d(input=x_reshape, filters=w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        x_pool1 = tf.nn.max_pool2d(input=x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 卷积层2
    with tf.compat.v1.variable_scope("conv2"):
        w_conv2 = init_weight([3, 3, 32, 64])
        b_conv2 = init_bias([64])
        x_relu2 = tf.nn.relu(tf.nn.conv2d(input=x_pool1, filters=w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        x_pool2 = tf.nn.max_pool2d(input=x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 卷积层3
    with tf.compat.v1.variable_scope("conv3"):
        w_conv3 = init_weight([3, 3, 64, 64])
        b_conv3 = init_bias([64])
        x_relu3 = tf.nn.relu(tf.nn.conv2d(input=x_pool2, filters=w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
        x_pool3 = tf.nn.max_pool2d(input=x_relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    '''
    # 卷积层4
    with tf.variable_scope("conv4"):
        w_conv4 = init_weight([3, 3, 64, 128])
        b_conv4 = init_bias([128])
        x_relu4 = tf.nn.relu(tf.nn.conv2d(x_pool3, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)
        x_pool4 = tf.nn.max_pool(x_relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # [3, 8, 128]
    # 卷积层5
    with tf.variable_scope("conv5"):
        w_conv5 = init_weight([3, 3, 128, 128])
        b_conv5 = init_bias([128])
        x_relu5 = tf.nn.relu(tf.nn.conv2d(x_pool4, w_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)
        x_pool5 = tf.nn.max_pool(x_relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # [1, 4, 128]
     '''

    # 全连接层
    with tf.compat.v1.variable_scope("fc"):
        w_fc = init_weight([8 * 17 * 64, 1024])
        b_fc = init_bias([1024])
        x_pool5_reshape = tf.reshape(x_pool3, [-1, 8 * 17 * 64])
        x_fc = tf.nn.relu(tf.matmul(x_pool5_reshape, w_fc) + b_fc)
        x_fc = tf.nn.dropout(x_fc, 1 - (0.75))

    # 输出层
    with tf.compat.v1.variable_scope("output"):
        w_out = init_weight([1024, CAPTCHA_NUM * CAPTCHA_SIZE])
        b_out = init_bias([CAPTCHA_NUM * CAPTCHA_SIZE])
        y_pre = tf.matmul(x_fc, w_out) + b_out

    return y_pre


def main():
    y_pre = model_2d()
    # 定义损失函数
    with tf.compat.v1.variable_scope("loss"):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=y_pre)
        loss = tf.reduce_mean(input_tensor=losses)

    # 选择优化器
    with tf.compat.v1.variable_scope("Optimizer"):
        train_op = tf.compat.v1.train.AdamOptimizer(FLAGS.lr_rate).minimize(loss)
        
    # 定义精度计算方式
    with tf.compat.v1.variable_scope("accu"):
        eq_list = tf.equal(tf.math.argmax(input=tf.reshape(Y, [-1, CAPTCHA_NUM, CAPTCHA_SIZE]), axis=2),
                           tf.math.argmax(input=tf.reshape(y_pre, [-1, CAPTCHA_NUM, CAPTCHA_SIZE]), axis=2))
        logi_results = tf.cast(eq_list, tf.float32)
        pre_accuracy = tf.reduce_mean(input_tensor=logi_results)
        pre_sum = tf.reduce_sum(input_tensor=logi_results, axis=1)
        right_sum = tf.reduce_sum(input_tensor=tf.ones_like(logi_results), axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pre_sum, right_sum), tf.float32))

    # 可视化loss, pre_accuracy, accuracy
    tf.compat.v1.summary.scalar("loss", loss)
    tf.compat.v1.summary.scalar("pre_acc", pre_accuracy)
    tf.compat.v1.summary.scalar("acc", accuracy)

    merged = tf.compat.v1.summary.merge_all()
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    # 开启会话
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        filewriter = tf.compat.v1.summary.FileWriter("log", graph=sess.graph)
        # 训练
        if FLAGS.istrain == 1:
            for i in range(FLAGS.iter_num):
                x_batch, y_batch_raw = gen_image(FLAGS.batch_size)
                # print(x_batch[2], y_batch[2])
                # img = Image.fromarray(np.reshape(x_batch[2],[50, 130]).astype('uint8'))
                # img.show()
                x_batch = np.reshape(x_batch, [-1, IMAGE_HEIGHT * IMAGE_WIDTH * 1])
                y_batch = np.reshape(y_to_onehot(y_batch_raw), [-1, CAPTCHA_NUM * CAPTCHA_SIZE])
                sess.run(train_op, feed_dict={X: x_batch, Y: y_batch})
                summary = sess.run(merged, feed_dict={X: x_batch, Y: y_batch})
                filewriter.add_summary(summary, i)
                pre_accu = sess.run(pre_accuracy, feed_dict={X: x_batch, Y: y_batch})
                accu = sess.run(accuracy, feed_dict={X: x_batch, Y: y_batch})
                count = int(i/FLAGS.iter_num*33)
                print("\riter: [%s][%4d/%d]\tloss: %7f\tpre_accu:%.3f\taccu:%.3f" % ('='*count+'>'+' '*(33-count),
                    i, FLAGS.iter_num, sess.run(loss, feed_dict={X: x_batch, Y: y_batch}), pre_accu, accu), end='')

                # 每200次保存一份备份模型
                if i % 500 == 0:
                    saver.save(sess, 'backup/fc_model')
                '''
                # 当精度值高于0.96时停止训练
                if accu > 0.96:
                    break
                '''
            saver.save(sess, 'ckpt/fc_model')
        # 测试
        elif FLAGS.istrain == 2:
            x_batch, y_batch_raw = gen_image(FLAGS.batch_size)
            x_batch = np.reshape(x_batch, [-1, IMAGE_HEIGHT * IMAGE_WIDTH * 1])
            y_batch = np.reshape(y_to_onehot(y_batch_raw), [-1, CAPTCHA_NUM * CAPTCHA_SIZE])
            # saver.restore(sess, 'backup/fc_model')    #测试备份模型
            saver.restore(sess, 'ckpt/fc_model')
            y_test = sess.run(y_pre, feed_dict={X: x_batch, Y: y_batch})
            y_test = decode_y_pre(tf.math.argmax(input=tf.reshape(y_test, [-1, CAPTCHA_NUM, CAPTCHA_SIZE]), axis=2).eval())
            r = 0
            for i in range(FLAGS.batch_size):
                print('iter: %4d\tt:%s\tp:%s\t%s' % (i, y_batch_raw[i], y_test[i], str(y_batch_raw[i]==y_test[i])))
                if y_batch_raw[i] == y_test[i]:
                    r = r + 1
            print('正确率: %3f' % (r / FLAGS.batch_size))
        # 检测
        elif FLAGS.istrain == 3:
            # saver.restore(sess, 'backup/fc_model')
            saver.restore(sess, 'ckpt/fc_model')
            filepath = FLAGS.img_path
            img = np.array(Image.open(filepath))
            img = convert2gray(img)
            y_test = sess.run(y_pre, feed_dict={X: img.reshape([1, IMAGE_HEIGHT * IMAGE_WIDTH])})
            y_test = decode_y_pre(tf.math.argmax(input=tf.reshape(y_test, [-1, CAPTCHA_NUM, CAPTCHA_SIZE]), axis=2).eval())
            print(y_test)

        elif FLAGS.istrain == 4:
            saveimg()

if __name__ == '__main__':
    X = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])  # 60*160
    Y = tf.compat.v1.placeholder(tf.float32, [None, CAPTCHA_NUM * CAPTCHA_SIZE])  # 4*10
    main()
    # saveimg() #保存一张验证码图片
