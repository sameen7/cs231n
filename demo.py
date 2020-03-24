# coding=utf-8
import numpy as np
from data_utils import load_CIFAR10, get_CIFAR10_data
import matplotlib.pyplot as plt
from  knn import KNearestNeighbor
from linear_classifier import LinearSVM
from linear_classifier import Softmax
from neural_net import TwoLayerNet, FullyConnectedNet, ThreeLayerConvNet
from vis_utils import visualize_grid
from features import *
from solver import Solver
from scipy.misc import imread, imresize
from layers import *



# x_train,y_train,x_test,y_test=load_CIFAR10('/Users/huziyi/CIFAR10/cifar-10')

data = get_CIFAR10_data()

for k, v in data.iteritems():
  print '%s: ' % k, v.shape



########################## 部分数据集展示 #######################
# classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
# num_claesses=len(classes)
# samples_per_class=7
# for y ,cls in enumerate(classes):
#     idxs=np.flatnonzero(y_train==y)
#     idxs=np.random.choice(idxs,samples_per_class,replace=False)
#     for i ,idx in enumerate(idxs):
#         plt_idx=i*num_claesses+y+1
#         plt.subplot(samples_per_class,num_claesses,plt_idx)
#         plt.imshow(x_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i ==0:
#             plt.title(cls)
# plt.show()

################################## knn ####################################
# knn划分小数据集
# num_training=5000
# mask=range(num_training)
# x_train=x_train[mask]
# y_train=y_train[mask]
# num_test=500
# mask=range(num_test)
# x_test=x_test[mask]
# y_test=y_test[mask]
#
# x_train=np.reshape(x_train,(x_train.shape[0],-1))
# x_test=np.reshape(x_test,(x_test.shape[0],-1))
#
#
# #knn
# classifier=KNearestNeighbor()
# classifier.train(x_train,y_train)
# dists=classifier.cumpute_distances_two_loops(x_test)
# y_test_pred = classifier.predict_labels(dists, k=1)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print 'got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

########################### 数据处理 ###############################
#训练集：49000 验证集：1000 测试集：1000
# num_training=49000
# num_validation=1000
# num_test=1000
# num_dev=500
# mask=range(num_training,num_training+num_validation)
# x_val=x_train[mask]
# y_val=y_train[mask]
# mask=range(num_training)
# x_train=x_train[mask]
# y_train=y_train[mask]
# mask=np.random.choice(num_training,num_dev,replace=False)
# x_dev=x_train[mask]
# y_dev=y_train[mask]
# mask=range(num_test)
# x_test=x_test[mask]
# y_test=y_test[mask]


# ### 特征数据处理 ###
# num_color_bins = 10 # Number of bins in the color histogram
# feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
# X_train_feats = extract_features(x_train, feature_fns, verbose=True)
# X_val_feats = extract_features(x_val, feature_fns)
# X_test_feats = extract_features(x_test, feature_fns)
#
# # Preprocessing: Subtract the mean feature
# mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
# X_train_feats -= mean_feat
# X_val_feats -= mean_feat
# X_test_feats -= mean_feat
#
# # Preprocessing: Divide by standard deviation. This ensures that each feature
# # has roughly the same scale.
# std_feat = np.std(X_train_feats, axis=0, keepdims=True)
# X_train_feats /= std_feat
# X_val_feats /= std_feat
# X_test_feats /= std_feat
#
# # Preprocessing: Add a bias dimension
# X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
# X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
# X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])


# ### 图像数据处理 ###
# x_train=np.reshape(x_train,(x_train.shape[0],-1))
# x_test=np.reshape(x_test,(x_test.shape[0],-1))
#
# x_val=np.reshape(x_val,(x_val.shape[0],-1))
# x_dev=np.reshape(x_dev,(x_dev.shape[0],-1))
#
# # 归一化处理 每个特征减去均值中心化
# mean_image=np.mean(x_train,axis=0)
# x_train-=mean_image
# x_val-=mean_image
# x_test-=mean_image
# x_dev-=mean_image



############################ svm  softmax 线性分类器 ###########################

# #w+b x增加一个维度 数值为1 则wx相当于wx+b
# x_train=np.hstack([x_train,np.ones((x_train.shape[0],1))])
# x_val=np.hstack([x_val,np.ones((x_val.shape[0],1))])
# x_test=np.hstack([x_test,np.ones((x_test.shape[0],1))])
# x_dev=np.hstack([x_dev,np.ones((x_dev.shape[0],1))])
#
#
# #svm梯度更新
# svm=LinearSVM()
# loss_hist_svm=svm.train(x_train,y_train,learning_rate=1e-7,reg=5e4,num_iters=1500,verbose=True)
#
# #预测
# y_train_pred_svm=svm.predict(x_train)
# print('training accuracy: %f '% (np.mean(y_train==y_train_pred_svm)))
# y_val_pred_svm=svm.predict(x_val)
# print('validation accuracy: %f'% (np.mean(y_val==y_val_pred_svm)))
#
# #softmax梯度更新
# softmax=Softmax()
# loss_hist_softmax=softmax.train(x_train,y_train,learning_rate=1e-7,reg=5e4,num_iters=1500,verbose=True)
#
# #softmax预测
# y_train_pred_softmax=softmax.predict(x_train)
# print('training accuracy: %f '% (np.mean(y_train==y_train_pred_softmax)))
# y_val_pred_softmax=svm.predict(x_val)
# print('validation accuracy: %f'% (np.mean(y_val==y_val_pred_softmax)))


# 权重可视化
# w=svm.W[:-1,:]
# w=w.reshape(32,32,3,10)
# w_min,w_max=np.min(w),np.max(w)
# classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     wimg=255.0*(w[:,:,:,i].squeeze()-w_min)/(w_max-w_min)
#     plt.imshow(wimg.astype('uint8'))
#     plt.axis('off')
#     plt.title(classes[i])
#
# plt.show()




########################### 二层神经网络 ############################

# # 训练   X_train_feats X_val_feats :特征   x_train x_val x_test: 普通
# isFeature = 1
# if(isFeature):
#     xtrain = X_train_feats
#     xval = X_val_feats
#     xtest = X_test_feats
#     input_size = X_train_feats.shape[1]
#     hidden_size = 500
#     reg = 0.001
#     lr = 0.5
# else:
#     xtrain = x_train
#     xval = x_val
#     xtest = x_test
#     input_size = 32*32*3
#     hidden_size = 75
#     reg = 0.75
#     lr = 1e-3
#
# # input_size=X_train_feats.shape[1] #32*32*3
# # hidden_size=500 #75
# num_classes=10
# net=TwoLayerNet(input_size,hidden_size,num_classes)
# stats=net.train(xtrain,y_train,xval,y_val,num_iters=1500,batch_size=200,learning_rate=lr,learning_rate_decay=0.95,reg=reg,verbose=True)
# train_acc = (net.predict(xtrain)==y_train).mean()
# print('train accuracy:',train_acc)
# val_acc=(net.predict(xval)==y_val).mean()
# print('valiadation accuracy:',val_acc)
# test_acc = (net.predict(xtest)==y_test).mean()
# print('teat accuracy:',test_acc)
#
# # loss accuracy 可视化
# plt.subplot(211)
# plt.plot(stats['loss_history'])
# plt.title('loss history')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.subplot(212)
# plt.plot(stats['train_acc_history'],label='train')
# plt.plot(stats['val_acc_history'],label='val')
# plt.title('classification accuracy history')
# plt.xlabel('epoch')
# plt.ylabel('classification accuracy')
# plt.show()


# # 超参数调整
# input_size=input_size=32*32*3
# num_classes=10
# hidden_size=[75,100,125]
# results={}
# best_val_acc=0
# best_net=None
# learning_rates=np.array([0.7,0.8,0.9,1.0,1.1])*1e-3
# regularization_strengths=[0.75,1.0,1.25]
# print('hyper-parameters adjustment running')
# for hs in hidden_size:
#     for lr in learning_rates:
#         for reg in regularization_strengths:
#             net=TwoLayerNet(input_size,hs,num_classes)
#             stats=net.train(x_train,y_train,x_val,y_val,num_iters=1500,batch_size=200,learning_rate=lr,learning_rate_decay=0.95,reg=reg,verbose=False)
#             val_acc=(net.predict(x_val)==y_val).mean()
#             if val_acc>best_val_acc:
#                 best_val_acc=val_acc
#                 best_net=net
#             results[(hs,lr,reg)]=val_acc
#
# print('hyper-parameters adjustment finshed')
# for hs,lr,reg in sorted(results):
#     val_acc=results[(hs,lr,reg)]
#     print('hs %d lr %e reg %e val accuracy: %f'% (hs,lr,reg,val_acc))
# print('best validation accuracy achieved during cross_validation: %f'%best_val_acc)

# # 权重可视化
# def show_net_weights(net):
#     W1=net.params['W1']
#     W1=W1.reshape(32,32,3,-1).transpose(3,0,1,2)
#     plt.imshow(visualize_grid(W1,padding=3).astype('uint8'))
#     plt.axis('off')
#     plt.show()
#
# show_net_weights(net)


########################################## 模块化二层神经网络 ##########################################3
# data={}
# data={'X_train':x_train,'y_train':y_train,'X_val':x_val,'y_val':y_val,'X_test':x_test,'y_test':y_test}
# model=TwoLayerNet(reg=1e-1)
# solver=None
# solver=Solver(model,data,update_rule='sgd',optim_config={'learning_rate':1e-3},lr_decay=0.8,num_epochs=10,batch_size=100,print_every=100)
# solver.train()
# scores=model.loss(data['X_test'])
# y_pred=np.argmax(scores,axis=1)
# acc=np.mean(y_pred==data['y_test'])
# print('test acc: %f'%acc)
#
# # 可视化
# plt.subplot(2, 1, 1)
# plt.title('training loss')
# plt.plot(solver.loss_history,'o')
# plt.xlabel('iteration')
# plt.subplot(2,1,2)
# plt.title('accuracy')
# plt.plot(solver.train_acc_history,'-o',label='train')
# plt.plot(solver.val_acc_history,'-o',label='val')
# plt.plot([0.5]*len(solver.val_acc_history),'k--')
# plt.xlabel('epoch')
# plt.legend(loc='lower right')#图例位置
# plt.gcf().set_size_inches(15,12)
# plt.show()

################################### 多层神经网络 #############################################
# data={}
# data={'X_train':x_train,'y_train':y_train,'X_val':x_val,'y_val':y_val,'X_test':x_test,'y_test':y_test}
# best_model=None
# X_val=data['X_val']
# y_val=data['y_val']
# X_test=data['X_test']
# y_test=data['y_test']
# learning_rate=3.1e-4
# weight_scale=2.5e-2#1e-5
# model=FullyConnectedNet([600, 500, 400, 300, 200, 100],weight_scale=weight_scale, dtype=np.float64,dropout=0.25, use_batchnorm=True, reg=1e-2)
# solver=Solver(model, data,print_every=500, num_epochs=30, batch_size=100,update_rule='adam',optim_config={'learning_rate': learning_rate,},lr_decay=0.9)
# solver.train()
# scores=model.loss(data['X_test'])
# y_pred=np.argmax(scores, axis=1)
# acc=np.mean(y_pred==data['y_test'])
# print ('test acc: %f'%(acc))
# best_model=model
#
#
# # 可视化
# plt.subplot(2, 1, 1)
# plt.plot(solver.loss_history)
# plt.title('Loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.subplot(2, 1, 2)
# plt.plot(solver.train_acc_history, label='train')
# plt.plot(solver.val_acc_history, label='val')
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch')
# plt.ylabel('Clasification accuracy')
# plt.show()
# y_test_pred=np.argmax(best_model.loss(X_test), axis=1)
# y_val_pred=np.argmax(best_model.loss(X_val), axis=1)
# print ('Validation set accuracy: ', (y_val_pred==y_val).mean())
# print ('Test set accuracy: ', (y_test_pred==y_test).mean())


############################################ BN层 ##########################################
# data={}
# data={'X_train':x_train,'y_train':y_train,'X_val':x_val,'y_val':y_val,'X_test':x_test,'y_test':y_test}
# num_train=1000
# hidden_dims=[100,100,100,100,100]
# small_data={
#     'X_train':data['X_train'][:num_train],
#     'y_train':data['y_train'][:num_train],
#     'X_val':data['X_val'],
#     'y_val':data['y_val']
# }
# weight_scale=2e-2
# bn_model=FullyConnectedNet(hidden_dims,weight_scale=weight_scale,use_batchnorm=True)
# model=FullyConnectedNet(hidden_dims,weight_scale=weight_scale,use_batchnorm=False)
# bn_solver=Solver(bn_model,small_data,num_epochs=10,batch_size=50,update_rule='adam',
#                  optim_config={'learning_rate':1e-3},verbose=True,print_every=200)
# bn_solver.train()
# solver=Solver(model,small_data,num_epochs=10,batch_size=50,update_rule='adam',
#                  optim_config={'learning_rate':1e-3},verbose=True,print_every=200)
# solver.train()
# bn_scores=bn_model.loss(data['X_test'])
# bn_y_pred=np.argmax(bn_scores, axis=1)
# acc=np.mean(bn_y_pred==data['y_test'])
# print ('bn_test acc: %f'%(acc))
# scores=model.loss(data['X_test'])
# y_pred=np.argmax(scores, axis=1)
# acc=np.mean(y_pred==data['y_test'])
# print ('test acc: %f'%(acc))
#
# # 可视化
# plt.subplot(3, 1, 1)
# plt.title('Training loss')
# plt.xlabel('Iteration')
#
# plt.subplot(3, 1, 2)
# plt.title('Training accuracy')
# plt.xlabel('Epoch')
#
# plt.subplot(3, 1, 3)
# plt.title('Validation accuracy')
# plt.xlabel('Epoch')
#
# plt.subplot(3, 1, 1)
# plt.plot(solver.loss_history, 'o', label='baseline')
# plt.plot(bn_solver.loss_history, 'o', label='batchnorm')
#
# plt.subplot(3, 1, 2)
# plt.plot(solver.train_acc_history, '-o', label='baseline')
# plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')
#
# plt.subplot(3, 1, 3)
# plt.plot(solver.val_acc_history, '-o', label='baseline')
# plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')
#
# for i in [1, 2, 3]:
#     plt.subplot(3, 1, i)
#     plt.legend(loc='upper center', ncol=4)
# plt.gcf().set_size_inches(15, 15)
# plt.show()




########################################### CNN 图像灰度化 边缘检测 #################################################
# kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
# # kitten is wide, and puppy is already square
# print kitten.shape , puppy.shape
# d = kitten.shape[1] - kitten.shape[0]
# kitten_cropped = kitten[:, d/2:-d/2, :]
# print kitten_cropped.shape , puppy.shape
#
# img_size = 200   # Make this smaller if it runs too slow
# x = np.zeros((2, 3, img_size, img_size))
# x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
# x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))
#
# # Set up a convolutional weights holding 2 filters, each 3x3
# w = np.zeros((2, 3, 3, 3))
#
# # The first filter converts the image to grayscale.
# # Set up the red, green, and blue channels of the filter.
# w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
# w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
# w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
#
# # Second filter detects horizontal edges in the blue channel.
# w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#
# # Vector of biases. We don't need any bias for the grayscale
# # filter, but for the edge detection filter we want to add 128
# # to each output so that nothing is negative.
# b = np.array([0, 128])
#
# # Compute the result of convolving each input in x with each filter in w,
# # offsetting by b, and storing the results in out.
# out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})
#
# def imshow_noax(img, normalize=True):
#     """ Tiny helper to show images as uint8 and remove axis labels """
#     if normalize:
#         img_max, img_min = np.max(img), np.min(img)
#         img = 255.0 * (img - img_min) / (img_max - img_min)
#     plt.imshow(img.astype('uint8'))
#     plt.gca().axis('off')
#
# # Show the original images and the results of the conv operation
# plt.subplot(2, 3, 1)
# imshow_noax(puppy, normalize=False)
# plt.title('Original image')
# plt.subplot(2, 3, 2)
# imshow_noax(out[0, 0])
# plt.title('Grayscale')
# plt.subplot(2, 3, 3)
# imshow_noax(out[0, 1])
# plt.title('Edges')
# plt.subplot(2, 3, 4)
# imshow_noax(kitten_cropped, normalize=False)
# plt.subplot(2, 3, 5)
# imshow_noax(out[1, 0])
# plt.subplot(2, 3, 6)
# imshow_noax(out[1, 1])
# plt.show()




################################################### CNN ######################################################

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()
scores=model.loss(data['X_test'])
y_pred=np.argmax(scores, axis=1)
acc=np.mean(y_pred==data['y_test'])
print ('test acc: %f'%(acc))


# filters可视化

grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
















