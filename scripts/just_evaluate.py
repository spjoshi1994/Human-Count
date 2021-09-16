import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import autokeras as ak
import numpy as np
import binary_ops_1
from binary_ops_1 import *


ckpt_model = "./ak_model/split_hg/callback__model_loss_0.44_acc_0.90.h5"
train_dir = "/home/pkarhade/training_keras/apps_training/hand_gesture"
color_mode="grayscale"
IMG_SHAPE = 32
CLASS_NAMES=None
BATCH_SIZE = 50000
image_gen_train = ImageDataGenerator(rescale=1./128)

test_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=False,
                                                     color_mode=color_mode,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     classes=CLASS_NAMES,
                                                     )

X_test, y_test = next(test_data_gen)
print(X_test.shape,y_test.shape)
print("min max value of X_test {} , {}".format(np.amin(X_test),np.amax(X_test)))
#np.save('linux_np.npy',X_test)
cust_obj1 = { "bo": binary_ops_1,"binary_ops": binary_ops_1,"lin_8b_quant": lin_8b_quant,\
                                                                    "FixedDropout" : FixedDropout,\
                                                                   "MyInitializer": MyInitializer,\
                                                                   "MyRegularizer": MyRegularizer,\
                                                                   "MyConstraints": MyConstraints,\
                                                                   "CastToFloat32":CastToFloat32}

net_model = tf.keras.models.load_model(ckpt_model ,custom_objects=cust_obj1)
ind=0
for i in range(-10,0,1):
    print(i,net_model.get_layer(index=i).name)
    if net_model.get_layer(index=i).name=="dense":
        print(net_model.get_layer(index=i))
        ind = i
        print("Found the dense layer index {}".format(ind))
if ind==0:
    print("No dense layer index found. Please check the name of the dense layer in this script")
intr_model = tf.keras.Model(net_model.input,net_model.get_layer(index=ind).output)
intr_output = intr_model.predict(X_test)
print(intr_output.shape)

### https://github.com/keras-team/keras/issues/2495 

print(np.amin(intr_output),np.amax(intr_output))
for i in range(intr_output.shape[0]):
    if np.amax(intr_output[i])>= 32:
        print("Value more than 32 found!")
        print(i,np.amax(intr_output[i]))
    if np.amin(intr_output[i])<-32:
        print("Value smaller than -32 found!")
        print(i,np.amin(intr_output[i]))
#last_layer = net_model.get_layer(index=1)
#print(net_model.layers)
#for l in net_model.layers :
    #if l.name=="dense":
        #print(l.output)

#y_out = net_model.predict(X_test)

#print(type(y_out))
#print(y_out.shape)
#print(np.amin(y_out),np.amax(y_out))
