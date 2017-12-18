

from keras import backend as K
import sys
sys.path.append('/root/lidor/Models/')
import argparse
import Models , LoadBatches
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
#os.environ["CUDA_VISIBLE_DEVICES"]="3" 

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str , default ="weights/best/" )
parser.add_argument("--train_images", type = str  , default ="patched/img/")
parser.add_argument("--train_annotations", type = str, default = "patched/anon/" )
parser.add_argument("--n_classes", type=int , default = 4)
parser.add_argument("--input_height", type=int , default = 300  )
parser.add_argument("--input_width", type=int , default = 300 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "patchedValidation/img/")
parser.add_argument("--val_annotations", type = str , default = "patchedValidation/anon/")

parser.add_argument("--epochs", type = int, default = 250 )
parser.add_argument("--batch_size", type = int, default = 5 )
parser.add_argument("--val_batch_size", type = int, default = 5 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "fcn8")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

print("getting model")
m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['accuracy'])
print("compiled model")

if len( load_weights ) > 0:
	m.load_weights(load_weights)


print ("Model output shape" ,  m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

# print("imageSegmentationGenerator")
# X,Y  = pageLoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


# if validate:
# 	print("validate mode")
# 	print("input_height",input_height)
# 	print("input_width",input_width)
# 	X_val,Y_val  = pageLoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

# mcp=ModelCheckpoint( filepath='bestmodel.h5',monitor='val_loss', save_best_only=True, save_weights_only=False,verbose=1)
# early_stopping=EarlyStopping(monitor='val_loss', patience=20)
# m.fit(X, Y, validation_data=(X_val,Y_val), batch_size=train_batch_size, epochs=200,callbacks=[mcp,early_stopping])


G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                           input_height, input_width, output_height, output_width)

if validate:
    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)

if not validate:
    for ep in range(epochs):
        m.fit_generator(G, 11224, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))
else:
    for ep in range(epochs):
        m.fit_generator(G, 11224, validation_data=G2, validation_steps=1000, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))


'''
if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 4  , epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( epochs ):
		m.fit_generator( G , 2  , validation_data=G2 , validation_steps=200 ,  epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep )  )
		m.save( save_weights_path + ".model." + str( ep ) )
'''