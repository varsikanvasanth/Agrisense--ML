"""Train AgriSense on ONLY 3 real PlantVillage crops: Tomato, Maize/Corn and Potato.

Dataset folders should be under data/plantvillage and use original PlantVillage class names,
e.g. Tomato___healthy, Corn_(maize)___Common_rust_, Potato___Late_blight.

Recommended CPU demo:
  python train_image_model.py --epochs 5 --max_images_per_class 300

For a fuller run:
  python train_image_model.py --epochs 10
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

SEED=42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/'data'/'plantvillage'
MODEL_DIR=ROOT/'models'; MODEL_DIR.mkdir(exist_ok=True)
ALLOWED_PREFIXES=('Tomato___','Corn_(maize)___','Potato___')
parser=argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default=5)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--image_size',type=int,default=224)
parser.add_argument('--max_images_per_class',type=int,default=0)
args=parser.parse_args()
if not DATA_DIR.exists(): raise SystemExit(f'Dataset not found: {DATA_DIR}')
class_dirs=sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith(ALLOWED_PREFIXES)])
if not class_dirs: raise SystemExit('No Tomato/Corn/Potato PlantVillage class folders found.')
items=[]
for c in class_dirs:
    files=sorted([p for p in c.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    if args.max_images_per_class: files=files[:args.max_images_per_class]
    items.extend((str(f),c.name) for f in files)
if len(items)<30: raise SystemExit('Too few images. Download the 3 crop dataset first.')
classes=sorted({y for _,y in items}); label_to_id={c:i for i,c in enumerate(classes)}
paths=np.array([x for x,_ in items]); y=np.array([label_to_id[y] for _,y in items])
# Stratified 80/10/10 split.
train_p,temp_p,train_y,temp_y=train_test_split(paths,y,test_size=0.20,stratify=y,random_state=SEED)
val_p,test_p,val_y,test_y=train_test_split(temp_p,temp_y,test_size=0.50,stratify=temp_y,random_state=SEED)

def make_ds(paths, labels, training=False):
    ds=tf.data.Dataset.from_tensor_slices((paths,labels))
    if training: ds=ds.shuffle(len(paths),seed=SEED)
    def load(path,label):
        img=tf.io.read_file(path); img=tf.image.decode_image(img,channels=3,expand_animations=False)
        img=tf.image.resize(img,(args.image_size,args.image_size)); img=tf.cast(img,tf.float32)/255.0
        return img,label
    ds=ds.map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
train_ds=make_ds(train_p,train_y,True); val_ds=make_ds(val_p,val_y); test_ds=make_ds(test_p,test_y)
augment=tf.keras.Sequential([layers.RandomFlip('horizontal'),layers.RandomRotation(.08),layers.RandomZoom(.1)])
base=MobileNetV2(input_shape=(args.image_size,args.image_size,3),include_top=False,weights='imagenet'); base.trainable=False
inp=layers.Input(shape=(args.image_size,args.image_size,3)); x=augment(inp); x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(.25)(x); out=layers.Dense(len(classes),activation='softmax')(x)
model=models.Model(inp,out); model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_ds,validation_data=val_ds,epochs=args.epochs)
loss,acc=model.evaluate(test_ds,verbose=1)
model.save(MODEL_DIR/'plant_disease_3crop.keras')
(MODEL_DIR/'class_names_3crop.json').write_text(json.dumps(classes,indent=2),encoding='utf-8')
metrics={'test_loss':float(loss),'test_accuracy':float(acc),'classes':classes,'num_classes':len(classes),'train_images':len(train_p),'validation_images':len(val_p),'test_images':len(test_p),'seed':SEED}
(MODEL_DIR/'3crop_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
print('\nREAL 3-CROP MODEL SAVED:',MODEL_DIR/'plant_disease_3crop.keras')
print('Classes:',len(classes)); print('Test accuracy:',round(acc*100,2),'%'); print('Test images:',len(test_p))
