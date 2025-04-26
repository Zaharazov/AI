# Отчёт по лабораторной работе
## Свёрточные нейронные сети

### Студент: 

| ФИО       | Роль в проекте                     | Оценка       |
|-----------|------------------------------------|--------------|
| Епифанов Е.В. | Выполнил ЛР |          |


> *Комментарии проверяющего*


## Часть I. Pet Faces

### 1. Постановка задачи  
**Датасет**: Faces (13 пород кошек + 23 породы собак).  
**Задачи**:  
1. Определение кошки или собаки (бинарная классификация)  
2. Определение породы кошки или собаки (мультиклассовая классификация)

---

### 2. Подготовка данных  

1. Скачал и распаковал архив:
   ```python 
   !wget -q http://www.soshnikov.com/permanent/data/petfaces.tar.gz
   !tar -xzf petfaces.tar.gz
   !rm petfaces.tar.gz 

2. Для мультиклассовой задачи: метки кодируются с помощью LabelEncoder (0–34)  


   Для бинарной задачи: создаются бинарные метки (0 — кошка, 1 — собака)


   Далее данные разделяются на обучающую и тестовую выборки с помощью StratifiedShuffleSplit

3. CNNClassifier:  
   Сверточный слой: Conv2d(3, 32, kernel_size=3, padding=1) → ReLU → MaxPool(2×2)


   Сверточный слой: Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → MaxPool(2×2)


   Сверточный слой: Conv2d(64, 128, kernel_size=3, padding=1) → ReLU → MaxPool(2×2) 


   Эпох: 20  


4. BinaryCNNClassifier
   Сверточный слой: Conv2d(3, 32, kernel_size=3, padding=1) → ReLU → MaxPool(2×2)


   Сверточный слой: Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → MaxPool(2×2)


   Сверточный слой: Conv2d(64, 128, kernel_size=3, padding=1) → ReLU → MaxPool(2×2)

   Эпох: 10
   
Бинарная точность: 0.9425  
Мультиклассовая точность: 0.5412  
Top-3: 0.7869  
Top-5: 0.8600  


## Часть II. Oxford Pets и Transfer Learing

### 1. Постановка задачи  
**Датасет**: Oxford-IIIT Pets  
35 классов пород (кошки + собаки)  
**Задачи**:  
  1. Обучить три классификатора пород: на основе VGG-16/19 и на основе ResNet.  
  2. Посчитать точность классификатора на тестовом датасете отдельно для каждого из классификаторов, для дальнейших действий выбрать сеть с лучшей точностью.  
  3. Посчитать точность двоичной классификации "кошки против собак" такой сетью на тестовом датасете. 
  4. Построить confusion matrix. 
  5. Посчитать top-3 и top-5 accuracy

---

### 2. Подготовка данных  
1. **Чтение данных**:

   Сначала скачивается датасет Oxford-IIIT Pets через torchvision.datasets.OxfordIIITPet — создаются train_dataset и test_dataset.

   Затем, отдельно:

   Из папки images/ собираются все файлы .jpg.

   Название породы берется из имени файла (до первого подчеркивания _).

   Создается DataFrame с двумя колонками:

   filename — имя файла изображения.

   label — имя породы (в нижнем регистре).

   Деление данных:

   train_df, val_df, test_df через train_test_split (в пропорции примерно 72%/8%/20%).
   
3. Генераторы:
```python
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=vgg_preprocess
)

val_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)
test_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)

train_generator = train_datagen.flow_from_dataframe(
    train_df, directory=base_dir, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), class_mode='sparse', batch_size=BATCH_SIZE
)

val_generator = val_datagen.flow_from_dataframe(
    val_df, directory=base_dir, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), class_mode='sparse', batch_size=BATCH_SIZE, shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df, directory=base_dir, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), class_mode='sparse', batch_size=BATCH_SIZE, shuffle=False
)
```
```python
resnet_train_gen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=resnet_preprocess
)

resnet_val_gen = ImageDataGenerator(preprocessing_function=resnet_preprocess)

resnet_train_generator = resnet_train_gen.flow_from_dataframe(
    train_df, directory=base_dir, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), class_mode='sparse', batch_size=BATCH_SIZE
)

resnet_val_generator = resnet_val_gen.flow_from_dataframe(
    val_df, directory=base_dir, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), class_mode='sparse', batch_size=BATCH_SIZE, shuffle=False
)
```
3. **Архитектура и Обучение**
```python
   from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_model(base_model_fn, input_shape=(224, 224, 3), num_classes=37, preprocess_fn=None, train_base=False):
    base_input = Input(shape=input_shape)
    x = preprocess_fn(base_input) if preprocess_fn else base_input

    base_model = base_model_fn(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = train_base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
```
```python
def train_model(model, train_gen, val_gen, model_name, epochs=10):
    checkpoint = ModelCheckpoint(f'{model_name}.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    earlystop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )

    return model, history
```

4. **Мультиклассовая оценка**
   VGG-16:    
   Accuracy: 0.5331529378890991
    
   Top-3 Accuracy: 0.800405953991881

   Top-5 Accuracy: 0.8903924221921515


   VGG-19:  
   Accuracy: 0.5622462630271912
   
   Top-3 accuracy: 0.8220568335588633
   
   Top-5 accuracy: 0.9012178619756428


   ResNet50:  
   Accuracy: 0.7543978095054626
   
   Top-3 accuracy: 0.9364005412719891
   
   Top-5 accuracy: 0.9682002706359946


6. **Бинарная классификация «кошки vs собаки»**
```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

cat_classes = [c.replace(' ', '_').lower() for c in [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair',
    'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll',
    'Russian Blue', 'Siamese', 'Sphynx'
]]

dog_classes = [d.replace(' ', '_').lower() for d in [
    'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
    'Boxer', 'Chihuahua', 'English Cocker Spaniel', 'English Setter',
    'German Shorthaired Pointer', 'Great Pyrenees', 'Havanese', 'Japanese Chin',
    'Keeshond', 'Leonberger', 'Miniature Pinscher', 'Newfoundland',
    'Pomeranian', 'Pug', 'Saint Bernard', 'Samoyed', 'Scottish Terrier',
    'Shiba Inu', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'
]]

image_dir = 'data/oxford-iiit-pet/images'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

data = []
for file in image_files:
    breed = file.rsplit('_', 1)[0].lower()
    if breed in cat_classes:
        label = 'cat'
    elif breed in dog_classes:
        label = 'dog'
    else:
        continue
    data.append({'filename': file, 'label': label})

df = pd.DataFrame(data)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32,
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

binary_model = Model(inputs=base_model.input, outputs=output)
binary_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

binary_model.fit(train_gen, validation_data=val_gen, epochs=5)

loss, accuracy = binary_model.evaluate(val_gen)
print(f"Validation Accuracy: {accuracy:.4f}")

val_gen.reset()
y_pred_probs = binary_model.predict(val_gen)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
disp.plot()
plt.show()
```  
Binary Accuracy ResNet: 0.6947  
