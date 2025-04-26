# Отчёт по лабораторной работе
## Свёрточные нейронные сети

### Студенты: 

| ФИО       | Роль в проекте                     | Оценка       |
|-----------|------------------------------------|--------------|
| Епифанов Е.В. | Выполнил ЛР |          |


> *Комментарии проверяющего*

   
# Отчет по лабораторной работе

## Часть I. Классификация Faces

### 1. Постановка задачи  
**Датасет**: Faces (13 пород кошек + 23 породы собак, ~200 изображений на породу).  
**Задачи**:  
1. Бинарная классификация «кошка vs собака».  
2. Мультиклассовая классификация пород (36 классов).

---

### 2. Подготовка данных  

1. Скачал и распаковал:
   ```python 
   wget http://www.soshnikov.com/permanent/data/petfaces.tar.gz  
   tar xfz petfaces.tar.gz  

Построили DataFrame df с колонками:  
filename — имя файла  
class_id — индекс породы (0–35)  
species — 0 = кошка, 1 = собака  

Разбили на train/test (80/20)   
   ```python
   train_test_split(df, stratify=df["class_id"], test_size=0.2, random_state=42)
```
Собственная CNN:  
Flatten → Dense(256) → Dropout → выходной слой  
Мультикласс: выход — Dense(36, activation="softmax")  
Бинарка: выход — Dense(1, activation="sigmoid")  
Оптимизатор: Adam(lr=1e-3 → 1e-4)  
Эпох: 20  

Бинарная точность	0.96  
Мультикласс Accuracy	0.49  
Top-3	0.72  
Top-5	0.80  


## Часть II. Transfer Learning на Oxford-IIIT Pets

### 1. Постановка задачи  
**Датасет**: Oxford-IIIT Pets  
36 классов пород (кошки + собаки), ~200 изображений на класс  
**Задачи**:  
  1. Мультиклассовая классификация пород с помощью VGG-16, VGG-19, ResNet50  
  2. Сравнение моделей по Accuracy, Top-3, Top-5  
  3. Бинарная классификация «кошка vs собака» на лучшей модели  
  4. (Опционально) Visualize зоны внимания (Grad-CAM)  
  5. (Для «хорошо/отлично») Обучить автоэнкодер или GAN для генерации лиц животных

---

### 2. Подготовка данных  
1**Чтение аннотаций**:
   ```python
   data = []
   with open("annotations/trainval.txt") as f:
       for line in f:
           fn, cid, sp = line.strip().split()[:3]
           data.append((fn + ".jpg", int(cid)-1, int(sp)-1))
   df = pd.DataFrame(data, columns=["filename","class_id","species"])
```
Train/Test Split (80/20 по class_id):
```python
   train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["class_id"], random_state=42
)
```
ImageDataGenerator:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.0
)
test_datagen = ImageDataGenerator(rescale=1./255)
```
Генераторы:
```python
train_gen = train_datagen.flow_from_dataframe(
    train_df, directory="images",
    x_col="filename", y_col="class_id",
    target_size=(224,224), batch_size=32,
    class_mode="sparse"
)
test_gen = test_datagen.flow_from_dataframe(
    test_df, directory="images",
    x_col="filename", y_col="class_id",
    target_size=(224,224), batch_size=32,
    class_mode="sparse", shuffle=False
)
```
3. **Архитектуры и обучение**
```python
   from tensorflow.keras.applications import VGG16, VGG19, ResNet50
   from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
   from tensorflow.keras.models import Model
   from tensorflow.keras.optimizers import Adam
   
   def build_model(base_cls):
       base = base_cls(weights='imagenet', include_top=False, input_shape=(224,224,3))
       base.trainable = False
       x = GlobalAveragePooling2D()(base.output)
       x = Dropout(0.3)(x)
       x = Dense(256, activation='relu')(x)
       out = Dense(37, activation='softmax')(x)
       model = Model(base.input, out)
       model.compile(optimizer=Adam(1e-4),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
       return model
   
   model_vgg16 = build_model(VGG16)
   model_vgg19 = build_model(VGG19)
   model_resnet = build_model(ResNet50)
   
   # Обучение (10 эпох)
   history16 = model_vgg16.fit(train_gen, validation_data=test_gen, epochs=10)
   history19 = model_vgg19.fit(train_gen, validation_data=test_gen, epochs=10)
   historyr = model_resnet.fit(train_gen, validation_data=test_gen, epochs=10)
```

4. **Мультиклассовая оценка**
   VGG-16:    
   Accuracy на тестовом датасете (VGG16): 0.3423913043478261  
   Top-3 Accuracy: 0.1549
   Top-5 Accuracy: 0.2405  
   VGG-19:  
   Accuracy на тестовом датасете (VGG-19): 0.47146739130434784  
   Top-3 accuracy: 0.7350544
   Top-5 accuracy: 0.8138587  
   ResNet50:  
   Accuracy на тестовом датасете: 0.0557  
   Top-3 accuracy: 0.1548913
   Top-5 accuracy: 0.24048913

5. **Бинарная классификация «кошки vs собаки»**
```python
import numpy as np
from sklearn.metrics import accuracy_score

def multiclass_to_binary_accuracy(model, generator, cat_idxs, dog_idxs):
    probs = model.predict(generator, steps=len(generator), verbose=1)

    sum_cat = probs[:, cat_idxs].sum(axis=1)
    sum_dog = probs[:, dog_idxs].sum(axis=1)

    preds = (sum_dog > sum_cat).astype(int)

    y_true = np.array([1 if cls in dog_idxs else 0 for cls in generator.classes])

    return accuracy_score(y_true, preds)
```
```python
cc_vgg16_bin  = multiclass_to_binary_accuracy(model_vgg16,  test_generator, cat_indices, dog_indices)
acc_vgg19_bin  = multiclass_to_binary_accuracy(model_vgg19,  test_generator, cat_indices, dog_indices)
acc_resnet_bin = multiclass_to_binary_accuracy(model_resnet, test_generator, cat_indices, dog_indices)

print(f"Binary Accuracy VGG-16 : {acc_vgg16_bin:.4f}")
print(f"Binary Accuracy VGG-19 : {acc_vgg19_bin:.4f}")
print(f"Binary Accuracy ResNet: {acc_resnet_bin:.4f}")
```
Binary Accuracy VGG-16 : 0.6766  
Binary Accuracy VGG-19 : 0.7908  
Binary Accuracy ResNet: 0.6739  
