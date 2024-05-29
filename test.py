from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json


# 加載模型
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights('weight.h5')


breed_name = ['scottish_deerhound','maltese_dog',
              'afghan_hound','entlebucher',"bernese_mountain_dog"
              ,"shih-tzu","great_pyrenees","pomeranian","basenji","samoyed"]
breed_name_chinese = ['蘇格蘭鹿獵犬', '馬爾濟斯犬', '阿富汗獵犬',
                       '恩特布赫山犬', '伯恩山犬', '獅子狗', '大白熊犬',
                         '波美拉尼犬', '巴辛吉犬', '薩摩耶犬']
# 讀取圖片
img = image.load_img('dog-breed-identification//train//1e5ce138280eabd214664398f19491b3.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

# 使用模型進行預測
predictions = model.predict(img)
# 輸出預測結果
print('Predicted:', breed_name_chinese[np.argmax(predictions[0])])