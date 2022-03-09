# coding: utf-8 
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import os
import pickle
 
def list_imgs(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

def plot_history(history, 
                save_graph_img_path, 
                fig_size_width, 
                fig_size_height, 
                lim_font_size):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
   
    epochs = range(len(acc))
       

def main():
    # ハイパーパラメータ
    batch_size = 5      # バッチサイズ
    num_classes = 5     # 分類クラス数(5種類)
    epochs = 19         # エポック数
    dropout_rate = 0.2  # 過学習防止用：入力の20%を0にする

    # 入力画像のパラメータ
    img_width = 64 # 入力画像の幅
    img_height = 64 # 入力画像の高さ
    img_ch = 3 # 3ch画像

    # データの保存先
    SAVE_DATA_DIR_PATH = "C:/Pythons/2.7.7 CLassification/ex1_data/"

    # ディレクトリがなければ作成
    os.makedirs(SAVE_DATA_DIR_PATH, exist_ok=True)

    # グラフ画像のサイズ
    FIG_SIZE_WIDTH = 12
    FIG_SIZE_HEIGHT = 10
    FIG_FONT_SIZE = 25

    data_x = []
    data_y = []

    FILE_PATH = 'C:/Pythons/2.7.7 CLassification/ex1_data/'
    files = os.listdir(FILE_PATH)

    # クラス0の画像データ群をロード
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "img0"):
        img = img_to_array(load_img(filepath, target_size=(img_width,img_height, img_ch)))
        data_x.append(img)
        data_y.append(0) # 教師データ

    # クラス1の画像データ群をロード
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "img1"):
        img = img_to_array(load_img(filepath, target_size=(img_width,img_height, img_ch)))
        data_x.append(img)
        data_y.append(1) # 教師データ

    # クラス2の画像データ群をロード
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "img2"):
        img = img_to_array(load_img(filepath, target_size=(img_width,img_height, img_ch)))
        data_x.append(img)
        data_y.append(2) # 教師データ

    # クラス3の画像データ群をロード
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "img3"):
        img = img_to_array(load_img(filepath, target_size=(img_width,img_height, img_ch)))
        data_x.append(img)
        data_y.append(3) # 教師データ

    # クラス4の画像データ群をロード
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "img4"):
        img = img_to_array(load_img(filepath, target_size=(img_width,img_height, img_ch)))
        data_x.append(img)
        data_y.append(4) # 教師データ

    # NumPy配列に変換
    data_x = np.asarray(data_x)
    print(data_x)

    # 学習データはNumPy配列に変換
    data_y = np.asarray(data_y)

    # 学習用データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

    # 学習データはfloat32型に変換し、正規化(0～1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 正解ラベルをone hotエンコーディング
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # データセットの個数を表示
    print(x_train.shape, 'x train samples')
    print(x_test.shape, 'x test samples')
    print(y_train.shape, 'y train samples')
    print(y_test.shape, 'y test samples')

    """
    
    """
    # モデルの構築
    model = Sequential()

    # 入力層:32×32*3
    # 【2次元畳み込み層】
    
    model.add(Conv2D(32,(3,3), 
                padding='same', 
                input_shape=x_train.shape[1:],
                activation='relu'))

    # 【2次元畳み込み層】
    
    model.add(Conv2D(64,(3,3),
                padding='same',
                activation='relu'))

    # 【プーリング層】
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ドロップアウト(過学習防止用, dropout_rate=0.2なら20%のユニットを無効化）
    model.add(Dropout(dropout_rate))

    # 【2次元畳み込み層】
    
    model.add(Conv2D(64,(3,3),
                padding='same',
                activation='relu'))

    # 【2次元畳み込み層】
    
    model.add(Conv2D(64,(3,3),
                padding='same',
                activation='relu'))

    # 【プーリング層】
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ドロップアウト

    model.add(Dropout(dropout_rate))

    # 平坦化（次元削減）
    
    model.add(Flatten())

    # 全結合層 (Affine)
    
    model.add(Dense(512, activation='relu'))

    # ドロップアウト

    model.add(Dropout(dropout_rate))
    
    # 全結合層 (Softmax)
    
    model.add(Dense(num_classes, activation='softmax')) 

    # モデル構造の表示
    #model.summary()

    # コンパイル
    # 最適化：RMSpropを使用
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    # 構築したモデルで学習（学習データ:trainのうち、10％を検証データ:validationとして使用）
    # verbose=1:標準出力にログを表示

    history = model.fit(x_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_split=0.1)

    # パフォーマンスの評価

    score = model.evaluate(x_test, 
                            y_test,
                            verbose=0
                            )

    # パフォーマンス計測の結果
    # 損失値
    print('Test loss:', score[0])

    # 正答率
    print('Test accuracy:', score[1])
      
    # モデル構造の保存
    open(SAVE_DATA_DIR_PATH  + "model.json","w").write(model.to_json())  

    # 学習済みの重みを保存
    model.save_weights(SAVE_DATA_DIR_PATH + "weight.hdf5")

    # 学習履歴を保存
    with open(SAVE_DATA_DIR_PATH + "history.json", 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    main()