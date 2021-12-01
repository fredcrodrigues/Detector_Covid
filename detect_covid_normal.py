from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
import os

path = r'/home/fredson/project_covid/dataset/normal/Normal-2.png'


## no may eat up all your disks
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def detect():

    data = []
    labels = []

    img = cv.imread(path)


    row, col, c = img.shape

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    image = cv.resize(img, (224,224), interpolation=cv.INTER_CUBIC)

    data.append(image)

    n_image = np.array(data)/255.0

    print(n_image.shape)

    for image in n_image:
       
        model = load_model('models/c_detect.h5')
        image = np.expand_dims(image, axis=0)

        covid, normal = model.predict(image)[0]
        
        label = "Pneumonia Detectado" if covid > normal else "Pneumonia Nao detectado"
        color = (0, 255, 0) if label == "Covid Detectado" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(covid, normal) * 100)

        cv.putText( img , label, (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv.rectangle( img , (0, 0), (row, col), color, 2)
   
    cv.imwrite('result/image/Image_detect.png',  img )

if __name__ == "__main__":
	detect()
