import streamlit as st
import pydicom
import numpy as np
import cv2
from tensorflow import keras

#pydicom.config.convert_wrong_length_to_UN = True
#pydicom.config.Settings.infer_sq_for_un_vr = True

st.set_option('deprecation.showfileUploaderEncoding', False)
from PIL import Image
image = Image.open('Capture.PNG')

st.sidebar.image(image,channels="RGB",width =250)
class WrongFileType(ValueError):
    pass

def main():
    st.title("CLASSIFICATION MEDICAL DICOM")
    #st.sidebar.title("Configuration")
    st.sidebar.text("YOU ONLY CHOSE LUNG IMAGE")
    mode1 = st.sidebar.radio(
        "Select type of input",
        ('Image', 'Dicom'))
    if mode1 == 'Dicom':
        dicom_bytes = st.sidebar.file_uploader("Upload file", type=["dcm","dicom"])
        # Config
        print(dicom_bytes)
        classes = ['AORTIC ENLARGEMENT ', 'COVID 19', 'OPACITY', 'NORMAL']
        model = keras.models.load_model('RES_128_proposed.h5')

        mode = st.sidebar.radio(
            "Select input source",
            ('View Image', 'View information'))

        if not dicom_bytes:
            raise st.stop()

        try:
            dicom_header = pydicom.read_file(dicom_bytes, force=True)
            image_dicom = dicom_header.pixel_array/4095
        except:
            st.write(WrongFileType("Does not appear to be a DICOM file"))

        if mode == 'View Image':

            if st.sidebar.button("Load Image"):
                st.image(image_dicom,width =400)

            if st.sidebar.button("Predicted"):
                st.image(image_dicom,width =400)
                #print(image_dicom.shape)
                my_data2 = cv2.resize(image_dicom, (128, 128))
                a = my_data2.reshape(-1, 128, 128, 1)
                # pass the image through the network to obtain our predictions
                preds = model.predict(a)
                if max(preds[0]) <= 0.7:
                    st.text("THIS IS NOT A FILE OF LUNG DICOM")

                else:
                    label = classes[np.argmax(preds)]
                    st.text("THE RESULT OF DICOM IS: " + label + " WITH ACCURACY IS " + str(max(preds[0])*100) + " %")

        if mode == 'View information' and st.sidebar.button("Load Information"):
            view = dicom_header.__repr__().split("\n")
            view = "\n".join(view)
            f"""
                ```
                {view}
                ```
                """

    if mode1 == 'Image':
        image_bytes = st.sidebar.file_uploader("Upload file", type=["jpg", "png"])
        # Config
        if image_bytes is not None:
            classes = ['COVID', 'NORMAL', 'PNEUMONIA']
            model = keras.models.load_model('RESNET50_128_proposed.h5')

            def load_image(img):
                im = Image.open(img)
                image = np.array(im)
                return image
            image = load_image(image_bytes)

            #print(image.shape)
            #print(image)

            if st.sidebar.button("Load Image"):
                    st.image(image_bytes, width=400)

            if st.sidebar.button("Predicted"):
                    st.image(image_bytes, width=400)
                    #print(image_bytes.shape)
                    my_data = cv2.resize(image, (128, 128))
                    my_data1 = my_data/255
                   
                    my_data3 = my_data1.reshape(-1, 128, 128,3)
                    # pass the image through the network to obtain our predictions
                    preds = model.predict(my_data3)
                    print(max(preds[0]))
                    if max(preds[0]) <= 0.8:
                        st.text("THIS IS NOT A FILE OF LUNG IMAGE")

                    else:
                        label = classes[np.argmax(preds)]
                        st.text("THE RESULT OF IMAGE IS: " + label + " WITH ACCURACY IS " + str(max(preds[0])*100) + " %")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
