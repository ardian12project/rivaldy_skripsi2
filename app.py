import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import base64
class_names=[
  'Eksim',
  'Herpes',
  'Jerawat',
  'Panu',
  'Sehat']
penjelasan=[
'Penggunaan antihistamin. Metode ini dapat meredakan eksim yang diinduksikan oleh alergi. Namun, perlu diperhatikan jika obat ini dapat menyebabkan kantuk jika dikonsumsi secara oral. Dokter kerap menyarankan untuk menggunakan obat secara topikal.',
'Pengobatan dilakukan dengan berfokus menghilangkan bekas lepuhan dan mencegah penyebaran virus. Meski koreng dan lepuhan dapat hilang dengan sendirinya, pengobatan yang dilakukan dapat mengurangi komplikasi yang bisa saja dialami oleh pengidap. Dokter mungkin juga meresepkan obat topikal atau obat oral seperti Asiklovir ,Famsiklovir, dan Valasiklovir.',
'Pengobatan jerawat bisa membutuhkan waktu 1–12 minggu, tergantung pada tingkat keparahan jerawat. Obat yang digunakan untuk mengatasi jerawat disesuaikan dengan jenis jerawat dan kondisi pasien. Obat yang digunakan biasanya berupa obat oles dengan kandungan zat seperti Benzoyl peroxide, Tretinoin, Asam azaleat, dan Antibiotik oles',
'Pengobatan panu adalah dengan pemberian obat antijamur dalam bentuk oles atau minum. Obat antijamur oles yang dapat diresepkan dokter antara lain ketoconazole atau selenium sulfide. Jika infeksi jamur makin luas dan parah, pasien akan diberikan tablet antijamur berupa fluconazole atau itraconazole.',
'Wajah anda sehat ,selalu pastikan wajah anda agar bersih dan sehat.'
]

# background image to streamlit

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# set_png_as_page_bg('grey-polygon-desktop-wallpaper-wallpaper-png-favpng-7M2rAXHUKfaSg6GE2cgp9BwA9.jpg')

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('Cendekia-pest-72.18.h5', compile=False)
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

# 1. as sidebar menu
selected2 = option_menu(None, ["Home", "Upload", "About"], 
    icons=['house', 'cloud-upload', "list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected2 == "Home":
  # col1, col2, col3 = st.columns(3)
  # with col1:
  #   st.write(' ')
  # with col2:
  #   st.image("20180911125756.png", width=200,use_column_width=True)
  # with col3:
  #   st.write(' ')
  st.markdown("## Klasifikasi Penyakit Wajah")
  st.markdown("""
ini adalah aplikasi deep learning (Convolutional Neural Network) untuk melakukan klasifikasi penyakit wajah, aplikasi ini hanya bisa mendeteksi penyakit pada wajah manusia. berikut adalah penyakit yang di klasifikasi:
1. Eksim
2. Herpes
3. Jerawat
4. Panu
""")
  st.image('images1.png', width= 730,use_column_width=True)

elif selected2 == "Upload":
  st.subheader("klasifikasi penyakit wajah")
  file = st.file_uploader("Silahkan upload gambar wajah anda" ,type=["jpg","png","jpeg"])
  import cv2
  from PIL import Image, ImageOps
  st.set_option('deprecation.showfileUploaderEncoding', False)
  def import_and_predict(image_data, model):
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        return prediction

  if file is None:
    st.text("silahkan upload gambar")
  else:
    image = Image.open(file)
    st.image(image, width=300)
    predictions = import_and_predict(image, model)
    score=np.array(predictions[0])
    st.subheader(
    "Wajah anda terdeteksi {}."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    st.markdown("Langkah Pengobatan :".format(penjelasan[np.argmax(score)]) )
    st.markdown("{}".format(penjelasan[np.argmax(score)]) )

elif selected2 == "About":
  st.header("Tentang Aplikasi")
  st.subheader("Aplikasi ini dibuat oleh Muhammad Rivaldy")
  st.markdown("Data source: https://www.kaggle.com/datasets/hilmiher/face-disease")
  # col1, col2, col3 = st.columns(3)
  # with col1:
  #   st.image("images0.jpg", width=100)
  # with col2:
  #   st.image("images1.jpg", width=100)
  # with col3:
  #   st.image("images2.jpg", width=100)

  # col1, col2, col3 = st.columns(3)
  # with col1:
  #   st.image("images3.jpg", width=100)
  # with col2:
  #   st.image("images4.jpg", width=100)
  # with col3:
  #   st.image("images5.jpg", width=100)

  # col1, col2, col3 = st.columns(3)
  # with col1:
  #   st.image("images6.jpg", width=100)
  # with col2:
  #   st.image("images7.jpg", width=100)
  # with col3:
  #   st.image("images8.jpg", width=100)
