from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from efficientnet.keras import EfficientNetB1

app = Flask(__name__)

# Parameters
input_size = (100,100)
channel = (3,)
input_shape = input_size + channel

#labelnya
labels = ['Bean', 'Bitter_Gourd',
		 'Bottle_Gourd', 'Brinjal',
		 'Broccoli', 'Cabbage',
		 'Capsicum', 'Carrot',
		 'Cauliflower', 'Cucumber',
		 'Papaya', 'Potato',
		 'Pumpkin', 'Radish',
		 'Tomato']

dic = {0 : 'Cat', 1 : 'Dog'}

model = load_model('model_13.h5', compile=False)

#model.make_predict_function()

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

def predict_label(img_path):
	i = image.load_img(img_path)
	x = preprocess(i,input_size)
	x = reshape([x])
	y = model.predict(x)
	confident = np.max(y)*100
	return labels[np.argmax(y)], confident


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/hasil", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p,y = predict_label(img_path)
		if p == "Carrot":
			return render_template("carrot.html", prediction = p, img_path = img_path,y=y)
		elif p == "Bean":
			return render_template("bean.html", prediction = p, img_path = img_path,y=y)
		elif p == "Bottle_Gourd":
			return render_template("bottle gourd.html", prediction = 'Bottle Gourd', img_path = img_path,y=y)
		elif p == "Bitter_Gourd":
			return render_template("bitter gourd.html", prediction = 'Bitter Gourd', img_path = img_path,y=y)
		elif p == "Brinjal":
			return render_template("brinjal.html", prediction = p, img_path = img_path,y=y)
		elif p == "Broccoli":
			return render_template("broccoli.html", prediction = p, img_path = img_path,y=y)
		elif p == "Cabbage":
			return render_template("cabbage.html", prediction = p, img_path = img_path,y=y)
		elif p == "Capsicum":
			return render_template("capsicum.html", prediction = p, img_path = img_path,y=y)
		elif p == "Cauliflower":
			return render_template("cauliflower.html", prediction = p, img_path = img_path,y=y)
		elif p == "Cucumber":
			return render_template("cucumber.html", prediction = p, img_path = img_path,y=y)
		elif p == "Papaya":
			return render_template("papaya.html", prediction = p, img_path = img_path,y=y)
		elif p == "Potato":
			return render_template("potato.html", prediction = p, img_path = img_path,y=y)
		elif p == "Pumpkin":
			return render_template("pumpkin.html", prediction = p, img_path = img_path,y=y)
		elif p == "Radish":
			return render_template("radish.html", prediction = p, img_path = img_path,y=y)
		elif p == "Tomato":
			return render_template("tomato.html", prediction = p, img_path = img_path,y=y)
		else:
			return render_template("index.html")

	return render_template("index.html", prediction = p, img_path = img_path,y=y)

@app.route("/teams")
def teams_page():
	return render_template("team.html")

if __name__ =='__main__':
	#app.debug = True
	app.run()