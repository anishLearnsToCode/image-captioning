import pickle
from RNNDecoder import RNN_Decoder
from CNNEncoder import CNN_Encoder
import tensorflow as tf
from flask import Flask, request, jsonify
import time
import numpy as np


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


top_k = 5000
# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
#                                                   oov_token="<unk>",
#                                                   filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

tokenizer = pickle.load(open('model/tokenizer.p', 'rb'))

max_length = pickle.load(open('model/max_length.p', 'rb'))
attention_features_shape = pickle.load(open('model/attention_features.p', 'rb'))

embedding_dim = 256
units = 512
vocab_size = top_k + 1

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


app = Flask(__name__)


@app.route("/caption")
def AutoImageCaption():
    image_url = request.args.get('image')
    print('image_url')
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file(str(int(time.time())) + image_extension, origin=image_url)
    result, attention_plot = evaluate(image_path)
    data = {'Prediction Caption:': ' '.join(result)}
    return jsonify(data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
