import tensorflow as tf
import tensorflow_federated as tff
import fed_compression

from datetime import datetime
from config import *
from pathlib import Path
from utils import plot_graph
from matplotlib import pyplot as plt
from dataset import load as loding_dataset

from tensorflow.keras import losses, metrics, optimizers

now = datetime.now()
date_time = now.strftime("%d.%m.%Y__%H.%M.%S")

this_dir = Path.cwd()
print(this_dir)
model_dir = this_dir / "saved_models" / experiment_name / str(datetime)
output_dir = this_dir / "results" / experiment_name / str(datetime)

if not model_dir.exists():
    model_dir.mkdir(parents=True)

if not output_dir.exists():
    output_dir.mkdir(parents=True)


federated_train_data, preprocessed_sample_dataset = loding_dataset(phase='train')


def create_keras_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=64, kernel_size=(3, 3), padding="same",
                                     activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
    model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
    model.add(tf.keras.layers.Dense(units=NumClass, activation="softmax"))

    ##############################################################################################
    ##############################################################################################

    return model


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.

    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_sample_dataset.element_spec,
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()])


iterative_process = fed_compression.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: optimizers.Adam(learning_rate=client_lr),
    server_optimizer_fn=lambda: optimizers.SGD(learning_rate=server_lr))


print(str(iterative_process.initialize.type_signature))
state = iterative_process.initialize()


x_test, y_test = loding_dataset(phase='test')

tff_train_acc = []
tff_val_acc = []
tff_train_loss = []
tff_val_loss = []

eval_model = None
for round_num in range(1, NUM_ROUNDS+1):
    state, tff_metrics = iterative_process.next(state, federated_train_data)
    eval_model = create_keras_model()
    eval_model.compile(optimizer=optimizers.Adam(learning_rate=client_lr),
                       loss=losses.SparseCategoricalCrossentropy(),
                       metrics=[metrics.SparseCategoricalAccuracy()])

    tff.learning.assign_weights_to_keras_model(eval_model, state.model)

    ev_result = eval_model.evaluate(x_test, y_test, verbose=0)
    print('round {:2d}, metrics={}'.format(round_num, tff_metrics))
    print(f"Eval loss : {ev_result[0]} and Eval accuracy : {ev_result[1]}")
    tff_train_acc.append(float(tff_metrics.sparse_categorical_accuracy))
    tff_val_acc.append(ev_result[1])
    tff_train_loss.append(float(tff_metrics.loss))
    tff_val_loss.append(ev_result[0])

metric_collection = {"sparse_categorical_accuracy": tff_train_acc,
                     "val_sparse_categorical_accuracy": tff_val_acc,
                     "loss": tff_train_loss,
                     "val_loss": tff_val_loss}

if eval_model:
    eval_model.save(model_dir / (experiment_name + ".h5"))
else:
    print("training didn't started")
    exit()

fig = plt.figure(figsize=(10, 6))
plot_graph(list(range(1, 26))[4::5], tff_train_acc, label='Train Accuracy')
plot_graph(list(range(1, 26))[4::5], tff_val_acc, label='Validation Accuracy')
plt.legend()
plt.savefig(output_dir / "federated_model_Accuracy.png")

plt.figure(figsize=(10, 6))
plot_graph(list(range(1, 26))[4::5], tff_train_loss, label='Train loss')
plot_graph(list(range(1, 26))[4::5], tff_val_loss, label='Validation loss')
plt.legend()
plt.savefig(output_dir / "federated_model_loss.png")



# saving metric values to text file

txt_file_path = output_dir / (experiment_name + ".txt")
with open(txt_file_path.as_posix(), "w") as handle:
    content = []
    for key, val in metric_collection.items():
        line_content = key
        val = [str(k) for k in val]
        line_content = line_content + " " + " ".join(val)
        content.append(line_content)
    handle.write("\n".join(content))