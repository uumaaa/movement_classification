import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import (
    Dense,
    Softmax,
    InputLayer,
    GRU,
)

from keras.api.optimizers import SGD
from keras.api.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import random
import numpy as np
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt


def format_frames(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return tf.convert_to_tensor(frame.flatten(), tf.float16) / 255.0


def frames_from_video_file(video_path, n_frames):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(video_path)

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = n_frames

    if need_length > video_length:
        start = 0
    else:
        max_start = int(video_length - need_length)
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    if not ret:
        return None

    result.append(format_frames(frame))

    for _ in range(n_frames - 1):
        ret, frame = src.read()
        if ret:
            frame = format_frames(frame)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)
    return result


def process_videos(base_path, n_frames=100):
    dataset = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    label_to_int = {}
    for data_split in ["train", "val", "test"]:
        split_path = os.path.join(base_path, data_split)
        for class_name in os.listdir((split_path)):
            class_path = os.path.join(split_path, class_name)
            print(f"{class_path}\n\n...\n\n")
            if os.path.isdir(class_path):
                for video in os.listdir(class_path):
                    video_path = os.path.join(class_path, video)
                    if video_path.endswith((".mp4")):
                        frames = frames_from_video_file(video_path, n_frames)
                        if frames is not None:
                            dataset[data_split].append(frames)
                            labels[data_split].append(class_name)
    unique_classes = np.unique(labels["train"])
    label_to_int = dict([(unique_classes[_], _) for _ in range(len(unique_classes))])
    return dataset, labels, label_to_int


class FrameGenerator:
    def __init__(self, videos, labels, label_to_int, n_frames):
        """Returns a set of frames with their associated labels.

        Args:
          path: Video file paths.
          n_frames: Number of frames.
          training: Boolean to determine if training dataset is being created.
        """
        self.n_frames = n_frames
        self.videos = videos
        self.labels = labels
        self.label_to_int = label_to_int

    def __call__(self):
        for video_frames, label in zip(self.videos, self.labels):
            yield video_frames, self.label_to_int[label]


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
base_path = "working_dataset"
dataset, labels, label_to_int = process_videos(base_path)

# Example of accessing the data
print(f"Unique name classes {label_to_int}")
print(f"Number of training videos: {len(dataset['train'])}")
print(f"Number of validation videos: {len(dataset['val'])}")
print(f"Number of test videos: {len(dataset['test'])}")


output_signature = (
    tf.TensorSpec(shape=(100, 224 * 224), dtype=tf.float16),
    tf.TensorSpec(shape=(), dtype=tf.int16),
)
train_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        dataset["train"],
        labels["train"],
        label_to_int,
        100,
    ),
    output_signature=output_signature,
)
test_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        dataset["test"],
        labels["test"],
        label_to_int,
        100,
    ),
    output_signature=output_signature,
)
val_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        dataset["val"],
        labels["val"],
        label_to_int,
        100,
    ),
    output_signature=output_signature,
)


AUTOTUNE = tf.data.AUTOTUNE
batch_size = 16
train_ds = (
    train_ds.cache()
    .shuffle(train_ds.cardinality(), reshuffle_each_iteration=True)
    .prefetch(buffer_size=AUTOTUNE)
    .batch(batch_size)
)
test_ds = (
    test_ds.cache()
    .shuffle(test_ds.cardinality(), reshuffle_each_iteration=True)
    .prefetch(buffer_size=AUTOTUNE)
    .batch(batch_size)
)

val_ds = (
    val_ds.cache()
    .shuffle(val_ds.cardinality(), reshuffle_each_iteration=True)
    .prefetch(buffer_size=AUTOTUNE)
    .batch(batch_size)
)


model = Sequential(
    [
        InputLayer(input_shape=(100, 224 * 224)),
        GRU(32, return_sequences=False),
        Dense(9),
        Softmax(),
    ]
)

model.summary()

model.compile(
    optimizer=SGD(learning_rate=0.08, momentum=0.3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=30,
    validation_data=test_ds,
    callbacks=EarlyStopping(patience=10, monitor="val_loss", verbose=1),
    verbose=1,
    batch_size=batch_size,
)

# Predict on the test dataset
y_pred = np.argmax(model.predict(test_ds, batch_size=batch_size, verbose=1), axis=1)

# Assuming y_true is the true labels of the test set
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(y_true, y_pred, digits=4)
print("Classification Report:")
print(class_report)

# Plot confusion matrix


plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[str(_) for _ in range(9)],
    yticklabels=[str(_) for _ in range(9)],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
