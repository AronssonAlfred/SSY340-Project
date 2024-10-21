import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np


# Create Class Structure of Emotion Detection Network
class EmotionDetectionModel2(nn.Module):
    
    def __init__(self, img_size, num_classes = 7):
        super(EmotionDetectionModel2, self).__init__()

        self._num_classes = num_classes  # Set num_classes from the argument
        self._img_channels = 1
        self.dropout_factor1 = 0.2
        self.dropout_factor2 = 0.4

        # First convolution block -----
        self.conv1 = nn.Conv2d(in_channels=self._img_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.4)
        
        
        # Second convolution block -----
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.4)
        
        
        # Third convolution block -----
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.5)

        # Fully connected layers
        # Calculate the output size of the feature maps after the convolutional layers
        self.flatten_size = self._calculate_conv_output_size(img_size)

        # Fully connected layer 1
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.6)

        # Fully connected layer 2
        self.fc2 = nn.Linear(128, self._num_classes)

    def _calculate_conv_output_size(self, img_size):
        # Helper method to compute the spatial dimensions after the convolution and pooling layers
        size = img_size // 8  # Since 3 pooling layers, we reduce the size by a factor of 8
        return size * size * 256  # 256 filters at the final conv layer

    def save_model(self, train_losses, train_accs, val_losses, val_accs):
        # Get the current date and time
        date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

        # Save the model state and training history
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "train_losses": train_losses,
                "train_accs": train_accs,
                "val_losses": val_losses,
                "val_accs": val_accs,
            },
            f"./EmotionDetectionModel-{date_and_time}.ckpt",  # Use f-string for formatting
        )

    def forward(self, x):
        # First convolution block -----
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second convolution block -----
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third convolution block -----
        x = F.elu(self.bn5(self.conv5(x)))
        x = F.elu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.elu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x

# Load the trained YOLOv8 model for face detection
face_model = YOLO('object-detection/runs/detect/train7/weights/best.pt')

# Load the trained Emotion Detection model
img_size = 48  # Input size of the emotion detection model
emotion_model = EmotionDetectionModel2(48, 7)
checkpoint_path = 'Emotion classification/FER2013_model2.ckpt'  # Path to the trained emotion detection model checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
emotion_model.load_state_dict(checkpoint['model_state_dict'])
emotion_model.eval()  # Set the model to evaluation mode


# Define emotion labels as a dictionary with indices as keys
emotion_labels_dict = {
    0: "angry",
    1: "sad",
    2: "sad",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "happy"
}

# Load emoji images for each emotion
emoji_paths = {
    "angry": "FinalProduct/Emojis/angry.png",
    "sad": "FinalProduct/Emojis/sad.png",
    "happy": "FinalProduct/Emojis/happy.png",
    "neutral": "FinalProduct/Emojis/neutral.png"
}
emojis = {label: cv2.imread(path, cv2.IMREAD_UNCHANGED) for label, path in emoji_paths.items()}

def overlay_emoji(frame, emoji, x, y, w, h):
    # Resize the emoji to fit the face bounding box
    emoji_resized = cv2.resize(emoji, (w, h))

    # Get dimensions and split alpha channel
    eh, ew, ec = emoji_resized.shape
    if ec == 4:  # If emoji has an alpha channel
        overlay_img = emoji_resized[:, :, :3]
        alpha_mask = emoji_resized[:, :, 3] / 255.0
    else:
        overlay_img = emoji_resized
        alpha_mask = np.ones((eh, ew), dtype=np.float32)

    # Blend the emoji with the frame
    for c in range(0, 3):
        frame[y:y + eh, x:x + ew, c] = (alpha_mask * overlay_img[:, :, c] +
                                        (1 - alpha_mask) * frame[y:y + eh, x:x + ew, c])

def main():
    # Access the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    cv2.namedWindow('Real-Time Face and Emotion Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Real-Time Face and Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Press 'q' to quit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Step 1: Run face detection on the frame
        results = face_model(frame)

        # Step 2: Iterate through detected faces
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_face = frame[y1:y2, x1:x2]

                if cropped_face.size == 0:
                    continue

                # Preprocess the cropped face for emotion detection
                cropped_face_resized = cv2.resize(cropped_face, (img_size, img_size))
                cropped_face_gray = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY)
                cropped_face_tensor = torch.tensor(cropped_face_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

                # Normalize using the mean and std
                train_mean = torch.tensor([0.5077])
                train_std = torch.tensor([0.2550])
                cropped_face_tensor = (cropped_face_tensor - train_mean) / train_std

                # Step 3: Run emotion detection on the cropped face
                with torch.no_grad():
                    emotion_prediction = emotion_model(cropped_face_tensor)
                    emotion_index = torch.argmax(emotion_prediction).item()
                    emotion_label = emotion_labels_dict[emotion_index]

                # Step 4: Overlay the corresponding emoji on the face
                if emotion_label in emojis:
                    emoji = emojis[emotion_label]
                    overlay_emoji(frame, emoji, x1, y1, x2 - x1, y2 - y1)

        # Display the frame with emoji overlay
        cv2.imshow('Real-Time Face and Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
