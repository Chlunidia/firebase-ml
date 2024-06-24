# Firebase ML Image Classification with Custom Model in Kotlin

## Preview
<div align="center">
  <img src="https://github.com/Chlunidia/firebase-ml/assets/115222445/a62b5a8b-02e8-4411-89a5-a8847fe7c59d" alt="Example Image" width="200">
</div>

## Introduction
I developed this project to learn about implementing Firebase ML in Kotlin. This project demonstrates how to use Firebase Machine Learning to perform image classification with a custom model using Kotlin. The goal is to provide an example of integrating Firebase ML Kit into an Android application for classifying images.

## Features
- Image classification using a custom TensorFlow Lite model.
- Integration with Firebase ML Kit.
- Simple Kotlin implementation.

## Technologies Used
- **Kotlin**: Main programming language.
- **Firebase ML Kit**: For deploying and managing the machine learning model.
- **TensorFlow Lite**: Custom model for image classification.
- **Android Studio**: IDE for development.

## Getting Started

### Prerequisites
- Android Studio installed on your development machine.
- A Firebase project setup (if not, create one at [Firebase Console](https://console.firebase.google.com/)).
- Basic knowledge of Kotlin and Android development.

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Chlunidia/firebase-ml.git
   cd firebase-ml

2. **Open the Project in Android Studio**
   - Open Android Studio and select "Open an existing Android Studio project".
   - Navigate to the cloned directory and select it.

3. **Add Firebase to Your Android Project**
   - Follow the instructions in the [Firebase documentation](https://firebase.google.com/docs/android/setup) to add Firebase to your Android project.
   - Download the `google-services.json` file from the Firebase Console and place it in the `app` directory of your project.

### Configuration
1. **Set Up Firebase ML Kit**
   - In the Firebase Console, navigate to the Machine Learning section.
   - Upload your TensorFlow Lite model to Firebase.
   - Note the model name provided by Firebase.

2. **Open the Project in Android Studio**
   - Open Android Studio and select "Open an existing Android Studio project".
   - Navigate to the cloned directory and select it.

3. **Add Firebase to Your Android Project**
   - Follow the instructions in the [Firebase documentation](https://firebase.google.com/docs/android/setup) to add Firebase to your Android project.
   - Download the `google-services.json` file from the Firebase Console and place it in the `app` directory of your project.

4. **Modify Code to Use Your Model from Firebase**
   - Open your Kotlin project in Android Studio.
   - Locate the section of your code where the model is referenced.
   - Update the code to download the model from Firebase. Example:
     ```kotlin
     // Set conditions for downloading the model
     val conditions = FirebaseModelDownloadConditions.Builder()
         .requireWifi()
         .build()

     // Download the model from Firebase
     FirebaseModelDownloader.getInstance()
         .getModel("your_model_name", DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND, conditions)
         .addOnSuccessListener { model: CustomModel? ->
             // Download complete. Depending on your requirements, you could also check if the model is already downloaded.
             val modelFile = model?.file
             if (modelFile != null) {
                 // Use the model file to initialize a TensorFlow Lite interpreter
                 val interpreter = Interpreter(modelFile)
                 // Use the interpreter to run inference
             }
         }
         .addOnFailureListener { exception ->
             // Handle any errors during the download
             Log.e("ModelDownload", "Model download failed", exception)
         }
     ```
   - Ensure you have the necessary permissions and dependencies in your `AndroidManifest.xml` and `build.gradle` files.

### Running the Project
- Connect your Android device or start an emulator.
- Click the "Run" button in Android Studio to build and run the project.
- Ensure your device/emulator is connected to the internet to download the model from Firebase.

## Usage
1. **Classifying Images**
   - Open the app and navigate to the image classification section.
   - Select an image from the gallery.
   - The app will process the image using the custom TensorFlow Lite model downloaded from Firebase and display the classification results.

2. **Customizing the Model**
   - If you wish to use a different TensorFlow Lite model:
     - Upload the new model to Firebase ML Kit.
     - Update the model name in your code to reference the new model.
     - Rebuild and run the application.

## Acknowledgements
- Thanks to [Firebase](https://firebase.google.com/) for providing powerful tools to integrate machine learning into mobile applications.
- Inspiration and guidance from various online tutorials and communities.
