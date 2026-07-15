Place the two TensorFlow Lite model files in this folder, using these exact names
(referenced by MainActivity.java / ObjectDetector.java / OCRDetector.java):

  best_float16.tflite      - object detector: "Reading", "Thermometer"
  latest_ocr_best.tflite    - digit OCR detector: "0".."9"

Both files must be present for the app to run. This assets/ folder is configured
as noCompress in app/build.gradle.kts so the models can be memory-mapped directly
by the TensorFlow Lite Interpreter.
