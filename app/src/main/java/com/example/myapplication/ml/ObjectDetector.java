package com.example.myapplication.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import com.example.myapplication.model.BoundingBox;
import com.example.myapplication.model.Detection;
import com.example.myapplication.utils.ImageUtils;
import com.example.myapplication.utils.Letterbox;
import com.example.myapplication.utils.NMS;
import com.example.myapplication.utils.ScaleBoxes;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * TensorFlow Lite wrapper for best_float16.tflite (classes: "Reading", "Thermometer").
 * Reproduces the Python pipeline: letterbox to the model's own input dimensions (read from
 * the tensor itself), float32 normalize, run inference, decode raw YOLO output, scale boxes
 * back to the original image, confidence filter, NMS.
 */
public class ObjectDetector implements AutoCloseable {

    public static final float CONFIDENCE_THRESHOLD = 0.25f;
    public static final float IOU_THRESHOLD = 0.45f;

    public static final String[] LABELS = {"Reading", "Thermometer"};
    public static final String READING_LABEL = "Reading";

    private final Interpreter interpreter;
    private final int inputWidth;
    private final int inputHeight;
    private final boolean inputChannelsFirst;

    public ObjectDetector(Context context, String modelFileName) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(loadModelFile(context, modelFileName), options);

        int[] inputShape = interpreter.getInputTensor(0).shape();
        // NCHW ([1, 3, H, W]) vs NHWC ([1, H, W, 3]): the channel dimension (1 or 3) tells them apart.
        inputChannelsFirst = inputShape[1] == 1 || inputShape[1] == 3;
        if (inputChannelsFirst) {
            inputHeight = inputShape[2];
            inputWidth = inputShape[3];
        } else {
            inputHeight = inputShape[1];
            inputWidth = inputShape[2];
        }
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int getInputHeight() {
        return inputHeight;
    }

    private static MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFileName);
        try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    public List<Detection> detect(Bitmap originalImage) {
        Letterbox.Result letterboxResult = Letterbox.apply(originalImage, inputWidth, inputHeight);
        ByteBuffer inputBuffer = inputChannelsFirst
                ? ImageUtils.bitmapToNormalizedFloatBufferNCHW(letterboxResult.bitmap)
                : ImageUtils.bitmapToNormalizedFloatBuffer(letterboxResult.bitmap);

        int[] outputShape = interpreter.getOutputTensor(0).shape();
        float[][][] output = new float[1][outputShape[1]][outputShape[2]];
        interpreter.run(inputBuffer, output);

        List<Detection> candidates = new ArrayList<>();

        if (outputShape[1] == 6 && outputShape[2] != 6) {
            // End-to-end / NMS-free export (e.g. YOLO26's default head): [1, 6, numDetections],
            // each column already fully decoded as [x1, y1, x2, y2, confidence, classId].
            decodeEndToEnd(output, outputShape[2], true, letterboxResult, candidates);
        } else if (outputShape[2] == 6 && outputShape[1] != 6) {
            decodeEndToEnd(output, outputShape[1], false, letterboxResult, candidates);
        } else {
            decodeLegacyGrid(output, outputShape, letterboxResult, candidates);
        }

        return NMS.apply(candidates, IOU_THRESHOLD);
    }

    private void decodeEndToEnd(float[][][] output, int numDetections, boolean columnsFirst,
                                 Letterbox.Result letterboxResult, List<Detection> candidates) {
        boolean normalized = areEndToEndCoordinatesNormalized(output, numDetections, columnsFirst);

        for (int d = 0; d < numDetections; d++) {
            float x1;
            float y1;
            float x2;
            float y2;
            float confidence;
            int classId;

            if (columnsFirst) {
                x1 = output[0][0][d];
                y1 = output[0][1][d];
                x2 = output[0][2][d];
                y2 = output[0][3][d];
                confidence = output[0][4][d];
                classId = Math.round(output[0][5][d]);
            } else {
                x1 = output[0][d][0];
                y1 = output[0][d][1];
                x2 = output[0][d][2];
                y2 = output[0][d][3];
                confidence = output[0][d][4];
                classId = Math.round(output[0][d][5]);
            }

            if (confidence < CONFIDENCE_THRESHOLD) {
                continue;
            }

            if (normalized) {
                x1 *= inputWidth;
                x2 *= inputWidth;
                y1 *= inputHeight;
                y2 *= inputHeight;
            }

            BoundingBox letterboxedBox = new BoundingBox(x1, y1, x2, y2);
            BoundingBox originalBox = ScaleBoxes.scaleToOriginal(letterboxedBox, letterboxResult);
            String label = (classId >= 0 && classId < LABELS.length) ? LABELS[classId] : String.valueOf(classId);
            candidates.add(new Detection(classId, label, confidence, originalBox));
        }
    }

    /**
     * Some end-to-end exports emit box corners normalized to [0, 1] instead of pixel
     * coordinates in the model's input space; scan every detection slot's x2/y2 once to tell
     * which convention this particular model uses, since both are common in the wild.
     */
    private boolean areEndToEndCoordinatesNormalized(float[][][] output, int numDetections, boolean columnsFirst) {
        float maxCoord = 0f;
        for (int d = 0; d < numDetections; d++) {
            float x2 = columnsFirst ? output[0][2][d] : output[0][d][2];
            float y2 = columnsFirst ? output[0][3][d] : output[0][d][3];
            maxCoord = Math.max(maxCoord, Math.max(x2, y2));
        }
        return maxCoord <= 1.5f;
    }

    private void decodeLegacyGrid(float[][][] output, int[] outputShape,
                                   Letterbox.Result letterboxResult, List<Detection> candidates) {
        // Raw Ultralytics grid output (no built-in NMS): [1, 4 + numClasses, numAnchors] or
        // [1, numAnchors, 4 + numClasses], requiring manual per-class argmax + NMS.
        boolean channelsFirst = outputShape[1] < outputShape[2];
        int channels = channelsFirst ? outputShape[1] : outputShape[2];
        int numAnchors = channelsFirst ? outputShape[2] : outputShape[1];
        int numClasses = channels - 4;

        for (int a = 0; a < numAnchors; a++) {
            float cx;
            float cy;
            float w;
            float h;
            int bestClass = -1;
            float bestScore = -1f;

            if (channelsFirst) {
                cx = output[0][0][a];
                cy = output[0][1][a];
                w = output[0][2][a];
                h = output[0][3][a];
                for (int c = 0; c < numClasses; c++) {
                    float score = output[0][4 + c][a];
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = c;
                    }
                }
            } else {
                cx = output[0][a][0];
                cy = output[0][a][1];
                w = output[0][a][2];
                h = output[0][a][3];
                for (int c = 0; c < numClasses; c++) {
                    float score = output[0][a][4 + c];
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = c;
                    }
                }
            }

            if (bestScore < CONFIDENCE_THRESHOLD) {
                continue;
            }

            BoundingBox letterboxedBox = BoundingBox.fromCenterWidthHeight(cx, cy, w, h);
            BoundingBox originalBox = ScaleBoxes.scaleToOriginal(letterboxedBox, letterboxResult);
            String label = (bestClass >= 0 && bestClass < LABELS.length) ? LABELS[bestClass] : String.valueOf(bestClass);
            candidates.add(new Detection(bestClass, label, bestScore, originalBox));
        }
    }

    public Detection findBestReadingDetection(List<Detection> detections) {
        Detection best = null;
        for (Detection detection : detections) {
            if (!READING_LABEL.equals(detection.label)) {
                continue;
            }
            if (best == null || detection.confidence > best.confidence) {
                best = detection;
            }
        }
        return best;
    }

    @Override
    public void close() {
        interpreter.close();
    }
}
