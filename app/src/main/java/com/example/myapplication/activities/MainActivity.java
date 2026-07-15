package com.example.myapplication.activities;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.example.myapplication.R;
import com.example.myapplication.ml.ObjectDetector;
import com.example.myapplication.ml.OCRDetector;
import com.example.myapplication.model.Detection;
import com.example.myapplication.utils.DigitSorter;
import com.example.myapplication.utils.ImageUtils;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "TemperatureOCR";
    private static final String OBJECT_MODEL_FILE = "best_float16.tflite";
    private static final String OCR_MODEL_FILE = "latest_ocr_best.tflite";

    private ImageView imgOriginal;
    private ImageView imgCropped;
    private ImageView imgOcr;
    private TextView txtResult;
    private ProgressBar progressBar;
    private Button btnCamera;
    private Button btnGallery;
    private Button btnCapture;
    private Button btnCancelCamera;
    private FrameLayout cameraContainer;
    private PreviewView previewView;

    private ObjectDetector objectDetector;
    private OCRDetector ocrDetector;
    private ExecutorService inferenceExecutor;
    private ExecutorService cameraExecutor;

    private ProcessCameraProvider cameraProvider;
    private ImageCapture imageCapture;

    private ActivityResultLauncher<String> cameraPermissionLauncher;
    private ActivityResultLauncher<String> galleryLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bindViews();
        initExecutorsAndModels();
        initLaunchers();
        initListeners();
    }

    private void bindViews() {
        imgOriginal = findViewById(R.id.imgOriginal);
        imgCropped = findViewById(R.id.imgCropped);
        imgOcr = findViewById(R.id.imgOcr);
        txtResult = findViewById(R.id.txtResult);
        progressBar = findViewById(R.id.progressBar);
        btnCamera = findViewById(R.id.btnCamera);
        btnGallery = findViewById(R.id.btnGallery);
        btnCapture = findViewById(R.id.btnCapture);
        btnCancelCamera = findViewById(R.id.btnCancelCamera);
        cameraContainer = findViewById(R.id.cameraContainer);
        previewView = findViewById(R.id.previewView);
    }

    private void initExecutorsAndModels() {
        inferenceExecutor = Executors.newSingleThreadExecutor();
        cameraExecutor = Executors.newSingleThreadExecutor();
        try {
            objectDetector = new ObjectDetector(this, OBJECT_MODEL_FILE);
            ocrDetector = new OCRDetector(this, OCR_MODEL_FILE);
        } catch (IOException e) {
            Log.e(TAG, "Failed to load TFLite models", e);
            Toast.makeText(this, "Failed to load models: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void initLaunchers() {
        cameraPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                granted -> {
                    if (granted) {
                        openCamera();
                    } else {
                        Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show();
                    }
                });

        galleryLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        handleImage(uri);
                    }
                });
    }

    private void initListeners() {
        btnCamera.setOnClickListener(v -> requestCameraAndOpen());
        btnGallery.setOnClickListener(v -> galleryLauncher.launch("image/*"));
        btnCapture.setOnClickListener(v -> capturePhoto());
        btnCancelCamera.setOnClickListener(v -> closeCamera());
    }

    private void requestCameraAndOpen() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void openCamera() {
        cameraContainer.setVisibility(View.VISIBLE);
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Failed to start camera", e);
                Toast.makeText(this, "Unable to start camera", Toast.LENGTH_SHORT).show();
                cameraContainer.setVisibility(View.GONE);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build();

        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);
    }

    private void capturePhoto() {
        if (imageCapture == null) {
            return;
        }

        File outputFile;
        try {
            outputFile = File.createTempFile("capture_", ".jpg", getCacheDir());
        } catch (IOException e) {
            Log.e(TAG, "Failed to create temp capture file", e);
            Toast.makeText(this, "Failed to prepare capture", Toast.LENGTH_SHORT).show();
            return;
        }

        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(outputFile).build();

        imageCapture.takePicture(outputOptions, cameraExecutor, new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                Uri capturedUri = Uri.fromFile(outputFile);
                runOnUiThread(() -> {
                    closeCamera();
                    handleImage(capturedUri);
                });
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Capture failed", exception);
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Capture failed", Toast.LENGTH_SHORT).show());
            }
        });
    }

    private void closeCamera() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        cameraContainer.setVisibility(View.GONE);
    }

    private void handleImage(Uri uri) {
        setLoading(true);
        inferenceExecutor.execute(() -> {
            try {
                Bitmap originalBitmap = ImageUtils.loadBitmapFromUri(this, uri);
                runPipeline(originalBitmap);
            } catch (IOException e) {
                Log.e(TAG, "Failed to load image", e);
                runOnUiThread(() -> {
                    setLoading(false);
                    Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                });
            }
        });
    }

    private void runPipeline(Bitmap originalBitmap) {
        List<Detection> detections = objectDetector.detect(originalBitmap);
        Detection bestReading = objectDetector.findBestReadingDetection(detections);
        Bitmap boxedBitmap = ImageUtils.drawDetections(originalBitmap, detections);

        if (bestReading == null) {
            runOnUiThread(() -> {
                setLoading(false);
                imgOriginal.setImageBitmap(boxedBitmap);
                imgCropped.setImageBitmap(null);
                imgOcr.setImageBitmap(null);
                txtResult.setText(R.string.result_no_reading);
            });
            return;
        }

        Bitmap croppedBitmap = ImageUtils.cropBitmap(originalBitmap, bestReading.box);
        Bitmap blackAndWhiteBitmap = ImageUtils.toBlackAndWhite(croppedBitmap);
        Bitmap resizedBitmap = ImageUtils.resizeBitmap(blackAndWhiteBitmap, ocrDetector.getInputWidth(), ocrDetector.getInputHeight());

        List<Detection> digitDetections = ocrDetector.detect(resizedBitmap);
        Bitmap ocrBoxedBitmap = ImageUtils.drawDetections(resizedBitmap, digitDetections);
        String temperature = DigitSorter.buildTemperatureString(digitDetections);

        runOnUiThread(() -> {
            setLoading(false);
            imgOriginal.setImageBitmap(boxedBitmap);
            imgCropped.setImageBitmap(croppedBitmap);
            imgOcr.setImageBitmap(ocrBoxedBitmap);
            txtResult.setText(temperature.isEmpty() ? getString(R.string.result_no_digits) : temperature);
        });
    }

    private void setLoading(boolean loading) {
        runOnUiThread(() -> progressBar.setVisibility(loading ? View.VISIBLE : View.GONE));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (objectDetector != null) {
            objectDetector.close();
        }
        if (ocrDetector != null) {
            ocrDetector.close();
        }
        inferenceExecutor.shutdown();
        cameraExecutor.shutdown();
    }
}
