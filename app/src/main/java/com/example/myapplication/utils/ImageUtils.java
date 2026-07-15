package com.example.myapplication.utils;

import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.ExifInterface;
import android.net.Uri;

import com.example.myapplication.model.BoundingBox;
import com.example.myapplication.model.Detection;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Locale;

public final class ImageUtils {

    private ImageUtils() {
    }

    public static Bitmap loadBitmapFromUri(Context context, Uri uri) throws IOException {
        ContentResolver resolver = context.getContentResolver();
        Bitmap bitmap;
        try (InputStream input = resolver.openInputStream(uri)) {
            bitmap = BitmapFactory.decodeStream(input);
        }
        if (bitmap == null) {
            throw new IOException("Unable to decode bitmap from uri: " + uri);
        }

        int rotation = readExifRotationDegrees(context, uri);
        if (rotation != 0) {
            bitmap = rotateBitmap(bitmap, rotation);
        }
        if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        }
        return bitmap;
    }

    private static int readExifRotationDegrees(Context context, Uri uri) {
        try (InputStream input = context.getContentResolver().openInputStream(uri)) {
            if (input == null) {
                return 0;
            }
            ExifInterface exif = new ExifInterface(input);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    return 90;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    return 180;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    return 270;
                default:
                    return 0;
            }
        } catch (IOException e) {
            return 0;
        }
    }

    public static Bitmap rotateBitmap(Bitmap source, int degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    public static Bitmap resizeBitmap(Bitmap source, int width, int height) {
        return Bitmap.createScaledBitmap(source, width, height, true);
    }

    public static Bitmap cropBitmap(Bitmap source, BoundingBox box) {
        int left = Math.max(0, Math.round(box.x1));
        int top = Math.max(0, Math.round(box.y1));
        int right = Math.min(source.getWidth(), Math.round(box.x2));
        int bottom = Math.min(source.getHeight(), Math.round(box.y2));

        int width = Math.max(1, right - left);
        int height = Math.max(1, bottom - top);
        if (left + width > source.getWidth()) {
            width = source.getWidth() - left;
        }
        if (top + height > source.getHeight()) {
            height = source.getHeight() - top;
        }
        return Bitmap.createBitmap(source, left, top, width, height);
    }

    /**
     * Converts an already-square bitmap (e.g. a 640x640 letterboxed frame or a 512x512 OCR
     * crop) into an NHWC, RGB, float32 ByteBuffer normalized to [0, 1] — matching the
     * Ultralytics preprocessing pipeline (no mean/std subtraction).
     */
    public static ByteBuffer bitmapToNormalizedFloatBuffer(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * width * height * 3);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int pixel : pixels) {
            buffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f);
            buffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);
            buffer.putFloat((pixel & 0xFF) / 255.0f);
        }
        buffer.rewind();
        return buffer;
    }

    /**
     * Same normalization as {@link #bitmapToNormalizedFloatBuffer(Bitmap)}, but written out
     * as three contiguous per-channel planes (R plane, then G plane, then B plane) instead of
     * interleaved per-pixel — for models whose input tensor is NCHW ([1, 3, H, W]) rather
     * than the more common NHWC ([1, H, W, 3]).
     */
    public static ByteBuffer bitmapToNormalizedFloatBufferNCHW(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * width * height * 3);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int pixel : pixels) {
            buffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f);
        }
        for (int pixel : pixels) {
            buffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);
        }
        for (int pixel : pixels) {
            buffer.putFloat((pixel & 0xFF) / 255.0f);
        }
        buffer.rewind();
        return buffer;
    }

    /**
     * Grayscales and binarizes an image the way Tesseract-style OCR preprocessing does:
     * luminosity grayscale, then a global Otsu threshold splitting every pixel to pure
     * black or pure white. Returned bitmap is still ARGB_8888 (R=G=B per pixel) so it can
     * feed straight into {@link #bitmapToNormalizedFloatBuffer(Bitmap)}/NCHW unchanged.
     */
    public static Bitmap toBlackAndWhite(Bitmap source) {
        int width = source.getWidth();
        int height = source.getHeight();
        int[] pixels = new int[width * height];
        source.getPixels(pixels, 0, width, 0, 0, width, height);

        int[] gray = new int[pixels.length];
        int[] histogram = new int[256];
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            int value = Math.round(0.299f * r + 0.587f * g + 0.114f * b);
            gray[i] = value;
            histogram[value]++;
        }

        int threshold = otsuThreshold(histogram, pixels.length);

        int[] output = new int[pixels.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = gray[i] >= threshold ? 0xFFFFFFFF : 0xFF000000;
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, width, 0, 0, width, height);
        return result;
    }

    private static int otsuThreshold(int[] histogram, int totalPixels) {
        long sum = 0;
        for (int t = 0; t < 256; t++) {
            sum += (long) t * histogram[t];
        }

        long sumBackground = 0;
        int weightBackground = 0;
        double maxVariance = 0;
        int threshold = 0;

        for (int t = 0; t < 256; t++) {
            weightBackground += histogram[t];
            if (weightBackground == 0) {
                continue;
            }
            int weightForeground = totalPixels - weightBackground;
            if (weightForeground == 0) {
                break;
            }

            sumBackground += (long) t * histogram[t];

            double meanBackground = (double) sumBackground / weightBackground;
            double meanForeground = (double) (sum - sumBackground) / weightForeground;
            double meanDiff = meanBackground - meanForeground;
            double betweenClassVariance = (double) weightBackground * weightForeground * meanDiff * meanDiff;

            if (betweenClassVariance > maxVariance) {
                maxVariance = betweenClassVariance;
                threshold = t;
            }
        }
        return threshold;
    }

    public static Bitmap drawDetections(Bitmap source, List<Detection> detections) {
        Bitmap output = source.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(output);

        Paint boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(Math.max(3f, output.getWidth() / 200f));
        boxPaint.setColor(Color.RED);
        boxPaint.setAntiAlias(true);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(Math.max(24f, output.getWidth() / 25f));
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setAntiAlias(true);

        for (Detection detection : detections) {
            BoundingBox box = detection.box;
            canvas.drawRect(box.x1, box.y1, box.x2, box.y2, boxPaint);
            String label = String.format(Locale.US, "%s %.2f", detection.label, detection.confidence);
            canvas.drawText(label, box.x1, Math.max(textPaint.getTextSize(), box.y1 - 8f), textPaint);
        }
        return output;
    }
}
