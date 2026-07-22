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

    public static Bitmap rotateBitmap(Bitmap source, float degrees) {
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

    private static final double CLAHE_CLIP_LIMIT = 3.0;
    private static final int CLAHE_TILE_GRID_SIZE = 8;
    private static final double UNSHARP_BLUR_SIGMA = 3.0;
    private static final double UNSHARP_IMAGE_WEIGHT = 2.2;
    private static final double UNSHARP_BLUR_WEIGHT = -1.2;
    private static final int BILATERAL_DIAMETER = 9;
    private static final double BILATERAL_SIGMA_COLOR = 75.0;
    private static final double BILATERAL_SIGMA_SPACE = 75.0;
    private static final double CONTRAST_ALPHA = 1.6;
    private static final double CONTRAST_BETA = -20.0;

    /**
     * Grayscale + resize + min/max contrast stretch + CLAHE + unsharp mask + bilateral
     * denoise + a final linear contrast boost — matching the reference OpenCV LCD
     * preprocessing pipeline. Optionally inverts after CLAHE for LCDs with light digits on a
     * dark background. Returned bitmap is still ARGB_8888 (R=G=B per pixel) so it can feed
     * straight into {@link #bitmapToNormalizedFloatBuffer(Bitmap)}/NCHW unchanged.
     */
    public static Bitmap preprocessLcd(Bitmap source, int targetWidth, int targetHeight, boolean invertDigits) {
        Bitmap resized = resizeBitmap(source, targetWidth, targetHeight);
        int width = resized.getWidth();
        int height = resized.getHeight();
        int[] pixels = new int[width * height];
        resized.getPixels(pixels, 0, width, 0, 0, width, height);

        int[] gray = new int[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            gray[i] = Math.round(0.299f * r + 0.587f * g + 0.114f * b);
        }

        int[] stretched = normalizeMinMax(gray);
        int[] equalized = applyClahe(stretched, width, height, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE);

        if (invertDigits) {
            for (int i = 0; i < equalized.length; i++) {
                equalized[i] = 255 - equalized[i];
            }
        }

        int[] sharpened = unsharpMask(equalized, width, height, UNSHARP_BLUR_SIGMA, UNSHARP_IMAGE_WEIGHT, UNSHARP_BLUR_WEIGHT);
        int[] denoised = bilateralFilter(sharpened, width, height, BILATERAL_DIAMETER, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE);
        int[] contrasted = convertScaleAbs(denoised, CONTRAST_ALPHA, CONTRAST_BETA);

        int[] output = new int[pixels.length];
        for (int i = 0; i < output.length; i++) {
            int v = contrasted[i];
            output[i] = 0xFF000000 | (v << 16) | (v << 8) | v;
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, width, 0, 0, width, height);
        return result;
    }

    /** Linear contrast stretch, matching cv2.normalize(..., 0, 255, NORM_MINMAX): remaps [min, max] to [0, 255]. */
    private static int[] normalizeMinMax(int[] gray) {
        int min = 255;
        int max = 0;
        for (int value : gray) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        int range = max - min;
        if (range == 0) {
            return gray.clone();
        }

        int[] output = new int[gray.length];
        for (int i = 0; i < gray.length; i++) {
            output[i] = Math.round((gray[i] - min) * 255f / range);
        }
        return output;
    }

    /**
     * Contrast Limited Adaptive Histogram Equalization: builds a clipped-histogram mapping
     * per tile, then bilinearly interpolates between the four nearest tile mappings for each
     * pixel so tile boundaries don't produce visible seams.
     */
    private static int[] applyClahe(int[] gray, int width, int height, double clipLimit, int tilesX, int tilesY) {
        int[] tileStartX = new int[tilesX + 1];
        int[] tileStartY = new int[tilesY + 1];
        for (int i = 0; i <= tilesX; i++) {
            tileStartX[i] = i * width / tilesX;
        }
        for (int j = 0; j <= tilesY; j++) {
            tileStartY[j] = j * height / tilesY;
        }

        int[][] tileMappings = new int[tilesX * tilesY][256];
        for (int ty = 0; ty < tilesY; ty++) {
            for (int tx = 0; tx < tilesX; tx++) {
                int x0 = tileStartX[tx];
                int x1 = tileStartX[tx + 1];
                int y0 = tileStartY[ty];
                int y1 = tileStartY[ty + 1];
                int tileSize = (x1 - x0) * (y1 - y0);

                int[] histogram = new int[256];
                for (int y = y0; y < y1; y++) {
                    int rowOffset = y * width;
                    for (int x = x0; x < x1; x++) {
                        histogram[gray[rowOffset + x]]++;
                    }
                }

                int clipThreshold = Math.max(1, (int) Math.round(clipLimit * tileSize / 256.0));
                int excess = 0;
                for (int i = 0; i < 256; i++) {
                    if (histogram[i] > clipThreshold) {
                        excess += histogram[i] - clipThreshold;
                        histogram[i] = clipThreshold;
                    }
                }
                int redistribute = excess / 256;
                int remainder = excess % 256;
                for (int i = 0; i < 256; i++) {
                    histogram[i] += redistribute + (i < remainder ? 1 : 0);
                }

                int[] mapping = tileMappings[ty * tilesX + tx];
                int cumulative = 0;
                for (int i = 0; i < 256; i++) {
                    cumulative += histogram[i];
                    mapping[i] = Math.round(cumulative * 255f / tileSize);
                }
            }
        }

        double[] centerX = new double[tilesX];
        double[] centerY = new double[tilesY];
        for (int tx = 0; tx < tilesX; tx++) {
            centerX[tx] = (tileStartX[tx] + tileStartX[tx + 1]) / 2.0;
        }
        for (int ty = 0; ty < tilesY; ty++) {
            centerY[ty] = (tileStartY[ty] + tileStartY[ty + 1]) / 2.0;
        }

        int[] output = new int[gray.length];
        for (int y = 0; y < height; y++) {
            int ty0 = findLowerTileIndex(centerY, y, tilesY);
            int ty1 = Math.min(ty0 + 1, tilesY - 1);
            double fy = ty1 == ty0 ? 0 : clamp01((y - centerY[ty0]) / (centerY[ty1] - centerY[ty0]));

            for (int x = 0; x < width; x++) {
                int tx0 = findLowerTileIndex(centerX, x, tilesX);
                int tx1 = Math.min(tx0 + 1, tilesX - 1);
                double fx = tx1 == tx0 ? 0 : clamp01((x - centerX[tx0]) / (centerX[tx1] - centerX[tx0]));

                int value = gray[y * width + x];
                int v00 = tileMappings[ty0 * tilesX + tx0][value];
                int v01 = tileMappings[ty0 * tilesX + tx1][value];
                int v10 = tileMappings[ty1 * tilesX + tx0][value];
                int v11 = tileMappings[ty1 * tilesX + tx1][value];

                double top = v00 * (1 - fx) + v01 * fx;
                double bottom = v10 * (1 - fx) + v11 * fx;
                output[y * width + x] = (int) Math.round(top * (1 - fy) + bottom * fy);
            }
        }
        return output;
    }

    private static int findLowerTileIndex(double[] centers, int coord, int count) {
        if (coord <= centers[0]) {
            return 0;
        }
        if (coord >= centers[count - 1]) {
            return count - 1;
        }
        for (int i = 0; i < count - 1; i++) {
            if (coord >= centers[i] && coord < centers[i + 1]) {
                return i;
            }
        }
        return count - 1;
    }

    private static double clamp01(double value) {
        return Math.max(0, Math.min(1, value));
    }

    /**
     * Unsharp mask, matching cv2.GaussianBlur(sigma) + cv2.addWeighted(img, imageWeight,
     * blur, blurWeight, 0): blurs the image, then pushes each pixel away from its blurred
     * (low-frequency) version to exaggerate edges.
     */
    private static int[] unsharpMask(int[] gray, int width, int height, double blurSigma, double imageWeight, double blurWeight) {
        int[] blurred = gaussianBlur(gray, width, height, blurSigma);
        int[] output = new int[gray.length];
        for (int i = 0; i < gray.length; i++) {
            double value = gray[i] * imageWeight + blurred[i] * blurWeight;
            output[i] = Math.max(0, Math.min(255, (int) Math.round(value)));
        }
        return output;
    }

    /** Separable Gaussian blur with OpenCV's auto kernel size for ksize=(0,0): round(sigma*6+1)|1. */
    private static int[] gaussianBlur(int[] gray, int width, int height, double sigma) {
        int kernelSize = ((int) Math.round(sigma * 6 + 1)) | 1;
        double[] kernel = buildGaussianKernel1D(kernelSize, sigma);
        int radius = kernelSize / 2;

        int[] horizontal = new int[gray.length];
        for (int y = 0; y < height; y++) {
            int rowOffset = y * width;
            for (int x = 0; x < width; x++) {
                double sum = 0;
                for (int k = -radius; k <= radius; k++) {
                    int sx = Math.max(0, Math.min(width - 1, x + k));
                    sum += gray[rowOffset + sx] * kernel[k + radius];
                }
                horizontal[rowOffset + x] = (int) Math.round(sum);
            }
        }

        int[] output = new int[gray.length];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0;
                for (int k = -radius; k <= radius; k++) {
                    int sy = Math.max(0, Math.min(height - 1, y + k));
                    sum += horizontal[sy * width + x] * kernel[k + radius];
                }
                output[y * width + x] = Math.max(0, Math.min(255, (int) Math.round(sum)));
            }
        }
        return output;
    }

    private static double[] buildGaussianKernel1D(int size, double sigma) {
        double[] kernel = new double[size];
        int radius = size / 2;
        double sum = 0;
        for (int i = -radius; i <= radius; i++) {
            double value = Math.exp(-(i * i) / (2 * sigma * sigma));
            kernel[i + radius] = value;
            sum += value;
        }
        for (int i = 0; i < size; i++) {
            kernel[i] /= sum;
        }
        return kernel;
    }

    /**
     * Bilateral filter, matching cv2.bilateralFilter: smooths within a diameter x diameter
     * window, weighting each neighbor by both spatial distance and intensity difference so
     * edges (digit strokes) are preserved while flatter regions get denoised.
     */
    private static int[] bilateralFilter(int[] gray, int width, int height, int diameter, double sigmaColor, double sigmaSpace) {
        int radius = diameter / 2;
        double[] spatialWeights = new double[diameter * diameter];
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                double distSq = (double) dx * dx + (double) dy * dy;
                spatialWeights[(dy + radius) * diameter + (dx + radius)] = Math.exp(-distSq / (2 * sigmaSpace * sigmaSpace));
            }
        }

        int[] output = new int[gray.length];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int center = gray[y * width + x];
                double weightSum = 0;
                double valueSum = 0;

                for (int dy = -radius; dy <= radius; dy++) {
                    int ny = Math.max(0, Math.min(height - 1, y + dy));
                    for (int dx = -radius; dx <= radius; dx++) {
                        int nx = Math.max(0, Math.min(width - 1, x + dx));
                        int neighbor = gray[ny * width + nx];
                        double colorDiff = neighbor - center;
                        double weight = spatialWeights[(dy + radius) * diameter + (dx + radius)]
                                * Math.exp(-(colorDiff * colorDiff) / (2 * sigmaColor * sigmaColor));
                        weightSum += weight;
                        valueSum += weight * neighbor;
                    }
                }
                output[y * width + x] = (int) Math.round(valueSum / weightSum);
            }
        }
        return output;
    }

    /** Linear contrast boost with saturation, matching cv2.convertScaleAbs(alpha, beta): |v*alpha + beta|, clamped to [0, 255]. */
    private static int[] convertScaleAbs(int[] gray, double alpha, double beta) {
        int[] output = new int[gray.length];
        for (int i = 0; i < gray.length; i++) {
            double value = Math.abs(gray[i] * alpha + beta);
            output[i] = Math.max(0, Math.min(255, (int) Math.round(value)));
        }
        return output;
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
