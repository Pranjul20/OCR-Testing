package com.example.myapplication.utils;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

public final class Letterbox {

    private Letterbox() {
    }

    public static final class Result {
        public final Bitmap bitmap;
        public final float scale;
        public final float padLeft;
        public final float padTop;
        public final int originalWidth;
        public final int originalHeight;

        public Result(Bitmap bitmap, float scale, float padLeft, float padTop, int originalWidth, int originalHeight) {
            this.bitmap = bitmap;
            this.scale = scale;
            this.padLeft = padLeft;
            this.padTop = padTop;
            this.originalWidth = originalWidth;
            this.originalHeight = originalHeight;
        }
    }

    /**
     * Reproduces Ultralytics' LetterBox transform (auto=False, scaleFill=False, scaleup=True, center=True).
     */
    public static Result apply(Bitmap source, int targetWidth, int targetHeight) {
        int originalWidth = source.getWidth();
        int originalHeight = source.getHeight();

        float r = Math.min((float) targetHeight / (float) originalHeight, (float) targetWidth / (float) originalWidth);

        int newUnpadW = Math.round(originalWidth * r);
        int newUnpadH = Math.round(originalHeight * r);

        float dw = (targetWidth - newUnpadW) / 2f;
        float dh = (targetHeight - newUnpadH) / 2f;

        Bitmap resized = (newUnpadW != originalWidth || newUnpadH != originalHeight)
                ? Bitmap.createScaledBitmap(source, newUnpadW, newUnpadH, true)
                : source;

        int top = Math.round(dh - 0.1f);
        int bottom = Math.round(dh + 0.1f);
        int left = Math.round(dw - 0.1f);
        int right = Math.round(dw + 0.1f);

        int paddedWidth = newUnpadW + left + right;
        int paddedHeight = newUnpadH + top + bottom;

        Bitmap output = Bitmap.createBitmap(Math.max(paddedWidth, targetWidth), Math.max(paddedHeight, targetHeight), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        canvas.drawColor(Color.rgb(114, 114, 114));
        canvas.drawBitmap(resized, left, top, new Paint(Paint.FILTER_BITMAP_FLAG));

        Bitmap finalBitmap = output;
        if (output.getWidth() != targetWidth || output.getHeight() != targetHeight) {
            finalBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888);
            Canvas finalCanvas = new Canvas(finalBitmap);
            finalCanvas.drawColor(Color.rgb(114, 114, 114));
            finalCanvas.drawBitmap(output, 0, 0, null);
        }

        return new Result(finalBitmap, r, left, top, originalWidth, originalHeight);
    }
}
