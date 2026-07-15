package com.example.myapplication.utils;

import com.example.myapplication.model.BoundingBox;

public final class ScaleBoxes {

    private ScaleBoxes() {
    }

    /**
     * Reverses Letterbox.apply(): maps a box from the padded/scaled 640x640 (or NxN) space
     * back to the coordinate system of the original, un-letterboxed image.
     */
    public static BoundingBox scaleToOriginal(BoundingBox letterboxedBox, Letterbox.Result meta) {
        float x1 = (letterboxedBox.x1 - meta.padLeft) / meta.scale;
        float y1 = (letterboxedBox.y1 - meta.padTop) / meta.scale;
        float x2 = (letterboxedBox.x2 - meta.padLeft) / meta.scale;
        float y2 = (letterboxedBox.y2 - meta.padTop) / meta.scale;

        BoundingBox box = new BoundingBox(x1, y1, x2, y2);
        return box.clip(meta.originalWidth, meta.originalHeight);
    }
}
